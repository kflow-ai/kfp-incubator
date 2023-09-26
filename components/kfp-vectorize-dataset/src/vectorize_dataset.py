import itertools
import logging
from os import path
from tempfile import TemporaryDirectory
from typing import Iterable, List, Sequence, TypeVar

import fsspec
import llama_index.vector_stores
import ray
from fsspec.asyn import AsyncFileSystem
from kfp import compiler, dsl
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.data_structs import IndexDict
from llama_index.llms import MockLLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import BaseNode

## The default concurrency for the number of concurrent
## ray tasks
DEFAULT_CONCURRENCY = 150

## The largest number of tasks we'll wait for at a time
READY_BATCH_SIZE = 10


T = TypeVar("T")

## Custom embeddings are currently not supported
## We use the BGE embeddings for now since they're at
## the top of the leaderboards.
EMBEDDING = "BAAI/bge-base-en"


logging.basicConfig(level=logging.INFO)


def partition(lst: Sequence[T], size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def download_files(root_uri: str, files: List[str], targetDir: str):
    logging.info(f"Downloading files to {targetDir}")
    fs = fsspec.filesystem(root_uri.split("://", 1)[0])
    fs.get([path.join(root_uri, f) for f in files], targetDir)


def vectorize_fileset(root_uri: str, files: List[str]):
    logging.basicConfig(level=logging.INFO)
    with TemporaryDirectory() as temp_dir:
        download_files(root_uri, files, temp_dir)

        embedding = HuggingFaceBgeEmbeddings(model_name=EMBEDDING)

        node_parser = SimpleNodeParser.from_defaults()
        docs = SimpleDirectoryReader(temp_dir, recursive=True).load_data()

        logging.info(f"Vectorizing {len(docs)} documents")

        logging.info("Parsing nodes from documents.")
        nodes = node_parser.get_nodes_from_documents(docs, show_progress=True)

        logging.info(f"Embedding {len(nodes)} nodes")
        for idx, node in enumerate(nodes):
            nodes[idx].embedding = embedding.embed_documents([node.text])[0]

    return nodes


def persist_nodes(nodes: List[BaseNode], vectordb_cls: str, vectordb_kwargs: dict):
    if vectordb_cls is None:
        logging.warn("Unable to persist nodes, there is no vector store specified")
        return

    if len(nodes) == 0:
        return

    cls = getattr(llama_index.vector_stores, vectordb_cls)

    vectordb_kwargs["dim"] = len(nodes[0].embedding)
    vector_store = cls(**vectordb_kwargs)

    service_context = ServiceContext.from_defaults(
        llm=MockLLM(), embed_model=HuggingFaceBgeEmbeddings(model_name=EMBEDDING)
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_store_index = VectorStoreIndex(
        storage_context=storage_context,
        index_struct=IndexDict(),
        service_context=service_context,
    )
    logging.info(f"Persisting {len(nodes)} nodes to vector store")
    vector_store_index.insert_nodes(nodes)


def first(iter: Iterable, n=1):
    return [next(iter) for _ in range(n)]


def ray_vectorize_dataset(
    ray_address: str,
    root_uri: str,
    files: List[str],
    batch_size=1000,
    vectordb_cls: str = None,
    vectordb_kwargs: dict = None,
    concurrency: int = DEFAULT_CONCURRENCY,
):
    with open(
        path.join(path.dirname(__file__), "runtime-requirements.txt")
    ) as requirements_file:
        runtime_requirements = requirements_file.read().splitlines()

    # Filter out ray so that we don't overwrite the ray on the ray workers,
    # but we do need ray on the kubeflow worker (and we don't want to have to
    # maintain a separate requirements list)
    requirements = list(filter(lambda x: not "ray" in x, runtime_requirements))

    runtime_env = {"pip": requirements}
    ray.init(address=ray_address, runtime_env=runtime_env)

    ##  Make remote versions of the functions we'll need
    remote_vectorize_fileset = ray.remote(vectorize_fileset)

    n = first(partition(files, size=batch_size), 2)

    ## Partition the file lists into batches and submit them to ray
    result_refs = []

    for p in first(partition(files, size=batch_size), 3):
        results = None
        if len(result_refs) >= concurrency:
            ready_refs, result_refs = ray.wait(
                result_refs, num_returns=min(READY_BATCH_SIZE, len(result_refs))
            )
            results = ray.get(ready_refs)

        result_refs.append(remote_vectorize_fileset.remote(root_uri, p))
        if results:
            persist_nodes(
                list(itertools.chain(*results)),
                vectordb_cls=vectordb_cls,
                vectordb_kwargs=vectordb_kwargs,
            )

    while result_refs:
        ready_refs, result_refs = ray.wait(
            result_refs, num_returns=min(READY_BATCH_SIZE, len(result_refs))
        )
        results = ray.get(ready_refs)
        persist_nodes(
            list(itertools.chain(*results)),
            vectordb_cls=vectordb_cls,
            vectordb_kwargs=vectordb_kwargs,
        )


def get_fs(url: str) -> AsyncFileSystem:
    return fsspec.filesystem(url.split("://", 1)[0])


def url_as_path(url: str) -> str:
    """Converts a URL to a path."""
    return url.split("://", 1)[-1]


@dsl.component(
    target_image="us-central1-docker.pkg.dev/kflow-artifacts/kfp-components/kfp-vectorize-dataset:latest",
    base_image="python:3.11-slim",
    packages_to_install=[
        "gcsfs~=2023.9",
        "s3fs~=2023.9",
        "fsspec~=2023.9",
        "ray[client]~=2.7",
        "llama_index~=0.8.29",
        "langchain~=0.0.298",
        "sentence-transformers~=2.2",
        "pymilvus~=2.3",
    ],
)
def vectorize_dataset(
    dataset_url: str,
    vectordb_cls: str,
    vectordb_kwargs: dict,
    ray_address: str,
    batch_size: int,
    concurrency: int,
):
    """
    Vectorizes each file ina  dataset and persists them to a datastore

    If `ray_address` is provided, then the component will use ray tasks to vectorize batches
    of the files in parallel. Otherwise, it will vectorize the files sequentially.

    Args:
        dataset_url: The URL of the dataset to vectorize. This should be a directory of separate documents.
            All files in the directory and any subdirectory will be vectorized. The URL should be in the form
            of a supported fsspec URL (e.g. `gs://` for Google Cloud Storage, `s3://` for S3, etc.)
        vectordb_cls: The class of the vector store to persist the vectors to. This should be a class from
            `llama_index.vector_stores`. If `None`, then the vectors will not be persisted.
        vectordb_kwargs: The keyword arguments to pass to the vector store class constructor.
        ray_address: The address of the ray cluster to use for parallelization. If `None`, then the files
            will be vectorized sequentially.
        batch_size: The number of files to vectorize in each batch. This is only used if `ray_address` is
            provided.
        concurrency: The maximum number of concurrent ray tasks to run. This is only used if `ray_address`
            is provided.
    """
    fs = get_fs(dataset_url)
    dataset_path = url_as_path(dataset_url)

    all_files = list(
        itertools.chain(
            *[
                [path.join(dirpath.replace(dataset_path, ""), f) for f in files]
                for dirpath, _, files in fs.walk(dataset_path)
            ]
        )
    )

    if ray_address is not None:
        ray_vectorize_dataset(
            ray_address,
            dataset_url,
            all_files,
            vectordb_cls=vectordb_cls,
            vectordb_kwargs=vectordb_kwargs,
            batch_size=batch_size,
            concurrency=concurrency,
        )
    else:
        nodes = vectorize_fileset(dataset_url, all_files)
        persist_nodes(nodes, vectordb_cls=vectordb_cls, vectordb_kwargs=vectordb_kwargs)


if __name__ == "__main__":
    compiler.Compiler().compile(
        vectorize_dataset, path.join(path.dirname(__file__), "..", "component.yaml")
    )
