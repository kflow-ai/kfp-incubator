import itertools
import logging
from os import path
from typing import List, Sequence

import llama_index.vector_stores
import ray
from kfp import compiler, dsl
from langchain.embeddings.fake import FakeEmbeddings
from llama_index import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.data_structs import IndexDict
from llama_index.llms import MockLLM
import vectorize_fileset

logging.basicConfig(level=logging.INFO)


## The default concurrency for the number of concurrent
## ray tasks
DEFAULT_CPU_CONCURRENCY = 150

DEFAULT_GPU_CONCURRENCY = 10

## The largest number of tasks we'll wait for at a time
READY_BATCH_SIZE = 1


def get_fs(url: str):
    import fsspec

    return fsspec.filesystem(url.split("://", 1)[0])


def url_as_path(url: str) -> str:
    """Converts a URL to a path."""
    return url.split("://", 1)[-1]


def persist_nodes(nodes: List, vectordb_cls: str, vectordb_kwargs: dict):
    if vectordb_cls is None:
        logging.warn("Unable to persist nodes, there is no vector store specified")
        return

    if len(nodes) == 0:
        return

    cls = getattr(llama_index.vector_stores, vectordb_cls)

    vectordb_kwargs["dim"] = len(nodes[0].embedding)
    vector_store = cls(**vectordb_kwargs)

    service_context = ServiceContext.from_defaults(
        llm=MockLLM(), embed_model=FakeEmbeddings(size=len(nodes[0].embedding))
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_store_index = VectorStoreIndex(
        storage_context=storage_context,
        index_struct=IndexDict(),
        service_context=service_context,
    )
    logging.info(f"Persisting {len(nodes)} nodes to vector store")
    vector_store_index.insert_nodes(nodes)


def partition(lst: Sequence, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def ray_vectorize_dataset(
    ray_address: str,
    root_uri: str,
    files: List[str],
    batch_size=1000,
    vectordb_cls: str = None,
    vectordb_kwargs: dict = None,
    concurrency: int = None,
    use_gpu: bool = False,
):
    runtime_env = {
        "working_dir": ".",
        "py_modules": [vectorize_fileset],
        "conda": {
            "dependencies": [
                "pip",
                {
                    "pip": [
                        "gcsfs~=2023.9",
                        "s3fs~=2023.9",
                        "fsspec~=2023.9",
                        "llama_index~=0.8.29",
                        "langchain~=0.0.298",
                        "sentence-transformers~=2.2",
                        "nltk",
                    ]
                },
            ],
        },
    }
    ray.init(address=ray_address, runtime_env=runtime_env)

    num_cpus = 2 if not use_gpu else 1
    num_gpus = 1 if use_gpu else 0
    ##  Make remote versions of the functions we'll need
    remote_vectorize_fileset = ray.remote(vectorize_fileset.vectorize_fileset)
    remote_vectorize_fileset = remote_vectorize_fileset.options(
        num_cpus=num_cpus, num_gpus=num_gpus
    )

    if concurrency is None:
        concurrency = DEFAULT_GPU_CONCURRENCY if use_gpu else DEFAULT_CPU_CONCURRENCY

    ## Partition the file lists into batches and submit them to ray
    result_refs = []

    for p in partition(files, size=batch_size):
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


@dsl.component(
    target_image="us-central1-docker.pkg.dev/kflow-artifacts/kfp-components/kfp-vectorize-dataset:latest",
    base_image="python:3.10-slim",
    packages_to_install=[
        "ray[client]~=2.7",
        "gcsfs~=2023.9",
        "s3fs~=2023.9",
        "fsspec~=2023.9",
        "llama_index~=0.8.29",
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
    use_gpu: bool,
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
    dataset_path = dataset_path.rstrip("/") + "/"  ## Ensure the path ends with a slash

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
            cpuconcurrency=concurrency,
            use_gpu=use_gpu,
        )
    else:
        nodes = vectorize_fileset(dataset_url, all_files)
        persist_nodes(nodes, vectordb_cls=vectordb_cls, vectordb_kwargs=vectordb_kwargs)


if __name__ == "__main__":
    compiler.Compiler().compile(
        vectorize_dataset, path.join(path.dirname(__file__), "..", "component.yaml")
    )
