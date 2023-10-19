import logging
from os import path
from tempfile import TemporaryDirectory
from typing import List

import fsspec
import nltk
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

## Custom embeddings are currently not supported
## We use the BGE embeddings for now since they're at
## the top of the leaderboards.
EMBEDDING = "BAAI/bge-base-en"


def download_files(root_uri: str, files: List[str], targetDir: str):
    logging.info(f"Downloading files to {targetDir}")
    fs = fsspec.filesystem(root_uri.split("://", 1)[0])
    fs.get([path.join(root_uri, f) for f in files], targetDir)


def vectorize_fileset(root_uri: str, files: List[str]):
    logging.basicConfig(level=logging.INFO)

    with TemporaryDirectory() as temp_dir, TemporaryDirectory() as nltk_temp_dir:
        nltk.data.path.append(nltk_temp_dir)
        nltk.download("punkt", download_dir=nltk_temp_dir)

        download_files(root_uri, files, temp_dir)

        embedding = HuggingFaceBgeEmbeddings(model_name=EMBEDDING)

        node_parser = SimpleNodeParser.from_defaults()
        docs = SimpleDirectoryReader(temp_dir, recursive=True).load_data()

        logging.info(f"Vectorizing {len(docs)} documents")

        logging.info("Parsing nodes from documents.")
        nodes = node_parser.get_nodes_from_documents(docs, show_progress=True)

        logging.info(f"Embedding {len(nodes)} nodes")
        for idx, embedding in enumerate(
            embedding.embed_documents([n.text for n in nodes])
        ):
            nodes[idx].embedding = embedding

    return nodes
