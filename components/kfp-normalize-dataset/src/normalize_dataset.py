import gzip
import logging
import shutil
import tarfile
import zipfile
from os import path
from tempfile import TemporaryDirectory, TemporaryFile, gettempdir

import magic
from fsspec.asyn import AsyncFileSystem
from gcsfs.core import GCSFileSystem
from kfp import compiler, dsl
from s3fs import S3FileSystem

logging.basicConfig(level=logging.INFO)


def get_fs(url: str) -> AsyncFileSystem:
    if url.startswith("s3"):
        return S3FileSystem()
    elif url.startswith("gs"):
        return GCSFileSystem()
    else:
        raise ValueError(f"Unsupported filesystem for url: {url}")


def url_as_path(url: str) -> str:
    """Converts a URL to a path."""
    return url.split("://", 1)[-1]


def is_dir(url: str) -> bool:
    """Checks if a URL is a directory."""
    return url.endswith("/")


def pull_dataset_file(source_url: str, target_file: str, fs: AsyncFileSystem) -> None:
    print("Pulling file from %s and storing in %s " % (source_url, target_file))
    path = url_as_path(source_url)
    ## If it's a directory, then we'll just leave the files where they are
    ## and let the next component handle them, otherwise, we'll extract
    ## the files from the dataset file and put them back into storage.
    fs.get(path, target_file, recursive=False)


def extract_gzip(target_file: str, target_dir: str) -> None:
    logging.info("Extracting gzip %s to %s" % (target_file, target_dir))
    file_type = magic.Magic(mime=True, uncompress=True).from_file(target_file)
    if file_type == "application/x-tar":
        logging.info("Gzip contains tar file, extracting tar file")
        with tarfile.open(target_file, "r:gz") as tar:
            tar.extractall(path=target_dir, filter="data")
    else:
        with TemporaryFile() as tempfile, gzip.open(target_file, "rb") as gz:
            shutil.copyfileobj(gz, tempfile)
            extract_dataset(target_fd=tempfile.fileno(), target_dir=target_dir)


def extract_zip(target_file: str, target_dir: str) -> None:
    logging.info("Extracting zip %s to %s" % (target_file, target_dir))

    with zipfile.ZipFile(target_file, "r") as zip:
        zip.extractall(path=target_dir)


def extract_from_text_file(target_file: str, target_dir: str) -> None:
    logging.info(f"Creating files from text file {target_file} to {target_dir}")
    ## If it's a text file, we'll split it on triple new lines as the document separator
    ## and store each document in a separate file in the target directory
    buffer = ""
    with open(target_file, "r") as input:
        read = input.read(4096)
        while read:
            page_separator_idx = read.find("\n\n\n")
            if page_separator_idx != -1:
                buffer += read[:page_separator_idx]
                with open(path.join(target_dir, f"{hash(buffer)}.txt"), "w") as output:
                    output.write(buffer)
                buffer = read[page_separator_idx + 3 :]
            else:
                buffer += read

            read = input.read(4096)

    if len(buffer) > 0:
        with open(path.join(target_dir, f"{hash(buffer)}.txt"), "w") as output:
            output.write(buffer)


def extract_dataset(
    target_dir: str, target_file: str = None, target_fd: int = None
) -> None:
    if target_file is not None:
        file_type = magic.from_file(target_file, mime=True)
    elif target_fd is not None:
        file_type = magic.from_descriptor(target_fd, mime=True)
    else:
        raise ValueError("Either target_file or target_fd must be specified")

    match file_type:
        case "application/gzip":
            ## gzip is a single file so we want to decompress it, then re-run the extraction process
            ## If the file inside the gzip is a tar, then just use tarfile to extract it, otherwise use gzip
            extract_gzip(target_file, target_dir)

        case "application/zip":
            ## zips are multiple files so we just want to decomress it to our target location
            extract_zip(target_file, target_dir)

        case "application/x-tar":
            ## tar files are multiple files so we just want to extract it to our target location
            logging.info(f"Extracting tar file {target_file} to {target_dir}")
            with tarfile.open(target_file, "r") as tar:
                tar.extractall(target_dir, filter="data")

        case "plain/text":
            extract_from_text_file(target_file, target_dir)

        case _:
            raise ValueError(f"Unsupported file type: {file_type}")


@dsl.component(
    target_image="us-central1-docker.pkg.dev/kflow-artifacts/kfp-components/kfp-normalize-dataset:latest",
    base_image="python:3.11-slim",
    packages_to_install=["gcsfs", "s3fs", "fsspec", "python-magic"],
)
def normalize_dataset(
    dataset_url: str, job_name: str, dataset_location: dsl.OutputPath(str)
):
    """
    Normalizes a platform ui dataset by pulling it from its storage location, extracts documents from the file either by
    uncompressing/untaring it or extracting it from a text file with triple new lines as the document separator, and
    storing it into a pipeline location in storage.
    """
    if not is_dir(dataset_url):
        filename = path.join(gettempdir(), dataset_url.split("/")[-1])
        fs = get_fs(dataset_url)
        pull_dataset_file(dataset_url, filename, fs)

        with TemporaryDirectory() as target_dir:
            logging.info(f"Extracting dataset from {dataset_url} to {target_dir}")
            extract_dataset(target_dir=target_dir, target_file=filename)

            [scheme, _, bucket, *_] = dataset_url.split("/")
            target_location = f"{scheme}//{bucket}/pipelines/{job_name}/"
            logging.info(f"Uploading dataset to {target_location}")
            fs.put(f"{target_dir}/", target_location, recursive=True)

        with open(dataset_location, "w") as output:
            output.write(target_location)
    else:
        with open(dataset_location, "w") as output:
            output.write(dataset_url)


if __name__ == "__main__":
    compiler.Compiler().compile(
        normalize_dataset, path.join(path.dirname(__file__), "..", "component.yaml")
    )
