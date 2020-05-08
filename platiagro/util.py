# -*- coding: utf-8 -*-
from os import getenv
from os.path import basename

from IPython.lib import kernel
from minio import Minio
from minio.error import BucketAlreadyOwnedByYou
from notebook.services.contents.filemanager import FileContentsManager
from requests import get
from s3fs.core import S3FileSystem

BUCKET_NAME = "anonymous"
MINIO_ENDPOINT = getenv("MINIO_ENDPOINT", "minio-service.kubeflow:9000")
MINIO_ACCESS_KEY = getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = getenv("MINIO_SECRET_KEY", "minio123")
JUPYTER_ENDPOINT = getenv("JUPYTER_ENDPOINT", "http://server.anonymous:80/notebook/anonymous/server")

MINIO_CLIENT = Minio(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

S3FS = S3FileSystem(
    key=MINIO_ACCESS_KEY,
    secret=MINIO_SECRET_KEY,
    use_ssl=False,
    client_kwargs={
        "endpoint_url": f"http://{MINIO_ENDPOINT}",
    },
)


def make_bucket(name):
    """Creates the bucket in MinIO. Ignores exception if bucket already exists.

    Args:
        name: the bucket name.
    """
    try:
        MINIO_CLIENT.make_bucket(name)
    except BucketAlreadyOwnedByYou:
        pass


def get_experiment_id():
    """Looks for an experiment id in various locations.

    1st env variable "EXPERIMENT_ID".
    2nd notebook metadata.

    Returns:
        str: the experiment uuid.

    Raises:
        TypeError: when experiment id is undefinded everywhere.
    """
    experiment_id = getenv("EXPERIMENT_ID")

    if experiment_id is not None:
        return experiment_id

    try:
        # get kernel id from running kernel
        connection_file_path = kernel.get_connection_file()
        connection_file = basename(connection_file_path)
        kernel_id = connection_file.split("-", 1)[1].split(".")[0]

        # then extract experiment id from notebook metadata
        sessions = get(f"{JUPYTER_ENDPOINT}/sessions")
        for sess in sessions:
            if sess["kernel"]["id"] == kernel_id:
                filename = sess["notebook"]["name"]
                file = FileContentsManager().get(filename)
                return file["content"]["metadata"].get("experiment_id")
    except (RuntimeError, ConnectionError):
        pass

    raise TypeError("experiment_id is undefined")


def get_operator_id():
    """Looks for an operator id in various locations.

    1st env variable "OPERATOR_ID".
    2nd notebook metadata.

    Returns:
        str: the operator uuid.

    Raises:
        TypeError: when operator id is undefinded everywhere.
    """
    operator_id = getenv("OPERATOR_ID")

    if operator_id is not None:
        return operator_id

    try:
        # get kernel id from running kernel
        connection_file_path = kernel.get_connection_file()
        connection_file = basename(connection_file_path)
        kernel_id = connection_file.split("-", 1)[1].split(".")[0]

        # then extract experiment id from notebook metadata
        sessions = get(f"{JUPYTER_ENDPOINT}/sessions").json()
        for sess in sessions:
            if sess["kernel"]["id"] == kernel_id:
                filename = sess["notebook"]["name"]
                file = FileContentsManager().get(filename)
                return file["content"]["metadata"].get("operator_id")
    except (RuntimeError, ConnectionError):
        pass

    raise TypeError("operator_id is undefined")
