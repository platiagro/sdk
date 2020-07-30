# -*- coding: utf-8 -*-
from os import getenv
from os.path import basename
from typing import Optional

from IPython.lib import kernel
from json import loads
from minio import Minio
from minio.error import BucketAlreadyOwnedByYou, NoSuchBucket, NoSuchKey
from notebook.services.contents.filemanager import FileContentsManager
from requests import get
from s3fs.core import S3FileSystem
from traitlets.config import MultipleInstanceError
from typing import Dict

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


def make_bucket(name: str):
    """Creates the bucket in MinIO. Ignores exception if bucket already exists.

    Args:
        name (str): the bucket name.
    """
    try:
        MINIO_CLIENT.make_bucket(name)
    except BucketAlreadyOwnedByYou:
        pass


def get_experiment_id(raise_for_none: bool = True, default: Optional[str] = None):
    """Looks for an experiment id in various locations.

    1st env variable "EXPERIMENT_ID".
    2nd notebook metadata.

    Args:
        raise_for_none (bool): Whether to raise TypeError if experiment id is undefined. Defaults to True.
        default (str): A default value to return experiment id is undefined. Defaults to None.

    Returns:
        str: the experiment uuid.

    Raises:
        TypeError: when raise_for_none is True and experiment id is undefinded.
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
        sessions = get(f"{JUPYTER_ENDPOINT}/api/sessions").json()
        for sess in sessions:
            if sess["kernel"]["id"] == kernel_id:
                filename = sess["notebook"]["name"]
                file = FileContentsManager().get(filename)
                return file["content"]["metadata"].get("experiment_id")
    except (RuntimeError, ConnectionError):
        pass

    if raise_for_none:
        raise TypeError("experiment_id is undefined")

    return default


def get_operator_id(raise_for_none: bool = True, default: Optional[str] = None):
    """Looks for an operator id in various locations.

    1st env variable "OPERATOR_ID".
    2nd notebook metadata.

    Args:
        raise_for_none (bool): Whether to raise TypeError if operator id is undefined. Defaults to True.
        default (str): A default value to return operator id is undefined. Defaults to None.

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
        sessions = get(f"{JUPYTER_ENDPOINT}/api/sessions").json()
        for sess in sessions:
            if sess["kernel"]["id"] == kernel_id:
                filename = sess["notebook"]["name"]
                file = FileContentsManager().get(filename)
                return file["content"]["metadata"].get("operator_id")
    except (RuntimeError, ConnectionError, MultipleInstanceError):
        pass

    if raise_for_none:
        raise TypeError("operator_id is undefined")

    return default


def get_run_id(raise_for_none: bool = False, default: Optional[str] = None):
    """Looks for an run id in env variable "RUN_ID".

    Args:
        raise_for_none (bool): Whether to raise TypeError if run id is undefined. Defaults to False.
        default (str): A default value to return run id is undefined. Defaults to None.

    Returns:
        str: the experiment uuid.

    Raises:
        TypeError: when raise_for_none is True and run id is undefinded.
    """
    run_id = getenv("RUN_ID")

    if run_id is not None:
        return run_id

    if raise_for_none:
        raise TypeError("run_id is undefined")

    return default


def stat_metadata(experiment_id: str, operator_id: str) -> Dict[str, str]:
    """Retrieves the metadata.

    Args:
        experiment_id (str): the experiment uuid.
        operator_id (str): the operator uuid.

    Returns:
        dict: The metadata.

    Raises:
        FileNotFoundError: If metadata does not exist in the object storage.
    """
    metadata = {}
    object_name = f'experiments/{experiment_id}/operators/{operator_id}/.metadata'
    try:
        # reads the .metadata file
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
        # decodes the metadata (which is in JSON format)
        metadata = loads(data.read())

    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError("The specified metadata does not exist")

    return metadata


def metadata_exists(name: str, run_id: str, operator_id: str = None) -> bool:
    """Test whether a metadata file path of a given run_id, or an operator of a run,
    exists in the object storage.

    Args:
        name (str): the dataset name.
        run_id (str): the run_id uuid.
        operator_id (str, optional): the operator uuid. Defaults to None.

    Returns:
        bool: True if metadata path exists in the obeject storage, otherwise, False.
    """
    if run_id and operator_id:
        object_name = f'datasets/{name}/runs/{run_id}/operators/{operator_id}/{name}/{name}.metadata'
    else:
        object_name = f'datasets/{name}/runs/{run_id}/{run_id}.metadata'

    try:
        # reads the .metadata file
        MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
        return True
    except (NoSuchBucket, NoSuchKey):
        return False
