# -*- coding: utf-8 -*-
from os import getenv
from typing import Optional

from json import loads
from minio import Minio
from minio.error import S3Error
from s3fs.core import S3FileSystem
from typing import Dict
import logging

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
    except S3Error as err:
        if err.code == "BucketAlreadyOwnedByYou":
            logging.warning("The bucket already exists.")


def get_experiment_id(raise_for_none: bool = False, default: Optional[str] = None):
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

    if raise_for_none:
        raise TypeError("experiment_id is undefined")

    return default


def get_operator_id(raise_for_none: bool = False, default: Optional[str] = None):
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

    if raise_for_none:
        raise TypeError("operator_id is undefined")

    return default


def get_run_id(raise_for_none: bool = False, default: Optional[str] = None):
    """Looks for an run id in env variable "RUN_ID".

    Args:
        raise_for_none (bool): Whether to raise TypeError if run id is undefined. Defaults to False.
        default (str): A default value to return run id is undefined. Defaults to None.

    Returns:
        str: the run uuid.

    Raises:
        TypeError: when raise_for_none is True and run id is undefinded.
    """
    run_id = getenv("RUN_ID")

    if run_id is not None:
        return run_id

    if raise_for_none:
        raise TypeError("run_id is undefined")

    return default


def get_deployment_id(raise_for_none: bool = False, default: Optional[str] = None):
    """Looks for an run id in env variable "DEPLOYMENT_ID".

    Args:
        raise_for_none (bool): Whether to raise TypeError if deployment id is undefined. Defaults to False.
        default (str): A default value to return deployment id is undefined. Defaults to None.

    Returns:
        str: the deployment uuid.

    Raises:
        TypeError: when raise_for_none is True and deployment id is undefinded.
    """
    deployment_id = getenv("DEPLOYMENT_ID")

    if deployment_id is not None:
        return deployment_id

    if raise_for_none:
        raise TypeError("deployment_id is undefined")

    return default


def get_monitoring_id(raise_for_none: bool = False, default: Optional[str] = None):
    """Looks for an run id in env variable "MONITORING_ID".

    Args:
        raise_for_none (bool): Whether to raise TypeError if monitoring id is undefined. Defaults to False.
        default (str): A default value to return monitoring id is undefined. Defaults to None.

    Returns:
        str: the monitoring uuid.

    Raises:
        TypeError: when raise_for_none is True and monitoring id is undefinded.
    """
    monitoring_id = getenv("MONITORING_ID")

    if monitoring_id is not None:
        return monitoring_id

    if raise_for_none:
        raise TypeError("monitoring_id is undefined")

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

    except S3Error as err:
        if err.code == "NoSuchBucket" or err.code == "NoSuchKey":
            raise FileNotFoundError("The specified metadata does not exist")

    return metadata


def metadata_exists(name: str, run_id: str = None, operator_id: str = None) -> bool:
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
    elif run_id:
        object_name = f'datasets/{name}/runs/{run_id}/{run_id}.metadata'
    else:
        object_name = f'datasets/{name}/{name}.metadata'

    try:
        # reads the .metadata file
        MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
        return True
    except S3Error as err:
        if err.code == "NoSuchBucket" or err.code == "NoSuchKey":
            return False


def operator_filepath(name: str,
                      experiment_id: Optional[str] = None,
                      operator_id: Optional[str] = None,
                      run_id: Optional[str] = None) -> str:
    """Builds the filepath of a given operator.
    Args:
        name (str): the file name.
        experiment_id (str, optional): the experiment uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.
        run_id (str, optional): the run id. Defaults to None.
    Returns:
        str: The object name.
    """
    if run_id:
        path = f'experiments/{experiment_id}/operators/{operator_id}/{run_id}/{name}'
    else:
        path = f'experiments/{experiment_id}/operators/{operator_id}/{name}'
    return path
