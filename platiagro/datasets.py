# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps, loads
from typing import List, Dict, Optional

import pandas as pd
from minio.error import NoSuchBucket, NoSuchKey

from .featuretypes import infer_featuretypes
from .util import BUCKET_NAME, MINIO_CLIENT, S3FS, make_bucket, get_operator_id, get_run_id

PREFIX = "datasets"


def list_datasets() -> List[str]:
    """Lists dataset names from object storage.

    Returns:
        list: A list of all datasets names.
    """
    datasets = []

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    objects = MINIO_CLIENT.list_objects_v2(BUCKET_NAME, PREFIX + "/")

    for obj in objects:
        name = obj.object_name[len(PREFIX) + 1:-1]
        datasets.append(name)

    return datasets


def load_dataset(name: str,
                 run_id: Optional[str] = None,
                 operator_id: Optional[str] = None) -> pd.DataFrame:
    """Retrieves a dataset as a pandas.DataFrame.

    If run_id exists, then loads the dataset from the specified run.

    Args:
        name (str): the dataset name.
        run_id (str, optional): the run id of training pipeline. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.

    Returns:
        pandas.DataFrame: A `pandas.DataFrame`.

    Raises:
        FileNotFoundError: If dataset does not exist in the object storage.
    """
    if run_id is None:
        # gets run_id from env variable
        # Attention: returns None if env is unset
        run_id = get_run_id()
    elif run_id == "latest":
        metadata = stat_dataset(name)
        run_id = metadata.get("run_id")

    if operator_id is None:
        # gets operator_id from env variables
        # Attention: returns None if env is unset
        operator_id = get_operator_id(raise_for_none=False)

    # builds the path to the dataset file
    path = data_filepath(name, run_id, operator_id)

    try:
        dataset = pd.read_csv(S3FS.open(path), header=0, index_col=False)
    except FileNotFoundError:
        raise FileNotFoundError("The specified dataset does not exist")

    return dataset


def save_dataset(name: str,
                 df: pd.DataFrame,
                 metadata: Optional[Dict[str, str]] = None,
                 read_only: bool = False,
                 run_id: Optional[str] = None,
                 operator_id: Optional[str] = None):
    """Saves a dataset and its metadata to the object storage.

    Args:
        name (str): the dataset name.
        df (pandas.DataFrame): the dataset as a `pandas.DataFrame`.
        metadata (dict, optional): metadata about the dataset. Defaults to None.
        read_only (bool, optional): whether the dataset will be read only. Defaults to False.
        run_id (str, optional): the run id. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.

    Raises:
        PermissionError: If dataset was read only.
    """
    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    if run_id is None:
        # gets run_id from env variables
        # Attention: returns None if env is unset
        run_id = get_run_id()

    if operator_id is None:
        # gets operator_id from env variables
        # Attention: returns None if env is unset
        operator_id = get_operator_id(raise_for_none=False)

    try:
        # gets metadata (if dataset exists)
        metadata = stat_dataset(name, run_id)
        was_read_only = metadata.get("read_only", False)
    except FileNotFoundError:
        was_read_only = False

    if was_read_only:
        raise PermissionError("The specified dataset was marked as read only")

    # builds metadata dict:
    # sets filename, read_only, columns, featuretypes, run_id
    if metadata is None:
        metadata = {}

    metadata["filename"] = name
    metadata["read_only"] = read_only
    metadata["columns"] = df.columns.tolist()

    if "featuretypes" not in metadata:
        metadata["featuretypes"] = infer_featuretypes(df)

    if run_id:
        metadata["run_id"] = run_id

        # When saving a dataset of a run, also
        # set the run_id in datasets/<name>.metadata
        # This enables load_dataset by run="latest"
        try:
            root_metadata = stat_dataset(name)
        except FileNotFoundError:
            root_metadata = {}

        root_metadata["run_id"] = run_id
        object_name = metadata_filepath(name)
        # encodes metadata to JSON format
        buffer = BytesIO(dumps(root_metadata).encode())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )

    # uploads file to MinIO
    path = data_filepath(name, run_id, operator_id)
    df.to_csv(S3FS.open(path, "w"), header=True, index=False)

    object_name = metadata_filepath(name, run_id, operator_id)
    # encodes metadata to JSON format
    buffer = BytesIO(dumps(metadata).encode())
    MINIO_CLIENT.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=buffer,
        length=buffer.getbuffer().nbytes,
    )


def stat_dataset(name: str,
                 run_id: Optional[str] = None,
                 operator_id: Optional[str] = None) -> Dict[str, str]:
    """Retrieves the metadata of a dataset.

    Args:
        name (str): the dataset name.
        run_id (str, optional): the run id of trainning pipeline. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.

    Returns:
        dict: The metadata.

    Raises:
        FileNotFoundError: If dataset does not exist in the object storage.
    """
    metadata = {}

    if run_id == "latest":
        metadata = stat_dataset(name)
        run_id = metadata.get("run_id")

    # gets the filepath of the dataset
    object_name = metadata_filepath(name, run_id, operator_id)

    try:
        # reads the .metadata file
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
        # decodes the metadata (which is in JSON format)
        metadata = loads(data.read())

    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError("The specified dataset does not exist")

    return metadata


def data_filepath(name: str,
                  run_id: Optional[str] = None,
                  operator_id: Optional[str] = None) -> str:
    """Builds the filepath of a given dataset.

    Args:
        name (str): the dataset name.
        run_id (str, optional): the run uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.
    Returns:
        str: The object name.
    """
    if run_id and operator_id:
        path = f"{BUCKET_NAME}/{PREFIX}/{name}/runs/{run_id}/operators/{operator_id}/{name}/{name}"
    else:
        # {name}/{name} is intentional!
        # Otherwise, any attempt to save a dataset inside a run would cause:
        # Object-prefix is already an object, please choose a different object-prefix name
        path = f"{BUCKET_NAME}/{PREFIX}/{name}/{name}"

    return path


def metadata_filepath(name: str,
                      run_id: Optional[str] = None,
                      operator_id: Optional[str] = None) -> str:
    """Builds the filepath of metadata of a given dataset.

    Args:
        name (str): the dataset name.
        run_id (str, optional): the run uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.

    Returns:
        str: The object name.
    """
    path = data_filepath(name, run_id, operator_id).lstrip(f"{BUCKET_NAME}/")
    path = f"{path}.metadata"

    return path
