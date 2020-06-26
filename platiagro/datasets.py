# -*- coding: utf-8 -*-
import os
import tempfile
from io import BytesIO
from json import dumps, loads
from typing import List, Dict, BinaryIO, Optional, Union

import pandas as pd
from minio.error import NoSuchBucket, NoSuchKey

from .featuretypes import CATEGORICAL, DATETIME, infer_featuretypes
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
                 operator_id: Optional[str] = None) -> Union[pd.DataFrame, BinaryIO]:
    """Retrieves the contents of a dataset.

    If run_id exists, then loads the dataset from the specified run.
    If the dataset does not exist for given run_id/operator_id return the
    'original' dataset

    Args:
        name (str): the dataset name.
        run_id (str, optional): the run id of training pipeline. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.

    Returns:
        The contents of a dataset. Either a `pandas.DataFrame` or an `BinaryIO` buffer.

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

    if run_id and operator_id:
        # when the dataset does not exist for given run_id/operator_id
        # must return the 'original' dataset
        try:
            metadata = stat_dataset(name, run_id, operator_id)
        except FileNotFoundError:
            # unset run_id so data_filepath points to the 'original' dataset
            run_id = None

    # builds the path to the dataset file
    path = data_filepath(name, run_id, operator_id)

    try:
        metadata = stat_dataset(name, run_id, operator_id)
        dataset = pd.read_csv(S3FS.open(path), header=0, index_col=False)

        dtypes = dict(
            (column, "object")
            for column, ftype in zip(metadata["columns"], metadata["featuretypes"])
            if ftype in [CATEGORICAL, DATETIME]
        )
        dataset = dataset.astype(dtypes)
    except (UnicodeDecodeError, pd.errors.EmptyDataError):
        # reads the raw file
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=path.lstrip(f"{BUCKET_NAME}/"),
        )
        return BytesIO(data.read())
    except FileNotFoundError:
        raise FileNotFoundError("The specified dataset does not exist")

    return dataset


def save_dataset(name: str,
                 data: Union[pd.DataFrame, BinaryIO] = None,
                 df: pd.DataFrame = None,
                 metadata: Optional[Dict[str, str]] = None,
                 read_only: bool = False,
                 run_id: Optional[str] = None,
                 operator_id: Optional[str] = None):
    """Saves a dataset and its metadata to the object storage.

    Args:
        name (str): the dataset name.
        data (pandas.DataFrame, BinaryIO, optional): the dataset contents as a
            pandas.DataFrame or an `BinaryIO` buffer. Defaults to None.
        df (pandas.DataFrame, optional): the dataset contents as an `pandas.DataFrame`.
            df exists only for compatibility with existing components.
            Use "data" for all types of datasets. Defaults to None.
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
        stored_metadata = stat_dataset(name, run_id)
        # update stored metadata values
        if metadata:
            stored_metadata.update(metadata)
        metadata = stored_metadata
        was_read_only = metadata.get("read_only", False)
    except FileNotFoundError:
        was_read_only = False

    if was_read_only:
        raise PermissionError("The specified dataset was marked as read only")

    # builds metadata dict:
    # sets filename, read_only, run_id
    if metadata is None:
        metadata = {}

    metadata["filename"] = name
    metadata["read_only"] = read_only

    # df exists only for compatibility with existing components
    # from now on one must use "data" for all types of datasets
    if df is not None:
        data = df

    if isinstance(data, pd.DataFrame):
        # sets metadata specific for pandas.DataFrame:
        # columns, featuretypes
        metadata["columns"] = data.columns.tolist()

        if "featuretypes" not in metadata:
            metadata["featuretypes"] = infer_featuretypes(data)

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

    path = data_filepath(name, run_id, operator_id)

    if isinstance(data, pd.DataFrame):
        # uploads dataframe to MinIO as a .csv file
        temp_file = tempfile.NamedTemporaryFile(dir='.', delete=False)
        data.to_csv(temp_file.name, header=True, index=False)
        MINIO_CLIENT.fput_object(
            bucket_name=BUCKET_NAME,
            object_name=path.lstrip(f"{BUCKET_NAME}/"),
            file_path=temp_file.name
        )
        temp_file.close()
        os.remove(temp_file.name)
    else:
        # uploads raw data to MinIO
        buffer = BytesIO(data.read())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name=path.lstrip(f"{BUCKET_NAME}/"),
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )

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


def download_dataset(name: str, path: str):
    """Downloads the given dataset to the path.

    Args:
        name (str): the dataset name.
        path (str): destination path.
    """
    dataset = load_dataset(name)

    if isinstance(dataset, pd.DataFrame):
        dataset.to_csv(path)
    else:
        f = open(path, 'wb')
        f.write(dataset.getvalue())
        f.close()


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
