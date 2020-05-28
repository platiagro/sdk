# -*- coding: utf-8 -*-
from datetime import datetime
from io import BytesIO
from json import dumps, loads
from typing import List, Dict, Optional

import pandas as pd
from minio.error import NoSuchBucket, NoSuchKey

from .featuretypes import infer_featuretypes
from .util import BUCKET_NAME, MINIO_CLIENT, S3FS, make_bucket, get_operator_id, get_run_id

PREFIX = "datasets"
METADATA_FILE = ".metadata"
RUNS_PREFIX = "runs"


def list_datasets() -> List[str]:
    """Lists all datasets from object storage.

    Returns:
        list: A list of all datasets names.
    """
    datasets = []

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    datasets_folders = MINIO_CLIENT.list_objects_v2(BUCKET_NAME, PREFIX + "/")
    for folder in datasets_folders:
        dataset_folder = MINIO_CLIENT.list_objects_v2(BUCKET_NAME, folder.object_name)
        for dataset_file in dataset_folder:
            if dataset_file.object_name.endswith(METADATA_FILE):
                name = dataset_file.object_name[(len(PREFIX) + 1):(-len(METADATA_FILE) - 1)]
                datasets.append(name)
    return datasets


def load_dataset(name: str,
                 run_id: Optional[str] = None,
                 operator_id: Optional[str] = None) -> pd.DataFrame:
    """Retrieves a dataset as a pandas.DataFrame.

    If run_id exist try to load dataset from runs

    Args:
        name (str): the dataset name.
        run_id (str, optional): the run id of trainning pipeline. Defaults to None.
        operator_id (str, optional): the operator uuid.  Defaults to None.

    Returns:
        pandas.DataFrame: A `pandas.DataFrame`.

    Raises:
        FileNotFoundError: If dataset does not exist in the object storage.
    """
    if run_id is None:
        run_id = get_run_id()
    elif run_id == 'latest':
        runs_metadata = stat_runs_dataset(name)
        run_id = runs_metadata["run_id"]

    if run_id:
        try:
            # gets the filename from metadata
            metadata = stat_dataset(name, run_id)
            filename = None
            if operator_id:
                for key in metadata:
                    value = metadata[key]
                    if type(value) is dict and "operatorId" in value:
                        operatorId = value["operatorId"]
                        if operator_id == operatorId:
                            filename = value["filename"]
            else:
                filename = metadata["filename"]

            path = f'{BUCKET_NAME}/{PREFIX}/{name}/{RUNS_PREFIX}/{run_id}/{filename}'
            return pd.read_csv(S3FS.open(path), header=0, index_col=False)
        except FileNotFoundError:
            if operator_id:
                raise FileNotFoundError("The specified dataset does not exist")
            else:
                pass

    try:
        # gets the filename from metadata
        metadata = stat_dataset(name)
        filename = metadata["filename"]
        path = f'{BUCKET_NAME}/{PREFIX}/{name}/{filename}'
        return pd.read_csv(S3FS.open(path), header=0, index_col=False)
    except FileNotFoundError:
        raise FileNotFoundError("The specified dataset does not exist")


def save_dataset(name: str,
                 df: pd.DataFrame,
                 metadata: Optional[Dict[str, str]] = None,
                 read_only: bool = False,
                 run_id: Optional[str] = None):
    """Saves a dataset and its metadata to the object storage.

    Args:
        name (str): the dataset name.
        df (pandas.DataFrame): the dataset as a `pandas.DataFrame`.
        metadata (dict, optional): metadata about the dataset. Defaults to None.
        read_only (bool, optional): whether the dataset will be read only. Defaults to False.
        run_id (str, optional): the run id of trainning pipeline. Defaults to None.

    Raises:
        PermissionError: If dataset was read only.
    """
    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    if run_id is None:
        run_id = get_run_id()
    elif run_id == 'latest':
        runs_metadata = stat_runs_dataset(name)
        run_id = runs_metadata["run_id"]

    # gets metadata (if dataset already exists)
    try:
        metadata = stat_dataset(name, run_id)
        was_read_only = metadata["read_only"]
    except FileNotFoundError:
        was_read_only = False

    if was_read_only:
        raise PermissionError("The specified dataset was marked as read only")

    # generates a filename using current UTC datetime
    filename_prefix = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    filename = filename_prefix + ".csv"

    # stores metadata: columns, filename, read_only, featuretypes
    object_metadata_name = f'{PREFIX}/{name}/{METADATA_FILE}'
    if metadata is None:
        metadata = {}
    metadata["filename"] = filename
    metadata["read_only"] = read_only
    metadata["columns"] = df.columns.tolist()
    if "featuretypes" not in metadata:
        metadata["featuretypes"] = infer_featuretypes(df)

    if run_id:
        operator_id = get_operator_id()

        # uploads file to MinIO
        # eg. anonymous/datasets/iris/runs/f16e4274/19700101000000000000.csv
        path = f'{BUCKET_NAME}/{PREFIX}/{name}/{RUNS_PREFIX}/{run_id}/{filename}'
        df.to_csv(S3FS.open(path, "w"), header=True, index=False)

        # save the last run id on runs metadata
        runs_metadata = {}
        runs_metadata["run_id"] = run_id
        buffer = BytesIO(dumps(runs_metadata).encode())
        object_name = f'{PREFIX}/{name}/{RUNS_PREFIX}/{METADATA_FILE}'
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )

        # add dataset run history in metadata
        run_metadata = {}
        run_metadata["operatorId"] = operator_id
        run_metadata["filename"] = filename
        run_metadata["columns"] = df.columns.tolist()
        metadata[filename_prefix] = run_metadata
        object_metadata_name = f'{PREFIX}/{name}/{RUNS_PREFIX}/{run_id}/{METADATA_FILE}'
    else:
        # uploads file to MinIO
        # eg. anonymous/datasets/iris/19700101000000000000.csv
        path = f'{BUCKET_NAME}/{PREFIX}/{name}/{filename}'
        df.to_csv(S3FS.open(path, "w"), header=True, index=False)

    # encodes metadata to JSON format
    buffer = BytesIO(dumps(metadata).encode())

    # uploads metadata to MinIO
    MINIO_CLIENT.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_metadata_name,
        data=buffer,
        length=buffer.getbuffer().nbytes,
    )


def stat_dataset(name: str, run_id: Optional[str] = None) -> Dict[str, str]:
    """Retrieves the metadata of a dataset.

    Args:
        name (str): the dataset name.
        run_id (str, optional): the run id of trainning pipeline. Defaults to None.

    Returns:
        dict: The metadata.

    Raises:
        FileNotFoundError: If dataset does not exist in the object storage.
    """
    metadata = {}
    try:
        # reads the .metadata file
        object_name = f'{PREFIX}/{name}/{METADATA_FILE}'
        if run_id:
            object_name = f'{PREFIX}/{name}/{RUNS_PREFIX}/{run_id}/{METADATA_FILE}'
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
        # decodes the metadata (which is in JSON format)
        metadata = loads(data.read())

    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError("The specified dataset does not exist")

    return metadata


def stat_runs_dataset(name: str) -> Dict[str, str]:
    """Retrieves the metadata of a dataset runs folder.

    Args:
        name (str): the dataset name.

    Returns:
        dict: The metadata.

    Raises:
        FileNotFoundError: If dataset does not exist in the object storage.
    """
    metadata = {}
    try:
        # reads the .metadata file
        object_name = f'{PREFIX}/{name}/{RUNS_PREFIX}/{METADATA_FILE}'
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
        # decodes the metadata (which is in JSON format)
        metadata = loads(data.read())
    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError("The specified run dataset does not exist")
    return metadata
