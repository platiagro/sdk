# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps, loads
from os.path import join
from typing import List, Dict, Optional

import pandas as pd
from minio.error import NoSuchBucket, NoSuchKey

from .util import BUCKET_NAME, MINIO_CLIENT, S3FS, make_bucket

PREFIX = "datasets"
FILE_EXTENSION = ".csv.gz"
METADATA_EXTENSION = ".metadata"
COMPRESSION = "gzip"


def list_datasets() -> List[str]:
    """Lists all datasets from object storage.

    Returns:
        list: A list of all datasets names.
    """
    datasets = []

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    objects = MINIO_CLIENT.list_objects_v2(BUCKET_NAME, PREFIX + "/")
    for obj in objects:
        name = obj.object_name[len(PREFIX) + 1:]
        datasets.append(name)
    return datasets


def load_dataset(name: str) -> pd.DataFrame:
    """Retrieves a dataset as a pandas.DataFrame.

    Args:
        name (str): the dataset name.

    Returns:
        pandas.DataFrame: A `pandas.DataFrame`.

    Raises:
        FileNotFoundError: If dataset does not exist in the object storage.
    """
    try:
        path = join("s3://", BUCKET_NAME, PREFIX, name + FILE_EXTENSION)
        return pd.read_csv(
            S3FS.open(path),
            header=0,
            index_col=False,
            compression=COMPRESSION,
        )
    except FileNotFoundError:
        raise FileNotFoundError("The specified dataset does not exist")


def save_dataset(name: str,
                 df: pd.DataFrame,
                 metadata: Optional[Dict[str, str]] = None):
    """Saves a dataset and its metadata to the object storage.

    Args:
        name (str): the dataset name.
        df (pandas.DataFrame): the dataset as a `pandas.DataFrame`.
        metadata (dict, optional): metadata about the dataset. Defaults to None.
    """
    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    # uploads file to MinIO
    path = join(BUCKET_NAME, PREFIX, name + FILE_EXTENSION)
    df.to_csv(
        S3FS.open(path, "w"),
        header=True,
        index=False,
        compression=COMPRESSION,
    )

    if metadata is None:
        metadata = {}

    # encodes metadata to JSON format
    buffer = BytesIO(dumps(metadata).encode())

    # uploads metadata to MinIO
    object_name = join(PREFIX, name + METADATA_EXTENSION)
    MINIO_CLIENT.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=buffer,
        length=buffer.getbuffer().nbytes,
    )


def stat_dataset(name: str) -> Dict[str, str]:
    """Retrieves the metadata of a dataset.

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
        object_name = join(PREFIX, name + METADATA_EXTENSION)
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
        # decodes the metadata (which is in JSON format)
        metadata = loads(data.read())

        # also reads the 1st line of data: column names
        path = join("s3://", BUCKET_NAME, PREFIX, name + FILE_EXTENSION)
        df = pd.read_csv(
            S3FS.open(path),
            header=0,
            index_col=False,
            nrows=1,
            compression=COMPRESSION,
        )
        columns = df.columns.tolist()
        metadata["columns"] = columns

    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError("The specified dataset does not exist")

    return metadata
