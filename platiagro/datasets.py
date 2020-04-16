# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps, loads
from os import SEEK_SET
from os.path import join
from typing import List, Dict, Optional

import pandas as pd
from minio.error import NoSuchBucket, NoSuchKey

from .util import BUCKET_NAME, MINIO_CLIENT, make_bucket

PREFIX = "datasets"


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
        object_name = join(PREFIX, name)

        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError("No such file or directory: '{}'".format(name))

    csv_buffer = BytesIO()
    for d in data.stream(32*1024):
        csv_buffer.write(d)
    csv_buffer.seek(0, SEEK_SET)
    df = pd.read_csv(csv_buffer, header=0, index_col=False)
    return df


def save_dataset(name: str,
                 df: pd.DataFrame,
                 metadata: Optional[Dict[str, str]] = None):
    """Saves a dataset and its metadata to the object storage.

    Args:
        name (str): the dataset name.
        df (pandas.DataFrame): the dataset as a `pandas.DataFrame`.
        metadata (dict, optional): metadata about the dataset. Defaults to None.
    """
    object_name = join(PREFIX, name)

    if metadata is None:
        metadata = {}

    # tries to encode metadata as json
    # obs: MinIO requires the metadata to be a Dict[str, str]
    for k, v in metadata.items():
        metadata[str(k)] = dumps(v)

    # converts DataFrame to bytes-like
    csv_bytes = df.to_csv(header=True, index=False).encode("utf-8")
    csv_buffer = BytesIO(csv_bytes)
    file_length = len(csv_bytes)

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    # uploads file to MinIO
    MINIO_CLIENT.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=csv_buffer,
        length=file_length,
        metadata=metadata,
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
    try:
        object_name = join(PREFIX, name)

        stat = MINIO_CLIENT.stat_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )

        metadata = {}
        for k, v in stat.metadata.items():
            if k.startswith("X-Amz-Meta-"):
                key = k[len("X-Amz-Meta-"):].lower()
                metadata[key] = loads(v)

        # reads first line of data (columns)
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
        buffer = ""
        for d in data.stream(32*1024):
            dstr = d.decode("utf-8")
            index = dstr.find("\n")
            buffer += dstr[:index]
            if index > -1:
                break
        columns = buffer.split(",")
        metadata["columns"] = columns

    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError("No such file or directory: '{}'".format(name))

    return metadata
