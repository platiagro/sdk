# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps, loads
from os import SEEK_SET, getenv
from os.path import join
from typing import List, Tuple

import pandas as pd
from minio import Minio
from minio.error import NoSuchBucket, NoSuchKey

from .featuretypes import infer_featuretypes

BUCKET_NAME = "anonymous"
PREFIX = "datasets"

client = Minio(
    endpoint=getenv("MINIO_ENDPOINT", "minio-service.kubeflow"),
    access_key=getenv("MINIO_ACCESS_KEY", "minio"),
    secret_key=getenv("MINIO_SECRET_KEY", "minio123"),
    region=getenv("MINIO_REGION_NAME", "us-east-1"),
    secure=False,
)


def load_dataset(name: str) -> Tuple[pd.DataFrame, List]:
    """Retrieves a dataset and its feature types.

    Args:
        name (str): the dataset name.

    Returns:
        tuple: A `pandas.DataFrame` and a list of feature types.
    """
    try:
        object_name = join(PREFIX, name)
        stat = client.stat_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )

        columns = loads(stat.metadata["X-Amz-Meta-Columns"])
        featuretypes = loads(stat.metadata["X-Amz-Meta-Featuretypes"])

        data = client.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError("No such file or directory: '{}'".format(name))

    csv_buffer = BytesIO()
    for d in data.stream(32*1024):
        csv_buffer.write(d)
    csv_buffer.seek(0, SEEK_SET)

    df = pd.read_csv(csv_buffer, header=None, names=columns, index_col=False)
    return df, featuretypes


def save_dataset(name: str, df: pd.DataFrame):
    """Saves a dataset and its feature types.

    Args:
        name (str): the dataset name.
        df (pandas.DataFrame): the dataset as a `pandas.DataFrame`.
    """
    object_name = join(PREFIX, name)

    columns = df.columns.values.tolist()
    featuretypes = infer_featuretypes(df)

    # will store columns and featuretypes as metadata
    metadata = {
        "columns": dumps(columns),
        "featuretypes": dumps(featuretypes),
    }

    # converts DataFrame to bytes-like
    csv_bytes = df.to_csv().encode("utf-8")
    csv_buffer = BytesIO(csv_bytes)
    file_length = len(csv_bytes)

    try:
        # uploads file to MinIO
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            data=csv_buffer,
            length=file_length,
            metadata=metadata,
        )
    except NoSuchBucket:
        raise FileNotFoundError("No such file or directory: '{}'".format(name))
