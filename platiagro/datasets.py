# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps, loads
from os import SEEK_SET
from os.path import join
from typing import List, Tuple, Optional

import pandas as pd
from minio.error import NoSuchBucket, NoSuchKey

from .featuretypes import infer_featuretypes, validate_featuretypes
from .util import BUCKET_NAME, MINIO_CLIENT, make_bucket

PREFIX = "datasets"


def load_dataset(name: str) -> Tuple[pd.DataFrame, List]:
    """Retrieves a dataset and its feature types.

    Args:
        name (str): the dataset name.

    Returns:
        tuple: A `pandas.DataFrame` and a list of feature types.
    """
    try:
        object_name = join(PREFIX, name)
        stat = MINIO_CLIENT.stat_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )

        columns = loads(stat.metadata["X-Amz-Meta-Columns"])
        featuretypes = loads(stat.metadata["X-Amz-Meta-Featuretypes"])

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

    df = pd.read_csv(csv_buffer, header=None, names=columns, index_col=False)
    return df, featuretypes


def save_dataset(name: str, df: pd.DataFrame, featuretypes: Optional[List] = None):
    """Saves a dataset and its feature types.

    Args:
        name (str): the dataset name.
        df (pandas.DataFrame): the dataset as a `pandas.DataFrame`.
        featuretypes (list, optional): the feature types. Defaults to None.
    """
    object_name = join(PREFIX, name)

    columns = df.columns.values.tolist()
    if featuretypes is None:
        featuretypes = infer_featuretypes(df)
    else:
        if len(columns) != len(featuretypes):
            raise ValueError("featuretypes must be the same length as the DataFrame columns")
        validate_featuretypes(featuretypes)

    # will store columns and featuretypes as metadata
    metadata = {
        "columns": dumps(columns),
        "featuretypes": dumps(featuretypes),
    }

    # converts DataFrame to bytes-like
    csv_bytes = df.to_csv(header=False, index=False).encode("utf-8")
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
