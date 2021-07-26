# -*- coding: utf-8 -*-
from io import BytesIO
from json import load, loads, dumps
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from minio.error import S3Error
import logging

from platiagro.util import BUCKET_NAME, MINIO_CLIENT, get_experiment_id, \
    get_operator_id, make_bucket, get_run_id, stat_metadata, operator_filepath

METRICS_FILE = "metrics.json"


def list_metrics(experiment_id: Optional[str] = None,
                 operator_id: Optional[str] = None,
                 run_id: Optional[str] = None) -> List[Dict[str, object]]:
    """Lists metrics from object storage.
    Args:
        experiment_id (str, optional): the experiment uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.
        run_id (str, optional): the run id. Defaults to None.
    Returns:
        list: A list of metrics.
    Raises:
        TypeError: when experiment_id is undefined in args and env.
        TypeError: when operator_id is undefined in args and env.
    """
    if experiment_id is None:
        experiment_id = get_experiment_id()

    if operator_id is None:
        operator_id = get_operator_id()

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    if run_id is None:
        # gets run_id from env variable
        # Attention: returns None if env is unset
        run_id = get_run_id()
    elif run_id == "latest":
        try:
            metadata = stat_metadata(experiment_id, operator_id)
            run_id = metadata.get("run_id")
        except FileNotFoundError:
            return []

    try:
        object_name = operator_filepath(METRICS_FILE, experiment_id, operator_id, run_id)
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
    except S3Error as err:
        if err.code == "NoSuchBucket" or err.code == "NoSuchKey":
            raise FileNotFoundError(f"No such file or directory: '{experiment_id}'")

    return load(data)


def save_metrics(experiment_id: Optional[str] = None,
                 operator_id: Optional[str] = None,
                 run_id: Optional[str] = None,
                 **kwargs):
    """Saves metrics of an experiment to the object storage.
    Args:
        experiment_id (str, optional): the experiment uuid. Defaults to None
        operator_id (str, optional): the operator uuid. Defaults to None
        run_id (str, optional): the run id. Defaults to None.
        **kwargs: the metrics dict.
    Raises:
        TypeError: when experiment_id is undefined in args and env.
        TypeError: when operator_id is undefined in args and env.
    """
    if experiment_id is None:
        experiment_id = get_experiment_id()

    if operator_id is None:
        operator_id = get_operator_id()

    if run_id is None:
        # gets run_id from env variables
        # Attention: returns None if env is unset
        run_id = get_run_id()

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    if run_id:
        metadata = {}
        try:
            metadata = stat_metadata(experiment_id, operator_id)
            if run_id == "latest":
                run_id = metadata.get("run_id")
        except FileNotFoundError:
            pass
        metadata["run_id"] = run_id

        # encodes metadata to JSON format and uploads to MinIO
        buffer = BytesIO(dumps(metadata).encode())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name=f'experiments/{experiment_id}/operators/{operator_id}/.metadata',
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )

    object_name = operator_filepath(METRICS_FILE, experiment_id, operator_id, run_id)

    encoded_metrics = []

    # retrieves the metrics saved previosuly
    try:
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
        encoded_metrics = loads(data.read())
    except S3Error as err:
        if err.code == "NoSuchKey":
            pass

    # appends new metrics
    encoded_metrics.extend(_encode_metrics(kwargs))

    # puts metrics into buffer
    buffer = BytesIO(dumps(encoded_metrics).encode())

    # uploads metrics to MinIO
    MINIO_CLIENT.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=buffer,
        length=buffer.getbuffer().nbytes,
    )


def _encode_metrics(metrics: Dict[str, object]) -> List[Dict[str, object]]:
    """Prepares metric values for JSON encoding.

    Args:
        metrics (dict): the dictionary of metrics.

    Returns:
        (list): the list of metrics after encoding.

    Raises:
        TypeError: If a metric is not any of these types:
        int, float, str, numpy.ndarray, pandas.DataFrame, pandas.Series.
    """
    encoded_metrics = []
    for k, v in metrics.items():
        if isinstance(v, (np.ndarray, pd.Series)):
            encoded_metrics.append({k: v.tolist()})
        elif isinstance(v, pd.DataFrame):
            encoded_metrics.append({k: v.values.tolist()})
        elif isinstance(v, (int, float, str)):
            encoded_metrics.append({k: v})
        else:
            raise TypeError(f"{k} is not any of these types: int, float, str, numpy.ndarray, pandas.DataFrame, pandas.Series")
    return encoded_metrics
