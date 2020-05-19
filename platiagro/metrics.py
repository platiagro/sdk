# -*- coding: utf-8 -*-
from io import BytesIO
from json import load, loads, dumps
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from minio.error import NoSuchBucket, NoSuchKey

from .figures import save_figure
from .util import BUCKET_NAME, MINIO_CLIENT, get_experiment_id, \
    get_operator_id, make_bucket

PREFIX = "experiments"
METRICS_FILE = "metrics.json"
CONFUSION_MATRIX = "confusion_matrix"


def list_metrics(experiment_id: Optional[str] = None) -> List[Dict[str, object]]:
    """Lists all metrics of an experiment.

    Args:
        experiment_id (str, optional): the experiment uuid. Defaults to None.

    Returns:
        list: A list of metrics.
    """
    if experiment_id is None:
        experiment_id = get_experiment_id()

    try:
        object_name = f'{PREFIX}/{experiment_id}/{METRICS_FILE}'
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError(f"No such file or directory: '{experiment_id}'")

    return load(data)


def save_metrics(reset: bool = False,
                 experiment_id: Optional[str] = None,
                 operator_id: Optional[str] = None,
                 **kwargs):
    """Saves metrics of an experiment to the object storage.

    Args:
        reset (bool): whether to reset the metrics. Defaults to False.
        experiment_id (str, optional): the experiment uuid. Defaults to None
        operator_id (str, optional): the operator uuid. Defaults to None
        **kwargs: the metrics dict.
    """
    if experiment_id is None:
        experiment_id = get_experiment_id()

    object_name = f'{PREFIX}/{experiment_id}/{METRICS_FILE}'

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    encoded_metrics = []

    # retrieves the metrics saved previosuly
    if not reset:
        try:
            data = MINIO_CLIENT.get_object(
                bucket_name=BUCKET_NAME,
                object_name=object_name,
            )
            encoded_metrics = loads(data.read())
        except NoSuchKey:
            pass

    # appends new metrics
    encoded_metrics.extend(encode_metrics(kwargs))

    # puts metrics into buffer
    buffer = BytesIO(dumps(encoded_metrics).encode())

    # uploads metrics to MinIO
    MINIO_CLIENT.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=buffer,
        length=buffer.getbuffer().nbytes,
    )

    # makes plots for some metrics
    if CONFUSION_MATRIX in kwargs:

        if operator_id is None:
            operator_id = get_operator_id()

        confusion_matrix = kwargs[CONFUSION_MATRIX]
        plt.clf()
        plot = plot_confusion_matrix(confusion_matrix)
        save_figure(experiment_id=experiment_id,
                    operator_id=operator_id,
                    figure=plot.figure)
        plt.clf()


def encode_metrics(metrics: Dict[str, object]) -> List[Dict[str, object]]:
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


def plot_confusion_matrix(df: pd.DataFrame):
    """Plots a confusion matrix.

    Args:
        df (pandas.DataFrame): the confusion matrix.

    Returns:
        (matplotlib.Axes): the axes object.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("{} must be a pandas.DataFrame", CONFUSION_MATRIX)

    df.index.name = "Classes Verdadeiras"
    df.columns.name = "Classes Previstas"
    ax = sns.heatmap(df,
                     annot=True,
                     annot_kws={"fontsize": 14},
                     cbar=False,
                     cmap="Greens")
    ax.set_xlabel(df.columns.name, fontsize=16, rotation=0, labelpad=20)
    ax.set_ylabel(df.index.name, fontsize=16, labelpad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    plt.tight_layout()
    return ax
