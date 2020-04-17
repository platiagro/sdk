# -*- coding: utf-8 -*-
from io import BytesIO
from json import loads, dumps
from os.path import join
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from minio.error import NoSuchKey

from .figures import save_figure
from .util import BUCKET_NAME, MINIO_CLIENT, make_bucket

PREFIX = "experiments"
METRICS_FILE = "metrics.json"
CONFUSION_MATRIX = "confusion_matrix"


def save_metrics(experiment_id: str, operator_id: str, reset: bool = False,
                 **kwargs):
    """Saves metrics of an experiment to the object storage.

    Args:
        experiment_id (str): the experiment uuid.
        operator_id (str): the operator uuid.
        reset (bool): whether to reset the metrics. default: False.
        **kwargs: the metrics dict.
    """
    object_name = join(PREFIX, experiment_id, METRICS_FILE)

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
            buffer = b""
            for d in data.stream(32*1024):
                buffer += d
            encoded_metrics = loads(buffer.decode())
        except NoSuchKey:
            pass

    # appends new metrics
    encoded_metrics.append(encode_metrics(kwargs))

    # puts metrics into buffer
    buffer = BytesIO()
    buffer.write(dumps(encoded_metrics).encode())
    buffer.seek(0)
    length = buffer.getbuffer().nbytes

    # uploads metrics to MinIO
    MINIO_CLIENT.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=buffer,
        length=length,
    )

    # makes plots for some metrics
    if CONFUSION_MATRIX in kwargs:
        confusion_matrix = kwargs[CONFUSION_MATRIX]
        plt.clf()
        plot = plot_confusion_matrix(confusion_matrix)
        save_figure(experiment_id=experiment_id, operator_id=operator_id,
                    figure=plot.figure)
        plt.clf()


def encode_metrics(metrics: Dict) -> Dict:
    """Prepares metric values for JSON encoding.

    Args:
        metrics (dict): the dictionary of metrics.

    Returns:
        (dict): the dictionary of metrics after encoding.

    Raises:
        TypeError: If a metric is not any of these types: int, float, str, numpy.ndarray, pandas.DataFrame.
    """
    encoded_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            encoded_metrics[k] = v.tolist()
        elif isinstance(v, pd.DataFrame):
            encoded_metrics[k] = v.to_dict()
        elif isinstance(v, (int, float, str)):
            encoded_metrics[k] = v
        else:
            raise TypeError("{k} is not any of these types: int, float, str, numpy.ndarray, pandas.DataFrame")
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
