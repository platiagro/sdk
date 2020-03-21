# -*- coding: utf-8 -*-
from base64 import b64encode
from io import BytesIO
from os.path import join
from tempfile import _get_candidate_names
from typing import List

import matplotlib.figure

from .util import BUCKET_NAME, MINIO_CLIENT, make_bucket

PREFIX_1 = "experiments"
PREFIX_2 = "operators"


def list_figures(experiment_id: str, operator_id: str) -> List[str]:
    """Lists all figures from object storage as data URI scheme.

    Args:
        experiment_id (str): the experiment uuid.
        operator_id (str): the operator uuid.

    Returns:
        list: A list of data URIs.
    """
    figures = []

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    prefix = join(PREFIX_1, experiment_id, PREFIX_2, operator_id, "figure-")
    objects = MINIO_CLIENT.list_objects_v2(BUCKET_NAME, prefix)
    for obj in objects:
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=obj.object_name,
        )
        buffer = b""
        for d in data.stream(32*1024):
            buffer += d
        encoded_figure = b64encode(buffer).decode("utf8")
        figure = "data:image/png;base64,{}".format(encoded_figure)
        figures.append(figure)
    return figures


def save_figure(experiment_id: str, operator_id: str,
                figure: matplotlib.figure.Figure):
    """Saves a matplotlib figure to the object storage.

    Args:
        experiment_id (str): the experiment uuid.
        operator_id (str): the operator uuid.
        figure (matplotlib.figure.Figure): a matplotlib figure.

    Raises:
        TypeError: when a figure is not a matplotlib figure.
    """
    if not isinstance(figure, matplotlib.figure.Figure):
        raise TypeError("figure must be a matplotlib figure")

    figure_name = "figure-{}.png".format(next(_get_candidate_names()))
    object_name = join(PREFIX_1, experiment_id, PREFIX_2, operator_id, figure_name)

    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    buffer.seek(0)
    length = buffer.getbuffer().nbytes

    # uploads metrics to MinIO
    MINIO_CLIENT.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=buffer,
        length=length,
    )
