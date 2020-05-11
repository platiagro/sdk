# -*- coding: utf-8 -*-
from base64 import b64encode
from io import BytesIO
from os.path import join
from tempfile import _get_candidate_names
from typing import List, Optional

import matplotlib.figure

from .util import BUCKET_NAME, MINIO_CLIENT, get_experiment_id, \
    get_operator_id, make_bucket

PREFIX_1 = "experiments"
PREFIX_2 = "operators"


def list_figures(experiment_id: Optional[str] = None,
                 operator_id: Optional[str] = None) -> List[str]:
    """Lists all figures from object storage as data URI scheme.

    Args:
        experiment_id (str, optional): the experiment uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.

    Returns:
        list: A list of data URIs.
    """
    if experiment_id is None:
        experiment_id = get_experiment_id()

    if operator_id is None:
        operator_id = get_operator_id()

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
        encoded_figure = b64encode(data.read()).decode()
        figure = f"data:image/png;base64,{encoded_figure}"
        figures.append(figure)
    return figures


def save_figure(figure: matplotlib.figure.Figure,
                experiment_id: Optional[str] = None,
                operator_id: Optional[str] = None):
    """Saves a matplotlib figure to the object storage.

    Args:
        figure (matplotlib.figure.Figure): a matplotlib figure.
        experiment_id (str, optional): the experiment uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.

    Raises:
        TypeError: when a figure is not a matplotlib figure.
    """
    if experiment_id is None:
        experiment_id = get_experiment_id()

    if operator_id is None:
        operator_id = get_operator_id()

    if not isinstance(figure, matplotlib.figure.Figure):
        raise TypeError("figure must be a matplotlib figure")

    random_str = next(_get_candidate_names())
    figure_name = f"figure-{random_str}.png"
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
