# -*- coding: utf-8 -*-
from io import BytesIO
from os.path import join
from tempfile import _get_candidate_names
from typing import List

import matplotlib.figure

from .util import BUCKET_NAME, MINIO_CLIENT, make_bucket

PREFIX = "experiments"


def list_figures(experiment_id: str) -> List[str]:
    """Lists all figures from object storage.

    Args
        experiment_id (str): the experiment name.

    Returns:
        A list of all figures names.
    """
    figures = []

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    prefix = join(PREFIX, experiment_id, "figure-")
    objects = MINIO_CLIENT.list_objects_v2(BUCKET_NAME, prefix)
    for obj in objects:
        name = obj.object_name[len(PREFIX) + 1:]
        figures.append(name)
    return figures


def save_figure(experiment_id: str, figure: matplotlib.figure.Figure):
    """Saves a matplotlib figure to the object storage.

    Args
        experiment_id (str): the experiment name.
        figure (matplotlib.figure.Figure): a matplotlib figure.
    """
    if not isinstance(figure, matplotlib.figure.Figure):
        raise TypeError("figure must be a matplotlib figure")

    figure_name = "figure-{}.png".format(next(_get_candidate_names()))
    object_name = join(PREFIX, experiment_id, figure_name)

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
