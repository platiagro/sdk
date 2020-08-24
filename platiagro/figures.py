# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps
from tempfile import _get_candidate_names
from typing import List, Optional

import base64
import matplotlib.figure

from .util import BUCKET_NAME, MINIO_CLIENT, make_bucket, \
    get_experiment_id, get_operator_id, get_run_id, stat_metadata

PREFIX_1 = "experiments"
PREFIX_2 = "operators"


def list_figures(experiment_id: Optional[str] = None,
                 operator_id: Optional[str] = None,
                 run_id: Optional[str] = None) -> List[str]:
    """Lists all figures from object storage as data URI scheme.

    Args:
        experiment_id (str, optional): the experiment uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.
        run_id (str, optional): the run id. Defaults to None.

    Returns:
        list: A list of data URIs.
    """
    if experiment_id is None:
        experiment_id = get_experiment_id()

    if operator_id is None:
        operator_id = get_operator_id()

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

    figures = []

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    prefix = figure_filepath('figure-', experiment_id, operator_id, run_id)
    objects = MINIO_CLIENT.list_objects_v2(BUCKET_NAME, prefix)
    for obj in objects:
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=obj.object_name,
        )
        figures.append(data.read().decode())
    return figures


def save_figure(figure: [bytes, matplotlib.figure.Figure, str],
                extension: Optional[str] = None,
                experiment_id: Optional[str] = None,
                operator_id: Optional[str] = None,
                run_id: Optional[str] = None):
    """Saves a matplotlib figure to the object storage.

    Args:
        figure (matplotlib.figure.Figure): a matplotlib figure.
        experiment_id (str, optional): the experiment uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.
        run_id (str, optional): the run id. Defaults to None.

    Raises:
        TypeError: when a figure is not a matplotlib figure.
    """
    if experiment_id is None:
        # gets experiment_id from env variables
        experiment_id = get_experiment_id()

    if operator_id is None:
        # gets operator_id from env variables
        operator_id = get_operator_id()

    if run_id is None:
        # gets run_id from env variables
        # Attention: returns None if env is unset
        run_id = get_run_id()

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
            object_name=f'{PREFIX_1}/{experiment_id}/{PREFIX_2}/{operator_id}/.metadata',
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )

    random_str = next(_get_candidate_names())

    if isinstance(figure, matplotlib.figure.Figure):
        buffer = BytesIO()
        figure.savefig(buffer, format="svg")
        buffer.seek(0)
        figure_name = f"figure-{random_str}.svg"
    else:
        buffer = BytesIO(base64.b64decode(figure))
        figure_name = f"figure-{random_str}.{extension}"

    length = buffer.getbuffer().nbytes

    # uploads figure to MinIO
    object_name = figure_filepath(figure_name, experiment_id, operator_id, run_id)
    MINIO_CLIENT.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=buffer,
        length=length,
    )


def figure_filepath(name: str,
                    experiment_id: Optional[str] = None,
                    operator_id: Optional[str] = None,
                    run_id: Optional[str] = None) -> str:
    """Builds the filepath of a given figure.

    Args:
        name (str): the figure name.
        experiment_id (str, optional): the experiment uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.
        run_id (str, optional): the run id. Defaults to None.
    Returns:
        str: The object name.
    """
    if run_id:
        path = f'{PREFIX_1}/{experiment_id}/{PREFIX_2}/{operator_id}/{run_id}/{name}'
    else:
        path = f'{PREFIX_1}/{experiment_id}/{PREFIX_2}/{operator_id}/{name}'

    return path
