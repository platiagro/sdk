# -*- coding: utf-8 -*-
from io import BytesIO
from os import SEEK_SET
from typing import Dict, Optional

from joblib import dump, load
from minio.error import NoSuchBucket, NoSuchKey

from .util import BUCKET_NAME, MINIO_CLIENT, get_experiment_id, \
    get_operator_id, make_bucket

PREFIX_1 = "experiments"
PREFIX_2 = "operators"
MODEL_FILE = "model.joblib"


def load_model(experiment_id: Optional[str] = None,
               operator_id: Optional[str] = None) -> Dict[str, object]:
    """Retrieves a model from object storage.

    Args:
        experiment_id (str, optional): the experiment uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.

    Returns:
        dict: A dictionary of models.

    Raises:
        TypeError: when experiment_id is undefined in args and env.
        TypeError: when operator_id is undefined in args and env.
    """
    if experiment_id is None:
        experiment_id = get_experiment_id()

    if operator_id is None:
        operator_id = get_operator_id()

    try:
        object_name = f"{PREFIX_1}/{experiment_id}/{PREFIX_2}/{operator_id}/{MODEL_FILE}"
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
    except (NoSuchBucket, NoSuchKey):
        return {}

    buffer = BytesIO(data.read())

    return load(buffer)


def save_model(**kwargs):
    """Serializes and saves models.

    Args:
        **kwargs: the models as keyword arguments.

    Raises:
        TypeError: when a figure is not a matplotlib figure.

    Raises:
        TypeError: when experiment_id is undefined in args and env.
        TypeError: when operator_id is undefined in args and env.
    """
    experiment_id = kwargs.get("experiment_id")
    if experiment_id is None:
        experiment_id = get_experiment_id()

    operator_id = kwargs.get("operator_id")
    if operator_id is None:
        operator_id = get_operator_id()

    object_name = f"{PREFIX_1}/{experiment_id}/{PREFIX_2}/{operator_id}/{MODEL_FILE}"

    model_buffer = BytesIO()
    dump(kwargs, model_buffer)
    model_buffer.seek(0, SEEK_SET)

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    # uploads file to MinIO
    MINIO_CLIENT.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=model_buffer,
        length=model_buffer.getbuffer().nbytes,
    )
