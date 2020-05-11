# -*- coding: utf-8 -*-
from io import BytesIO
from os import SEEK_SET
from os.path import join
from typing import Dict, Optional

from joblib import dump, load
from minio.error import NoSuchBucket, NoSuchKey

from .util import BUCKET_NAME, MINIO_CLIENT, get_experiment_id, make_bucket

PREFIX = "experiments"
MODEL_FILE = "model.joblib"


def load_model(experiment_id: Optional[str] = None) -> Dict[str, object]:
    """Retrieves a model.

    Args:
        experiment_id (str, optional): the experiment uuid. Defaults to None.

    Returns:
        dict: A dictionary of models.
    """
    if experiment_id is None:
        experiment_id = get_experiment_id()

    try:
        object_name = join(PREFIX, experiment_id, MODEL_FILE)
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError(f"No such file or directory: '{experiment_id}'")

    buffer = BytesIO(data.read())

    return load(buffer)


def save_model(**kwargs):
    """Serializes and saves models.

    Args:
        experiment_id (str, optional): the experiment uuid. Defaults to None.
        **kwargs: the models as keyword arguments.
    """
    experiment_id = kwargs.get("experiment_id")
    if experiment_id is None:
        experiment_id = get_experiment_id()

    object_name = join(PREFIX, experiment_id, MODEL_FILE)

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
