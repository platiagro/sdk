# -*- coding: utf-8 -*-
from io import BytesIO
from os import SEEK_SET
from os.path import join

from joblib import dump, load
from minio.error import NoSuchBucket, NoSuchKey

from .util import BUCKET_NAME, MINIO_CLIENT, make_bucket

PREFIX = "experiments"
MODEL_FILE = "model"


def load_model(experiment_id: str) -> object:
    """Retrieves a model.

    Args:
        experiment_id (str): the experiment id.

    Returns:
        object: A model.
    """
    try:
        object_name = join(PREFIX, experiment_id, MODEL_FILE)
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError("No such file or directory: '{}'".format(experiment_id))

    model_buffer = BytesIO()
    for d in data.stream(32*1024):
        model_buffer.write(d)
    model_buffer.seek(0, SEEK_SET)
    model = load(model_buffer)

    return model


def save_model(experiment_id: str, model: object):
    """Serializes and saves a model.

    Args:
        experiment_id (str): the experiment id.
        model (object): the model.
    """
    object_name = join(PREFIX, experiment_id, MODEL_FILE)

    model_buffer = BytesIO()
    dump(model, model_buffer)
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
