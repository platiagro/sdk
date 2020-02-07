# -*- coding: utf-8 -*-
from io import BytesIO
from os import SEEK_SET
from os.path import join

from joblib import dump, load
from minio.error import NoSuchBucket, NoSuchKey

from .util import BUCKET_NAME, MINIO_CLIENT, make_bucket

PREFIX = "models"


def load_model(name: str) -> object:
    """Retrieves a model.

    Args:
        name (str): the model name.

    Returns:
        object: A model.
    """
    try:
        object_name = join(PREFIX, name)
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError("No such file or directory: '{}'".format(name))

    model_buffer = BytesIO()
    for d in data.stream(32*1024):
        model_buffer.write(d)
    model_buffer.seek(0, SEEK_SET)
    model = load(model_buffer)

    return model


def save_model(name: str, model: object):
    """Serializes and saves a model.

    Args:
        name (str): the dataset name.
        model (object): the model.
    """
    object_name = join(PREFIX, name)

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
