# -*- coding: utf-8 -*-
from io import BytesIO
from os import SEEK_SET, getenv
from os.path import join

from joblib import dump, load
from minio import Minio
from minio.error import NoSuchBucket, NoSuchKey

BUCKET_NAME = "anonymous"
PREFIX = "models"

client = Minio(
    endpoint=getenv("MINIO_ENDPOINT", "minio-service.kubeflow"),
    access_key=getenv("MINIO_ACCESS_KEY", "minio"),
    secret_key=getenv("MINIO_SECRET_KEY", "minio123"),
    region=getenv("MINIO_REGION_NAME", "us-east-1"),
    secure=False,
)


def load_model(name: str) -> object:
    """Retrieves a model.

    Args:
        name (str): the model name.

    Returns:
        object: A model.
    """
    try:
        object_name = join(PREFIX, name)
        data = client.get_object(
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
    try:
        object_name = join(PREFIX, name)

        model_buffer = BytesIO()
        dump(model, model_buffer)
        model_buffer.seek(0, SEEK_SET)

        # uploads file to MinIO
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            data=model_buffer,
            length=model_buffer.getbuffer().nbytes,
        )
    except NoSuchBucket:
        raise FileNotFoundError("No such file or directory: '{}'".format(name))
