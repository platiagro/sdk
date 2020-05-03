# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps, loads
from os import SEEK_SET
from os.path import join
from typing import Dict, Optional

from joblib import dump, load
from minio.error import NoSuchBucket, NoSuchKey

from .util import BUCKET_NAME, MINIO_CLIENT, make_bucket

PREFIX = "experiments"
MODEL_FILE = "model"


def load_model(experiment_id: str) -> object:
    """Retrieves a model.

    Args:
        experiment_id (str): the experiment uuid.

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
        raise FileNotFoundError(f"No such file or directory: '{experiment_id}'")

    model_buffer = BytesIO()
    for d in data.stream(32*1024):
        model_buffer.write(d)
    model_buffer.seek(0, SEEK_SET)
    model = load(model_buffer)

    return model


def save_model(experiment_id: str,
               model: object,
               metadata: Optional[Dict[str, str]] = None):
    """Serializes and saves a model.

    Args:
        experiment_id (str): the experiment uuid.
        model (object): the model.
        metadata (dict, optional): metadata about the dataset. Defaults to None.
    """
    object_name = join(PREFIX, experiment_id, MODEL_FILE)

    model_buffer = BytesIO()
    dump(model, model_buffer)
    model_buffer.seek(0, SEEK_SET)

    if metadata is None:
        metadata = {}

    # tries to encode metadata as json
    # obs: MinIO requires the metadata to be a Dict[str, str]
    for k, v in metadata.items():
        metadata[str(k)] = dumps(v)

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    # uploads file to MinIO
    MINIO_CLIENT.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=model_buffer,
        length=model_buffer.getbuffer().nbytes,
        metadata=metadata,
    )


def stat_model(experiment_id: str) -> Dict[str, str]:
    """Retrieves the metadata of a model.

    Args:
        experiment_id (str): the experiment uuid.

    Returns:
        dict: The metadata.

    Raises:
        FileNotFoundError: If model does not exist in the object storage.
    """
    try:
        object_name = join(PREFIX, experiment_id, MODEL_FILE)
        stat = MINIO_CLIENT.stat_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )

        metadata = {}
        for k, v in stat.metadata.items():
            if k.startswith("X-Amz-Meta-"):
                key = k[len("X-Amz-Meta-"):].lower()
                metadata[key] = loads(v)
    except (NoSuchBucket, NoSuchKey):
        raise FileNotFoundError(f"No such file or directory: '{experiment_id}'")

    return metadata
