# -*- coding: utf-8 -*-
from minio.error import S3Error

from platiagro.util import BUCKET_NAME, MINIO_CLIENT

PREFIX = "artifacts"


def download_artifact(name: str, path: str):
    """Downloads the given artifact to the path.

    Args:
        name (str): the dataset name.
        path (str): destination path.

    Raises:
        FileNotFoundError
    """
    try:
        MINIO_CLIENT.fget_object(
            bucket_name=BUCKET_NAME,
            object_name=f"{PREFIX}/{name}",
            file_path=path,
        )
    except S3Error as err:
        if err.code == "NoSuchBucket" or err.code == "NoSuchKey":
            raise FileNotFoundError("The specified artifact does not exist")
