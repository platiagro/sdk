# -*- coding: utf-8 -*-
from os import getenv

from minio import Minio
from minio.error import BucketAlreadyOwnedByYou
from s3fs.core import S3FileSystem

BUCKET_NAME = "anonymous"
MINIO_ENDPOINT = getenv("MINIO_ENDPOINT", "minio-service.kubeflow:9000")
MINIO_ACCESS_KEY = getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = getenv("MINIO_SECRET_KEY", "minio123")

MINIO_CLIENT = Minio(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

S3FS = S3FileSystem(
    key=MINIO_ACCESS_KEY,
    secret=MINIO_SECRET_KEY,
    use_ssl=False,
    client_kwargs={
        "endpoint_url": f"http://{MINIO_ENDPOINT}",
    },
)


def make_bucket(name):
    """Creates the bucket in MinIO. Ignores exception if bucket already exists.

    Args:
        name: the bucket name
    """
    try:
        MINIO_CLIENT.make_bucket(name)
    except BucketAlreadyOwnedByYou:
        pass
