# -*- coding: utf-8 -*-
from os import getenv

from minio import Minio
from minio.error import BucketAlreadyOwnedByYou
from s3fs.core import S3FileSystem

BUCKET_NAME = "anonymous"

MINIO_CLIENT = Minio(
    endpoint=getenv("MINIO_ENDPOINT", "minio-service.kubeflow:9000"),
    access_key=getenv("MINIO_ACCESS_KEY", "minio"),
    secret_key=getenv("MINIO_SECRET_KEY", "minio123"),
    region=getenv("MINIO_REGION_NAME", "us-east-1"),
    secure=False,
)

S3FS = S3FileSystem(
    key=getenv("MINIO_ACCESS_KEY", "minio"),
    secret=getenv("MINIO_SECRET_KEY", "minio123"),
    use_ssl=False,
    client_kwargs={
        "endpoint_url": "http://{}".format(getenv("MINIO_ENDPOINT", "minio-service.kubeflow:9000")),
    }
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
