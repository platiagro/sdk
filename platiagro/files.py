# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps, loads
from os import SEEK_SET, getenv
from os.path import join
from typing import List, Tuple, Dict, Optional
import pandas as pd
from minio.error import NoSuchBucket, NoSuchKey
#from .util import BUCKET_NAME, MINIO_CLIENT, make_bucket
from minio import Minio



MINIO_CLIENT = Minio(
    endpoint=getenv("MINIO_ENDPOINT", "minio-service.kubeflow:9000"),
    access_key=getenv("MINIO_ACCESS_KEY", "minio"),
    secret_key=getenv("MINIO_SECRET_KEY", "minio123"),
    region=getenv("MINIO_REGION_NAME", "us-east-1"),
    secure=False,
)


def list_files(bucket_name: str, pref: str) -> List[str]:
    """Lists all files from object storage.

    Returns:
        A list of all files in a folder.
    """
    file_names = []
    if len(bucket_name) == 0:
        return "bucket_name should not be empty"
    if len(pref) ==0:
        return "pref should not be empty"

    try:
        objects = MINIO_CLIENT.list_objects_v2(bucket_name, prefix=pref, recursive=True)
        for obj in objects:
            name = obj.object_name[len(pref) + 1:]
            file_names.append(name)
    except:
        return "list_files failed"
    return file_names


def load_file(BUCKET_NAME: str, pref: str, file_name: str) -> object:
    """Retrieves a file as a pandas.DataFrame.

    Args:
        name (str): the file name.

    Returns:
        object: A file.

    Raises:
        FileNotFoundError: If file does not exist in the object storage.
    """
    if len(BUCKET_NAME) == 0:
        raise ValueError("Bucet should not be empty")
    if len(pref) == 0:
        raise ValueError("Prefix should not be empry")
    if len(file_name) == 0:
        raise ValueError("File name should ot be empty")

    try:
        object_name=pref + "/" + file_name

        print(BUCKET_NAME)
        print(pref)
        print(file_name)
        print(object_name)

        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
    except:
        raise FileNotFoundError("File not found")

    file_loaded = BytesIO()
    for d in data.stream(32*1024):
        file_loaded.write(d)
    file_loaded.seek(0, SEEK_SET)

    return file_loaded


def save_file(BUCKET_NAME: str, pref: str, file_name: str, input_data: BytesIO):
    """Saves a file and its metadata to the object storage.

    Args:
        name (str): the file name.
        df (pandas.DataFrame): the dataset as a `pandas.DataFrame`.
        metadata (dict, optional): metadata about the dataset. Defaults to None.
    """

    if len(BUCKET_NAME) == 0:
        raise ValueError("bucket_name should not be empty")
    if len(pref) == 0:
        raise ValueError("pref should not be empty")
    if len(file_name) == 0:
        raise ValueError("file_name should not be empty")
    file_length = input_data.getbuffer().nbytes
    if file_length == 0:
        raise ValueError("input_data should not be empty")
    object_name = pref + '/' + file_name

    try:
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name=pref + "/" + file_name,
            data=input_data,
            length=input_data.getbuffer().nbytes,
            metadata=None,
        )
    except Exception as e:
        print(str(e))
        raise FileNotFoundError("File found")



