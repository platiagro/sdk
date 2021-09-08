# -*- coding: utf-8 -*-
import base64
import io
from typing import Any

from minio.error import S3Error
from minio.helpers import ObjectWriteResult
from urllib3.response import HTTPResponse

CSV_DATA = (
    "SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species\n"
    "5.1,3.5,1.4,0.2,Iris-setosa\n"
    "4.9,3.0,1.4,0.2,Iris-setosa\n"
    "4.7,3.2,1.3,0.2,Iris-setosa\n"
    "4.6,3.1,1.5,0.2,Iris-setosa\n"
).encode()

BINARY_DATA = b"\x89PNG\r\n"

NO_SUCH_KEY_ERROR = S3Error(
    code="NoSuchKey",
    message="",
    resource=None,
    request_id=None,
    host_id=None,
    response=None,
)

FIGURE_SVG = (
    "<svg viewBox='0 0 125 80' xmlns='http://www.w3.org/2000/svg'>\n"
    '<text y="75" font-size="100" font-family="serif"><![CDATA[10]]></text>\n'
    "</svg>\n"
)
FIGURE_SVG_BASE64 = base64.b64encode(FIGURE_SVG.encode()).decode()

FIGURE_HTML = "<html><body>HELLO!</body></html>"
FIGURE_HTML_BASE64 = base64.b64encode(FIGURE_HTML.encode()).decode()


def get_object_side_effect(bucket_name: str, object_name: str, **kwargs):
    """
    Returns a mock object when accessing bucket objects.

    Parameters
    ----------
    bucket_name : str
    object_name : str
    **kwargs

    Returns
    -------
    HTTPResponse
    """
    if object_name.endswith(".metadata"):
        filename = object_name[: -len(".metadata")].split("/")[-1]
        body = f'{{"filename": "{filename}"}}'.encode()
    elif object_name.endswith(".csv"):
        body = CSV_DATA
    else:
        body = BINARY_DATA

    return HTTPResponse(body=io.BytesIO(body), preload_content=False)


def put_object_side_effect(
    bucket_name: str, object_name: str, data: Any, length: int, **kwargs
):
    """
    Returns a mock object when adding objects to bucket.

    Parameters
    ----------
    bucket_name : str
    object_name : str
    data : Any
    length : int
    **kwargs

    Returns
    -------
    ObjectWriteResult
    """
    return ObjectWriteResult(
        bucket_name=bucket_name,
        object_name=object_name,
        version_id=0,
        etag="",
        http_headers={},
    )


def fput_object_side_effect(
    bucket_name: str, object_name: str, file_path: str, **kwargs
):
    """
    Returns a mock object when adding objects to bucket.

    Parameters
    ----------
    bucket_name : str
    object_name : str
    file_path : str
    **kwargs

    Returns
    -------
    ObjectWriteResult
    """
    return ObjectWriteResult(
        bucket_name=bucket_name,
        object_name=object_name,
        version_id=0,
        etag="",
        http_headers={},
    )
