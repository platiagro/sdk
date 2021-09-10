# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps
from typing import List, Optional, Union
from datetime import datetime

import base64

from platiagro.util import (
    BUCKET_NAME,
    MINIO_CLIENT,
    make_bucket,
    get_experiment_id,
    get_operator_id,
    get_run_id,
    get_deployment_id,
    get_monitoring_id,
    stat_metadata,
    operator_filepath,
)


def list_figures(
    experiment_id: Optional[str] = None,
    operator_id: Optional[str] = None,
    run_id: Optional[str] = None,
    deployment_id: Optional[str] = None,
    monitoring_id: Optional[str] = None,
) -> List[str]:
    """Lists all figures from object storage as data URI scheme.

    Args:
        experiment_id (str, optional): the experiment uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.
        run_id (str, optional): the run id. Defaults to None.

    Returns:
        list: A list of data URIs.
    """
    if experiment_id is None:
        experiment_id = get_experiment_id()

    if operator_id is None:
        operator_id = get_operator_id()

    if deployment_id is None:
        deployment_id = get_deployment_id()

    if monitoring_id is None:
        monitoring_id = get_monitoring_id()

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    if run_id is None:
        # gets run_id from env variable
        # Attention: returns None if env is unset
        run_id = get_run_id()
    elif run_id == "latest":
        try:
            metadata = stat_metadata(experiment_id, operator_id)
            run_id = metadata.get("run_id")
        except FileNotFoundError:
            return []

    figures = []

    if deployment_id is not None:
        prefix = f"deployments/{deployment_id}/monitorings/{monitoring_id}/figure-"
    else:
        prefix = operator_filepath("figure-", experiment_id, operator_id, run_id)

    objects = MINIO_CLIENT.list_objects(BUCKET_NAME, prefix)

    for obj in objects:
        data = MINIO_CLIENT.get_object(
            bucket_name=BUCKET_NAME,
            object_name=obj.object_name,
        )
        encoded_figure = base64.b64encode(data.read()).decode()
        file_extension = obj.object_name.split(".")[1]
        if file_extension == "html":
            figure = f"data:text/html;base64,{encoded_figure}"
        elif file_extension == "svg":
            figure = f"data:image/svg+xml;base64,{encoded_figure}"
        else:
            figure = f"data:image/{file_extension};base64,{encoded_figure}"
        figures.append(figure)
    return figures


def save_figure(
    figure: Union[bytes, str],
    extension: Optional[str] = None,
    experiment_id: Optional[str] = None,
    operator_id: Optional[str] = None,
    run_id: Optional[str] = None,
    deployment_id: Optional[str] = None,
    monitoring_id: Optional[str] = None,
):
    """Saves a figure to the object storage.

    Args:
        figure (bytes, str):
            a base64 bytes or a base64 string.
        extension (str, optional): the file extension when base64 is send. Defaults to None.
        experiment_id (str, optional): the experiment uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.
        run_id (str, optional): the run id. Defaults to None.
        deployment_id (str, optional): the deployment id. Defaults to None.
        monitoring_id (str, optional): the monitoring id. Defaults to None.
    """
    if experiment_id is None:
        # gets experiment_id from env variables
        experiment_id = get_experiment_id()

    if operator_id is None:
        # gets operator_id from env variables
        operator_id = get_operator_id()

    if run_id is None:
        # gets run_id from env variables
        # Attention: returns None if env is unset
        run_id = get_run_id()

    if deployment_id is None:
        # gets run_id from env variables
        # Attention: returns None if env is unset
        deployment_id = get_deployment_id()

    if monitoring_id is None:
        # gets run_id from env variables
        # Attention: returns None if env is unset
        monitoring_id = get_monitoring_id()

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    if run_id:
        metadata = {}
        try:
            metadata = stat_metadata(experiment_id, operator_id)
            if run_id == "latest":
                run_id = metadata.get("run_id")
        except FileNotFoundError:
            pass
        metadata["run_id"] = run_id

        # encodes metadata to JSON format and uploads to MinIO
        buffer = BytesIO(dumps(metadata).encode())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/{experiment_id}/operators/{operator_id}/.metadata",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )

    if extension == "html":
        buffer = BytesIO(figure.encode())
    else:
        buffer = BytesIO(base64.b64decode(figure))

    pref = datetime.now().strftime("%y%m%d%H%M%S%f")
    figure_name = f"figure-{pref}.{extension}"

    length = buffer.getbuffer().nbytes

    # uploads figure to MinIO
    if deployment_id is not None:
        object_name = (
            f"deployments/{deployment_id}/monitorings/{monitoring_id}/{figure_name}"
        )
    else:
        object_name = operator_filepath(figure_name, experiment_id, operator_id, run_id)

    MINIO_CLIENT.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=buffer,
        length=length,
    )


def delete_figures(
    experiment_id: Optional[str] = None,
    operator_id: Optional[str] = None,
    run_id: Optional[str] = None,
    deployment_id: Optional[str] = None,
    monitoring_id: Optional[str] = None,
) -> List[str]:
    """Delete a figure to the object storage.

    Args:
        experiment_id (str, optional): the experiment uuid. Defaults to None.
        operator_id (str, optional): the operator uuid. Defaults to None.
        run_id (str, optional): the run id. Defaults to None.
        deployment_id (str, optional): the deployment id. Defaults to None.
        monitoring_id (str, optional): the monitoring id. Defaults to None.
    """

    if experiment_id is None:
        experiment_id = get_experiment_id()

    if operator_id is None:
        operator_id = get_operator_id()

    if deployment_id is None:
        deployment_id = get_deployment_id()

    if monitoring_id is None:
        monitoring_id = get_monitoring_id()

    # ensures MinIO bucket exists
    make_bucket(BUCKET_NAME)

    if run_id:
        metadata = {}
        try:
            metadata = stat_metadata(experiment_id, operator_id)
            if run_id == "latest":
                run_id = metadata.get("run_id")
        except FileNotFoundError:
            pass

    if deployment_id is not None:
        prefix = f"deployments/{deployment_id}/monitorings/{monitoring_id}/figure-"
    else:
        prefix = operator_filepath("figure-", experiment_id, operator_id, run_id)

    objects = MINIO_CLIENT.list_objects(BUCKET_NAME, prefix)

    for obj in objects:
        MINIO_CLIENT.remove_object(
            bucket_name=BUCKET_NAME,
            object_name=obj.object_name,
        )
