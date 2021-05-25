# -*- coding: utf-8 -*-
import requests
from platiagro.client._util import PROJECTS_ENDPOINT


def create_task(name, **kwargs):
    """API to create tasks.

    Args:
        name (str): the task name.
        **kwargs
            Arbitrary keyword arguments.

    Returns:
        requests.Response: the response returned by PlatIAgro Projects API.
    """
    description = kwargs.get("description")
    tags = kwargs.get("tags")
    copy_from = kwargs.get("copy_from")
    image = kwargs.get("image")
    commands = kwargs.get("commands")
    arguments = kwargs.get("arguments")
    parameters = kwargs.get("parameters")
    experiment_notebook = kwargs.get("experiment_notebook")
    deployment_notebook = kwargs.get("deployment_notebook")
    cpu_limit = kwargs.get("cpu_limit")
    cpu_request = kwargs.get("cpu_request")
    memory_limit = kwargs.get("memory_limit")
    memory_request = kwargs.get("memory_request")
    is_default = kwargs.get("is_default")

    data = {
            "name": name,
            "description": description,
            "tags": tags,
            "copy_from": copy_from,
            "image": image,
            "commands": commands,
            "arguments": arguments,
            "parameters": parameters,
            "experimentNotebook": experiment_notebook,
            "deploymentNotebook": deployment_notebook,
            "cpuLimit": cpu_limit,
            "cpuRequest": cpu_request,
            "memoryLimit": memory_limit,
            "memoryRequest": memory_request,
            "isDefault": is_default
        }

    response = requests.post(url=f"{PROJECTS_ENDPOINT}/tasks", json=data)

    return response
