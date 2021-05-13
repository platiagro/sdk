# -*- coding: utf-8 -*-
import json
import requests
from platiagro.util import PROJECTS_ENDPOINT


def create_task(name, **kwargs):
    """API to create tasks.

    Args:
        name (str): the task name.
        **kwargs
            Arbitrary keyword arguments.

    Returns:
        dict: a dictionary with information about the created task.
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
            "experiment_notebook": experiment_notebook,
            "deployment_notebook": deployment_notebook,
            "cpu_limit": cpu_limit,
            "cpu_request": cpu_request,
            "memory_limit": memory_limit,
            "memory_request": memory_request,
            "is_default": is_default
        }

    response = requests.post(url=f"{PROJECTS_ENDPOINT}/tasks", data=json.dumps(data))

    return response.json()
