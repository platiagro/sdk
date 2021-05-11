# -*- coding: utf-8 -*-
import json
import requests
from typing import Dict, List, Optional
from platiagro.util import PROJECTS_ENDPOINT


def create_task(name: str,
                description: Optional[str] = None,
                tags: Optional[List[str]] = None,
                copy_from: Optional[str] = None,
                image: Optional[str] = None,
                commands: Optional[str] = None,
                arguments: Optional[str] = None,
                parameters: Optional[List] = None,
                experiment_notebook: Optional[Dict] = None,
                deployment_notebook: Optional[Dict] = None,
                cpu_limit: Optional[str] = None,
                cpu_request: Optional[str] = None,
                memory_limit: Optional[str] = None,
                memory_request: Optional[str] = None,
                is_default: Optional[bool] = None):
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

    response = requests.post(url=PROJECTS_ENDPOINT, data=json.dumps(data))

    return response.json()
