# -*- coding: utf-8 -*-
import requests

from platiagro.client._util import PROJECTS_ENDPOINT


def list_projects():
    """Lists all projects.

    Returns:
        requests.Response: list of projects.
    """
    response = requests.get(url=f'{PROJECTS_ENDPOINT}/projects')
    return response


def get_project_by_name(project_name: str):
    """Gets the project by name.

    Args:
        project_name (str): the project name.

    Returns:
        dict: the project corresponding to the given project name.

    Raises:
        ValueError: if the given project name doesn't exist.
    """
    projects = list_projects().json()

    for project in projects["projects"]:
        if project["name"] == project_name:
            return project

    raise ValueError("project for the given name doesn't exist")
