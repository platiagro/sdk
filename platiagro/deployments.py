# -*- coding: utf-8 -*-
import requests
from platiagro.util import PROJECTS_ENDPOINT


def list_projects():
    """Lists the projects.

    Returns:
        requests.Response: the response returned list Projects.
    """
    response = requests.get(url=f"{PROJECTS_ENDPOINT}/projects")
    return response


def get_project_by_name(project_name: str):
    """Lists projects by name.

    Args:
        project_name (str): the project name.

    Returns:
        requests.Response: the response returned list of projects.

    Raises:
        ValueError: If the given project name doesn't exist.
    """
    projects = list_projects().json()

    for project in projects["projects"]:
        if project["name"] == project_name:
            return project

    raise ValueError("project for the given name doesn't exist")


def list_deployments(project_name: str):
    """Lists the deployments.

    Args:
        project_name (str): the project name.

    Returns:
        requests.Response: the response returned list of deployments.
    """
    project = get_project_by_name(project_name)
    project_id = project["uuid"]
    response = requests.get(url=f"{PROJECTS_ENDPOINT}/projects/{project_id}/deployments")
    return response


def get_deployment_by_name(project_name: str, deployment_name: str):
    """Lists deployments by name.

    Args:
        deployment_name (str): the deployment name.
        project_name (str): the project name.

    Returns:
        requests.Response: the response returned list of deployments by name.
    
    Raises:
        ValueError: If the given deployment name doesn't exist.
    """
    deployments = list_deployments(project_name).json()

    for deployment in deployments["deployments"]:
        if deployment["name"] == deployment_name:
            return deployment

    raise ValueError("deployment for the given name doesn't exist")


def run_deployments(project_name: str, deployment_name: str):
    """Runs a deployement.

    Args:
        deployment_name (str): the deployment name.
        project_name (str): the project name.

    Returns:
        requests.Response: the response returned dictionary with information about the deployment runs.
    """
    project_id = get_project_by_name(project_name)["uuid"]
    deployment_id = get_deployment_by_name(project_name, deployment_name)["uuid"]
    response = requests.post(url=f"{PROJECTS_ENDPOINT}/projects/{project_id}/deployments/{deployment_id}/runs")
    return response
