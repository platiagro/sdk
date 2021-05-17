# -*- coding: utf-8 -*-
import requests
from platiagro.util import PROJECTS_ENDPOINT


def list_projects():
    """Lists the projects.

    Returns:
        requests.Response: the response returned list Projects.
    """
    response = requests.get(url=f"{PROJECTS_ENDPOINT}/projects")
    return response.json()


def get_project_name(project_name: str):
    """Lists projects by name.

    Args:
        project_name (str): the project name.

    Returns:
        list: returns the list of projects.
    """
    projects = list_projects()

    for project in projects["projects"]:
        if project["name"] == project_name:
            return project


def list_deployments(project_name: str):
    """Lists the deployments.

    Args:
        project_name (str): the project name.

    Returns:
        requests.Response: returns the list of deployments.
    """
    project = get_project_name(project_name)
    project_id = project["uuid"]
    response = requests.get(url=f"{PROJECTS_ENDPOINT}/projects/{project_id}/deployments")
    return response.json()


def get_deployment_name(project_name: str, deployment_name: str):
    """Lists deployments by name.

    Args:
        deployment_name (str): the deployment name.
        project_name (str): the project name.

    Returns:
        list: returns the list of deployments by name.
    """
    deployments = list_deployments(project_name)
    for deployment in deployments["deployments"]:
        if deployment["name"] == deployment_name:
            return deployment


def run_deployments(project_name: str, deployment_name: str):
    """Runs a deployement.

    Args:
        deployment_name (str): the deployment name.
        project_name (str): the project name.

    Returns:
        requests.Response: a dictionary with information about the deployment runs.
    """
    project = get_project_name(project_name)
    deployment = get_deployment_name(project_name, deployment_name)
    project_id = project["uuid"]
    deployment_id = deployment["uuid"]
    response = requests.post(url=f"{PROJECTS_ENDPOINT}/projects/{project_id}/deployments/{deployment_id}/runs", json={})
    return response.json()
