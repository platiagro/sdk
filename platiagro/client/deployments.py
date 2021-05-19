# -*- coding: utf-8 -*-
import requests

from platiagro.client.projects import get_project_by_name
from platiagro.client._util import PROJECTS_ENDPOINT


def list_deployments(project_id: str):
    """Lists all deployments.

    Args:
        project_id (str): the project uuid.

    Returns:
        requests.Response: list of deployments.
    """
    response = requests.get(url=f"{PROJECTS_ENDPOINT}/projects/{project_id}/deployments")
    return response


def get_deployment_by_name(project_id: str, deployment_name: str):
    """Gets the deployment by name.

    Args:
        project_id (str): the project uuid.
        deployment_name (str): the deployment name.

    Returns:
        dict: the deployment corresponding to the given name.
    
    Raises:
        ValueError: if the given deployment name doesn't exist.
    """
    deployments = list_deployments(project_id=project_id).json()

    for deployment in deployments["deployments"]:
        if deployment["name"] == deployment_name:
            return deployment

    raise ValueError("deployment for the given name doesn't exist")

def run_deployment(project_name: str, deployment_name: str):
    """Runs the given deployement.

    Args:
        project_name (str): the project name.
        deployment_name (str): the deployment name.

    Returns:
        requests.Response: the details of deployment.
    """
    project = get_project_by_name(project_name=project_name)
    project_id = project["uuid"]

    deployment = get_deployment_by_name(project_id=project_id,
                                        deployment_name=deployment_name)

    response = requests.post(
        url=f'{PROJECTS_ENDPOINT}/projects/{project_id}/deployments/{deployment["uuid"]}/runs'
    )
    return response