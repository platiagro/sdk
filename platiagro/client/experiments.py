# -*- coding: utf-8 -*-
import requests

from platiagro.client.projects import get_project_by_name
from platiagro.client._util import PROJECTS_ENDPOINT


def list_experiments(project_id: str):
    """Lists all experiments.

    Args:
        project_id (str): the project uuid.

    Returns:
        requests.Response: list of experiments.
    """
    response = requests.get(url=f'{PROJECTS_ENDPOINT}/projects/{project_id}/experiments')
    return response


def get_experiment_by_name(project_id: str, experiment_name: str):
    """Gets the experiment by name.

    Args:
        project_id (str): the project uuid.
        experiment_name (str): the experiment name.

    Returns:
        dict: the experiment corresponding to the given name.

    Raises:
        ValueError: if the given experiment name doesn't exist.
    """
    experiments = list_experiments(project_id=project_id).json()

    for experiment in experiments["experiments"]:
        if experiment["name"] == experiment_name:
            return experiment

    raise ValueError("experiment for the given name doesn't exist")


def run_experiment(project_name: str, experiment_name: str):
    """Run the given experiment.

    Args:
        project_name (str): the project name.
        experiment_name (str): the experiment name.

    Returns:
        requests.Response: the details of experiment.
    """
    project = get_project_by_name(project_name=project_name)
    project_id = project["uuid"]

    experiment = get_experiment_by_name(project_id=project_id,
                                        experiment_name=experiment_name)

    response = requests.post(
        url=f'{PROJECTS_ENDPOINT}/projects/{project_id}/experiments/{experiment["uuid"]}/runs'
    )
    return response