# -*- coding: utf-8 -*-
import os

import requests
from werkzeug.exceptions import HTTPException

DATASETS_ENDPOINT = os.getenv("DATASETS_ENDPOINT", "datasets.platiagro:8080")
PROJECTS_ENDPOINT = os.getenv("PROJECTS_ENDPOINT", "projects.platiagro:8080")


def run(project_name: str, experiment_name: str, dataset_path: str = None):
    """Runs an experiment. Does not wait for a run to complete.

    Args:
        project_name (str): the project name.
        experiment_name (str): the experiment name.
        dataset_path (str): the path to a dataset file.

    Raises:
        ValueError: when either the project or experiment does not exist.
        RuntimeError: when PlatIAgro returns an error.
    """
    try:
        try:
            response = requests.get(f"http://{PROJECTS_ENDPOINT}/projects")
            response.raise_for_status()
            project_id = next((p["uuid"] for p in response.json()["projects"] if p["name"] == project_name))
        except StopIteration:
            raise ValueError(f"project {project_name} does not exist")

        try:
            response = requests.get(f"http://{PROJECTS_ENDPOINT}/projects/{project_id}/experiments")
            response.raise_for_status()
            experiment_id = next((e["uuid"] for e in response.json() if e["name"] == experiment_name))
        except StopIteration:
            raise ValueError(f"experiment {experiment_name} does not exist")

        if dataset_path is not None:
            set_dataset(project_id, experiment_id, dataset_path)

        # Starts async run
        response = requests.post(
            f"http://{PROJECTS_ENDPOINT}/projects/{project_id}/experiments/{experiment_id}/runs",
            json={},
        )
        response.raise_for_status()
    except HTTPException:
        raise RuntimeError("an error occurred while accessing PlatIAgro")


def set_dataset(project_id: str, experiment_id: str, dataset_path: str = None):
    """Uploads and sets a dataset in an experiment.

    Args:
        project_id (str): the project id.
        experiment_id (str): the experiment id.
        dataset_path (str): the path to a dataset file.

    Raises:
        ValueError: when either the project or experiment does not exist.
    """
    if not os.path.isfile(dataset_path):
        return

    # Faz upload do dataset
    response = requests.post(
        f"http://{DATASETS_ENDPOINT}/datasets",
        files={"file": open(dataset_path, "rb")}
    )
    response.raise_for_status()
    dataset = response.json().get("name")

    try:
        # Finds an operator that has a parameter named "dataset"
        response = requests.get(f"http://{PROJECTS_ENDPOINT}/projects/{project_id}/experiments/{experiment_id}/operators")
        response.raise_for_status()
        operator_id = next((o["uuid"] for o in response.json() if "dataset" in o["parameters"]))
    except StopIteration:
        return

    # Edita o dataset no fluxo
    response = requests.patch(
        f"http://{PROJECTS_ENDPOINT}/projects/{project_id}/experiments/{experiment_id}/operators/{operator_id}",
        json={
            "parameters": {
                "dataset": dataset,
                "target": "",
            },
        },
    )
    response.raise_for_status()
