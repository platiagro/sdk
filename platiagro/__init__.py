from platiagro.artifacts import download_artifact
from platiagro import client
from platiagro.datasets import list_datasets, load_dataset, save_dataset, \
    stat_dataset, download_dataset, update_dataset_metadata
from platiagro.featuretypes import infer_featuretypes, validate_featuretypes, \
    DATETIME, CATEGORICAL, NUMERICAL
from platiagro.figures import list_figures, save_figure
from platiagro.metrics import list_metrics, save_metrics
from platiagro.models import load_model, save_model
from platiagro.tasks import create_task
from platiagro.deployments import list_projects, get_project_by_name, list_deployments, \
    get_deployment_by_name, run_deployments

__all__ = ["download_artifact",
           "list_datasets", "load_dataset", "save_dataset", "stat_dataset",
           "download_dataset", "update_dataset_metadata",
           "infer_featuretypes", "validate_featuretypes",
           "DATETIME", "CATEGORICAL", "NUMERICAL",
           "list_figures", "save_figure",
           "list_metrics", "save_metrics",
           "load_model", "save_model",
           "create_task",
           "list_projects", "get_project_by_name", "list_deployments", "get_deployment_by_name", "run_deployments"]

__version__ = "0.2.0"
