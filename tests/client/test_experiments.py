# -*- coding: utf-8 -*-
import pytest
from unittest import mock, TestCase

from platiagro.client.experiments import get_experiment_by_name, list_experiments

PROJECT_ID = "c862813f-13e8-4adb-a430-56a1444209a0"

EXPERIMENT_NAME = "experiment00"
EXPERIMENTS_LIST = {
    "experiments": [
        {
            "uuid": "bc4a0874-4a6b-4e20-bd7e-ed00c51fd8ea",
            "name": "experiment01"
        },
        {
            "uuid": "a8ab15b1-7a90-4f18-b18d-e14f7422c938",
            "name": EXPERIMENT_NAME
        }
    ]
}

class TestRuns(TestCase):
    
    @mock.patch("requests.get")
    def test_list_experiments(self, mock_get):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = EXPERIMENTS_LIST
        mock_get.return_value = mock_response

        result = list_experiments(project_id=PROJECT_ID)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), EXPERIMENTS_LIST)

    @mock.patch("platiagro.client.experiments.list_experiments")
    def test_get_experiment_by_name(self, mock_list_experiments):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = EXPERIMENTS_LIST
        mock_list_experiments.return_value = mock_response

        with pytest.raises(ValueError):
            assert get_experiment_by_name(project_id=PROJECT_ID, experiment_name="unk")

        result = get_experiment_by_name(project_id=PROJECT_ID, experiment_name=EXPERIMENT_NAME)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], EXPERIMENT_NAME)

    # TODO: create test for run_experiment, needs to mock all function calls