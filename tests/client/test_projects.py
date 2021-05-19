# -*- coding: utf-8 -*-
import pytest
from unittest import mock, TestCase

from platiagro.client.projects import get_project_by_name, list_projects

PROJECT_NAME = "project00"
PROJECTS_LIST = {
    "projects": [
        {
            "uuid": "c862813f-13e8-4adb-a430-56a1444209a0",
            "name": "project01"
        },
        {
            "uuid": "e2ea6c34-2ace-4a5d-a97c-ca60e9ef92ec",
            "name": PROJECT_NAME
        }
    ]
}

class TestRuns(TestCase):

    @mock.patch("requests.get")
    def test_list_projects(self, mock_get):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = PROJECTS_LIST
        mock_get.return_value = mock_response

        result = list_projects()
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), PROJECTS_LIST)

    @mock.patch("platiagro.client.projects.list_projects")
    def test_get_experiment_by_name(self, mock_list_projects):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = PROJECTS_LIST
        mock_list_projects.return_value = mock_response

        with pytest.raises(ValueError):
            assert get_project_by_name(project_name="unk")

        result = get_project_by_name(project_name=PROJECT_NAME)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], PROJECT_NAME)