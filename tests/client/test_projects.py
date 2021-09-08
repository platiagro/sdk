# -*- coding: utf-8 -*-
import unittest
import unittest.mock as mock

import platiagro.client.projects

PROJECT_NAME = "project00"
PROJECTS_LIST = {
    "projects": [
        {"uuid": "c862813f-13e8-4adb-a430-56a1444209a0", "name": "project01"},
        {"uuid": "e2ea6c34-2ace-4a5d-a97c-ca60e9ef92ec", "name": PROJECT_NAME},
    ]
}


class TestProjects(unittest.TestCase):
    @mock.patch("requests.get")
    def test_list_projects(self, mock_get):
        """
        Should return a list of two projects.
        """
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = PROJECTS_LIST
        mock_get.return_value = mock_response

        result = platiagro.client.projects.list_projects()
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), PROJECTS_LIST)

    @mock.patch("platiagro.client.projects.list_projects")
    def test_get_project_by_name_value_error(self, mock_list_projects):
        """
        Should raise an exception when project_name does not exist.
        """
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = PROJECTS_LIST
        mock_list_projects.return_value = mock_response

        with self.assertRaises(ValueError):
            assert platiagro.client.projects.get_project_by_name(project_name="unk")

    @mock.patch("platiagro.client.projects.list_projects")
    def test_get_project_by_name_success(self, mock_list_projects):
        """
        Should return project_name successfully.
        """
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = PROJECTS_LIST
        mock_list_projects.return_value = mock_response

        result = platiagro.client.projects.get_project_by_name(
            project_name=PROJECT_NAME
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], PROJECT_NAME)
