# -*- coding: utf-8 -*-
import unittest
import unittest.mock as mock

import platiagro.client.deployments

PROJECT_ID = "c862813f-13e8-4adb-a430-56a1444209a0"

DEPLOYMENT_NAME = "deployment00"
DEPLOYMENTS_LIST = {
    "deployments": [
        {"uuid": "b793c517-2f95-41f0-a4bb-25da50d6e052", "name": "deployment01"},
        {"uuid": "ea5661ab-28fe-4531-b310-8a37bd77197b", "name": DEPLOYMENT_NAME},
    ]
}


class TestDeployments(unittest.TestCase):
    @mock.patch("platiagro.client.deployments.requests.get")
    def test_list_deployments(self, mock_get):
        """
        Should return a list of two deployments.
        """
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = DEPLOYMENTS_LIST
        mock_get.return_value = mock_response

        result = platiagro.client.deployments.list_deployments(project_id=PROJECT_ID)

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), DEPLOYMENTS_LIST)

    @mock.patch("platiagro.client.deployments.list_deployments")
    def test_get_deployment_by_name_value_error(self, mock_list_deployments):
        """
        Should raise an exception when deployment_name does not exist.
        """
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = DEPLOYMENTS_LIST
        mock_list_deployments.return_value = mock_response

        with self.assertRaises(ValueError):
            assert platiagro.client.deployments.get_deployment_by_name(
                project_id=PROJECT_ID, deployment_name="unk"
            )

    @mock.patch("platiagro.client.deployments.list_deployments")
    def test_get_deployment_by_name_success(self, mock_list_deployments):
        """
        Should return deployment_name successfully.
        """
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = DEPLOYMENTS_LIST
        mock_list_deployments.return_value = mock_response

        result = platiagro.client.deployments.get_deployment_by_name(
            project_id=PROJECT_ID, deployment_name=DEPLOYMENT_NAME
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], DEPLOYMENT_NAME)

    # TODO: Fix this test.
    # Estamos com dificuldade para mockar o teste run_deployments, estamos procurando uma alternativa
