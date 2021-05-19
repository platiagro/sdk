import pytest
from unittest import mock, TestCase

from platiagro.client.deployments import list_deployments, get_deployment_by_name

PROJECT_ID = "c862813f-13e8-4adb-a430-56a1444209a0"

DEPLOYMENT_NAME = "deployment00"
DEPLOYMENTS_LIST = {
    "deployments": [
        {
            "uuid": "b793c517-2f95-41f0-a4bb-25da50d6e052",
            "name": "deployment01"
        },
        {
            "uuid": "ea5661ab-28fe-4531-b310-8a37bd77197b",
            "name": DEPLOYMENT_NAME
        }
    ]
}


class TestRuns(TestCase):

    @mock.patch("platiagro.client.deployments.requests.get")
    def test_list_deployments(self, mock_get):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = DEPLOYMENTS_LIST
        mock_get.return_value = mock_response

        result = list_deployments(project_id=PROJECT_ID)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), DEPLOYMENTS_LIST)

    @mock.patch("platiagro.client.deployments.list_deployments")
    def test_get_experiment_by_name(self, mock_list_deployments):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = DEPLOYMENTS_LIST
        mock_list_deployments.return_value = mock_response

        with pytest.raises(ValueError):
            assert get_deployment_by_name(project_id=PROJECT_ID, deployment_name="unk")

        result = get_deployment_by_name(project_id=PROJECT_ID, deployment_name=DEPLOYMENT_NAME)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], DEPLOYMENT_NAME)

    # TODO: Fix this test.
    @mock.patch("platiagro.deployments.requests.post")
    @mock.patch("platiagro.deployments.get_project_by_name")
    @mock.patch("platiagro.deployments.get_deployment_by_name")
    def test_run_deployments(self, mock_post, mock_get_project, mock_get_deployments):
        mock_response = mock.Mock(
            status_code=200
            )
        mock_get_project.return_value = {
            "uuid": "bc4a0874-4a6b-4e20-bd7e-ed00c51fd8ea",
            "name": "projects01"
        }

        mock_get_deployments.return_value = {
            "uuid": "c1406cc2-f82e-4d97-b82a-274880b2ce2d",
            "name": "deployments01"
        }

        mock_post.json.return_value = {
            "uuid": "f72a180a-2b8f-4a16-b2c6-e93a27ff8da0",
            "operators": {
                "a4a16d55-f745-4e06-9273-d26c16b20269": {
                    "status": "Pending",
                    "parameters": {
                        "dataset": "null",
                        "features_to_filter": [
                            "nox"
                        ]
                    },
                    "taskId": "c03dfde0-e6bb-4e83-a370-8caa99c79cff"
                },
                "deployment": {
                    "status": "Pending",
                    "parameters": {}
                }
            },
            "createdAt": "2021-05-18T00:21:16+00:00",
            "deploymentId": "c2ac8d9b-8a0a-4a41-ac8b-48a5685f6c86"
        }
        mock_post.return_value = mock_response
        # Estamos com dificuldade para mockar o teste run_deployments, estamos procurando uma alternativa
        # result = run_deployments("projects01", "deployments01")

        # self.assertTrue(isinstance(result, dict))
        # self.assertEqual(result.status_code, 200)