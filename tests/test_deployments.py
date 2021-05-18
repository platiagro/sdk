from unittest import TestCase, mock, main
import requests
from platiagro.util import PROJECTS_ENDPOINT
from platiagro.deployments import list_projects, get_project_by_name, list_deployments, \
    get_deployment_by_name


class TestRuns(TestCase):
    @mock.patch("platiagro.deployments.requests.get")
    def test_mock_list_projects(self, mock_get):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "name": "projects01"
        }
        mock_get.return_value = mock_response

        response = list_projects()
        expected = {"name": "projects01"}
        self.assertDictEqual(expected, response)

    @mock.patch("platiagro.deployments.requests.get")
    def test_get_project_name(self, mock_get):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "projects": [
                {"name": "projects01"},
                {"name": "projects02"}
            ]
        }
        mock_get.return_value = mock_response

        result = get_project_by_name("projects01")
        self.assertIsInstance(result, dict)

    @mock.patch("platiagro.deployments.requests.get")
    def test_mock_list_deployments(self, mock_get):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "projects": [
                {
                    "uuid": "bc4a0874-4a6b-4e20-bd7e-ed00c51fd8ea",
                    "name": "projects01"
                },
                {
                    "uuid": "a8ab15b1-7a90-4f18-b18d-e14f7422c938",
                    "name": "projects02"
                }
            ]
        }
        mock_get.return_value = mock_response

        result = list_deployments("projects01")
        self.assertTrue(isinstance(result, dict))
        self.assertIsInstance(result, dict)

    @mock.patch("platiagro.deployments.get_deployment_by_name")
    @mock.patch("platiagro.deployments.requests.get")
    def test_get_deployment_name(self, mock_get, mock_requests):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "projects": [
                {
                    "uuid": "bc4a0874-4a6b-4e20-bd7e-ed00c51fd8ea",
                    "name": "projects01"
                },
                {
                    "uuid": "a8ab15b1-7a90-4f18-b18d-e14f7422c938",
                    "name": "projects02"
                }
            ],
            "deployments": [
                {
                    "uuid": "c1406cc2-f82e-4d97-b82a-274880b2ce2d",
                    "name": "deployments01"
                }
            ]
        }
        mock_get.return_value = mock_response
        mock_requests.return_value = mock_response

        result = get_deployment_by_name("projects01", "deployments01")
        self.assertIsInstance(mock_response.json.return_value, dict)
        self.assertIsInstance(result, dict)

    @mock.patch("platiagro.deployments.requests.post")
    def test_run_deployments(self, mock_post):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
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

        response = requests.post(
            url=f"{PROJECTS_ENDPOINT}/bc4a0874-4a6b-4e20-bd7e-ed00c51fd8ea/deployments/c1406cc2-f82e-4d97-b82a-274880b2ce2d/runs")
        self.assertFalse(isinstance(response, list))
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    main()
