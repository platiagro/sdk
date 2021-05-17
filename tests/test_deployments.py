from unittest import TestCase, mock, main
import requests
from platiagro.util import PROJECTS_ENDPOINT
from platiagro.deployments import list_projects, get_project_by_name, list_deployments, \
    get_deployment_by_name, run_deployments


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
        for project in mock_response.json.return_value["projects"]:
            if project["name"] == "projects01":
                return project

        result = get_project_by_name(project)
        self.assertTrue(isinstance(result, dict))

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

    @mock.patch("platiagro.deployments.requests.get")
    def test_get_deployment_name(self, mock_get):
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
        for deployment in mock_response.json.return_value["deployments"]:
            if deployment["name"] == "deployments01":
                return deployment

        result = get_deployment_by_name("projects01", deployment)
        self.assertTrue(isinstance(result, dict))

    @mock.patch("platiagro.deployments.requests.post")
    def test_run_deployments(self, mock_post):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "runs": [{
                "uuid": "7c186204-23d8-4a83-8e51-68f99145e06a",
                "operators": {
                    "c5078756-feab-4dfd-8d05-b8966024dd99": {
                        "status": "Succeeded",
                        "parameters": {
                            "dataset": "null",
                            "features_to_filter": [
                                "nox"
                            ]
                        },
                        "taskId": "c03dfde0-e6bb-4e83-a370-8caa99c79cff"
                    },
                    "deployment": {
                        "status": "Succeeded",
                        "parameters": {}
                    }
                },
                "createdAt": "2021-05-17T17:50:13+00:00"
            }]
        }
        mock_post.return_value = mock_response

        response = requests.post(
            url=f"{PROJECTS_ENDPOINT}/bc4a0874-4a6b-4e20-bd7e-ed00c51fd8ea/deployments/c1406cc2-f82e-4d97-b82a-274880b2ce2d/runs", json=mock_post.return_value)
        mock_post.assert_called_with(
            url=f"{PROJECTS_ENDPOINT}/bc4a0874-4a6b-4e20-bd7e-ed00c51fd8ea/deployments/c1406cc2-f82e-4d97-b82a-274880b2ce2d/runs", json=mock_post.return_value)
        self.assertFalse(isinstance(response, list))
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    main()
