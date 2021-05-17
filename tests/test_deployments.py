from unittest import TestCase, mock, main
from platiagro.deployments import list_projects, get_project_name, list_deployments, \
    get_deployment_name, run_deployments


class TestTasks(TestCase):
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

        result = get_project_name("projects01")
        self.assertTrue(isinstance(result, dict))

    @mock.patch("platiagro.deployments.requests.get")
    def test_mock_list_deployments(self, mock_get):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "projects": [
                {
                    "name": "projects02",
                    "uuid": "bc4a0874-4a6b-4e20-bd7e-ed00c51fd8ea"
                },
                {
                    "name": "projects01",
                    "uuid": "a8ab15b1-7a90-4f18-b18d-e14f7422c938"
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

        result = get_deployment_name("projects02", "deployments01")
        self.assertTrue(isinstance(result, dict))

    @mock.patch("platiagro.deployments.requests.get")
    def test_run_deployments(self, mock_get):
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

        result = run_deployments("projects01", "deployments01")
        self.assertEqual(result.status_code, 200)


if __name__ == "__main__":
    main()
