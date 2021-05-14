from unittest import TestCase, mock, main
from platiagro.deployments import list_projects, get_project_name, list_deployments, \
    get_deployment_name, run_deployments


class TestTasks(TestCase):
    def test_list_projects(self):
        result = list_projects()
        self.assertTrue(isinstance(result, dict))

    @mock.patch("platiagro.deployments.requests.get")
    def test_mock_list_projects(self, mock_get):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "name": "Teste"
        }
        mock_get.return_value = mock_response

        response = list_projects()
        expected = {"name": "Teste"}
        self.assertDictEqual(expected, response)

    @mock.patch("platiagro.deployments.requests.get")
    def test_get_project_name(self, mock_get):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "projects": [
                {"name": "teste"},
                {"name": "teste2"}
            ]
        }
        mock_get.return_value = mock_response

        result = get_project_name("teste")
        self.assertTrue(isinstance(result, dict))

    @mock.patch("platiagro.deployments.requests.get")
    def test_mock_list_deployments(self, mock_get):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "projects": [
                {
                    "name": "teste",
                    "uuid": "123456"
                },
                {
                    "name": "teste2",
                    "uuid": "12345678"
                }
            ]
        }
        mock_get.return_value = mock_response

        result = list_deployments("teste")
        self.assertTrue(isinstance(result, dict))

    @mock.patch("platiagro.deployments.requests.get")
    def test_get_deployment_name(self, mock_get):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "projects": [
                {
                    "name": "teste",
                    "uuid": "123456"
                },
                {
                    "name": "teste2",
                    "uuid": "123456"
                }
            ],
            "deployments": [
                {
                    "uuid": "c1406cc2-f82e-4d97-b82a-274880b2ce2d",
                    "name": "teste"
                }
            ]
        }
        mock_get.return_value = mock_response

        result = get_deployment_name("teste", "teste2")
        self.assertTrue(isinstance(result, dict))

    @mock.patch("platiagro.deployments.requests.get")
    def test_run_deployments(self, mock_get):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "projects": [
                {
                    "name": "teste",
                    "uuid": "123456"
                },
                {
                    "name": "teste2",
                    "uuid": "123456"
                }
            ],
            "deployments": [
                {
                    "uuid": "c1406cc2-f82e-4d97-b82a-274880b2ce2d",
                    "name": "teste"
                }
            ]
        }
        mock_get.return_value = mock_response
        mock_get.return_value.status_code = 200

        result = run_deployments("teste", "teste2")
        self.assertTrue(isinstance(result, dict))


if __name__ == "__main__":
    main()
