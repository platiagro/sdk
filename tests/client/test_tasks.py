from unittest import mock, TestCase
from platiagro.client.tasks import create_task


class TestTasks(TestCase):
    @mock.patch("platiagro.client.tasks.requests.post")
    def test_create_tasks(self, mock_post):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "name": "TestTasks"
        }
        mock_post.return_value = mock_response

        response = create_task("TestTasks")
        self.assertEqual(response.status_code, 200)
