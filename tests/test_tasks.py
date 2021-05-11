from unittest import TestCase, mock, main
from platiagro.tasks import create_task


class TestTasks(TestCase):
    @mock.patch("platiagro.tasks.requests.post")
    def test_create_tasks(self, mock_post):
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {
            "name": "TestTasks"
        }
        mock_post.return_value = mock_response

        response = create_task("TestTasks")
        expected = {"name": "TestTasks"}
        self.assertDictEqual(expected, response)


if __name__ == "__main__":
    main()
