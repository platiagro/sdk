from unittest import TestCase, mock, main
from platiagro.tasks import create_task


class TestTasks(TestCase):
    @mock.patch("platiagro.tasks.requests.post")
    def test_create_tasks(self, mock_post):
        my_mocke_response = mock.Mock(status_code=200)
        my_mocke_response.json.return_value = {
            "name": "TestTasks"
        }
        mock_post.return_value = my_mocke_response

        response = create_task("TestTasks")
        expected = {"name": "TestTasks"}
        self.assertDictEqual(expected, response)


if __name__ == "__main__":
    main()
