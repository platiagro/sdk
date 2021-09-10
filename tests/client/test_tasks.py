# -*- coding: utf-8 -*-
import unittest
import unittest.mock as mock

import platiagro.client.tasks


class TestTasks(unittest.TestCase):
    @mock.patch("platiagro.client.tasks.requests.post")
    def test_create_tasks(self, mock_post):
        """
        Should create a task.
        """
        mock_response = mock.Mock(status_code=200)
        mock_response.json.return_value = {"name": "TestTasks"}
        mock_post.return_value = mock_response

        response = platiagro.client.tasks.create_task("TestTasks")
        self.assertEqual(response.status_code, 200)
