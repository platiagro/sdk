# -*- coding: utf-8 -*-
from unittest import TestCase
import unittest

from platiagro.tasks import create_task


class TestTasks(TestCase):
    def test_create_tasks(self):
        response = create_task(
            "teste-task",
        )
        self.assertIsInstance(response, dict)


if __name__ == "__main__":
    unittest.main()
