from os import name
from unittest import TestCase
from platiagro.tasks import create_task


class TestTasks(TestCase):
    def test_create_tasks(self):
        response = create_task(name="TesteTasks")
        self.assertIsInstance(response, dict)


if __name__ == "__main__":
    TestCase.main()
