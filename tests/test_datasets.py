# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps
from unittest import TestCase
from uuid import uuid4

from minio.error import BucketAlreadyOwnedByYou
import pandas as pd

from platiagro import list_datasets, load_dataset, save_dataset, stat_dataset, \
    DATETIME, CATEGORICAL, NUMERICAL
from platiagro.util import BUCKET_NAME, MINIO_CLIENT

RUN_ID = str(uuid4())
OPERATOR_ID = str(uuid4())


class TestDatasets(TestCase):

    def setUp(self):
        """Prepares a dataset for tests."""
        self.make_bucket()
        self.empty_bucket()
        self.create_mock_dataset()

    def tearDown(self):
        self.empty_bucket()

    def empty_bucket(self):
        for obj in MINIO_CLIENT.list_objects(BUCKET_NAME, prefix="", recursive=True):
            MINIO_CLIENT.remove_object(BUCKET_NAME, obj.object_name)

    def make_bucket(self):
        try:
            MINIO_CLIENT.make_bucket(BUCKET_NAME)
        except BucketAlreadyOwnedByYou:
            pass

    def mock_columns(self, size=1e3):
        return [f"col{i}" for i in range(int(size))]

    def mock_values(self, size=1e3):
        values = ["01/01/2000", 5.1, 3.5, 1.4, 0.2, "Iris-setosa"]
        return [values[i % len(values)] for i in range(int(size))]

    def mock_featuretypes(self, size=1e3):
        ftypes = [DATETIME, NUMERICAL, NUMERICAL,
                  NUMERICAL, NUMERICAL, CATEGORICAL]
        return [ftypes[i % len(ftypes)] for i in range(int(size))]

    def create_mock_dataset(self, size=1e2):
        header = ",".join(self.mock_columns()) + "\n"
        rows = "\n".join([",".join([str(v) for v in self.mock_values()])
                          for x in range(int(size))])
        buffer = BytesIO((header + rows).encode())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="datasets/mock.csv/mock.csv",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )
        metadata = {
            "columns": self.mock_columns(),
            "featuretypes": self.mock_featuretypes(),
            "filename": "mock.csv",
            "run_id": RUN_ID,
        }
        buffer = BytesIO(dumps(metadata).encode())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="datasets/mock.csv/mock.csv.metadata",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )
        MINIO_CLIENT.copy_object(
            bucket_name=BUCKET_NAME,
            object_name=f"datasets/mock.csv/runs/{RUN_ID}/operators/{OPERATOR_ID}/mock.csv/mock.csv",
            object_source=f"/{BUCKET_NAME}/datasets/mock.csv/mock.csv",
        )
        MINIO_CLIENT.copy_object(
            bucket_name=BUCKET_NAME,
            object_name=f"datasets/mock.csv/runs/{RUN_ID}/operators/{OPERATOR_ID}/mock.csv/mock.csv.metadata",
            object_source=f"/{BUCKET_NAME}/datasets/mock.csv/mock.csv.metadata",
        )

    def test_list_datasets(self):
        result = list_datasets()
        self.assertTrue(isinstance(result, list))

    def test_load_dataset(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset("UNK")

        result = load_dataset("mock.csv")
        expected = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        self.assertTrue(result.equals(expected))

        result = load_dataset("mock.csv", run_id=RUN_ID, operator_id=OPERATOR_ID)
        expected = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        self.assertTrue(result.equals(expected))

        result = load_dataset("mock.csv", run_id="latest", operator_id=OPERATOR_ID)
        expected = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        self.assertTrue(result.equals(expected))

    def test_save_dataset(self):
        df = pd.DataFrame({"col0": []})
        save_dataset("test.csv", df)

        df = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        save_dataset("test.csv", df)

        df = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        save_dataset("test.csv", df, metadata={
            "featuretypes": [CATEGORICAL for ft in self.mock_featuretypes()],
        })

        df = pd.DataFrame({"col0": []})
        save_dataset("test.csv", df, read_only=True)

        with self.assertRaises(PermissionError):
            df = pd.DataFrame({"col0": []})
            save_dataset("test.csv", df)

        df = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        save_dataset("newtest.csv", df, run_id=RUN_ID, operator_id=OPERATOR_ID)

    def test_stat_dataset(self):
        with self.assertRaises(FileNotFoundError):
            stat_dataset("UNK")

        result = stat_dataset("mock.csv")
        expected = {
            "columns": self.mock_columns(),
            "featuretypes": self.mock_featuretypes(),
            "filename": "mock.csv",
            "run_id": RUN_ID,
        }
        self.assertDictEqual(result, expected)

        result = stat_dataset("mock.csv", run_id="latest", operator_id=OPERATOR_ID)
        expected = {
            "columns": self.mock_columns(),
            "featuretypes": self.mock_featuretypes(),
            "filename": "mock.csv",
            "run_id": RUN_ID,
        }
        self.assertDictEqual(result, expected)

        result = stat_dataset("mock.csv", run_id=RUN_ID, operator_id=OPERATOR_ID)
        expected = {
            "columns": self.mock_columns(),
            "featuretypes": self.mock_featuretypes(),
            "filename": "mock.csv",
            "run_id": RUN_ID,
        }
        self.assertDictEqual(result, expected)
