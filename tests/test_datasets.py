# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps
from os import getcwd
from os.path import join, exists
from unittest import TestCase

from minio.error import BucketAlreadyOwnedByYou
import pandas as pd

from platiagro import download_dataset, list_datasets, load_dataset, \
    save_dataset, stat_dataset, DATETIME, CATEGORICAL, NUMERICAL
from platiagro.util import BUCKET_NAME, MINIO_CLIENT


class TestDatasets(TestCase):

    def setUp(self):
        """Prepares a dataset for tests."""
        self.make_bucket()
        self.empty_bucket()
        self.create_mock_dataset()

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
            object_name="datasets/mock/19700101000000000000.csv",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )
        metadata = {
            "columns": self.mock_columns(),
            "featuretypes": self.mock_featuretypes(),
            "filename": "19700101000000000000.csv",
        }
        buffer = BytesIO(dumps(metadata).encode())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="datasets/mock/.metadata",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )

    def test_list_datasets(self):
        result = list_datasets()
        self.assertTrue(isinstance(result, list))

    def test_load_dataset(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset("UNK")

        expected = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        result = load_dataset("mock")
        self.assertTrue(result.equals(expected))

    def test_download_dataset(self):
        with self.assertRaises(FileNotFoundError):
            download_dataset("UNK")

        expected = join(getcwd(), "mock")
        result = download_dataset("mock")
        self.assertEqual(result, expected)
        self.assertTrue(exists(result))

    def test_save_dataset(self):
        df = pd.DataFrame({"col0": []})
        save_dataset("test", df)

        df = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        save_dataset("test", df)

        df = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        save_dataset("test", df, metadata={
            "featuretypes": [CATEGORICAL for ft in self.mock_featuretypes()],
        })

        df = pd.DataFrame({"col0": []})
        save_dataset("test", df, read_only=True)

        with self.assertRaises(PermissionError):
            df = pd.DataFrame({"col0": []})
            save_dataset("test", df)

    def test_stat_dataset(self):
        with self.assertRaises(FileNotFoundError):
            stat_dataset("UNK")

        expected = {
            "columns": self.mock_columns(),
            "featuretypes": self.mock_featuretypes(),
            "filename": "19700101000000000000.csv",
        }
        result = stat_dataset("mock")
        self.assertDictEqual(result, expected)
