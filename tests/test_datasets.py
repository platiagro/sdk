# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps
from os import getenv
from unittest import TestCase

import pandas as pd
from minio import Minio

from platiagro import list_datasets, load_dataset, save_dataset, stat_dataset
from platiagro.util import BUCKET_NAME, MINIO_CLIENT


class TestDatasets(TestCase):

    def setUp(self):
        """Prepares a dataset for tests."""
        self.make_bucket()
        self.empty_bucket()
        self.create_mock_dataset()

    def empty_bucket(self):
        try:
            for obj in MINIO_CLIENT.list_objects(BUCKET_NAME, prefix="", recursive=True):
                MINIO_CLIENT.remove_object(BUCKET_NAME, obj.object_name)
        except:
            pass

    def make_bucket(self):
        try:
            MINIO_CLIENT.make_bucket(BUCKET_NAME)
        except:
            pass

    def create_mock_dataset(self):
        file = BytesIO(b"01/01/2000,5.1,3.5,1.4,0.2,Iris-setosa\n" +
                       b"01/01/2001,4.9,3.0,1.4,0.2,Iris-setosa\n" +
                       b"01/01/2002,4.7,3.2,1.3,0.2,Iris-setosa\n" +
                       b"01/01/2003,4.6,3.1,1.5,0.2,Iris-setosa")
        columns = ["col0", "col1", "col2", "col3", "col4", "col5"]
        featuretypes = ["DateTime", "Numerical", "Numerical",
                        "Numerical", "Numerical", "Categorical"]
        metadata = {
            "columns": dumps(columns),
            "featuretypes": dumps(featuretypes),
            "filename": dumps("iris.data"),
        }
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="datasets/iris",
            data=file,
            length=file.getbuffer().nbytes,
            metadata=metadata,
        )

    def test_list_datasets(self):
        result = list_datasets()
        self.assertTrue(isinstance(result, list))

    def test_load_dataset(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset("UNK")

        expected = pd.DataFrame({
            "col0": ["01/01/2000", "01/01/2001", "01/01/2002", "01/01/2003"],
            "col1": [5.1, 4.9, 4.7, 4.6],
            "col2": [3.5, 3.0, 3.2, 3.1],
            "col3": [1.4, 1.4, 1.3, 1.5],
            "col4": [0.2, 0.2, 0.2, 0.2],
            "col5": ["Iris-setosa", "Iris-setosa", "Iris-setosa", "Iris-setosa"],
        })
        result = load_dataset("iris")
        self.assertTrue(result.equals(expected))

    def test_save_dataset(self):
        df = pd.DataFrame({"col0": []})
        save_dataset("test", df)

        df = pd.DataFrame({
            "col0": ["2000-01-01", "2001-01-01", "2002-01-01", "2003-01-01"],
            "col1": [5.1, 4.9, 4.7, 4.6],
            "col2": [3.5, 3.0, 3.2, 3.1],
            "col3": [1.4, 1.4, 1.3, 1.5],
            "col4": [0.2, 0.2, 0.2, 0.2],
            "col5": ["Iris-setosa", "Iris-setosa", "Iris-setosa", "Iris-setosa"],
        })
        save_dataset("test", df)

        df = pd.DataFrame({"col0": []})
        save_dataset("test", df, metadata={
            "filename": "test.data",
            "featuretypes": ["DateTime", "Numerical", "Numerical",
                             "Numerical", "Numerical", "Categorical"],
        })

    def test_stat_dataset(self):
        with self.assertRaises(FileNotFoundError):
            stat_dataset("UNK")

        expected = {
            "columns": ["col0", "col1", "col2", "col3", "col4", "col5"],
            "featuretypes": ["DateTime", "Numerical", "Numerical",
                             "Numerical", "Numerical", "Categorical"],
            "filename": "iris.data",
        }
        result = stat_dataset("iris")
        self.assertDictEqual(result, expected)
