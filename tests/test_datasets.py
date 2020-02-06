# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps
from os import getenv
from unittest import TestCase

import pandas as pd
from minio import Minio

from platiagro import BUCKET_NAME, client, load_dataset, save_dataset


class TestDatasets(TestCase):

    def setUp(self):
        """Prepares a dataset for tests."""
        self.make_bucket()
        self.create_mock_dataset()

    def make_bucket(self):
        try:
            client.make_bucket(BUCKET_NAME)
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
        }
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name="datasets/iris",
            data=file,
            length=file.getbuffer().nbytes,
            metadata=metadata,
        )

    def test_load_dataset(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset("UNK")

        expected_df = pd.DataFrame({
            "col0": ["01/01/2000", "01/01/2001", "01/01/2002", "01/01/2003"],
            "col1": [5.1, 4.9, 4.7, 4.6],
            "col2": [3.5, 3.0, 3.2, 3.1],
            "col3": [1.4, 1.4, 1.3, 1.5],
            "col4": [0.2, 0.2, 0.2, 0.2],
            "col5": ["Iris-setosa", "Iris-setosa", "Iris-setosa", "Iris-setosa"],
        })
        expected_featuretypes = ["DateTime", "Numerical", "Numerical",
                                 "Numerical", "Numerical", "Categorical"]
        result_df, result_featuretypes = load_dataset("iris")
        self.assertTrue(result_df.equals(expected_df))
        self.assertListEqual(result_featuretypes, expected_featuretypes)

    def test_save_dataset(self):
        df = pd.DataFrame({"col0": []})
        save_dataset("test", df)

        df = pd.DataFrame({"col0": ["2000-01-01", "2001-01-01", "2002-01-01"]})
        save_dataset("test", df)
