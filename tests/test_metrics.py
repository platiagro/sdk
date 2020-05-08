# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps
from os import environ
from unittest import TestCase

import numpy as np
import pandas as pd
from minio.error import BucketAlreadyOwnedByYou
from platiagro import list_metrics, save_metrics
from platiagro.util import BUCKET_NAME, MINIO_CLIENT


class TestMetrics(TestCase):

    def setUp(self):
        """Prepares metrics for tests."""
        self.make_bucket()
        self.create_mock_metrics()

    def make_bucket(self):
        try:
            MINIO_CLIENT.make_bucket(BUCKET_NAME)
        except BucketAlreadyOwnedByYou:
            pass

    def create_mock_metrics(self):
        metric = [{"accuracy": 1.0}]
        buffer = BytesIO(dumps(metric).encode())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="experiments/mock/metrics.json",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )

    def test_load_metrics(self):
        with self.assertRaises(FileNotFoundError):
            list_metrics("UNK")

        with self.assertRaises(TypeError):
            list_metrics()

        environ["EXPERIMENT_ID"] = "mock"
        metrics = list_metrics()
        self.assertIsInstance(metrics, list)
        self.assertDictEqual(metrics[0], {"accuracy": 1.0})
        del environ["EXPERIMENT_ID"]

        metrics = list_metrics("mock")
        self.assertIsInstance(metrics, list)
        self.assertDictEqual(metrics[0], {"accuracy": 1.0})

    def test_save_metrics(self):
        with self.assertRaises(TypeError):
            save_metrics(accuracy=1.0)

        data = np.array([[2, 0, 0],
                         [0, 0, 1],
                         [1, 0, 2]])
        labels = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
        confusion_matrix = pd.DataFrame(data, columns=labels, index=labels)

        scores = pd.Series([1.0, 0.5, 0.1])

        environ["EXPERIMENT_ID"] = "test"
        save_metrics(accuracy=1.0)
        save_metrics(scores=scores)
        save_metrics(reset=True,
                     r2_score=-3.0)
        environ["OPERATOR_ID"] = "test"
        save_metrics(confusion_matrix=confusion_matrix)
        del environ["EXPERIMENT_ID"]

        with self.assertRaises(TypeError):
            save_metrics(experiment_id="test",
                         operator_id="test",
                         accuracy=lambda x: "WUT")

        save_metrics(experiment_id="test",
                     operator_id="test",
                     confusion_matrix=confusion_matrix)
        save_metrics(experiment_id="test",
                     operator_id="test",
                     scores=scores)
        save_metrics(experiment_id="test",
                     operator_id="test",
                     reset=True,
                     r2_score=-3.0)
