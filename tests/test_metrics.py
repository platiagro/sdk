# -*- coding: utf-8 -*-
from io import BytesIO
from json import dumps
from os import environ
from unittest import TestCase
from uuid import uuid4

import pandas as pd
from minio.error import S3Error
from platiagro import list_metrics, save_metrics
from platiagro.util import BUCKET_NAME, MINIO_CLIENT

EXPERIMENT_ID = str(uuid4())
OPERATOR_ID = str(uuid4())
RUN_ID = str(uuid4())


class TestMetrics(TestCase):

    def setUp(self):
        """Prepares metrics for tests."""
        self.make_bucket()
        self.create_mock_metrics()

    def make_bucket(self):
        try:
            MINIO_CLIENT.make_bucket(BUCKET_NAME)
        except S3Error as err:
            if err.code == "BucketAlreadyOwnedByYou":
                pass

    def create_mock_metrics(self):
        metric = [{"accuracy": 1.0}]
        buffer = BytesIO(dumps(metric).encode())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/{EXPERIMENT_ID}/operators/{OPERATOR_ID}/metrics.json",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )

    def test_list_metrics(self):
        with self.assertRaises(FileNotFoundError):
            list_metrics("UNK", "UNK")

        environ["EXPERIMENT_ID"] = EXPERIMENT_ID
        environ["OPERATOR_ID"] = OPERATOR_ID
        metrics = list_metrics()
        self.assertIsInstance(metrics, list)
        self.assertDictEqual(metrics[0], {"accuracy": 1.0})
        del environ["EXPERIMENT_ID"]
        del environ["OPERATOR_ID"]

        metrics = list_metrics(experiment_id=EXPERIMENT_ID, operator_id=OPERATOR_ID)
        self.assertIsInstance(metrics, list)
        self.assertDictEqual(metrics[0], {"accuracy": 1.0})

    def test_save_metrics(self):
        environ["EXPERIMENT_ID"] = EXPERIMENT_ID
        environ["OPERATOR_ID"] = OPERATOR_ID
        with self.assertRaises(TypeError):
            save_metrics(accuracy=lambda x: "WUT")

        scores = pd.Series([1.0, 0.5, 0.1])

        save_metrics(accuracy=0.5)
        save_metrics(scores=scores)
        save_metrics(r2_score=-3.0)
        del environ["EXPERIMENT_ID"]
        del environ["OPERATOR_ID"]

        save_metrics(experiment_id=EXPERIMENT_ID,
                     operator_id=OPERATOR_ID,
                     accuracy=0.5)
        save_metrics(experiment_id=EXPERIMENT_ID,
                     operator_id=OPERATOR_ID,
                     scores=scores)
        save_metrics(experiment_id=EXPERIMENT_ID,
                     operator_id=OPERATOR_ID,
                     r2_score=-3.0)

    def test_run_id(self):
        metrics = list_metrics(experiment_id=EXPERIMENT_ID,
                               operator_id=OPERATOR_ID,
                               run_id="latest")
        self.assertTrue(isinstance(metrics, list))
        self.assertEqual(metrics, [])

        environ["EXPERIMENT_ID"] = EXPERIMENT_ID
        environ["OPERATOR_ID"] = OPERATOR_ID
        environ["RUN_ID"] = RUN_ID

        with self.assertRaises(FileNotFoundError):
            list_metrics()

        save_metrics(accuracy=0.75)
        metrics = list_metrics()
        self.assertTrue(isinstance(metrics, list))
        self.assertDictEqual(metrics[0], {"accuracy": 0.75})

        del environ["EXPERIMENT_ID"]
        del environ["OPERATOR_ID"]
        del environ["RUN_ID"]

        save_metrics(experiment_id=EXPERIMENT_ID,
                     operator_id=OPERATOR_ID,
                     run_id=RUN_ID,
                     r2_score=1.0)
        metrics = list_metrics(experiment_id=EXPERIMENT_ID,
                               operator_id=OPERATOR_ID,
                               run_id=RUN_ID)
        self.assertTrue(isinstance(metrics, list))
        self.assertDictEqual(metrics[1], {"r2_score": 1.0})

        save_metrics(experiment_id=EXPERIMENT_ID,
                     operator_id=OPERATOR_ID,
                     run_id="latest",
                     r2_score=2.0)
        metrics = list_metrics(experiment_id=EXPERIMENT_ID,
                               operator_id=OPERATOR_ID,
                               run_id="latest")
        self.assertTrue(isinstance(metrics, list))
        self.assertDictEqual(metrics[2], {"r2_score": 2.0})
