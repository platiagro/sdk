# -*- coding: utf-8 -*-
import os
import unittest
import unittest.mock as mock

import pandas as pd

import platiagro
from platiagro.util import BUCKET_NAME, MINIO_CLIENT

import tests.util as util


class TestMetrics(unittest.TestCase):
    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.NO_SUCH_KEY_ERROR,
    )
    def test_list_metrics_not_found(self, mock_get_object, mock_make_bucket):
        """
        Should list a single metric "accuracy".
        """
        experiment_id = "UNK"
        operator_id = "UNK"
        with self.assertRaises(FileNotFoundError):
            platiagro.list_metrics(experiment_id=experiment_id, operator_id=operator_id)

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        mock_get_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/{experiment_id}/operators/{operator_id}/metrics.json",
        )

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    def test_list_metrics_success(self, mock_get_object, mock_make_bucket):
        """
        Should list a single metric "accuracy".
        """
        os.environ["EXPERIMENT_ID"] = "UNK"
        os.environ["OPERATOR_ID"] = "UNK"
        os.environ["RUN_ID"] = "latest"

        metrics = platiagro.list_metrics()
        self.assertIsInstance(metrics, list)
        self.assertDictEqual(metrics[0], {"accuracy": 1.0})

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        mock_get_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/{os.environ['EXPERIMENT_ID']}/operators/{os.environ['OPERATOR_ID']}/{os.environ['RUN_ID']}/metrics.json",
        )

        del os.environ["EXPERIMENT_ID"]
        del os.environ["OPERATOR_ID"]
        del os.environ["RUN_ID"]

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    @mock.patch.object(
        MINIO_CLIENT,
        "put_object",
        side_effect=util.put_object_side_effect,
    )
    def test_save_metrics_type_error(
        self, mock_put_object, mock_get_object, mock_make_bucket
    ):
        """
        Should raise TypeError when metric is an invalid object type.
        """
        os.environ["EXPERIMENT_ID"] = "UNK"
        os.environ["OPERATOR_ID"] = "UNK"
        os.environ["RUN_ID"] = "latest"

        with self.assertRaises(TypeError):
            platiagro.save_metrics(accuracy=lambda x: "WUT")

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        mock_get_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/{os.environ['EXPERIMENT_ID']}/operators/{os.environ['OPERATOR_ID']}/metrics.json",
        )

        del os.environ["EXPERIMENT_ID"]
        del os.environ["OPERATOR_ID"]
        del os.environ["RUN_ID"]

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    @mock.patch.object(
        MINIO_CLIENT,
        "put_object",
        side_effect=util.put_object_side_effect,
    )
    def test_save_metrics_success(
        self, mock_put_object, mock_get_object, mock_make_bucket
    ):
        """
        Should raise TypeError when metric is an invalid object type.
        """
        experiment_id = "UNK"
        operator_id = "UNK"
        run_id = "latest"
        accuracy = 0.5

        platiagro.save_metrics(
            accuracy=accuracy,
            experiment_id=experiment_id,
            operator_id=operator_id,
            run_id=run_id,
        )

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        mock_get_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/{experiment_id}/operators/{operator_id}/.metadata",
        )

        mock_put_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/{experiment_id}/operators/{operator_id}/metrics.json",
            data=mock.ANY,
            length=mock.ANY,
        )

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    @mock.patch.object(
        MINIO_CLIENT,
        "put_object",
        side_effect=util.put_object_side_effect,
    )
    def test_save_metrics_env_variables_success(
        self, mock_put_object, mock_get_object, mock_make_bucket
    ):
        """
        Should save metrics sucessfully.
        """
        os.environ["EXPERIMENT_ID"] = "UNK"
        os.environ["OPERATOR_ID"] = "UNK"
        os.environ["RUN_ID"] = "latest"
        accuracy = 0.5
        scores = pd.Series([1.0, 0.5, 0.1])
        r2_score = -3.0

        platiagro.save_metrics(accuracy=accuracy)
        platiagro.save_metrics(scores=scores)
        platiagro.save_metrics(r2_score=r2_score)

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        mock_get_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/{os.environ['EXPERIMENT_ID']}/operators/{os.environ['OPERATOR_ID']}/metrics.json",
        )

        mock_put_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/UNK/operators/UNK/metrics.json",
            data=mock.ANY,
            length=mock.ANY,
        )

        del os.environ["EXPERIMENT_ID"]
        del os.environ["OPERATOR_ID"]
        del os.environ["RUN_ID"]
