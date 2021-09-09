# -*- coding: utf-8 -*-
import os
import unittest
import unittest.mock as mock

import platiagro
from platiagro.util import BUCKET_NAME, MINIO_CLIENT

import tests.util as util


class TestModels(unittest.TestCase):
    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    def test_load_model_success(self, mock_get_object, mock_make_bucket):
        """
        Should return a MockModel object when experiment_id and operator_id exist.
        """
        experiment_id = "UNK"
        operator_id = "UNK"

        model = platiagro.load_model(
            experiment_id=experiment_id, operator_id=operator_id
        )

        self.assertIsInstance(model, dict)
        self.assertIsInstance(model["model"], util.MockModel)

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    def test_load_model_with_env_variables_success(
        self, mock_get_object, mock_make_bucket
    ):
        """
        Should return a MockModel object when experiment_id and operator_id exist.
        """
        os.environ["EXPERIMENT_ID"] = "UNK"
        os.environ["OPERATOR_ID"] = "UNK"

        model = platiagro.load_model()

        self.assertIsInstance(model, dict)
        self.assertIsInstance(model["model"], util.MockModel)

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "put_object",
        side_effect=util.put_object_side_effect,
    )
    def test_save_model_type_error(self, mock_put_object, mock_make_bucket):
        """
        Should raise an exception when given an invalid object type.
        """
        model = util.MockModel()

        with self.assertRaises(TypeError):
            platiagro.save_model(model)

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "put_object",
        side_effect=util.put_object_side_effect,
    )
    def test_save_model_success(self, mock_put_object, mock_make_bucket):
        """
        Should call .put_object using given variables.
        """
        experiment_id = "UNK"
        operator_id = "UNK"
        model = util.MockModel()

        platiagro.save_model(
            experiment_id=experiment_id, operator_id=operator_id, model=model
        )

        mock_put_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/{experiment_id}/operators/{operator_id}/model.joblib",
            data=mock.ANY,
            length=mock.ANY,
        )

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "put_object",
        side_effect=util.put_object_side_effect,
    )
    def test_save_model_with_env_variable_success(
        self, mock_put_object, mock_make_bucket
    ):
        """
        Should call .put_object using given variables.
        """
        os.environ["EXPERIMENT_ID"] = "UNK"
        os.environ["OPERATOR_ID"] = "UNK"
        model = util.MockModel()

        platiagro.save_model(model=model)

        mock_put_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/{os.environ['EXPERIMENT_ID']}/operators/{os.environ['OPERATOR_ID']}/model.joblib",
            data=mock.ANY,
            length=mock.ANY,
        )
