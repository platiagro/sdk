# -*- coding: utf-8 -*-
from io import BytesIO
from os import SEEK_SET, environ
from unittest import TestCase

from joblib import dump
from minio.error import BucketAlreadyOwnedByYou

from platiagro import load_model, save_model
from platiagro.util import BUCKET_NAME, MINIO_CLIENT


class MockModel:
    def predict(self, x):
        return True


class TestModels(TestCase):

    def setUp(self):
        """Prepares a model for tests."""
        self.make_bucket()
        self.create_mock_model()

    def make_bucket(self):
        try:
            MINIO_CLIENT.make_bucket(BUCKET_NAME)
        except BucketAlreadyOwnedByYou:
            pass

    def create_mock_model(self):
        model = {"model": MockModel()}
        buffer = BytesIO()
        dump(model, buffer)
        buffer.seek(0, SEEK_SET)
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="experiments/mock/model",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )

    def test_load_model(self):
        with self.assertRaises(FileNotFoundError):
            load_model("UNK")

        with self.assertRaises(TypeError):
            load_model()

        environ["EXPERIMENT_ID"] = "mock"
        model = load_model()
        self.assertIsInstance(model, dict)
        self.assertIsInstance(model["model"], MockModel)
        del environ["EXPERIMENT_ID"]

        model = load_model("mock")
        self.assertIsInstance(model, dict)
        self.assertIsInstance(model["model"], MockModel)

    def test_save_model(self):
        with self.assertRaises(TypeError):
            model = MockModel()
            save_model(model)

        environ["EXPERIMENT_ID"] = "test"
        model = MockModel()
        save_model(model=model)
        del environ["EXPERIMENT_ID"]

        model = MockModel()
        save_model(experiment_id="test",
                   model=model)

        model = MockModel()
        save_model(experiment_id="test",
                   model=model)
