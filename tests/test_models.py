# -*- coding: utf-8 -*-
from io import BytesIO
from os import SEEK_SET, getenv
from unittest import TestCase

from joblib import dump
from minio import Minio

from platiagro import load_model, save_model
from platiagro.util import BUCKET_NAME, MINIO_CLIENT


class MockModel:
    def predict(x):
        return True


class TestModels(TestCase):

    def setUp(self):
        """Prepares a dataset for tests."""
        self.make_bucket()
        self.create_mock_model()

    def make_bucket(self):
        try:
            MINIO_CLIENT.make_bucket(BUCKET_NAME)
        except:
            pass

    def create_mock_model(self):
        model = MockModel()
        model_buffer = BytesIO()
        dump(model, model_buffer)
        model_buffer.seek(0, SEEK_SET)
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="models/mock",
            data=model_buffer,
            length=model_buffer.getbuffer().nbytes,
        )

    def test_load_model(self):
        with self.assertRaises(FileNotFoundError):
            load_model("UNK")

        model = load_model("mock")
        self.assertIsInstance(model, MockModel)

    def test_save_model(self):
        model = MockModel()
        save_model("test", model)
