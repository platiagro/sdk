# -*- coding: utf-8 -*-
from io import BytesIO
from os import environ
from unittest import TestCase
from uuid import uuid4

import matplotlib.pyplot as plt
from minio.error import BucketAlreadyOwnedByYou
import numpy as np

from platiagro import list_figures, save_figure
from platiagro.util import BUCKET_NAME, MINIO_CLIENT

RUN_ID = str(uuid4())


class TestFigures(TestCase):

    def setUp(self):
        """Prepares a figure for tests."""
        self.make_bucket()
        self.empty_bucket()
        self.create_mock_figure()

    def empty_bucket(self):
        for obj in MINIO_CLIENT.list_objects(BUCKET_NAME, prefix="", recursive=True):
            MINIO_CLIENT.remove_object(BUCKET_NAME, obj.object_name)

    def make_bucket(self):
        try:
            MINIO_CLIENT.make_bucket(BUCKET_NAME)
        except BucketAlreadyOwnedByYou:
            pass

    def create_mock_figure(self):
        file = BytesIO(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\xc0\x00\x00\x00\xba\x08\x02\x00\x00\x00w\x07\xd5\xf7\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05\x00\x00\x00\tpHYs\x00\x00\x0e\xc3\x00\x00\x0e\xc3\x01\xc7o\xa8d\x00\x00\x01\xbaIDATx^\xed\xd21\x01\x00\x00\x0c\xc3\xa0\xf97\xdd\x89\xc8\x0b\x1a\xb8A \x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\x89@$\x02\x91\x08D"\x10\xc1\xf6\x1a:\xf5\xe1\x06\x89A\xdf\x00\x00\x00\x00IEND\xaeB`\x82')
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="experiments/test/operators/test/figure-123456.png",
            data=file,
            length=file.getbuffer().nbytes,
        )

    def test_list_figures(self):
        with self.assertRaises(TypeError):
            list_figures()

        environ["EXPERIMENT_ID"] = "test"
        with self.assertRaises(TypeError):
            list_figures()

        environ["OPERATOR_ID"] = "test"
        result = list_figures()
        self.assertTrue(isinstance(result, list))

        del environ["EXPERIMENT_ID"]
        del environ["OPERATOR_ID"]

        result = list_figures(experiment_id="test", operator_id="test")
        self.assertTrue(isinstance(result, list))

    def test_save_figure(self):
        with self.assertRaises(TypeError):
            save_figure(experiment_id="test", operator_id="test",
                        figure="path/to/figure")

        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)
        fig, ax = plt.subplots()
        ax.plot(t, s)

        with self.assertRaises(TypeError):
            save_figure(fig)

        environ["EXPERIMENT_ID"] = "test"
        with self.assertRaises(TypeError):
            save_figure(fig)

        environ["OPERATOR_ID"] = "test"
        save_figure(fig)

        del environ["EXPERIMENT_ID"]
        del environ["OPERATOR_ID"]

        save_figure(experiment_id="test", operator_id="test", figure=fig)

    def test_list_figures_run_id(self):
        environ["EXPERIMENT_ID"] = "test"
        environ["OPERATOR_ID"] = "test"
        environ["RUN_ID"] = RUN_ID
        result = list_figures()
        self.assertTrue(isinstance(result, list))

        del environ["EXPERIMENT_ID"]
        del environ["OPERATOR_ID"]
        del environ["RUN_ID"]

        result = list_figures(experiment_id="test", operator_id="test", run_id=RUN_ID)
        self.assertTrue(isinstance(result, list))

        result = list_figures(experiment_id="test", operator_id="test", run_id="latest")
        self.assertTrue(isinstance(result, list))

    def test_save_figure_run_id(self):
        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)
        fig, ax = plt.subplots()
        ax.plot(t, s)

        environ["EXPERIMENT_ID"] = "test"
        environ["OPERATOR_ID"] = "test"
        environ["RUN_ID"] = RUN_ID
        save_figure(fig)

        del environ["EXPERIMENT_ID"]
        del environ["OPERATOR_ID"]
        del environ["RUN_ID"]

        save_figure(figure=fig, experiment_id="test", operator_id="test", run_id=RUN_ID)
