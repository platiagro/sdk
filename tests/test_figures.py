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
        file = BytesIO(b'<svg viewBox=\'0 0 125 80\' xmlns=\'http://www.w3.org/2000/svg\'>\n  <text y="75" font-size="100" font-family="serif"><![CDATA[10]]></text>\n</svg>\n')
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="experiments/test/operators/test/figure-123456.svg",
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
