# -*- coding: utf-8 -*-
from io import BytesIO
from os import environ
from unittest import TestCase
from uuid import uuid4

import base64
from minio.error import BucketAlreadyOwnedByYou

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
        file = BytesIO(
            b'<svg viewBox=\'0 0 125 80\' xmlns=\'http://www.w3.org/2000/svg\'>\n'
            b'<text y="75" font-size="100" font-family="serif"><![CDATA[10]]></text>\n'
            b'</svg>\n'
            )
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

    def test_save_figure_base64(self):
        with open("./tests/figure.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            environ["EXPERIMENT_ID"] = "testFigureBase64"
            environ["OPERATOR_ID"] = "testFigureBase64"
            environ["RUN_ID"] = RUN_ID
            save_figure(figure=encoded_string.decode('utf-8'), extension='png')
            save_figure(figure=encoded_string.decode('utf-8'), extension='svg', run_id="latest")

        result = list_figures()
        self.assertTrue(len(result) == 2)
        result = list_figures(run_id="latest")
        self.assertTrue(len(result) == 2)

    def test_save_html_figure(self):
        environ["EXPERIMENT_ID"] = "testHtmlFigure"
        environ["OPERATOR_ID"] = "testHtmlFigure"
        environ["RUN_ID"] = RUN_ID
        html_figure = '<html><body></body></html>'
        save_figure(figure=html_figure, extension='html')

        expected = ['data:text/html;base64,PGh0bWw+PGJvZHk+PC9ib2R5PjwvaHRtbD4=']
        self.assertEqual(expected, list_figures())

        del environ["EXPERIMENT_ID"]
        del environ["OPERATOR_ID"]
        del environ["RUN_ID"]

    def test_save_html_figure_deploy_monit_id(self):
        environ["DEPLOYMENT_ID"] = "testHtmlFigure"
        environ["MONITORING_ID"] = "testHtmlFigure"
        html_figure = '<html><body></body></html>'
        save_figure(figure=html_figure, extension='html')

        expected = ['data:text/html;base64,PGh0bWw+PGJvZHk+PC9ib2R5PjwvaHRtbD4=']
        self.assertEqual(expected, list_figures())

        del environ["DEPLOYMENT_ID"]
        del environ["MONITORING_ID"]
