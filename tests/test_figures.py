# -*- coding: utf-8 -*-
import os
import unittest
import unittest.mock as mock

from minio.datatypes import Object

import platiagro
from platiagro.util import BUCKET_NAME, MINIO_CLIENT

import tests.util as util


class TestFigures(unittest.TestCase):
    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "list_objects",
        return_value=[
            Object(
                bucket_name=BUCKET_NAME,
                object_name=f"experiments/None/operators/None/None/figure.png",
            )
        ],
    )
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    def test_list_figures_success(
        self, mock_get_object, mock_list_objects, mock_make_bucket
    ):
        """
        Should list a single figure.
        """
        result = platiagro.list_figures()

        self.assertTrue(isinstance(result, list))
        self.assertEqual(result, [f"data:image/png;base64,iVBORw0K"])

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        mock_list_objects.assert_any_call(
            BUCKET_NAME, f"experiments/None/operators/None/figure-"
        )

        mock_get_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/None/operators/None/None/figure.png",
        )

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "list_objects",
        return_value=[
            Object(
                bucket_name=BUCKET_NAME,
                object_name=f"experiments/UNK/operators/UNK/latest/figure.png",
            )
        ],
    )
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    def test_list_figures_with_env_variables_success(
        self, mock_get_object, mock_list_objects, mock_make_bucket
    ):
        """
        Should list a single figure using env variables in path.
        """
        os.environ["EXPERIMENT_ID"] = "UNK"
        os.environ["OPERATOR_ID"] = "UNK"
        os.environ["RUN_ID"] = "latest"

        result = platiagro.list_figures()

        self.assertTrue(isinstance(result, list))
        self.assertEqual(result, [f"data:image/png;base64,iVBORw0K"])

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        mock_list_objects.assert_any_call(
            BUCKET_NAME,
            f"experiments/{os.environ['EXPERIMENT_ID']}/operators/{os.environ['OPERATOR_ID']}/{os.environ['RUN_ID']}/figure-",
        )

        mock_get_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/{os.environ['EXPERIMENT_ID']}/operators/{os.environ['OPERATOR_ID']}/{os.environ['RUN_ID']}/figure.png",
        )

        del os.environ["EXPERIMENT_ID"]
        del os.environ["OPERATOR_ID"]
        del os.environ["RUN_ID"]

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "put_object",
        side_effect=util.put_object_side_effect,
    )
    def test_save_figure_with_base64_success(self, mock_put_object, mock_make_bucket):
        """
        Should call .put_object passing a base64 string.
        """
        figure = util.FIGURE_SVG_BASE64
        extension = "svg"

        platiagro.save_figure(figure=figure, extension=extension)

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        # I wish we could assert object_name value, but it has a timestamp... :(
        mock_put_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=mock.ANY,
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
    def test_save_figure_with_base64_and_env_variables_success(
        self, mock_put_object, mock_get_object, mock_make_bucket
    ):
        """
        Should call .put_object passing a base64 string, and using env variables in path.
        """
        figure = util.FIGURE_SVG_BASE64
        extension = "svg"
        os.environ["EXPERIMENT_ID"] = "UNK"
        os.environ["OPERATOR_ID"] = "UNK"
        os.environ["RUN_ID"] = "latest"

        platiagro.save_figure(figure=figure, extension=extension)

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        mock_get_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"experiments/{os.environ['EXPERIMENT_ID']}/operators/{os.environ['OPERATOR_ID']}/.metadata",
        )

        mock_put_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=mock.ANY,
            data=mock.ANY,
            length=mock.ANY,
        )

        del os.environ["EXPERIMENT_ID"]
        del os.environ["OPERATOR_ID"]
        del os.environ["RUN_ID"]

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "put_object",
        side_effect=util.put_object_side_effect,
    )
    def test_save_figure_with_html_success(self, mock_put_object, mock_make_bucket):
        """
        Should call .put_object passing a string containing an html.
        """
        figure = util.FIGURE_HTML
        extension = "html"

        platiagro.save_figure(figure=figure, extension=extension)

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        # I wish we could assert object_name value, but it has a timestamp... :(
        mock_put_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=mock.ANY,
            data=mock.ANY,
            length=mock.ANY,
        )

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "put_object",
        side_effect=util.put_object_side_effect,
    )
    def test_save_figure_with_env_monitoring_success(
        self, mock_put_object, mock_make_bucket
    ):
        """
        Should call .put_object passing a string containing an html, and using env variables in path.
        """
        figure = util.FIGURE_HTML_BASE64
        extension = "html"
        os.environ["DEPLOYMENT_ID"] = "UNK"
        os.environ["MONITORING_ID"] = "UNK"

        platiagro.save_figure(figure=figure, extension=extension)

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        # I wish we could assert object_name value, but it has a timestamp... :(
        mock_put_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=mock.ANY,
            data=mock.ANY,
            length=mock.ANY,
        )

        del os.environ["DEPLOYMENT_ID"]
        del os.environ["MONITORING_ID"]

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "list_objects",
        return_value=[
            Object(
                bucket_name=BUCKET_NAME,
                object_name=f"experiments/UNK/operators/UNK/latest/figure.png",
            )
        ],
    )
    @mock.patch.object(
        MINIO_CLIENT,
        "remove_object",
    )
    def test_delete_figure_success(
        self, mock_remove_object, mock_list_objects, mock_make_bucket
    ):
        """
        Should call .remove_object using env variables in object_name.
        """
        os.environ["EXPERIMENT_ID"] = "UNK"
        os.environ["OPERATOR_ID"] = "UNK"
        os.environ["RUN_ID"] = "latest"

        platiagro.delete_figures()

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        mock_list_objects.assert_any_call(
            BUCKET_NAME,
            f"experiments/{os.environ['EXPERIMENT_ID']}/operators/{os.environ['OPERATOR_ID']}/figure-",
        )

        # I wish we could assert object_name value, but it has a timestamp... :(
        mock_remove_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=mock.ANY,
        )

        del os.environ["EXPERIMENT_ID"]
        del os.environ["OPERATOR_ID"]
        del os.environ["RUN_ID"]

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "list_objects",
        return_value=[
            Object(
                bucket_name=BUCKET_NAME,
                object_name=f"deployments/UNK/monitorings/UNK/figure.png",
            )
        ],
    )
    @mock.patch.object(
        MINIO_CLIENT,
        "remove_object",
    )
    def test_delete_figure_with_env_monitoring_success(
        self, mock_remove_object, mock_list_objects, mock_make_bucket
    ):
        """
        Should call .remove_object using env variables in object_name.
        """
        os.environ["DEPLOYMENT_ID"] = "UNK"
        os.environ["MONITORING_ID"] = "UNK"

        platiagro.delete_figures()

        mock_make_bucket.assert_any_call(BUCKET_NAME)

        mock_list_objects.assert_any_call(
            BUCKET_NAME,
            f"deployments/{os.environ['DEPLOYMENT_ID']}/monitorings/{os.environ['MONITORING_ID']}/figure-",
        )

        # I wish we could assert object_name value, but it has a timestamp... :(
        mock_remove_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=mock.ANY,
        )

        del os.environ["DEPLOYMENT_ID"]
        del os.environ["MONITORING_ID"]
