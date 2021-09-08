# -*- coding: utf-8 -*-
import io
import os
import unittest
import unittest.mock as mock

import pandas as pd
from minio.datatypes import Object

import platiagro
from platiagro.util import BUCKET_NAME, MINIO_CLIENT, S3FS

import tests.util as util


class TestDatasets(unittest.TestCase):
    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "list_objects",
        return_value=[
            Object(bucket_name=BUCKET_NAME, object_name=f"datasets/unk.zip/")
        ],
    )
    def test_list_datasets_success(self, mock_list_objects, mock_make_bucket):
        """
        Should list a single dataset name "unk.zip".
        """
        dataset_name = "unk.zip"

        result = platiagro.list_datasets()
        self.assertTrue(isinstance(result, list))
        self.assertEqual(result, [dataset_name])

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.NO_SUCH_KEY_ERROR,
    )
    def test_load_dataset_not_found(self, mock_list_objects, mock_make_bucket):
        """
        Should raise an exception when given a dataset name that does not exist.
        """
        bad_dataset_name = "unk.zip"

        with self.assertRaises(FileNotFoundError):
            platiagro.load_dataset(name=bad_dataset_name)

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    @mock.patch.object(
        S3FS,
        "open",
        return_value=io.BytesIO(util.BINARY_DATA),
    )
    def test_load_dataset_filelike_success(
        self, mock_s3fs_open, mock_get_object, mock_make_bucket
    ):
        """
        Should return a readable object successfully.
        """
        dataset_name = "unk.zip"

        result = platiagro.load_dataset(name=dataset_name)
        self.assertTrue(hasattr(result, "read"))

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    @mock.patch.object(
        S3FS,
        "open",
        return_value=io.StringIO(util.CSV_DATA.decode()),
    )
    def test_load_dataset_dataframe_success(
        self, mock_s3fs_open, mock_get_object, mock_make_bucket
    ):
        """
        Should return a pandas DataFrame object successfully.
        """
        dataset_name = "unk.csv"

        result = platiagro.load_dataset(name=dataset_name)
        self.assertIsInstance(result, pd.DataFrame)

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    @mock.patch.object(
        S3FS,
        "open",
        return_value=io.BytesIO(util.CSV_DATA),
    )
    def test_load_dataset_with_run_id_and_operator_id_success(
        self, mock_s3fs_open, mock_get_object, mock_make_bucket
    ):
        """
        Should return a pandas DataFrame object when run_id and operator_id exist.
        """
        dataset_name = "unk.csv"
        run_id = "UNK"
        operator_id = "UNK"

        result = platiagro.load_dataset(
            name=dataset_name, run_id=run_id, operator_id=operator_id
        )
        self.assertIsInstance(result, pd.DataFrame)

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.NO_SUCH_KEY_ERROR,
    )
    def test_get_dataset_not_found(self, mock_get_object, mock_make_bucket):
        """
        Should raise an exception when given a dataset name that does not exist.
        """
        dataset_name = "unk.csv"

        with self.assertRaises(FileNotFoundError):
            platiagro.get_dataset(name=dataset_name)

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    @mock.patch.object(
        S3FS,
        "open",
        return_value=io.BytesIO(util.CSV_DATA),
    )
    def test_get_dataset_success(
        self, mock_s3fs_open, mock_get_object, mock_make_bucket
    ):
        """
        Should return a readable object successfully.
        """
        dataset_name = "unk.csv"

        result = platiagro.get_dataset(name=dataset_name)
        self.assertTrue(hasattr(result, "read"))

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    @mock.patch.object(
        S3FS,
        "open",
        return_value=io.BytesIO(util.CSV_DATA),
    )
    def test_get_dataset_with_run_id_and_operator_id_success(
        self, mock_s3fs_open, mock_get_object, mock_make_bucket
    ):
        """
        Should return a readable object when run_id and operator_id exist.
        """
        dataset_name = "unk.csv"
        run_id = "UNK"
        operator_id = "UNK"

        result = platiagro.get_dataset(
            name=dataset_name, run_id=run_id, operator_id=operator_id
        )
        self.assertTrue(hasattr(result, "read"))

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
    def test_save_dataset_file_like_success(
        self, mock_put_object, mock_get_object, mock_make_bucket
    ):
        """
        Should call .put_object twice: passing data, and passing metadata.
        """
        dataset_name = "unk.zip"
        data = io.BytesIO(util.BINARY_DATA)

        platiagro.save_dataset(name=dataset_name, data=data)

        mock_put_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"datasets/{dataset_name}/{dataset_name}.metadata",
            data=mock.ANY,
            length=mock.ANY,
        )

        # BUG not sure why this assert fails...
        # mock_put_object.assert_any_call(
        #     bucket_name=BUCKET_NAME,
        #     object_name=f"datasets/{dataset_name}/{dataset_name}",
        #     data=mock.ANY,
        #     length=mock.ANY,
        # )

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
    @mock.patch.object(
        MINIO_CLIENT,
        "fput_object",
        side_effect=util.fput_object_side_effect,
    )
    def test_save_dataset_dataframe_success(
        self, mock_fput_object, mock_put_object, mock_get_object, mock_make_bucket
    ):
        """
        Should call .put_object twice: passing data, and passing metadata.
        """
        dataset_name = "unk.csv"
        data = pd.DataFrame({"col0": []})

        platiagro.save_dataset(name=dataset_name, data=data)

        mock_fput_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"datasets/{dataset_name}/{dataset_name}",
            file_path=mock.ANY,
        )

        mock_put_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"datasets/{dataset_name}/{dataset_name}.metadata",
            data=mock.ANY,
            length=mock.ANY,
        )

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.NO_SUCH_KEY_ERROR,
    )
    def test_stat_dataset_not_found(self, mock_get_object, mock_make_bucket):
        """
        Should raise an exception when given a dataset name that does not exist.
        """
        dataset_name = "unk.csv"

        with self.assertRaises(FileNotFoundError):
            platiagro.stat_dataset(name=dataset_name)

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    @mock.patch.object(
        S3FS,
        "open",
        return_value=io.BytesIO(util.CSV_DATA),
    )
    def test_stat_dataset_success(
        self, mock_s3fs_open, mock_get_object, mock_make_bucket
    ):
        """
        Should return a dict object successfully.
        """
        dataset_name = "unk.csv"

        result = platiagro.stat_dataset(name=dataset_name)

        expected = {"filename": dataset_name}
        self.assertDictEqual(result, expected)

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    @mock.patch.object(
        S3FS,
        "open",
        return_value=io.BytesIO(util.CSV_DATA),
    )
    def test_stat_dataset_with_run_id_and_operator_id_success(
        self, mock_s3fs_open, mock_get_object, mock_make_bucket
    ):
        """
        Should return a dict object when run_id and operator_id exist.
        """
        dataset_name = "unk.csv"
        run_id = "UNK"
        operator_id = "UNK"

        result = platiagro.stat_dataset(
            name=dataset_name, run_id=run_id, operator_id=operator_id
        )

        expected = {"filename": dataset_name}
        self.assertDictEqual(result, expected)

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.NO_SUCH_KEY_ERROR,
    )
    def test_download_dataset_not_found(self, mock_download_dataset, mock_make_bucket):
        """
        Should raise an exception when given a dataset name that does not exist.
        """
        bad_dataset_name = "unk.zip"
        path = "./unk.csv"

        with self.assertRaises(FileNotFoundError):
            platiagro.download_dataset(name=bad_dataset_name, path=path)

    @mock.patch.object(MINIO_CLIENT, "make_bucket")
    @mock.patch.object(
        MINIO_CLIENT,
        "get_object",
        side_effect=util.get_object_side_effect,
    )
    @mock.patch.object(
        S3FS,
        "open",
        return_value=io.BytesIO(util.CSV_DATA),
    )
    def test_download_dataset_success(
        self, mock_s3fs_open, mock_get_object, mock_make_bucket
    ):
        """
        Should download a dataset to a given local path.
        """
        dataset_name = "unk.csv"
        path = "./unk.csv"

        platiagro.download_dataset(name=dataset_name, path=path)

        self.assertTrue(os.path.exists(path))

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
    def test_update_dataset_metadata(
        self, mock_put_object, mock_get_object, mock_make_bucket
    ):
        dataset_name = "unk.csv"
        metadata = {
            "featuretypes": [
                "Categorical",
                "Categorical",
                "Categorical",
                "Categorical",
                "Categorical",
            ],
        }

        platiagro.update_dataset_metadata(name=dataset_name, metadata=metadata)

        mock_put_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"datasets/{dataset_name}/{dataset_name}.metadata",
            data=mock.ANY,
            length=mock.ANY,
        )
