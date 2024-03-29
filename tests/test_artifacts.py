# -*- coding: utf-8 -*-
import os
import unittest
import unittest.mock as mock

import platiagro
from platiagro.util import BUCKET_NAME, MINIO_CLIENT

import tests.util as util


class TestArtifacts(unittest.TestCase):
    @mock.patch.object(
        MINIO_CLIENT,
        "fget_object",
        side_effect=util.NO_SUCH_KEY_ERROR,
    )
    def test_download_artifact_not_found(self, mock_fget_object):
        """
        Should raise an exception when given an artifact name that does not exist.
        """
        bad_artifact_name = "unk.zip"
        local_path = "./unk.zip"

        with self.assertRaises(FileNotFoundError):
            platiagro.download_artifact(name=bad_artifact_name, path=local_path)

        mock_fget_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"artifacts/{bad_artifact_name}",
            file_path=local_path,
        )

    @mock.patch.object(
        MINIO_CLIENT,
        "fget_object",
        side_effect=lambda **kwargs: open("unk.zip", "w").close(),
    )
    def test_download_artifact_success(self, mock_fget_object):
        """
        Should download an artifact to a given local path.
        """
        artifact_name = "unk.zip"
        local_path = "./unk.zip"

        platiagro.download_artifact(name=artifact_name, path=local_path)
        self.assertTrue(os.path.exists(local_path))

        mock_fget_object.assert_any_call(
            bucket_name=BUCKET_NAME,
            object_name=f"artifacts/{artifact_name}",
            file_path=local_path,
        )
