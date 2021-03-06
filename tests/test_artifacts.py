# -*- coding: utf-8 -*-
import io
import os
from unittest import TestCase

from minio.error import BucketAlreadyOwnedByYou

from platiagro import download_artifact
from platiagro.util import BUCKET_NAME, MINIO_CLIENT


class TestArtifacts(TestCase):

    def setUp(self):
        self.make_bucket()
        buffer = io.BytesIO(b"mock")
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="artifacts/mock.txt",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )

    def tearDown(self):
        MINIO_CLIENT.remove_object(
            bucket_name=BUCKET_NAME,
            object_name="artifacts/mock.txt",
        )

    def make_bucket(self):
        try:
            MINIO_CLIENT.make_bucket(BUCKET_NAME)
        except BucketAlreadyOwnedByYou:
            pass

    def test_download_artifact(self):
        with self.assertRaises(FileNotFoundError):
            download_artifact("unk.zip", "./unk.zip")

        download_artifact("mock.txt", "./mock-dest.txt")
        self.assertTrue(os.path.exists("./mock-dest.txt"))
