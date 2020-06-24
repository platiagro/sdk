# -*- coding: utf-8 -*-
from base64 import b64decode
from io import BytesIO
from json import dumps
from unittest import TestCase
from uuid import uuid4
from zipfile import ZipFile

from minio.error import BucketAlreadyOwnedByYou
import pandas as pd

from platiagro import list_datasets, load_dataset, save_dataset, stat_dataset, \
    DATETIME, CATEGORICAL, NUMERICAL
from platiagro.util import BUCKET_NAME, MINIO_CLIENT

RUN_ID = str(uuid4())
OPERATOR_ID = str(uuid4())
MOCK_IMAGE = b64decode("R0lGODlhPQBEAPeoAJosM//AwO/AwHVYZ/z595kzAP/s7P+goOXMv8+fhw/v739/f+8PD98fH/8mJl+fn/9ZWb8/PzWlwv///6wWGbImAPgTEMImIN9gUFCEm/gDALULDN8PAD6atYdCTX9gUNKlj8wZAKUsAOzZz+UMAOsJAP/Z2ccMDA8PD/95eX5NWvsJCOVNQPtfX/8zM8+QePLl38MGBr8JCP+zs9myn/8GBqwpAP/GxgwJCPny78lzYLgjAJ8vAP9fX/+MjMUcAN8zM/9wcM8ZGcATEL+QePdZWf/29uc/P9cmJu9MTDImIN+/r7+/vz8/P8VNQGNugV8AAF9fX8swMNgTAFlDOICAgPNSUnNWSMQ5MBAQEJE3QPIGAM9AQMqGcG9vb6MhJsEdGM8vLx8fH98AANIWAMuQeL8fABkTEPPQ0OM5OSYdGFl5jo+Pj/+pqcsTE78wMFNGQLYmID4dGPvd3UBAQJmTkP+8vH9QUK+vr8ZWSHpzcJMmILdwcLOGcHRQUHxwcK9PT9DQ0O/v70w5MLypoG8wKOuwsP/g4P/Q0IcwKEswKMl8aJ9fX2xjdOtGRs/Pz+Dg4GImIP8gIH0sKEAwKKmTiKZ8aB/f39Wsl+LFt8dgUE9PT5x5aHBwcP+AgP+WltdgYMyZfyywz78AAAAAAAD///8AAP9mZv///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAKgALAAAAAA9AEQAAAj/AFEJHEiwoMGDCBMqXMiwocAbBww4nEhxoYkUpzJGrMixogkfGUNqlNixJEIDB0SqHGmyJSojM1bKZOmyop0gM3Oe2liTISKMOoPy7GnwY9CjIYcSRYm0aVKSLmE6nfq05QycVLPuhDrxBlCtYJUqNAq2bNWEBj6ZXRuyxZyDRtqwnXvkhACDV+euTeJm1Ki7A73qNWtFiF+/gA95Gly2CJLDhwEHMOUAAuOpLYDEgBxZ4GRTlC1fDnpkM+fOqD6DDj1aZpITp0dtGCDhr+fVuCu3zlg49ijaokTZTo27uG7Gjn2P+hI8+PDPERoUB318bWbfAJ5sUNFcuGRTYUqV/3ogfXp1rWlMc6awJjiAAd2fm4ogXjz56aypOoIde4OE5u/F9x199dlXnnGiHZWEYbGpsAEA3QXYnHwEFliKAgswgJ8LPeiUXGwedCAKABACCN+EA1pYIIYaFlcDhytd51sGAJbo3onOpajiihlO92KHGaUXGwWjUBChjSPiWJuOO/LYIm4v1tXfE6J4gCSJEZ7YgRYUNrkji9P55sF/ogxw5ZkSqIDaZBV6aSGYq/lGZplndkckZ98xoICbTcIJGQAZcNmdmUc210hs35nCyJ58fgmIKX5RQGOZowxaZwYA+JaoKQwswGijBV4C6SiTUmpphMspJx9unX4KaimjDv9aaXOEBteBqmuuxgEHoLX6Kqx+yXqqBANsgCtit4FWQAEkrNbpq7HSOmtwag5w57GrmlJBASEU18ADjUYb3ADTinIttsgSB1oJFfA63bduimuqKB1keqwUhoCSK374wbujvOSu4QG6UvxBRydcpKsav++Ca6G8A6Pr1x2kVMyHwsVxUALDq/krnrhPSOzXG1lUTIoffqGR7Goi2MAxbv6O2kEG56I7CSlRsEFKFVyovDJoIRTg7sugNRDGqCJzJgcKE0ywc0ELm6KBCCJo8DIPFeCWNGcyqNFE06ToAfV0HBRgxsvLThHn1oddQMrXj5DyAQgjEHSAJMWZwS3HPxT/QMbabI/iBCliMLEJKX2EEkomBAUCxRi42VDADxyTYDVogV+wSChqmKxEKCDAYFDFj4OmwbY7bDGdBhtrnTQYOigeChUmc1K3QTnAUfEgGFgAWt88hKA6aCRIXhxnQ1yg3BCayK44EWdkUQcBByEQChFXfCB776aQsG0BIlQgQgE8qO26X1h8cEUep8ngRBnOy74E9QgRgEAC8SvOfQkh7FDBDmS43PmGoIiKUUEGkMEC/PJHgxw0xH74yx/3XnaYRJgMB8obxQW6kL9QYEJ0FIFgByfIL7/IQAlvQwEpnAC7DtLNJCKUoO/w45c44GwCXiAFB/OXAATQryUxdN4LfFiwgjCNYg+kYMIEFkCKDs6PKAIJouyGWMS1FSKJOMRB/BoIxYJIUXFUxNwoIkEKPAgCBZSQHQ1A2EWDfDEUVLyADj5AChSIQW6gu10bE/JG2VnCZGfo4R4d0sdQoBAHhPjhIB94v/wRoRKQWGRHgrhGSQJxCS+0pCZbEhAAOw==")


class TestDatasets(TestCase):

    def setUp(self):
        """Prepares a dataset for tests."""
        self.make_bucket()
        self.empty_bucket()
        self.create_mock_dataset1()
        self.create_mock_dataset2()
        self.create_mock_dataset3()

    def tearDown(self):
        self.empty_bucket()

    def empty_bucket(self):
        for obj in MINIO_CLIENT.list_objects(BUCKET_NAME, prefix="", recursive=True):
            MINIO_CLIENT.remove_object(BUCKET_NAME, obj.object_name)

    def make_bucket(self):
        try:
            MINIO_CLIENT.make_bucket(BUCKET_NAME)
        except BucketAlreadyOwnedByYou:
            pass

    def mock_columns(self, size=1e3):
        return [f"col{i}" for i in range(int(size))]

    def mock_values(self, size=1e3):
        values = ["01/01/2000", 5.1, 3.5, 1.4, 0.2, "Iris-setosa"]
        return [values[i % len(values)] for i in range(int(size))]

    def mock_featuretypes(self, size=1e3):
        ftypes = [DATETIME, NUMERICAL, NUMERICAL,
                  NUMERICAL, NUMERICAL, CATEGORICAL]
        return [ftypes[i % len(ftypes)] for i in range(int(size))]

    def create_mock_dataset1(self, size=1e2):
        header = ",".join(self.mock_columns()) + "\n"
        rows = "\n".join([",".join([str(v) for v in self.mock_values()])
                          for x in range(int(size))])
        buffer = BytesIO((header + rows).encode())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="datasets/mock.csv/mock.csv",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )
        metadata = {
            "columns": self.mock_columns(),
            "featuretypes": self.mock_featuretypes(),
            "filename": "mock.csv",
            "run_id": RUN_ID,
        }
        buffer = BytesIO(dumps(metadata).encode())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="datasets/mock.csv/mock.csv.metadata",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )
        MINIO_CLIENT.copy_object(
            bucket_name=BUCKET_NAME,
            object_name=f"datasets/mock.csv/runs/{RUN_ID}/operators/{OPERATOR_ID}/mock.csv/mock.csv",
            object_source=f"/{BUCKET_NAME}/datasets/mock.csv/mock.csv",
        )
        MINIO_CLIENT.copy_object(
            bucket_name=BUCKET_NAME,
            object_name=f"datasets/mock.csv/runs/{RUN_ID}/operators/{OPERATOR_ID}/mock.csv/mock.csv.metadata",
            object_source=f"/{BUCKET_NAME}/datasets/mock.csv/mock.csv.metadata",
        )

    def create_mock_dataset2(self):
        with ZipFile("mock.zip", "w") as zipf:
            zipf.writestr("mock.gif", MOCK_IMAGE)

        MINIO_CLIENT.fput_object(
            bucket_name=BUCKET_NAME,
            object_name="datasets/mock.zip/mock.zip",
            file_path="mock.zip",
        )
        metadata = {
            "filename": "mock.zip",
        }
        buffer = BytesIO(dumps(metadata).encode())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="datasets/mock.zip/mock.zip.metadata",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )

    def create_mock_dataset3(self):
        with open("mock.jpg", 'wb') as imagef:
            imagef.write(MOCK_IMAGE)

        MINIO_CLIENT.fput_object(
            bucket_name=BUCKET_NAME,
            object_name="datasets/mock.jpg/mock.jpg",
            file_path="mock.jpg",
        )
        metadata = {
            "filename": "mock.jpg",
        }
        buffer = BytesIO(dumps(metadata).encode())
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name="datasets/mock.jpg/mock.jpg.metadata",
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )        

    def test_list_datasets(self):
        result = list_datasets()
        self.assertTrue(isinstance(result, list))

    def test_load_dataset(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset("UNK")

        # UnicodeDecodeError
        result = load_dataset("mock.zip")
        self.assertIsInstance(result, BytesIO)

        # EmptyDataError
        result = load_dataset("mock.jpg")
        self.assertIsInstance(result, BytesIO)

        result = load_dataset("mock.csv")
        expected = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        self.assertTrue(result.equals(expected))

        result = load_dataset("mock.csv", run_id=RUN_ID, operator_id=OPERATOR_ID)
        expected = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        self.assertTrue(result.equals(expected))

        result = load_dataset("mock.csv", run_id="latest", operator_id=OPERATOR_ID)
        expected = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        self.assertTrue(result.equals(expected))

    def test_save_dataset(self):
        data = BytesIO(MOCK_IMAGE)
        save_dataset("test.zip", data=data)

        df = pd.DataFrame({"col0": []})
        save_dataset("test.csv", df)

        df = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        save_dataset("test.csv", df)

        df = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        save_dataset("test.csv", df, metadata={
            "featuretypes": [CATEGORICAL for ft in self.mock_featuretypes()],
        })

        df = pd.DataFrame({"col0": []})
        save_dataset("test.csv", df, read_only=True)

        with self.assertRaises(PermissionError):
            df = pd.DataFrame({"col0": []})
            save_dataset("test.csv", df)

        df = pd.DataFrame(
            data=[self.mock_values() for x in range(int(1e2))],
            columns=self.mock_columns(),
        )
        save_dataset("newtest.csv", df=df, run_id=RUN_ID, operator_id=OPERATOR_ID)

    def test_stat_dataset(self):
        with self.assertRaises(FileNotFoundError):
            stat_dataset("UNK")

        result = stat_dataset("mock.zip")
        expected = {
            "filename": "mock.zip",
        }
        self.assertDictEqual(result, expected)

        result = stat_dataset("mock.csv")
        expected = {
            "columns": self.mock_columns(),
            "featuretypes": self.mock_featuretypes(),
            "filename": "mock.csv",
            "run_id": RUN_ID,
        }
        self.assertDictEqual(result, expected)

        result = stat_dataset("mock.csv", run_id="latest", operator_id=OPERATOR_ID)
        expected = {
            "columns": self.mock_columns(),
            "featuretypes": self.mock_featuretypes(),
            "filename": "mock.csv",
            "run_id": RUN_ID,
        }
        self.assertDictEqual(result, expected)

        result = stat_dataset("mock.csv", run_id=RUN_ID, operator_id=OPERATOR_ID)
        expected = {
            "columns": self.mock_columns(),
            "featuretypes": self.mock_featuretypes(),
            "filename": "mock.csv",
            "run_id": RUN_ID,
        }
        self.assertDictEqual(result, expected)
