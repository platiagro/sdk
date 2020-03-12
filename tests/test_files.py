# -*- coding: utf-8 -*-
import unittest
from io import BytesIO
from os import SEEK_SET, getenv
from unittest import TestCase

from joblib import dump
from minio import Minio

from platiagro import load_model, save_model
from platiagro.util import BUCKET_NAME, MINIO_CLIENT
from platiagro.files import list_files, save_file, load_file


BUCKET = "anonymous"
PREFIX = "predictions"
FILE_NAME = "arq_carregado.csv"
FILE_NAME2 = "arq_novo_para_teste.csv"
fileIO = BytesIO(b"01/01/2000,5.1,3.5,1.4,0.2,Iris-setosa\n" +
               b"01/01/2001,4.9,3.0,1.4,0.2,Iris-setosa\n" +
               b"01/01/2002,4.7,3.2,1.3,0.2,Iris-setosa\n" +
               b"01/01/2003,4.6,3.1,1.5,0.2,Iris-setosa")



class TestFiles(unittest.TestCase):

    client = Minio(
        getenv("MINIO_ENDPOINT"),
        access_key = getenv("MINIO_ACCESS_KEY"),
        secret_key = getenv("MINIO_SECRET_KEY"),
        region = getenv("MINIO_REGION_NAME", "us-east-1"),
        secure = False,
    )


    def setUp(self):
        """Prepares a dataset for tests."""
        self.make_bucket()
        try:
            save_file(BUCKET, PREFIX, FILE_NAME, fileIO)
            save_file(BUCKET, PREFIX, FILE_NAME2, fileIO)
        except:
            pass

    def make_bucket(self):
        try:
            MINIO_CLIENT.make_bucket("anonymous")
        except:
            pass


    def test_list_files_bucket_name_blank(self):
        self.assertEqual(list_files("", "x"), "bucket_name should not be empty")

    def test_list_files_bucket_name_unknown(self):
        self.assertEqual(list_files("nobody", "x"), "list_files failed")
        

    def test_list_files_pref_blank(self):
        self.assertEqual(list_files("x", ""), "pref should not be empty")

    def test_list_files_pref_unknown(self):
        self.assertEqual(list_files("x", "nobody"), "list_files failed")


    def test_list_files_ok(self):
        self.assertNotEqual(list_files(BUCKET, PREFIX), "0")
#--------------------------------------------------------------------------------------------
    def test_save_file_bucket_name_blank(self):
        with self.assertRaises(ValueError):
            save_file("", "x", "x", fileIO)

    def test_save_file_bucket_name_unknown(self):
        with self.assertRaises(FileNotFoundError):
            save_file("nobody", PREFIX, FILE_NAME, fileIO)


    def test_save_file_pref_blank(self):
        with self.assertRaises(ValueError):
            save_file("x", "", "x", fileIO)

    def test_save_file_pref_unknown(self):
        with self.assertRaises(FileNotFoundError):
            save_file(BUCKET, "nobody", FILE_NAME, fileIO)


    def test_save_file_file_name_blank(self):
        with self.assertRaises(ValueError):
            save_file("x", "x", "", fileIO)

    def test_save_file_file_name_unknown(self):
        with self.assertRaises(FileNotFoundError):
            save_file(BUCKET, "x", "nobody", fileIO)


    def test_save_file_ok(self):
        with self.assertRaises(FileNotFoundError):
            save_file(BUCKET, PREFIX, FILE_NAME, fileIO)
#--------------------------------------------------------------------------------------------
    def test_load_file_bucket_name_blank(self):
        with self.assertRaises(ValueError):
            load_file("", "x", FILE_NAME)

    def test_load_file_bucket_name_unknown(self):
        with self.assertRaises(FileNotFoundError):
            load_file("nobody", PREFIX, FILE_NAME)


    def test_load_file_pref_blank(self):
        with self.assertRaises(ValueError):
            load_file("x", "", FILE_NAME)

    def test_load_file_pref_unknown(self):
        with self.assertRaises(FileNotFoundError):
            load_file("x", "nobody", FILE_NAME)


    def test_load_file_file_name_blank(self):
        with self.assertRaises(ValueError):
            load_file("x", "x", "")

    def test_load_file_file_name_unknown(self):
        with self.assertRaises(FileNotFoundError):
            load_file(BUCKET, PREFIX, "nobody")


    def test_load_file_ok(self):
        with self.assertRaises(FileNotFoundError):
            load_file(BUCKET, PREFIX, FILE_NAME2)

