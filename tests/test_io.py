
import os
import shutil
from unittest import TestCase
from uuid import uuid4

from .test_util import download_zipfile
from platiagro.io import unzip_to_folder

RUN_ID = str(uuid4())

class TestIO(TestCase):

    def setUp(self):
        pass

    def test_zip_download(self):
        zip_file = 'ocr_dataset.zip'
        tmp_folder = './tmp/data'
        download_zipfile(zip_file)
        unzip_to_folder(f'{tmp_folder}/{zip_file}',tmp_folder)
        list_of_files = os.listdir(tmp_folder)
        print(list_of_files)

        if any(['.png' in f for f in list_of_files]):
            shutil.rmtree('tmp')
        else:
            raise Exception('Unzip Failed')


    


 