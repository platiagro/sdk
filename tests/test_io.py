# -*- coding: utf-8 -*-
import unittest
import unittest.mock as mock

import platiagro.io


class TestIO(unittest.TestCase):
    @mock.patch("platiagro.io.zipfile")
    def test_unzip_file(self, mock_zipfile):
        """
        Should extract zipfile to given path.
        """
        path_to_zip_file = "/path/to.zip"
        directory_to_extract_to = "/path/to/extract"

        platiagro.io.unzip_to_folder(
            path_to_zip_file=path_to_zip_file,
            directory_to_extract_to=directory_to_extract_to,
        )

        mock_zipfile.ZipFile.assert_called_once_with(path_to_zip_file, "r")
        mock_zipfile.ZipFile.return_value.__enter__.assert_called_once_with()
        mock_zipfile.ZipFile.return_value.__enter__.return_value.extractall.assert_called_once_with(
            "/path/to/extract"
        )
