# -*- coding: utf-8 -*-
import zipfile


def unzip_to_folder(path_to_zip_file: str, directory_to_extract_to: str):
    """Unzips .zip file into a given directory.

    Args:
        path_to_zip_file (str): path to zipfile.
        directory_to_extract_to (str): destination path.
    """
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
