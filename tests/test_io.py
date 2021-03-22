import platiagro.io


class TestIO:

    def test_unzip_file(self, mocker):
        mock_zipfile = mocker.MagicMock(name="zipfile")
        mocker.patch("platiagro.io.zipfile", new=mock_zipfile)

        platiagro.io.unzip_to_folder("/path/to.zip", "/path/to/extract")

        mock_zipfile.ZipFile.assert_called_once_with("/path/to.zip", "r")
        mock_zipfile.ZipFile.return_value.__enter__.assert_called_once_with()
        mock_zipfile.ZipFile.return_value.__enter__.return_value.extractall. \
            assert_called_once_with("/path/to/extract")
