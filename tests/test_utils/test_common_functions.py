"""
test_common_functions.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 26-02-2025
"""

import pytest
import os
import requests
from unittest.mock import patch, mock_open, MagicMock
import time
from comproscanner.utils.common_functions import (
    get_paper_metadata_from_oaworks,
    return_error_message,
    write_timeout_file,
)


class TestGetPaperMetadataFromOAWorks:
    """Test cases for get_paper_metadata_from_oaworks function"""

    @patch("requests.request")
    def test_successful_request(self, mock_request):
        """Test a successful API request"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": "Test Paper Title",
            "container-title": "Test Journal",
            "publisher": "Test Publisher",
        }
        mock_request.return_value = mock_response
        title, journal_name, publisher = get_paper_metadata_from_oaworks(
            "10.1000/test.doi"
        )
        assert title == "Test Paper Title"
        assert journal_name == "Test Journal"
        assert publisher == "Test Publisher"
        mock_request.assert_called_once_with(
            "GET", "https://bg.api.oa.works/metadata?id=10.1000/test.doi"
        )

    @patch("requests.request")
    def test_failed_request(self, mock_request):
        """Test a failed API request"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_request.return_value = mock_response
        title, journal_name, publisher = get_paper_metadata_from_oaworks(
            "10.1000/test.doi"
        )
        assert title == ""
        assert journal_name == ""
        assert publisher == ""

    @patch("requests.request")
    def test_exception_handling(self, mock_request):
        """Test exception handling"""
        mock_request.side_effect = Exception("Connection error")
        title, journal_name, publisher = get_paper_metadata_from_oaworks(
            "10.1000/test.doi"
        )
        assert title == ""
        assert journal_name == ""
        assert publisher == ""

    @patch("requests.request")
    def test_missing_data_fields(self, mock_request):
        """Test handling of missing data fields"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "container-title": "Test Journal",
        }
        mock_request.return_value = mock_response
        title, journal_name, publisher = get_paper_metadata_from_oaworks(
            "10.1000/test.doi"
        )
        assert title == ""
        assert journal_name == "Test Journal"
        assert publisher == ""


class TestReturnErrorMessage:
    """Test cases for return_error_message function"""

    def test_none_variable(self):
        """Test when the variable is None"""
        with pytest.raises(ValueError) as exc_info:
            return_error_message(None)
        assert "The variable is missing." in str(exc_info.value)

    def test_main_property_keyword_message(self):
        """Test error message for main_property_keyword"""
        message = return_error_message("main_property_keyword")
        assert "main_property_keyword cannot be None" in message
        assert "Example: 'piezoelectric'" in message

    def test_property_keywords_message(self):
        """Test error message for property_keywords"""
        message = return_error_message("property_keywords")
        assert "property_keywords cannot be None" in message
        assert "exact_keywords" in message
        assert "substring_keywords" in message

    def test_scopus_api_key_message(self):
        """Test error message for scopus_api_key"""
        message = return_error_message("scopus_api_key")
        assert "SCOPUS_API_KEY is not set in the environment variables" in message

    def test_wiley_api_key_message(self):
        """Test error message for wiley_api_key"""
        message = return_error_message("wiley_api_key")
        assert "WILEY_API_KEY is not set in the environment variables" in message

    def test_springer_api_key_message(self):
        """Test error message for springer_open_access_api_key"""
        message = return_error_message("springer_open_access_api_key")
        assert (
            "SPRINGER_OPENACCESS_API_KEY is not set in the environment variables"
            in message
        )


class TestWriteTimeoutFile:
    """Test cases for write_timeout_file function"""

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("time.sleep")
    def test_write_doi_to_file_existing_dir(
        self, mock_sleep, mock_file, mock_makedirs, mock_exists
    ):
        """Test writing DOI to timeout file when directory exists"""
        mock_exists.return_value = True
        write_timeout_file("10.1000/test.doi", "/path/to/timeout.txt")
        mock_exists.assert_called_once_with("/path/to")
        mock_makedirs.assert_not_called()
        mock_file.assert_called_once_with("/path/to/timeout.txt", "a")
        mock_file().write.assert_called_once_with("10.1000/test.doi\n")
        mock_sleep.assert_called_once_with(1)

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("time.sleep")
    def test_write_doi_to_file_create_dir(
        self, mock_sleep, mock_file, mock_makedirs, mock_exists
    ):
        """Test writing DOI to timeout file with directory creation"""
        mock_exists.return_value = False
        write_timeout_file("10.1000/test.doi", "/path/to/timeout.txt")
        mock_exists.assert_called_once_with("/path/to")
        mock_makedirs.assert_called_once_with("/path/to")
        mock_file.assert_called_once_with("/path/to/timeout.txt", "a")
        mock_file().write.assert_called_once_with("10.1000/test.doi\n")
        mock_sleep.assert_called_once_with(1)
