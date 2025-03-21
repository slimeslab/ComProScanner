"""
test_prepare_iop_files.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 20-03-2025
"""

import os
import shutil
import pytest
from unittest.mock import patch, MagicMock, mock_open
from lxml import etree
import pandas as pd
from comproscanner.utils.prepare_iop_files import PrepareIOPFiles
from comproscanner.utils.error_handler import (
    ValueErrorHandler,
    KeyboardInterruptHandler,
)


@pytest.fixture
def sample_df():
    """Fixture to create a sample DataFrame for testing"""
    return pd.DataFrame(
        {
            "doi": ["10.1088/test_1", "10.1088/test_2", "10.1088/test_3"],
            "publication_name": ["Journal A", "Journal B", "Journal C"],
        }
    )


@pytest.fixture
def mock_xml_content():
    """Fixture to provide sample XML content"""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <article>
        <front>
            <article-meta>
                <article-id pub-id-type="doi">10.1088/test_1</article-id>
            </article-meta>
        </front>
        <body>
            <p>Test content</p>
        </body>
    </article>"""


@pytest.fixture
def mock_xml_content_nobody():
    """Fixture to provide sample XML content without body tag"""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <article>
        <front>
            <article-meta>
                <article-id pub-id-type="doi">10.1088/test_2</article-id>
            </article-meta>
        </front>
    </article>"""


@pytest.fixture
def mock_logger():
    """Fixture to create a mock logger"""
    logger = MagicMock()
    logger.error = MagicMock()
    logger.info = MagicMock()
    logger.debug = MagicMock()
    logger.warning = MagicMock()
    logger.verbose = MagicMock()
    return logger


@pytest.fixture
def iop_paths(tmp_path):
    """Fixture to set up temporary file paths for testing"""
    # Create main directory
    iop_dir = tmp_path / "iop_test"
    iop_dir.mkdir()

    # Create a subdirectory
    sub_dir = iop_dir / "subfolder"
    sub_dir.mkdir()

    return {
        "main_dir": str(iop_dir),
        "sub_dir": str(sub_dir),
        "csv_path": str(tmp_path / "metadata.csv"),
    }


@pytest.fixture
def prepare_iop_files(mock_logger, iop_paths, sample_df):
    """Fixture to create a PrepareIOPFiles instance with mocked dependencies"""
    sample_df.to_csv(iop_paths["csv_path"], index=False)

    with patch("comproscanner.utils.prepare_iop_files.DefaultPaths") as mock_paths:
        mock_paths.return_value.METADATA_CSV_FILENAME = iop_paths["csv_path"]
        mock_paths.return_value.IOP_FOLDERPATH = iop_paths["main_dir"]

        iop_files = PrepareIOPFiles(main_property_keyword="test", logger=mock_logger)
        return iop_files


def test_init_without_keyword(mock_logger):
    """Test initialization without main property keyword"""
    with pytest.raises(ValueErrorHandler):
        PrepareIOPFiles(main_property_keyword=None, logger=mock_logger)


def test_init_without_logger():
    """Test initialization without logger"""
    with pytest.raises(ValueErrorHandler):
        PrepareIOPFiles(main_property_keyword="test", logger=None)


def test_return_xml_root(prepare_iop_files, mock_xml_content, tmp_path):
    """Test the _return_xml_root method with valid XML"""
    xml_file = tmp_path / "test.xml"
    with open(xml_file, "w") as f:
        f.write(mock_xml_content)

    root = prepare_iop_files._return_xml_root(str(xml_file))
    assert root is not None
    assert root.tag == "article"


def test_return_xml_root_error(prepare_iop_files, mock_logger):
    """Test the _return_xml_root method with invalid XML"""
    with patch("lxml.etree.parse", side_effect=Exception("XML parsing error")):
        result = prepare_iop_files._return_xml_root("nonexistent.xml")
        assert result is None
        prepare_iop_files.logger.error.assert_called_once()


def test_get_modified_doi(prepare_iop_files, mock_xml_content, tmp_path):
    """Test the _get_modified_doi method"""
    xml_file = tmp_path / "test.xml"
    with open(xml_file, "w") as f:
        f.write(mock_xml_content)

    modified_doi = prepare_iop_files._get_modified_doi(str(xml_file))
    assert modified_doi == "10.1088_test_1"


def test_get_modified_doi_error(prepare_iop_files):
    """Test the _get_modified_doi method with errors"""
    with patch.object(prepare_iop_files, "_return_xml_root", return_value=None):
        result = prepare_iop_files._get_modified_doi("nonexistent.xml")
        assert result is None


def test_check_body_presence_true(prepare_iop_files, mock_xml_content, tmp_path):
    """Test the _check_body_presence method with XML containing body"""
    xml_file = tmp_path / "test.xml"
    with open(xml_file, "w") as f:
        f.write(mock_xml_content)

    has_body = prepare_iop_files._check_body_presence(str(xml_file))
    assert has_body is True


def test_check_body_presence_false(
    prepare_iop_files, mock_xml_content_nobody, tmp_path
):
    """Test the _check_body_presence method with XML not containing body"""
    xml_file = tmp_path / "test_nobody.xml"
    with open(xml_file, "w") as f:
        f.write(mock_xml_content_nobody)

    has_body = prepare_iop_files._check_body_presence(str(xml_file))
    assert has_body is False


def test_check_body_presence_none(prepare_iop_files):
    """Test the _check_body_presence method with None root"""
    with patch.object(prepare_iop_files, "_return_xml_root", return_value=None):
        has_body = prepare_iop_files._check_body_presence("nonexistent.xml")
        assert has_body is False


def test_get_xml_files(
    prepare_iop_files, iop_paths, mock_xml_content, mock_xml_content_nobody
):
    """Test the _get_xml_files method"""
    # Create test XML files
    main_xml = os.path.join(iop_paths["main_dir"], "main.xml")
    with open(main_xml, "w") as f:
        f.write(mock_xml_content)

    sub_xml = os.path.join(iop_paths["sub_dir"], "sub.xml")
    with open(sub_xml, "w") as f:
        f.write(mock_xml_content_nobody)

    with patch.object(
        prepare_iop_files,
        "_get_modified_doi",
        side_effect=["10.1088_test_1", "10.1088_test_2"],
    ):
        xml_files = prepare_iop_files._get_xml_files(iop_paths["main_dir"])
        # Only the file in the subfolder should be returned
        assert len(xml_files) == 1
        assert xml_files[0][0] == iop_paths["sub_dir"]
        assert xml_files[0][1] == "sub.xml"


def test_process_zip_files(prepare_iop_files, iop_paths):
    """Test the _process_zip_files method"""
    with patch("os.path.abspath", return_value=iop_paths["main_dir"]):
        os.makedirs(iop_paths["main_dir"], exist_ok=True)
        with patch("os.walk", return_value=[(iop_paths["main_dir"], [], [])]):
            files_processed = prepare_iop_files._process_zip_files(
                iop_paths["main_dir"]
            )
            assert files_processed == 0


def test_prepare_files(prepare_iop_files):
    """Test the prepare_files method"""
    with patch.object(prepare_iop_files, "_process_zip_files", return_value=5):
        with patch.object(prepare_iop_files, "_combine_xml_files", return_value=3):
            prepare_iop_files.prepare_files()
            prepare_iop_files.logger.info.assert_any_call("Total files processed: 8")


def test_keyboard_interrupt_in_combine_xml_files(prepare_iop_files, iop_paths):
    """Test handling of KeyboardInterrupt in _combine_xml_files method"""
    with patch.object(
        prepare_iop_files,
        "_get_xml_files",
        return_value=[(iop_paths["sub_dir"], "test.xml")],
    ):
        with patch.object(
            prepare_iop_files,
            "_get_modified_doi",
            side_effect=KeyboardInterrupt("Test interrupt"),
        ):
            with pytest.raises(KeyboardInterruptHandler):
                prepare_iop_files._combine_xml_files(iop_paths["main_dir"])


def test_keyboard_interrupt_in_process_zip_files(prepare_iop_files, iop_paths):
    """Test handling of KeyboardInterrupt in _process_zip_files method"""
    with patch("os.walk", return_value=[(iop_paths["main_dir"], [], ["test.zip"])]):
        with patch("os.makedirs"):
            with patch(
                "shutil.unpack_archive", side_effect=KeyboardInterrupt("Test interrupt")
            ):
                with pytest.raises(KeyboardInterruptHandler):
                    prepare_iop_files._process_zip_files(iop_paths["main_dir"])


def test_error_folder_not_exists(prepare_iop_files):
    """Test error handling when folder does not exist"""
    with pytest.raises(ValueErrorHandler):
        prepare_iop_files._combine_xml_files("/nonexistent/path")

    with pytest.raises(ValueErrorHandler):
        prepare_iop_files._process_zip_files("/nonexistent/path")
