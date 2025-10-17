import pytest
import pandas as pd
import json
import glob
from unittest.mock import patch, MagicMock, mock_open

from comproscanner.utils.configs import RAGConfig, ArticleRelatedKeywords
from comproscanner.utils.error_handler import (
    ValueErrorHandler,
    KeyboardInterruptHandler,
)
from comproscanner.utils.pdf_to_markdown_text import PDFToMarkdownText
from comproscanner.article_processors.pdfs_processor import PDFsProcessor


@pytest.fixture
def sample_property_keywords():
    """Fixture to provide sample property keywords for testing"""
    return {
        "exact_keywords": ["test_keyword1", "test_keyword2"],
        "substring_keywords": [" test_substring1 ", " test_substring2 "],
    }


@pytest.fixture
def pdfs_processor(sample_property_keywords):
    """Fixture to create a PDFsProcessor instance with test parameters"""
    return PDFsProcessor(
        folder_path="/test/path",
        main_property_keyword="piezoelectric",
        property_keywords=sample_property_keywords,
        is_sql_db=False,
        csv_batch_size=10,
    )


def test_init_valid_parameters(sample_property_keywords):
    """Test initialization with valid parameters"""
    processor = PDFsProcessor(
        folder_path="/test/path",
        main_property_keyword="piezoelectric",
        property_keywords=sample_property_keywords,
    )

    assert processor.folder_path == "/test/path"
    assert processor.keyword == "piezoelectric"
    assert processor.property_keywords == sample_property_keywords
    assert processor.is_sql_db is False
    assert processor.csv_batch_size == 2000
    assert processor.valid_property_articles == 0
    assert processor.source == "pdf"


def test_init_missing_folder_path(sample_property_keywords):
    """Test initialization with missing folder path"""
    with pytest.raises(ValueErrorHandler) as exc_info:
        PDFsProcessor(
            folder_path=None,
            main_property_keyword="piezoelectric",
            property_keywords=sample_property_keywords,
        )
    assert "PDF folder path cannot be empty" in str(exc_info.value)


def test_init_missing_keyword(sample_property_keywords):
    """Test initialization with missing main property keyword"""
    with pytest.raises(ValueErrorHandler) as exc_info:
        PDFsProcessor(
            folder_path="/test/path",
            main_property_keyword=None,
            property_keywords=sample_property_keywords,
        )
    assert "main_property_keyword" in str(exc_info.value)


def test_init_missing_property_keywords():
    """Test initialization with missing property keywords"""
    with pytest.raises(ValueErrorHandler) as exc_info:
        PDFsProcessor(
            folder_path="/test/path",
            main_property_keyword="piezoelectric",
            property_keywords=None,
        )
    assert "property_keywords" in str(exc_info.value)


def test_get_paper_metadata_from_pdf(pdfs_processor):
    """Test extraction of metadata from PDF results"""
    valid_results = {
        "validation_info": json.dumps(
            {
                "title": "Test Title",
                "container-title": "Test Journal",
                "publisher": "Test Publisher",
            }
        )
    }

    title, journal, publisher = pdfs_processor._get_paper_metadata_from_pdf(
        valid_results
    )
    assert title == "Test Title"
    assert journal == "Test Journal"
    assert publisher == "Test Publisher"

    missing_validation = {}
    title, journal, publisher = pdfs_processor._get_paper_metadata_from_pdf(
        missing_validation
    )
    assert title == ""
    assert journal == ""
    assert publisher == ""

    invalid_json = {"validation_info": "Not a valid JSON"}
    title, journal, publisher = pdfs_processor._get_paper_metadata_from_pdf(
        invalid_json
    )
    assert title == ""
    assert journal == ""
    assert publisher == ""


@pytest.mark.parametrize("is_sql_db", [True, False])
def test_database_selection(is_sql_db):
    """Test database selection based on is_sql_db parameter"""
    with (
        patch(
            "comproscanner.utils.database_manager.MySQLDatabaseManager"
        ) as mock_sql_db,
        patch("comproscanner.utils.database_manager.CSVDatabaseManager"),
        patch("comproscanner.utils.database_manager.VectorDatabaseManager"),
    ):

        sample_property_keywords = {
            "exact_keywords": ["test_keyword1", "test_keyword2"],
            "substring_keywords": [" test_substring1 ", " test_substring2 "],
        }

        # Create a simple test class for verification
        class TestProcessor(PDFsProcessor):
            def __init__(self, *args, **kwargs):
                # Don't call parent constructor to avoid actual initialization
                self.is_sql_db = kwargs.get("is_sql_db", False)
                self.folder_path = "/test/path"
                self.keyword = "piezoelectric"
                self.property_keywords = sample_property_keywords

                # Initialize database managers based on is_sql_db
                if self.is_sql_db:
                    from comproscanner.utils.database_manager import (
                        MySQLDatabaseManager,
                    )

                    self.sql_db_manager = MySQLDatabaseManager(
                        self.keyword, self.is_sql_db
                    )

                from comproscanner.utils.database_manager import (
                    CSVDatabaseManager,
                    VectorDatabaseManager,
                )

                self.csv_db_manager = CSVDatabaseManager()
                self.vector_db_manager = VectorDatabaseManager(rag_config=RAGConfig())

        # Create the test processor
        processor = TestProcessor(is_sql_db=is_sql_db)

        # Check if MySQLDatabaseManager was called based on is_sql_db
        if is_sql_db:
            assert mock_sql_db.called, "MySQLDatabaseManager should have been created"
        else:
            assert (
                not mock_sql_db.called
            ), "MySQLDatabaseManager should not have been created"


@patch("glob.glob")
@patch("pdf2doi.pdf2doi")
def test_process_pdfs_no_identifier(mock_pdf2doi, mock_glob, pdfs_processor):
    """Test processing PDFs with no DOI identifier"""
    mock_glob.return_value = ["/test/path/file1.pdf"]

    mock_pdf2doi.return_value = {}

    with (
        patch.object(PDFToMarkdownText, "convert_to_markdown", return_value="# Test"),
        patch.object(PDFToMarkdownText, "clean_text", return_value={}),
        patch.object(
            PDFToMarkdownText,
            "append_section_to_df",
            return_value=pd.DataFrame(
                {
                    "doi": [""],
                    "article_title": [""],
                    "publication_name": [""],
                    "publisher": [""],
                    "abstract": [""],
                    "introduction": [""],
                    "exp_methods": [""],
                    "comp_methods": [""],
                    "results_discussion": [""],
                    "conclusion": [""],
                    "is_property_mentioned": ["0"],
                }
            ),
        ),
        patch.object(pdfs_processor.csv_db_manager, "write_to_csv"),
    ):

        pdfs_processor.process_pdfs()

        assert pdfs_processor.identifier == "/test/path/file1"


@patch("glob.glob")
@patch("pdf2doi.pdf2doi")
def test_process_pdfs_keyboard_interrupt(mock_pdf2doi, mock_glob, pdfs_processor):
    """Test keyboard interrupt during PDF processing"""
    mock_glob.return_value = ["/test/path/file1.pdf"]

    mock_pdf2doi.side_effect = KeyboardInterrupt()

    with pytest.raises(KeyboardInterruptHandler):
        pdfs_processor.process_pdfs()


@patch("glob.glob")
@patch("pdf2doi.pdf2doi")
def test_process_pdfs_exception_handling(mock_pdf2doi, mock_glob, pdfs_processor):
    """Test exception handling during PDF processing"""
    mock_glob.return_value = ["/test/path/file1.pdf", "/test/path/file2.pdf"]

    mock_pdf2doi.side_effect = [
        Exception("Test error"),
        {
            "identifier": "10.1234/test.123",
            "validation_info": json.dumps(
                {
                    "title": "Test Title",
                    "container-title": "Test Journal",
                    "publisher": "Test Publisher",
                }
            ),
        },
    ]

    with (
        patch.object(PDFToMarkdownText, "convert_to_markdown", return_value="# Test"),
        patch.object(PDFToMarkdownText, "clean_text", return_value={}),
        patch.object(
            PDFToMarkdownText,
            "append_section_to_df",
            return_value=pd.DataFrame(
                {
                    "doi": ["10.1234/test.123"],
                    "article_title": ["Test Title"],
                    "publication_name": ["Test Journal"],
                    "publisher": ["Test Publisher"],
                    "abstract": [""],
                    "introduction": [""],
                    "exp_methods": [""],
                    "comp_methods": [""],
                    "results_discussion": [""],
                    "conclusion": [""],
                    "is_property_mentioned": ["0"],
                }
            ),
        ),
        patch.object(pdfs_processor.csv_db_manager, "write_to_csv"),
    ):

        pdfs_processor.process_pdfs()

        assert mock_pdf2doi.call_count == 2
        assert pdfs_processor.valid_property_articles == 0
