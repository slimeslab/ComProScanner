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
    assert processor.csv_batch_size == 1
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


def test_extract_doi_from_text(pdfs_processor):
    """Test DOI extraction from text"""
    text_with_doi = "This paper has DOI: 10.1234/test.567"
    doi = pdfs_processor._extract_doi_from_text(text_with_doi)
    assert doi == "10.1234/test.567"

    text_without_doi = "This paper has no DOI"
    doi = pdfs_processor._extract_doi_from_text(text_without_doi)
    assert doi == ""

    text_with_multiple_dois = "DOIs: 10.1234/test.567 and 10.5678/another.123"
    doi = pdfs_processor._extract_doi_from_text(text_with_multiple_dois)
    assert doi == "10.1234/test.567"


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

        class TestProcessor(PDFsProcessor):
            def __init__(self, *args, **kwargs):
                self.is_sql_db = kwargs.get("is_sql_db", False)
                self.folder_path = "/test/path"
                self.keyword = "piezoelectric"
                self.property_keywords = sample_property_keywords

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

        processor = TestProcessor(is_sql_db=is_sql_db)

        if is_sql_db:
            assert mock_sql_db.called, "MySQLDatabaseManager should have been created"
        else:
            assert (
                not mock_sql_db.called
            ), "MySQLDatabaseManager should not have been created"


@patch("glob.glob")
@patch(
    "comproscanner.article_processors.pdfs_processor.get_paper_metadata_from_openalex"
)
def test_process_pdfs_with_doi(mock_metadata, mock_glob, pdfs_processor):
    """Test processing PDFs with DOI found"""
    mock_glob.return_value = ["/test/path/file1.pdf"]
    mock_metadata.return_value = ("Test Title", "Test Journal", "Test Publisher")

    with (
        patch.object(
            PDFToMarkdownText,
            "convert_to_markdown",
            return_value="DOI: 10.1234/test.567\n# Test content",
        ),
        patch.object(PDFToMarkdownText, "clean_text", return_value={}),
        patch.object(
            PDFToMarkdownText,
            "append_section_to_df",
            return_value=pd.DataFrame(
                {
                    "doi": ["10.1234/test.567"],
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
        assert pdfs_processor.identifier == "10.1234/test.567"
        mock_metadata.assert_called_once_with("10.1234/test.567")


@patch("glob.glob")
@patch(
    "comproscanner.article_processors.pdfs_processor.get_paper_metadata_from_openalex"
)
def test_process_pdfs_no_doi(mock_metadata, mock_glob, pdfs_processor):
    """Test processing PDFs with no DOI found"""
    mock_glob.return_value = ["/test/path/file1.pdf"]

    with (
        patch.object(
            PDFToMarkdownText, "convert_to_markdown", return_value="# Test content"
        ),
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
        assert pdfs_processor.identifier == "file1"
        mock_metadata.assert_not_called()


@patch("glob.glob")
@patch(
    "comproscanner.article_processors.pdfs_processor.get_paper_metadata_from_openalex"
)
def test_process_pdfs_exception_handling(mock_metadata, mock_glob, pdfs_processor):
    """Test exception handling during PDF processing"""
    mock_glob.return_value = ["/test/path/file1.pdf", "/test/path/file2.pdf"]
    mock_metadata.return_value = ("Test Title", "Test Journal", "Test Publisher")

    call_count = [0]

    def mock_convert(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("Test error")
        return "DOI: 10.1234/test.567\n# Test content"

    with (
        patch.object(
            PDFToMarkdownText, "convert_to_markdown", side_effect=mock_convert
        ),
        patch.object(PDFToMarkdownText, "clean_text", return_value={}),
        patch.object(
            PDFToMarkdownText,
            "append_section_to_df",
            return_value=pd.DataFrame(
                {
                    "doi": ["10.1234/test.567"],
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
        assert pdfs_processor.valid_property_articles == 0


@patch("glob.glob")
def test_process_pdfs_keyboard_interrupt(mock_glob, pdfs_processor):
    """Test keyboard interrupt during PDF processing"""
    mock_glob.return_value = ["/test/path/file1.pdf"]

    with patch.object(
        PDFToMarkdownText, "convert_to_markdown", side_effect=KeyboardInterrupt()
    ):
        with pytest.raises(KeyboardInterruptHandler):
            pdfs_processor.process_pdfs()


@patch("glob.glob")
@patch("comproscanner.utils.common_functions.get_paper_metadata_from_openalex")
def test_process_pdfs_exception_handling(mock_metadata, mock_glob, pdfs_processor):
    """Test exception handling during PDF processing"""
    mock_glob.return_value = ["/test/path/file1.pdf", "/test/path/file2.pdf"]
    mock_metadata.return_value = ("Test Title", "Test Journal", "Test Publisher")

    call_count = [0]

    def mock_convert(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("Test error")
        return "DOI: 10.1234/test.567\n# Test content"

    with (
        patch.object(
            PDFToMarkdownText, "convert_to_markdown", side_effect=mock_convert
        ),
        patch.object(PDFToMarkdownText, "clean_text", return_value={}),
        patch.object(
            PDFToMarkdownText,
            "append_section_to_df",
            return_value=pd.DataFrame(
                {
                    "doi": ["10.1234/test.567"],
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
        assert pdfs_processor.valid_property_articles == 0
