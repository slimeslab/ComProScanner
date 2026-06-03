import pytest
import pandas as pd
import json
import glob
import os
from unittest.mock import patch, MagicMock, mock_open

from comproscanner.utils.configs import RAGConfig, ArticleRelatedKeywords, DefaultPaths
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
        is_track_pdfs=False,
    )


@pytest.fixture
def pdfs_processor_with_tracking(sample_property_keywords):
    """Fixture with PDF tracking enabled for tracking-specific tests"""
    return PDFsProcessor(
        folder_path="/test/path",
        main_property_keyword="piezoelectric",
        property_keywords=sample_property_keywords,
        is_sql_db=False,
        csv_batch_size=10,
        is_track_pdfs=True,
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
    assert processor.save_failed_pdf_report is True
    assert processor.failed_pdf_report_path == os.path.join(
        "/test/path", "failed_pdf_filenames.txt"
    )
    assert processor.is_track_pdfs is True
    assert processor.track_pdfs_report_path == DefaultPaths("piezoelectric").PDF_PROCESSED_DOIS_FILENAME


def test_init_tracking_disabled(sample_property_keywords):
    """Test that tracking can be disabled"""
    processor = PDFsProcessor(
        folder_path="/test/path",
        main_property_keyword="piezoelectric",
        property_keywords=sample_property_keywords,
        is_track_pdfs=False,
    )
    assert processor.is_track_pdfs is False


def test_init_custom_track_pdfs_report_path(sample_property_keywords):
    """Test initialization with a custom DOI tracking file path"""
    processor = PDFsProcessor(
        folder_path="/test/path",
        main_property_keyword="piezoelectric",
        property_keywords=sample_property_keywords,
        track_pdfs_report_path="/custom/tracking.txt",
    )
    assert processor.track_pdfs_report_path == "/custom/tracking.txt"


def test_load_processed_pdfs_from_tracking_file(pdfs_processor_with_tracking):
    """_load_processed_pdfs reads basename+DOI pairs from the tracking file"""
    tracking_content = "paper1.pdf\t10.1234/a\npaper2.pdf\t10.5678/b\n"
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=tracking_content)),
    ):
        filenames, dois = pdfs_processor_with_tracking._load_processed_pdfs()
    assert filenames == {"paper1.pdf", "paper2.pdf"}
    assert dois == {"10.1234/a", "10.5678/b"}


def test_load_processed_pdfs_legacy_doi_only_format(pdfs_processor_with_tracking):
    """_load_processed_pdfs handles legacy tracking files that contain only DOIs"""
    tracking_content = "10.1234/a\n10.5678/b\n"
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=tracking_content)),
    ):
        filenames, dois = pdfs_processor_with_tracking._load_processed_pdfs()
    assert filenames == set()
    assert dois == {"10.1234/a", "10.5678/b"}


def test_load_processed_pdfs_fallback_to_csv(pdfs_processor_with_tracking):
    """_load_processed_pdfs falls back to the CSV when the tracking file is absent"""
    csv_data = pd.DataFrame({"doi": ["10.1234/a", "10.9999/c"]})
    with (
        patch("os.path.exists", side_effect=lambda p: p.endswith(".csv")),
        patch("pandas.read_csv", return_value=csv_data),
    ):
        filenames, dois = pdfs_processor_with_tracking._load_processed_pdfs()
    assert filenames == set()
    assert dois == {"10.1234/a", "10.9999/c"}


def test_load_processed_pdfs_no_sources(pdfs_processor_with_tracking):
    """_load_processed_pdfs returns empty sets when neither file exists"""
    with patch("os.path.exists", return_value=False):
        filenames, dois = pdfs_processor_with_tracking._load_processed_pdfs()
    assert filenames == set()
    assert dois == set()


def test_mark_pdf_processed_writes_to_file(pdfs_processor_with_tracking):
    """_mark_pdf_processed appends basename<TAB>doi to the tracking file"""
    m = mock_open()
    with (
        patch("os.makedirs"),
        patch("builtins.open", m),
    ):
        pdfs_processor_with_tracking._mark_pdf_processed("/some/path/paper1.pdf", "10.1234/test")
    m().write.assert_called_once_with("paper1.pdf\t10.1234/test\n")


def test_mark_pdf_processed_skipped_when_disabled(sample_property_keywords):
    """_mark_pdf_processed does nothing when is_track_pdfs is False"""
    processor = PDFsProcessor(
        folder_path="/test/path",
        main_property_keyword="piezoelectric",
        property_keywords=sample_property_keywords,
        is_track_pdfs=False,
    )
    m = mock_open()
    with patch("builtins.open", m):
        processor._mark_pdf_processed("/some/path/paper1.pdf", "10.1234/test")
    m.assert_not_called()


def test_init_custom_failed_pdf_report_path(sample_property_keywords):
    """Test initialization with custom failed PDF report settings"""
    processor = PDFsProcessor(
        folder_path="/test/path",
        main_property_keyword="piezoelectric",
        property_keywords=sample_property_keywords,
        save_failed_pdf_report=False,
        failed_pdf_report_path="/custom/failed_report.txt",
    )
    assert processor.save_failed_pdf_report is False
    assert processor.failed_pdf_report_path == "/custom/failed_report.txt"


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
@patch("comproscanner.article_processors.pdfs_processor.get_doi_from_crossref", return_value="")
@patch(
    "comproscanner.article_processors.pdfs_processor.get_paper_metadata_from_openalex"
)
def test_process_pdfs_no_doi(mock_metadata, mock_crossref, mock_glob, pdfs_processor):
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
def test_process_pdfs_skips_already_processed(mock_glob, pdfs_processor):
    """PDFs whose basename is in the tracking file are skipped before conversion"""
    mock_glob.return_value = ["/test/path/paper1.pdf"]

    with (
        patch(
            "comproscanner.article_processors.pdfs_processor.PDFsProcessor._load_processed_pdfs",
            return_value=({"paper1.pdf"}, {"10.1234/test.567"}),
        ),
        patch.object(PDFToMarkdownText, "convert_to_markdown") as mock_convert,
    ):
        pdfs_processor.process_pdfs()
        mock_convert.assert_not_called()


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
