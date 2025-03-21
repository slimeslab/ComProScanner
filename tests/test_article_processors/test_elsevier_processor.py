"""
test_elsevier_processor.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 02-03-2025
"""

import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open, PropertyMock
from lxml import etree
import requests
from sqlalchemy.exc import SQLAlchemyError
from comproscanner.article_processors.elsevier_processor import ElsevierArticleProcessor
from comproscanner.utils.error_handler import (
    ValueErrorHandler,
    BaseError,
    KeyboardInterruptHandler,
)
from comproscanner.utils.configs import RAGConfig


@pytest.fixture
def sample_xml_content():
    """Fixture to load sample XML content from a file"""
    sample_xml_path = os.path.join(
        os.path.dirname(__file__), "../test_apis_primary", "elsevier_test.xml"
    )

    with open(sample_xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    return xml_content


@pytest.fixture
def sample_df():
    """Fixture to create a sample DataFrame for testing"""
    return pd.DataFrame(
        {
            "doi": ["10.1000/test1", "10.1000/test2", "10.1000/test3"],
            "article_title": ["Title 1", "Title 2", "Title 3"],
            "publication_name": ["Journal A", "Journal B", "Journal C"],
            "metadata_publisher": ["Elsevier", "Elsevier", "Elsevier"],
            "general_publisher": ["elsevier", "elsevier", "elsevier"],
        }
    )


@pytest.fixture
def elsevier_processor(monkeypatch, share_scopus_api_key):
    """Fixture to create an ElsevierArticleProcessor instance with test parameters"""
    monkeypatch.setenv("SCOPUS_API_KEY", share_scopus_api_key)
    property_keywords = {
        "exact_keywords": ["d33", "piezoelectric coefficient"],
        "substring_keywords": [" d33 ", " piezo "],
    }
    with patch("os.path.exists", return_value=False), patch("os.makedirs"):
        return ElsevierArticleProcessor(
            main_property_keyword="piezoelectric",
            property_keywords=property_keywords,
            is_sql_db=False,
        )


def test_initialization(monkeypatch, share_scopus_api_key):
    """Test basic initialization with minimum required parameters"""
    monkeypatch.setenv("SCOPUS_API_KEY", share_scopus_api_key)
    property_keywords = {"exact_keywords": ["d33"], "substring_keywords": [" d33 "]}

    with patch("os.path.exists", return_value=False), patch("os.makedirs"):
        processor = ElsevierArticleProcessor(
            main_property_keyword="piezoelectric", property_keywords=property_keywords
        )

    assert processor.keyword == "piezoelectric"
    assert processor.property_keywords == property_keywords
    assert processor.api_key == share_scopus_api_key
    assert processor.is_sql_db is False
    assert processor.is_save_xml is False
    assert processor.valid_property_articles == 0
    assert processor.source == "elsevier"


def test_init_without_api_key(monkeypatch):
    """Test initialization without API key"""
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    monkeypatch.setattr(BaseError, "exit_program", lambda self: None)

    property_keywords = {"exact_keywords": ["d33"], "substring_keywords": [" d33 "]}

    with pytest.raises(ValueErrorHandler) as exc_info:
        ElsevierArticleProcessor(
            main_property_keyword="piezoelectric", property_keywords=property_keywords
        )
    assert "SCOPUS_API_KEY is not set in the environment variables" in str(
        exc_info.value
    )


def test_init_without_main_property_keyword(monkeypatch, share_scopus_api_key):
    """Test initialization without main property keyword"""
    monkeypatch.setenv("SCOPUS_API_KEY", share_scopus_api_key)
    monkeypatch.setattr(BaseError, "exit_program", lambda self: None)

    property_keywords = {"exact_keywords": ["d33"], "substring_keywords": [" d33 "]}

    with pytest.raises(ValueErrorHandler) as exc_info:
        ElsevierArticleProcessor(property_keywords=property_keywords)
    assert "main_property_keyword" in str(exc_info.value)


def test_init_without_property_keywords(monkeypatch, share_scopus_api_key):
    """Test initialization without property keywords"""
    monkeypatch.setenv("SCOPUS_API_KEY", share_scopus_api_key)
    monkeypatch.setattr(BaseError, "exit_program", lambda self: None)

    with pytest.raises(ValueErrorHandler) as exc_info:
        ElsevierArticleProcessor(main_property_keyword="piezoelectric")
    assert "property_keywords" in str(exc_info.value)


def test_load_and_preprocess_data(elsevier_processor, sample_df, monkeypatch):
    """Test load and preprocess data method"""
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: sample_df)
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    elsevier_processor._load_and_preprocess_data()
    assert len(elsevier_processor.df) == 3
    assert all(elsevier_processor.df["general_publisher"] == "elsevier")


def test_load_and_preprocess_data_with_row_limits(
    elsevier_processor, sample_df, monkeypatch
):
    """Test load and preprocess data method with row limits"""
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: sample_df)
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    elsevier_processor.start_row = 0
    elsevier_processor.end_row = 1
    elsevier_processor._load_and_preprocess_data()
    assert len(elsevier_processor.df) == 1
    assert elsevier_processor.df.iloc[0]["doi"] == "10.1000/test1"


def test_load_and_preprocess_data_with_processed_dois(
    elsevier_processor, sample_df, monkeypatch
):
    """Test load and preprocess data with already processed DOIs"""
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda path, **kwargs: (
            sample_df
            if path == elsevier_processor.metadata_csv_filename
            else pd.DataFrame({"doi": ["10.1000/test1"]})
        ),
    )
    monkeypatch.setattr(
        os.path, "exists", lambda path: path == elsevier_processor.csv_filepath
    )
    elsevier_processor._load_and_preprocess_data()
    assert len(elsevier_processor.df) == 2
    assert "10.1000/test1" not in elsevier_processor.df["doi"].values


def test_parse_response(elsevier_processor, sample_xml_content):
    """Test parsing XML response"""
    mock_response = MagicMock()
    mock_response.text = sample_xml_content

    with patch(
        "lxml.etree.fromstring",
        return_value=etree.fromstring(sample_xml_content.encode("utf-8")),
    ) as mock_fromstring:
        root = elsevier_processor._parse_response(mock_response)

        assert root is not None
        assert (
            root.tag
            == "{http://www.elsevier.com/xml/svapi/article/dtd}full-text-retrieval-response"
        )
        assert root.xpath('.//*[local-name()="description"]')[0].text is not None
        assert len(root.xpath('.//*[local-name()="section"]')) > 0


def test_parse_response_with_error(elsevier_processor):
    """Test parsing XML response with error"""
    mock_response = MagicMock()
    mock_response.text = "Invalid XML"

    with patch(
        "comproscanner.article_processors.elsevier_processor.logger.error"
    ) as mock_logger:
        root = elsevier_processor._parse_response(mock_response)

        assert root is None
        assert mock_logger.called


def test_send_request_success(elsevier_processor, monkeypatch):
    """Test sending request with successful response"""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("requests.get", return_value=mock_response) as mock_get:
        response = elsevier_processor._send_request("10.1000/test-doi")

        assert response == mock_response
        mock_get.assert_called_once()


def test_send_request_rate_limit(elsevier_processor, monkeypatch):
    """Test sending request with rate limit response"""
    mock_response = MagicMock()
    mock_response.status_code = 429

    with (
        patch("requests.get", return_value=mock_response) as mock_get,
        patch(
            "comproscanner.article_processors.elsevier_processor.logger.critical"
        ) as mock_logger,
    ):
        monkeypatch.setattr(
            elsevier_processor,
            "_send_request",
            lambda doi: None if mock_response.status_code == 429 else mock_response,
        )

        response = elsevier_processor._send_request("10.1000/test-doi")

        assert response is None
        elsevier_processor.is_exceeded = True
        assert elsevier_processor.is_exceeded is True


def test_send_request_timeout(elsevier_processor, monkeypatch):
    """Test sending request with timeout error"""
    with (
        patch("requests.get", side_effect=requests.exceptions.ReadTimeout) as mock_get,
        patch(
            "comproscanner.article_processors.elsevier_processor.write_timeout_file"
        ) as mock_write,
        patch(
            "comproscanner.article_processors.elsevier_processor.logger.error"
        ) as mock_logger,
    ):
        response = elsevier_processor._send_request("10.1000/test-doi")

        assert response is None
        assert mock_logger.called
        mock_write.assert_called_once()


def test_process_xml(elsevier_processor, sample_xml_content):
    """Test processing XML to extract sections"""
    parser = etree.XMLParser(encoding="utf-8")
    root = etree.fromstring(sample_xml_content.encode("utf-8"), parser=parser)
    expected_abstract = [etree.Element("description")]
    expected_abstract[0].text = "This is a test abstract about piezoelectric materials."
    expected_sections = []
    for title in ["Introduction", "Methods", "Results and Discussion", "Conclusion"]:
        section = etree.Element("section")
        title_elem = etree.SubElement(section, "section-title")
        title_elem.text = title
        expected_sections.append(section)

    with patch.object(
        elsevier_processor,
        "_process_xml",
        return_value=(expected_abstract, expected_sections),
    ):
        abstract, sections = elsevier_processor._process_xml(root)

        assert abstract is not None
        assert (
            abstract[0].text == "This is a test abstract about piezoelectric materials."
        )
        assert len(sections) == 4
        section_titles = [
            section.xpath('./child::*[local-name()="section-title"]')[0].text
            for section in sections
        ]
        assert "Introduction" in section_titles
        assert "Methods" in section_titles
        assert "Results and Discussion" in section_titles
        assert "Conclusion" in section_titles


def test_extract_paragraphs(elsevier_processor):
    """Test extracting paragraphs from a section"""
    section_xml = """
    <section>
        <section-title>Methods</section-title>
        <para>This is a regular paragraph.</para>
        <para>This is a computational paragraph with computational methods.</para>
    </section>
    """
    section = etree.fromstring(section_xml)
    with patch.object(
        elsevier_processor.article_related_keywords,
        "COMP_KEYWORDS",
        ["computational methods"],
    ):
        other_paragraphs, comp_paragraphs = elsevier_processor._extract_paragraphs(
            section
        )

        assert "This is a regular paragraph." in other_paragraphs
        assert (
            "This is a computational paragraph with computational methods."
            in comp_paragraphs
        )


@patch("pandas.DataFrame.to_csv")
@patch("os.path.exists", return_value=False)
@patch("os.makedirs")
def test_process_articles(
    mock_makedirs,
    mock_exists,
    mock_to_csv,
    elsevier_processor,
    sample_df,
    monkeypatch,
    sample_xml_content,
):
    """Test processing articles"""
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: sample_df)

    elsevier_processor.property_keywords = {
        "exact_keywords": ["property"],
        "substring_keywords": [" property "],
    }

    mock_row = pd.Series(
        {
            "doi": "10.1000/test1",
            "article_title": "Test Title",
            "publication_name": "Test Journal",
            "publisher": "Elsevier",
            "abstract": "Test abstract with property",
            "introduction": "Test intro",
            "exp_methods": "Test methods",
            "comp_methods": "Test computational methods",
            "results_discussion": "Test results",
            "conclusion": "Test conclusion",
            "is_property_mentioned": "1",
        }
    )

    elsevier_processor.csv_batch_size = 1
    elsevier_processor.sql_batch_size = 1
    elsevier_processor.valid_property_articles = 0

    def custom_append_sections(*args, **kwargs):
        """
        Modified mock function that returns a DataFrame (not a Series) to match the
        actual implementation in _append_sections_to_df
        """
        modified_doi = args[2].replace("/", "_")
        total_text = (
            "# ABSTRACT:\nTest abstract with property\n# INTRODUCTION:\nTest intro\n..."
        )
        elsevier_processor.vector_db_manager.create_database(
            db_name=modified_doi, article_text=total_text
        )
        return pd.DataFrame(
            [
                {
                    "doi": "10.1000/test1",
                    "article_title": "Test Title",
                    "publication_name": "Test Journal",
                    "publisher": "Elsevier",
                    "abstract": "Test abstract with property",
                    "introduction": "Test intro",
                    "exp_methods": "Test methods",
                    "comp_methods": "Test computational methods",
                    "results_discussion": "Test results",
                    "conclusion": "Test conclusion",
                    "is_property_mentioned": "1",
                }
            ]
        )

    with (
        patch.object(elsevier_processor, "_load_and_preprocess_data"),
        patch.object(elsevier_processor, "_send_request", return_value=MagicMock()),
        patch.object(elsevier_processor, "_parse_response", return_value=MagicMock()),
        patch.object(
            elsevier_processor, "_process_xml", return_value=(MagicMock(), [])
        ),
        patch.object(
            elsevier_processor,
            "_append_sections_to_df",
            side_effect=custom_append_sections,
        ),
        patch.object(
            elsevier_processor.csv_db_manager, "write_to_csv"
        ) as mock_write_to_csv,
        patch.object(
            elsevier_processor.vector_db_manager,
            "create_database",
        ) as mock_create_db,
        patch("comproscanner.article_processors.elsevier_processor.tqdm") as mock_tqdm,
        patch("os.path.isdir", return_value=False),
        patch("os.listdir", return_value=[]),
        patch("time.sleep", return_value=None),
    ):
        elsevier_processor.df = pd.DataFrame(
            {
                "doi": ["10.1000/test1"],
                "article_title": ["Test Title"],
                "publication_name": ["Test Journal"],
                "metadata_publisher": ["Elsevier"],
                "general_publisher": ["elsevier"],
            }
        )
        mock_tqdm.return_value = [(0, elsevier_processor.df.iloc[0])]
        elsevier_processor._process_articles()
        assert (
            mock_write_to_csv.called
        ), "CSV database manager write_to_csv method should be called"
        assert (
            mock_create_db.called
        ), "Vector database create_database method should be called"
        assert (
            elsevier_processor.valid_property_articles > 0
        ), "Valid property articles count should be incremented"


def test_process_with_timeout_handling(elsevier_processor, monkeypatch):
    """Test processing with timeout handling"""
    with (
        patch.object(elsevier_processor, "_process_articles") as mock_process,
        patch("os.path.isfile", return_value=False),
        patch(
            "comproscanner.article_processors.elsevier_processor.logger.debug"
        ) as mock_debug,
    ):
        elsevier_processor._process_with_timeout_handling()
        assert mock_process.call_count == 0
        mock_debug.assert_not_called()
    with (
        patch.object(elsevier_processor, "_process_articles") as mock_process,
        patch("os.path.isfile", side_effect=[True, False]),
        patch("os.path.exists", return_value=True),
        patch("os.remove") as mock_remove,
        patch("builtins.open", mock_open(read_data="10.1000/timeout-doi")),
        patch(
            "comproscanner.article_processors.elsevier_processor.logger.debug"
        ) as mock_debug,
    ):
        elsevier_processor._process_with_timeout_handling()
        assert mock_process.call_count == 1
        assert mock_debug.call_count == 1
        assert mock_remove.called


@patch("comproscanner.article_processors.elsevier_processor.logger.verbose")
@patch("comproscanner.article_processors.elsevier_processor.logger.debug")
@patch("comproscanner.article_processors.elsevier_processor.logger.info")
def test_process_elsevier_articles(
    mock_info, mock_debug, mock_verbose, elsevier_processor, monkeypatch
):
    """Test the main process_elsevier_articles method"""
    with (
        patch.object(elsevier_processor, "_process_articles") as mock_process,
        patch.object(
            elsevier_processor, "_process_with_timeout_handling"
        ) as mock_timeout,
    ):
        elsevier_processor.process_elsevier_articles()
        assert mock_process.called
        assert mock_timeout.called
        assert mock_verbose.call_count == 2
        assert mock_info.call_count == 1


def test_keyboard_interrupt_handling(elsevier_processor, sample_df, monkeypatch):
    """Test handling of keyboard interrupts during processing"""
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: sample_df)

    call_count = [0]

    def mock_send_request(doi):
        call_count[0] += 1
        if call_count[0] > 1:
            raise KeyboardInterrupt
        mock_resp = MagicMock()
        mock_resp.text = "Sample XML"
        return mock_resp

    monkeypatch.setattr(elsevier_processor, "_send_request", mock_send_request)

    with (
        patch("comproscanner.article_processors.elsevier_processor.tqdm") as mock_tqdm,
        patch(
            "comproscanner.article_processors.elsevier_processor.logger.error"
        ) as mock_logger,
    ):

        mock_tqdm.return_value = [(idx, row) for idx, row in sample_df.iterrows()]

        with pytest.raises(KeyboardInterruptHandler):
            elsevier_processor._process_articles()

        assert mock_logger.called


@pytest.mark.integration
def test_integration_flow(elsevier_processor, monkeypatch, sample_xml_content):
    """Integration test for the complete article processing flow"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = sample_xml_content

    sample_row = {
        "doi": "10.1000/test-integration",
        "article_title": "Integration Test Article",
        "publication_name": "Integration Journal",
        "metadata_publisher": "Elsevier",
        "general_publisher": "elsevier",
    }

    sample_df = pd.DataFrame([sample_row])
    mock_row = pd.DataFrame(
        [
            {
                "doi": "10.1000/test-integration",
                "article_title": "Integration Test Article",
                "publication_name": "Integration Journal",
                "publisher": "Elsevier",
                "abstract": "Test abstract with d33 piezoelectric coefficient",
                "introduction": "Test intro",
                "exp_methods": "Test methods",
                "comp_methods": "Test computational methods",
                "results_discussion": "Test results",
                "conclusion": "Test conclusion",
                "is_property_mentioned": "1",
            }
        ]
    )

    with (
        patch("pandas.read_csv", return_value=sample_df),
        patch("requests.get", return_value=mock_response),
        patch.object(elsevier_processor, "_parse_response", return_value=MagicMock()),
        patch.object(
            elsevier_processor, "_process_xml", return_value=(MagicMock(), [])
        ),
        patch.object(
            elsevier_processor, "_append_sections_to_df", return_value=mock_row
        ),
        patch.object(elsevier_processor.csv_db_manager, "write_to_csv"),
        patch.object(elsevier_processor.vector_db_manager, "create_database"),
        patch("os.path.exists", return_value=False),
        patch("os.makedirs"),
        patch("comproscanner.article_processors.elsevier_processor.tqdm") as mock_tqdm,
    ):
        mock_tqdm.return_value = [(0, sample_row)]
        elsevier_processor.valid_property_articles = 1
        elsevier_processor.process_elsevier_articles()
        assert elsevier_processor.valid_property_articles > 0


def test_load_preprocess_only_end_row(elsevier_processor, sample_df, monkeypatch):
    """Test preprocessing with only end_row specified (not start_row)."""
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: sample_df)
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    elsevier_processor.start_row = None
    elsevier_processor.end_row = 2
    elsevier_processor._load_and_preprocess_data()
    assert len(elsevier_processor.df) == 2
    assert elsevier_processor.df["doi"].tolist() == ["10.1000/test1", "10.1000/test2"]


def test_load_preprocess_only_start_row(elsevier_processor, sample_df, monkeypatch):
    """Test preprocessing with only start_row specified (not end_row)."""
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: sample_df)
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    elsevier_processor.start_row = 1
    elsevier_processor.end_row = None
    elsevier_processor._load_and_preprocess_data()
    assert len(elsevier_processor.df) == 2
    assert elsevier_processor.df["doi"].tolist() == ["10.1000/test2", "10.1000/test3"]


def test_csv_read_error_handling(elsevier_processor, sample_df, monkeypatch):
    """Test handling of CSV read errors."""
    read_csv_mock = MagicMock(side_effect=[sample_df, Exception("CSV read error")])
    monkeypatch.setattr(pd, "read_csv", read_csv_mock)
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    elsevier_processor._load_and_preprocess_data()
    assert elsevier_processor.df is not None
    assert len(elsevier_processor.df) > 0


def test_sql_read_error_handling(elsevier_processor, sample_df, monkeypatch):
    """Test handling of SQL database read errors."""
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: sample_df)
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    elsevier_processor.is_sql_db = True
    monkeypatch.setattr(
        elsevier_processor.sql_db_manager, "table_exists", lambda x: True
    )
    with patch(
        "sqlalchemy.create_engine", side_effect=SQLAlchemyError("Database error")
    ):
        elsevier_processor._load_and_preprocess_data()
    assert len(elsevier_processor.df) == 3


def test_save_xml(elsevier_processor):
    """Test saving XML response to a file."""
    mock_response = MagicMock()
    mock_response.content = b"<test>XML content</test>"
    with (
        patch("os.path.exists", return_value=False),
        patch("os.makedirs") as mock_makedirs,
        patch("builtins.open", mock_open()) as mock_file,
    ):
        elsevier_processor._save_xml(mock_response, "10.1016/j.test")
        mock_makedirs.assert_called_once()
        mock_file.assert_called_once_with(
            "downloaded_files/xmls/elsevier/10.1016_j.test.xml", "wb"
        )
        mock_file().write.assert_called_once_with(mock_response.content)


def test_generate_tables_success(elsevier_processor):
    """Test the _generate_tables method with valid data."""
    header_data = [["Header 1", "Header 2"], ["Col A", "Col B"]]
    column_number = [2, 2]
    all_table_data = [
        [["Data 1", "Data 2"], ["Data 3", "Data 4"]],
        [["A1", "B1"], ["A2", "B2"]],
    ]
    caption_data = ["First Table", "Second Table"]

    tables = elsevier_processor._generate_tables(
        header_data, column_number, all_table_data, caption_data
    )
    assert len(tables) == 2
    assert "Table 1.First Table" in tables[0]
    assert "Table 2.Second Table" in tables[1]
    assert "|Header 1|Header 2|" in tables[0]
    assert "|Col A|Col B|" in tables[1]
    assert "|Data 1|Data 2|" in tables[0]
    assert "|A1|B1|" in tables[1]


def test_generate_tables_with_list_items(elsevier_processor):
    """Test the _generate_tables method with list items in rows."""
    header_data = [["Header 1", "Header 2"]]
    column_number = [2]
    all_table_data = [
        [["Data 1", "Data 2"], [["List Item 1", "List Item 2"], "Data 4"]]
    ]
    caption_data = ["Table with List"]
    tables = elsevier_processor._generate_tables(
        header_data, column_number, all_table_data, caption_data
    )
    assert len(tables) == 1
    assert "Table 1.Table with List" in tables[0]
    assert "['List Item 1', 'List Item 2']" in tables[0]


def test_generate_tables_mismatch(elsevier_processor):
    """Test the _generate_tables method with mismatched data."""
    header_data = [["Header 1", "Header 2"]]
    column_number = [2]
    all_table_data = []
    caption_data = ["Mismatched Table"]
    result = elsevier_processor._generate_tables(
        header_data, column_number, all_table_data, caption_data
    )
    assert result == "Error: Data mismatch"
