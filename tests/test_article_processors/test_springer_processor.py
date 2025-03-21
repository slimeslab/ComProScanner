"""
test_springer_processor.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 19-03-2025
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from lxml import etree
import requests
from comproscanner.article_processors.springer_processor import SpringerArticleProcessor
from comproscanner.utils.error_handler import (
    ValueErrorHandler,
    KeyboardInterruptHandler,
)
from comproscanner.utils.configs import RAGConfig


@pytest.fixture
def sample_springer_response():
    """Fixture to load sample Springer article XML content from a file"""
    sample_xml_path = os.path.join(
        os.path.dirname(__file__), "../test_apis_primary", "springer_test.xml"
    )

    with open(sample_xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    return xml_content


@pytest.fixture
def sample_table_xml():
    """Fixture to load sample table XML content from a file"""
    sample_xml_path = os.path.join(
        os.path.dirname(__file__), "../test_apis_primary", "springer_table_test.xml"
    )

    with open(sample_xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    return xml_content


SAMPLE_METADATA_CSV = """doi,article_type,article_title,publication_name,metadata_publisher,general_publisher
10.1007/s12345-678-9101-2,Article,Sample Article Title,Sample Journal,Springer Nature,springer
10.1007/s98765-432-1098-7,Article,Another Article Title,Another Journal,Springer Nature,springer"""


@pytest.fixture
def mock_environment(monkeypatch):
    """Setup environment variables for testing"""
    monkeypatch.setenv("SPRINGER_OPENACCESS_API_KEY", "test_api_key")
    monkeypatch.setenv("SPRINGER_TDM_API_KEY", "test_tdm_api_key")


@pytest.fixture
def property_keywords():
    """Sample property keywords for testing"""
    return {
        "exact_keywords": ["piezoelectric", "piezoelectricity"],
        "substring_keywords": [" piezo ", " ferroelectric "],
    }


@pytest.fixture
def mock_csv_data(tmp_path):
    """Create a mock CSV file for testing"""
    csv_path = tmp_path / "metadata.csv"
    with open(csv_path, "w") as f:
        f.write(SAMPLE_METADATA_CSV)
    return str(csv_path)


@pytest.fixture
def processor(
    mock_environment, property_keywords, mock_csv_data, monkeypatch, tmp_path
):
    """Fixture for creating a SpringerArticleProcessor instance"""
    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)
    with patch("pandas.read_csv", return_value=pd.read_csv(mock_csv_data)):
        processor = SpringerArticleProcessor(
            main_property_keyword="piezoelectric",
            property_keywords=property_keywords,
            is_sql_db=False,
            is_save_xml=False,
        )
        processor.metadata_csv_filename = mock_csv_data
        processor.timeout_file = str(tmp_path / "timeout_dois.txt")

        return processor


class TestSpringerArticleProcessor:
    """Test class for SpringerArticleProcessor"""

    def test_initialization(self, mock_environment, property_keywords):
        """Test initialization of SpringerArticleProcessor with correct parameters"""

        processor = SpringerArticleProcessor(
            main_property_keyword="piezoelectric", property_keywords=property_keywords
        )

        assert processor.keyword == "piezoelectric"
        assert processor.property_keywords == property_keywords
        assert processor.openaccess_api_key == "test_api_key"
        assert processor.tdm_api_key == "test_tdm_api_key"
        assert processor.is_sql_db is False
        assert processor.is_save_xml is False
        assert processor.is_exceeded is False

    def test_initialization_missing_keyword(self, mock_environment, property_keywords):
        """Test initialization with missing main_property_keyword"""

        with pytest.raises(ValueErrorHandler) as exc_info:
            SpringerArticleProcessor(property_keywords=property_keywords)

        assert "main_property_keyword" in str(exc_info.value)

    def test_initialization_missing_property_keywords(self, mock_environment):
        """Test initialization with missing property_keywords"""

        with pytest.raises(ValueErrorHandler) as exc_info:
            SpringerArticleProcessor(main_property_keyword="piezoelectric")

        assert "property_keywords" in str(exc_info.value)

    def test_initialization_missing_api_key(self, monkeypatch, property_keywords):
        """Test initialization with missing API key"""

        monkeypatch.delenv("SPRINGER_OPENACCESS_API_KEY", raising=False)

        with pytest.raises(ValueErrorHandler) as exc_info:
            SpringerArticleProcessor(
                main_property_keyword="piezoelectric",
                property_keywords=property_keywords,
            )

        assert "springer_open_access_api_key" in str(exc_info.value).lower()

    def test_load_and_preprocess_data(self, processor, mock_csv_data):
        """Test loading and preprocessing data from CSV"""

        with patch("pandas.read_csv", return_value=pd.read_csv(mock_csv_data)):
            processor._load_and_preprocess_data()

        assert not processor.df.empty
        assert "doi" in processor.df.columns
        assert "general_publisher" in processor.df.columns
        assert all(
            publisher.lower() == "springer"
            for publisher in processor.df["general_publisher"]
        )

    def test_load_and_preprocess_data_with_row_limits(self, processor, mock_csv_data):
        """Test loading data with row limits"""
        processor.start_row = 0
        processor.end_row = 1
        with patch("pandas.read_csv", return_value=pd.read_csv(mock_csv_data)):
            processor._load_and_preprocess_data()
        assert len(processor.df) == 1

    def test_parse_response(self, processor, sample_springer_response):
        """Test parsing XML response"""

        mock_response = MagicMock()
        mock_response.text = sample_springer_response
        root = processor._parse_response(mock_response)
        assert root is not None
        assert root.tag == "article"
        assert root.xpath('.//*[local-name()="abstract"]')
        assert root.xpath('.//*[local-name()="body"]')

    def test_send_request_success(self, processor, sample_springer_response):
        """Test successful API request"""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_springer_response
        mock_response.content = sample_springer_response.encode("utf-8")

        with patch("requests.get", return_value=mock_response):
            response = processor._send_request("10.1007/s12345-678-9101-2")

        assert response is not None
        assert response.status_code == 200

    def test_send_request_rate_limit(self, processor):
        """Test API rate limit handling"""

        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("requests.get", return_value=mock_response):
            response = processor._send_request("10.1007/s12345-678-9101-2")
        assert response is None
        assert processor.is_exceeded is True

    def test_send_request_error(self, processor):
        """Test handling of request errors"""
        with patch(
            "requests.get", side_effect=requests.exceptions.RequestException("Error")
        ):
            response = processor._send_request("10.1007/s12345-678-9101-2")
        assert response is None

    def test_send_request_timeout(self, processor, tmp_path):
        """Test handling of request timeouts"""
        processor.timeout_file = str(tmp_path / "timeout_dois.txt")
        with patch(
            "requests.get", side_effect=requests.exceptions.ReadTimeout("Timeout")
        ):
            with patch("builtins.open", mock_open()) as mock_file:
                response = processor._send_request("10.1007/s12345-678-9101-2")
        assert response is None
        mock_file.assert_called_once()

    def test_modify_specific_element(self, processor):
        """Test modification of specific XML elements"""
        xml = """<p>Test with <sup>superscript</sup> element</p>"""
        root = etree.fromstring(xml)
        processor._modify_specific_element(root, "sup")
        assert b"<sup>" not in etree.tostring(root)
        assert b"E^{superscript}" in etree.tostring(root)

    def test_process_xml(self, processor, sample_springer_response):
        """Test processing of XML to extract sections"""
        root = etree.fromstring(sample_springer_response)
        abstract, sections = processor._process_xml(root)
        assert abstract is not None
        assert len(sections) > 0
        assert (
            abstract[0].xpath('.//*[local-name()="p"]')[0].text
            == "This is a sample abstract with piezoelectric property mentioned."
        )

    def test_extract_paragraphs(self, processor):
        """Test extraction of paragraphs from sections"""

        xml = """<sec><p>This is a regular paragraph.</p><p>This paragraph mentions computational methods.</p></sec>"""
        section = etree.fromstring(xml)
        other_paragraphs, comp_paragraphs = processor._extract_paragraphs(section)
        assert "regular paragraph" in other_paragraphs
        assert "computational methods" in comp_paragraphs

    def test_process_tables(self, processor, sample_table_xml):
        """Test processing of tables from XML"""
        root = etree.fromstring(sample_table_xml)
        tables = root.xpath(".//table")
        header_data, column_number, all_table_data, caption_data = (
            processor._process_tables(tables)
        )
        assert len(header_data) == 1
        assert header_data[0] == [
            "Material",
            "Piezoelectric Coefficient",
            "Temperature (K)",
        ]
        assert column_number[0] == 3
        assert len(all_table_data[0]) == 2
        assert "Sample Table 1" in caption_data[0]

    def test_generate_tables(self, processor):
        """Test generation of markdown tables"""

        header_data = [["Material", "Piezoelectric Coefficient", "Temperature (K)"]]
        column_number = [3]
        all_table_data = [[["PZT", "450", "300"], ["BaTiO3", "190", "300"]]]
        caption_data = ["Sample Table 1"]

        tables = processor._generate_tables(
            header_data, column_number, all_table_data, caption_data
        )

        assert len(tables) == 1
        assert "Table 1.Sample Table 1" in tables[0]
        assert "Material|Piezoelectric Coefficient|Temperature (K)" in tables[0]
        assert "PZT|450|300" in tables[0]
        assert "BaTiO3|190|300" in tables[0]

    def test_append_sections_to_df(self, processor):
        """Test appending sections to dataframe"""
        root = etree.fromstring(sample_springer_response)
        abstract = root.xpath('.//*[local-name()="abstract"]')
        sections = root.xpath('.//*[local-name()="body"]/sec')

        result_df = processor._append_sections_to_df(
            abstract,
            sections,
            "10.1007/s12345-678-9101-2",
            [],
            "Sample Article Title",
            "Sample Journal",
            "Springer Nature",
        )
        assert not result_df.empty
        assert result_df["doi"].iloc[0] == "10.1007/s12345-678-9101-2"
        assert result_df["article_title"].iloc[0] == "Sample Article Title"
        assert result_df["publication_name"].iloc[0] == "Sample Journal"
        assert result_df["publisher"].iloc[0] == "Springer Nature"
        assert result_df["is_property_mentioned"].iloc[0] == "1"

    @patch("pandas.read_csv")
    @patch("requests.get")
    def test_process_articles(
        self, mock_requests, mock_read_csv, processor, mock_csv_data
    ):
        """Test processing of articles"""
        mock_df = pd.read_csv(mock_csv_data)
        mock_read_csv.return_value = mock_df
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_springer_response
        mock_response.content = sample_springer_response.encode("utf-8")
        mock_requests.return_value = mock_response
        with patch("pandas.DataFrame.to_csv"):
            processor._process_articles()
        assert processor.valid_property_articles > 0

    @patch("os.path.isfile")
    @patch("os.path.exists")
    @patch(
        "builtins.open", new_callable=mock_open, read_data="10.1007/s12345-678-9101-2\n"
    )
    def test_process_with_timeout_handling(
        self, mock_file, mock_exists, mock_isfile, processor
    ):
        """Test processing with timeout handling"""
        mock_isfile.side_effect = [True, False]
        mock_exists.return_value = True
        with patch.object(processor, "_process_articles") as mock_process:
            processor._process_with_timeout_handling()
        assert mock_process.call_count == 1
        assert processor.doi_list == ["10.1007/s12345-678-9101-2"]

    @patch("os.path.isfile")
    def test_process_springer_articles(self, mock_isfile, processor):
        """Test the main processing workflow"""
        mock_isfile.return_value = False
        with patch.object(processor, "_process_articles") as mock_process:
            processor.process_springer_articles()
        mock_process.assert_called_once()
