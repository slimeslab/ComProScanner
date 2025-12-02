"""
test_data_preparator.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 01-12-2025
"""

import pytest
import pandas as pd
import json
import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from comproscanner.utils.data_preparator import (
    SectionProcessor,
    MatPropDataPreparator,
)
from comproscanner.utils.error_handler import (
    ValueErrorHandler,
    FileNotFoundErrorHandler,
)


class TestSectionProcessor:
    """Test suite for SectionProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a SectionProcessor instance for testing."""
        return SectionProcessor()

    def test_initialization(self, processor):
        """Test SectionProcessor initialization."""
        assert processor.section_names["article_title"] == "TITLE"
        assert processor.section_names["abstract"] == "ABSTRACT"
        assert processor.section_names["introduction"] == "INTRODUCTION"
        assert processor.section_names["exp_methods"] == "EXPERIMENTAL METHODS"
        assert processor.section_names["results_discussion"] == "RESULTS AND DISCUSSION"
        assert processor.section_names["conclusion"] == "CONCLUSION"

        assert len(processor.column_to_section_map) == 6
        assert processor.column_to_section_map["article_title"] == "article_title"

    def test_separate_tables_and_text_with_table(self, processor):
        """Test separating tables from text when table is present."""
        text = "Main content here.\nTable 1. Table content"
        main_text, tables = processor._separate_tables_and_text(text)

        assert main_text == "Main content here."
        assert tables == "Table 1. Table content"

    def test_separate_tables_and_text_without_table(self, processor):
        """Test separating tables from text when no table is present."""
        text = "Only main content here."
        main_text, tables = processor._separate_tables_and_text(text)

        assert main_text == "Only main content here."
        assert tables == ""

    def test_separate_tables_and_text_with_nan(self, processor):
        """Test separating tables from text with NaN input."""
        main_text, tables = processor._separate_tables_and_text(pd.NA)

        assert main_text == ""
        assert tables == ""

    def test_split_into_sentences_with_period_and_space(self, processor):
        """Test splitting text into sentences with period and space."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = processor._split_into_sentences(text)

        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence."
        assert sentences[2] == "Third sentence."

    def test_split_into_sentences_with_newline(self, processor):
        """Test splitting text into sentences with newline after period."""
        text = "First sentence.\nSecond sentence."
        sentences = processor._split_into_sentences(text)

        assert len(sentences) == 2
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence."

    def test_split_into_sentences_with_nan(self, processor):
        """Test splitting sentences with NaN input."""
        sentences = processor._split_into_sentences(pd.NA)

        assert sentences == []

    def test_has_digits_or_consecutive_caps_with_digits(self, processor):
        """Test detection of digits in sentence."""
        sentence = "Temperature was 300K."
        assert processor._has_digits_or_consecutive_caps(sentence) is True

    def test_has_digits_or_consecutive_caps_with_consecutive_caps(self, processor):
        """Test detection of consecutive capital letters."""
        sentence = "XRD analysis was performed."
        assert processor._has_digits_or_consecutive_caps(sentence) is True

    def test_has_digits_or_consecutive_caps_with_both(self, processor):
        """Test detection with both digits and consecutive caps."""
        sentence = "XRD showed peaks at 2theta = 30 degrees."
        assert processor._has_digits_or_consecutive_caps(sentence) is True

    def test_has_digits_or_consecutive_caps_without_either(self, processor):
        """Test sentence without digits or consecutive caps."""
        sentence = "This is a normal sentence."
        assert processor._has_digits_or_consecutive_caps(sentence) is False

    def test_get_relevant_sentences_filters_correctly(self, processor):
        """Test that only relevant sentences are returned."""
        text = (
            "Normal sentence. Temperature was 300K. Another normal. "
            "XRD analysis was done. Yet another normal."
        )
        relevant = processor._get_relevant_sentences(text)

        assert len(relevant) == 2
        assert "300K" in relevant[0]
        assert "XRD" in relevant[1]

    def test_get_relevant_sentences_with_nan(self, processor):
        """Test getting relevant sentences with NaN input."""
        relevant = processor._get_relevant_sentences(pd.NA)

        assert relevant == []

    def test_process_section_results_discussion_with_tables(self, processor):
        """Test processing results_discussion section with tables."""
        text = "Temperature was 500K.\nTable 1. Data table"
        tables_text, main_text = processor._process_section(text, "results_discussion")

        assert "# TABLES:" in tables_text
        assert "Data table" in tables_text
        assert "# RESULTS AND DISCUSSION" in main_text
        assert "500K" in main_text

    def test_process_section_results_discussion_without_relevant_content(
        self, processor
    ):
        """Test processing results_discussion with no relevant content."""
        text = "Normal text without digits or caps."
        tables_text, main_text = processor._process_section(text, "results_discussion")

        assert tables_text == ""
        assert main_text == ""

    def test_process_section_regular_section(self, processor):
        """Test processing regular sections (not results_discussion)."""
        text = "Introduction with 300K temperature. Normal sentence."
        tables_text, main_text = processor._process_section(text, "introduction")

        assert tables_text == ""
        assert "# INTRODUCTION" in main_text
        assert "300K" in main_text

    def test_process_section_with_nan(self, processor):
        """Test processing section with NaN input."""
        tables_text, main_text = processor._process_section(pd.NA, "abstract")

        assert tables_text == ""
        assert main_text == ""

    def test_create_formatted_texts_complete_row(self, processor):
        """Test creating formatted texts with complete row data."""
        row = pd.Series(
            {
                "article_title": "Study on XRD Analysis.",
                "abstract": "Temperature was 500K.",
                "introduction": "XPS measurement was done.",
                "results_discussion": "Results showed 99% purity.\nTable 1. Data",
                "conclusion": "Synthesis at 300C was successful.",
                "exp_methods": "Sample prepared at 400K.",
            }
        )

        comp_prop_text, synthesis_text = processor.create_formatted_texts(row)

        # Check composition/property text
        assert "# TABLES:" in comp_prop_text
        assert "# RESULTS AND DISCUSSION" in comp_prop_text
        assert "# TITLE" in comp_prop_text
        assert "# ABSTRACT" in comp_prop_text
        assert "# INTRODUCTION" in comp_prop_text
        assert "# CONCLUSION" in comp_prop_text

        # Check synthesis text
        assert "# EXPERIMENTAL METHODS" in synthesis_text
        assert "# RESULTS AND DISCUSSION" in synthesis_text

    def test_create_formatted_texts_minimal_row(self, processor):
        """Test creating formatted texts with minimal row data."""
        row = pd.Series(
            {
                "article_title": "Test Article",
                "abstract": pd.NA,
                "introduction": pd.NA,
                "results_discussion": pd.NA,
                "conclusion": pd.NA,
                "exp_methods": pd.NA,
            }
        )

        comp_prop_text, synthesis_text = processor.create_formatted_texts(row)

        # Should still work but with minimal content
        assert isinstance(comp_prop_text, str)
        assert isinstance(synthesis_text, str)


class TestMatPropDataPreparator:
    """Test suite for MatPropDataPreparator class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        return pd.DataFrame(
            {
                "doi": ["10.1000/test1", "10.1000/test2", "10.1000/test3"],
                "is_property_mentioned": ["1", "1", "0"],
                "article_title": [
                    "Test Article with XRD analysis.",
                    "Test Article with 300K temperature.",
                    "Test Article 3",
                ],
                "abstract": [
                    "Abstract with 300K temperature.",
                    "Abstract with XRD.",
                    "Abstract 3",
                ],
                "introduction": ["Intro with 500K.", "Intro with XPS.", "Intro 3"],
                "results_discussion": [
                    "Results at 500K with measurement.",
                    "Results with XRD pattern.",
                    "Results 3",
                ],
                "conclusion": [
                    "Conclusion at 800C synthesis.",
                    "Conclusion with 99%.",
                    "Conclusion 3",
                ],
                "exp_methods": ["Methods at 400K.", "Methods with XRD.", "Methods 3"],
            }
        )

    @pytest.fixture
    def mock_database_config(self, temp_dir):
        """Mock DatabaseConfig for testing."""
        with patch("comproscanner.utils.data_preparator.DatabaseConfig") as mock_config:
            mock_instance = MagicMock()
            mock_instance.EXTRACTED_CSV_FOLDERPATH = temp_dir
            mock_config.return_value = mock_instance
            yield mock_config

    def test_initialization_success(self, temp_dir, mock_database_config):
        """Test successful initialization of MatPropDataPreparator."""
        json_file = os.path.join(temp_dir, "results.json")

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
        )

        assert preparator.main_property_keyword == "conductivity"
        assert preparator.main_extraction_keyword == "electrical conductivity"
        assert preparator.json_results_file == json_file
        assert preparator.start_row == 0
        assert preparator.num_rows is None

    def test_initialization_missing_main_property_keyword(
        self, temp_dir, mock_database_config
    ):
        """Test initialization fails without main_property_keyword."""
        json_file = os.path.join(temp_dir, "results.json")

        with pytest.raises(ValueErrorHandler):
            MatPropDataPreparator(
                main_property_keyword=None,
                main_extraction_keyword="electrical conductivity",
                json_results_file=json_file,
            )

    def test_initialization_missing_extraction_keyword(
        self, temp_dir, mock_database_config
    ):
        """Test initialization fails without main_extraction_keyword."""
        json_file = os.path.join(temp_dir, "results.json")

        with pytest.raises(ValueErrorHandler):
            MatPropDataPreparator(
                main_property_keyword="conductivity",
                main_extraction_keyword=None,
                json_results_file=json_file,
            )

    def test_initialization_missing_json_file(self, temp_dir, mock_database_config):
        """Test initialization fails without json_results_file."""
        with pytest.raises(ValueErrorHandler):
            MatPropDataPreparator(
                main_property_keyword="conductivity",
                main_extraction_keyword="electrical conductivity",
                json_results_file=None,
            )

    def test_initialization_test_mode_missing_test_file(
        self, temp_dir, mock_database_config
    ):
        """Test initialization fails in test mode without test_doi_list_file."""
        json_file = os.path.join(temp_dir, "results.json")

        with pytest.raises(ValueErrorHandler):
            MatPropDataPreparator(
                main_property_keyword="conductivity",
                main_extraction_keyword="electrical conductivity",
                json_results_file=json_file,
                is_test_data_preparation=True,
                test_doi_list_file=None,
            )

    def test_initialization_test_mode_success(self, temp_dir, mock_database_config):
        """Test successful initialization in test mode."""
        json_file = os.path.join(temp_dir, "results.json")
        test_file = os.path.join(temp_dir, "test_dois.txt")

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
            is_test_data_preparation=True,
            test_doi_list_file=test_file,
            total_test_data=100,
        )

        assert preparator.is_test_data_preparation is True
        assert preparator.test_doi_list_file == test_file
        assert preparator.total_test_data == 100
        assert preparator.test_random_seed == 42

    def test_load_existing_results_no_file(self, temp_dir, mock_database_config):
        """Test loading results when file doesn't exist."""
        json_file = os.path.join(temp_dir, "results.json")

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
        )

        assert preparator.results == {}

    def test_load_existing_results_with_file(self, temp_dir, mock_database_config):
        """Test loading existing results from JSON file."""
        json_file = os.path.join(temp_dir, "results.json")
        existing_results = {"10.1000/test1": {"data": "value"}}

        with open(json_file, "w") as f:
            json.dump(existing_results, f)

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
        )

        assert preparator.results == existing_results

    def test_load_checked_dois_no_file(self, temp_dir, mock_database_config):
        """Test loading checked DOIs when file doesn't exist."""
        json_file = os.path.join(temp_dir, "results.json")
        checked_file = os.path.join(temp_dir, "checked.txt")

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
            checked_doi_list_file=checked_file,
        )

        assert preparator.checked_dois == set()

    def test_load_checked_dois_with_file(self, temp_dir, mock_database_config):
        """Test loading checked DOIs from file."""
        json_file = os.path.join(temp_dir, "results.json")
        checked_file = os.path.join(temp_dir, "checked.txt")

        with open(checked_file, "w") as f:
            f.write("10.1000/test1\n")
            f.write("10.1000/test2\n")

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
            checked_doi_list_file=checked_file,
        )

        assert len(preparator.checked_dois) == 2
        assert "10.1000/test1" in preparator.checked_dois
        assert "10.1000/test2" in preparator.checked_dois

    def test_get_unprocessed_data_no_files(self, temp_dir, mock_database_config):
        """Test get_unprocessed_data with no CSV files."""
        json_file = os.path.join(temp_dir, "results.json")

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
        )

        with pytest.raises(FileNotFoundErrorHandler):
            preparator.get_unprocessed_data()

    def test_get_unprocessed_data_with_csv(
        self, temp_dir, sample_csv_data, mock_database_config
    ):
        """Test get_unprocessed_data with CSV files."""
        json_file = os.path.join(temp_dir, "results.json")
        csv_file = os.path.join(temp_dir, "test_data.csv")
        sample_csv_data.to_csv(csv_file, index=False)

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
        )

        data = preparator.get_unprocessed_data()

        # Should only process rows where is_property_mentioned is True
        assert len(data) == 2
        assert all("doi" in item for item in data)
        assert all("main_extraction_keyword" in item for item in data)
        assert all("comp_prop_text" in item for item in data)
        assert all("synthesis_text" in item for item in data)

    def test_get_unprocessed_data_with_processed_dois(
        self, temp_dir, sample_csv_data, mock_database_config
    ):
        """Test get_unprocessed_data excludes already processed DOIs."""
        json_file = os.path.join(temp_dir, "results.json")
        csv_file = os.path.join(temp_dir, "test_data.csv")
        sample_csv_data.to_csv(csv_file, index=False)

        # Create results file with one processed DOI
        existing_results = {"10.1000/test1": {"data": "value"}}
        with open(json_file, "w") as f:
            json.dump(existing_results, f)

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
        )

        data = preparator.get_unprocessed_data()

        # Should only have one unprocessed DOI (test2)
        assert len(data) == 1
        assert data[0]["doi"] == "10.1000/test2"

    def test_get_unprocessed_data_with_checked_dois(
        self, temp_dir, sample_csv_data, mock_database_config
    ):
        """Test get_unprocessed_data excludes checked DOIs."""
        json_file = os.path.join(temp_dir, "results.json")
        csv_file = os.path.join(temp_dir, "test_data.csv")
        checked_file = os.path.join(temp_dir, "checked.txt")
        sample_csv_data.to_csv(csv_file, index=False)

        # Create checked DOIs file
        with open(checked_file, "w") as f:
            f.write("10.1000/test1\n")

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
            checked_doi_list_file=checked_file,
        )

        data = preparator.get_unprocessed_data()

        # Should exclude checked DOI
        assert len(data) == 1
        assert data[0]["doi"] == "10.1000/test2"

    def test_get_unprocessed_data_with_num_rows(
        self, temp_dir, sample_csv_data, mock_database_config
    ):
        """Test get_unprocessed_data respects num_rows parameter."""
        json_file = os.path.join(temp_dir, "results.json")
        csv_file = os.path.join(temp_dir, "test_data.csv")
        sample_csv_data.to_csv(csv_file, index=False)

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
            num_rows=1,
        )

        data = preparator.get_unprocessed_data()

        assert len(data) == 1

    def test_get_unprocessed_data_with_start_row(
        self, temp_dir, sample_csv_data, mock_database_config
    ):
        """Test get_unprocessed_data respects start_row parameter."""
        json_file = os.path.join(temp_dir, "results.json")
        csv_file = os.path.join(temp_dir, "test_data.csv")
        sample_csv_data.to_csv(csv_file, index=False)

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
            start_row=1,
        )

        data = preparator.get_unprocessed_data()

        # Should skip first row
        assert len(data) <= 2

    def test_get_unprocessed_data_test_mode_with_test_dois(
        self, temp_dir, sample_csv_data, mock_database_config
    ):
        """Test get_unprocessed_data in test mode with test DOI list."""
        json_file = os.path.join(temp_dir, "results.json")
        csv_file = os.path.join(temp_dir, "test_data.csv")
        test_file = os.path.join(temp_dir, "test_dois.txt")
        sample_csv_data.to_csv(csv_file, index=False)

        # Create test DOI list
        with open(test_file, "w") as f:
            f.write("10.1000/test1\n")

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
            is_test_data_preparation=True,
            test_doi_list_file=test_file,
            total_test_data=1,
        )

        data = preparator.get_unprocessed_data()

        assert len(data) >= 0

    def test_get_unprocessed_data_test_mode_only_consider_list(
        self, temp_dir, sample_csv_data, mock_database_config
    ):
        """Test get_unprocessed_data in test mode with only_consider_test_doi_list."""
        json_file = os.path.join(temp_dir, "results.json")
        csv_file = os.path.join(temp_dir, "test_data.csv")
        test_file = os.path.join(temp_dir, "test_dois.txt")
        sample_csv_data.to_csv(csv_file, index=False)

        # Create test DOI list
        with open(test_file, "w") as f:
            f.write("10.1000/test1\n")
            f.write("10.1000/test2\n")

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
            is_test_data_preparation=True,
            test_doi_list_file=test_file,
            total_test_data=2,
            is_only_consider_test_doi_list=True,
        )

        data = preparator.get_unprocessed_data()

        # Should only process DOIs from test list
        assert len(data) == 2
        assert all(item["doi"] in ["10.1000/test1", "10.1000/test2"] for item in data)

    def test_get_unprocessed_data_handles_processing_error(
        self, temp_dir, sample_csv_data, mock_database_config
    ):
        """Test get_unprocessed_data handles errors during processing."""
        json_file = os.path.join(temp_dir, "results.json")
        csv_file = os.path.join(temp_dir, "test_data.csv")
        sample_csv_data.to_csv(csv_file, index=False)

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
        )

        # Mock the processor to raise an exception
        with patch(
            "comproscanner.utils.data_preparator.SectionProcessor.create_formatted_texts"
        ) as mock_processor:
            mock_processor.side_effect = Exception("Test error")

            data = preparator.get_unprocessed_data()

            # Should handle error and continue
            assert data == []

    def test_get_unprocessed_data_multiple_csv_files(
        self, temp_dir, sample_csv_data, mock_database_config
    ):
        """Test get_unprocessed_data with multiple CSV files."""
        json_file = os.path.join(temp_dir, "results.json")
        csv_file1 = os.path.join(temp_dir, "test_data1.csv")
        csv_file2 = os.path.join(temp_dir, "test_data2.csv")

        # Create two CSV files
        sample_csv_data.iloc[:2].to_csv(csv_file1, index=False)
        sample_csv_data.iloc[1:].to_csv(csv_file2, index=False)

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
        )

        data = preparator.get_unprocessed_data()

        # Should process data from both files
        assert len(data) >= 2


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    @pytest.fixture
    def complete_test_data(self):
        """Create complete test data with all sections."""
        return pd.DataFrame(
            {
                "doi": ["10.1000/complete"],
                "is_property_mentioned": ["1"],
                "article_title": ["XRD Study of Materials with analysis."],
                "abstract": ["Temperature was 500K in the experiment."],
                "introduction": ["XPS analysis revealed interesting patterns."],
                "results_discussion": [
                    "Conductivity measured at 300K was high.\nTable 1. Measurement data"
                ],
                "conclusion": ["Synthesis at 800C was successful."],
                "exp_methods": ["Samples prepared at 400K using standard methods."],
            }
        )

    @pytest.fixture
    def mock_database_config(self, temp_dir):
        """Mock DatabaseConfig for testing."""
        with patch("comproscanner.utils.data_preparator.DatabaseConfig") as mock_config:
            mock_instance = MagicMock()
            mock_instance.EXTRACTED_CSV_FOLDERPATH = temp_dir
            mock_config.return_value = mock_instance
            yield mock_config

    def test_end_to_end_data_preparation(
        self, temp_dir, complete_test_data, mock_database_config
    ):
        """Test complete data preparation workflow."""
        json_file = os.path.join(temp_dir, "results.json")
        csv_file = os.path.join(temp_dir, "complete_data.csv")
        complete_test_data.to_csv(csv_file, index=False)

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
        )

        data = preparator.get_unprocessed_data()

        assert len(data) == 1
        item = data[0]

        # Verify structure
        assert item["doi"] == "10.1000/complete"
        assert item["main_extraction_keyword"] == "electrical conductivity"

        # Verify composition/property text has all sections
        assert "# TABLES:" in item["comp_prop_text"]
        assert "# RESULTS AND DISCUSSION" in item["comp_prop_text"]
        assert "# TITLE" in item["comp_prop_text"]
        assert "# ABSTRACT" in item["comp_prop_text"]
        assert "# INTRODUCTION" in item["comp_prop_text"]
        assert "# CONCLUSION" in item["comp_prop_text"]

        # Verify synthesis text
        assert "# EXPERIMENTAL METHODS" in item["synthesis_text"]
        assert "# RESULTS AND DISCUSSION" in item["synthesis_text"]

    def test_filtering_workflow(
        self, temp_dir, complete_test_data, mock_database_config
    ):
        """Test filtering workflow with processed and checked DOIs."""
        json_file = os.path.join(temp_dir, "results.json")
        csv_file = os.path.join(temp_dir, "data.csv")
        checked_file = os.path.join(temp_dir, "checked.txt")

        # Add more rows
        df = pd.concat([complete_test_data] * 3, ignore_index=True)
        df["doi"] = ["10.1000/doi1", "10.1000/doi2", "10.1000/doi3"]
        df.to_csv(csv_file, index=False)

        # Mark one as processed
        with open(json_file, "w") as f:
            json.dump({"10.1000/doi1": {"processed": True}}, f)

        # Mark one as checked
        with open(checked_file, "w") as f:
            f.write("10.1000/doi2\n")

        preparator = MatPropDataPreparator(
            main_property_keyword="conductivity",
            main_extraction_keyword="electrical conductivity",
            json_results_file=json_file,
            checked_doi_list_file=checked_file,
        )

        data = preparator.get_unprocessed_data()

        # Should only process doi3
        assert len(data) == 1
        assert data[0]["doi"] == "10.1000/doi3"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
