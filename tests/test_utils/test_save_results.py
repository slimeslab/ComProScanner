"""
test_save_results.py

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
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from comproscanner.utils.save_results import SaveResults


class TestSaveResults:
    """Test suite for SaveResults class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    @pytest.fixture
    def json_file(self, temp_dir):
        """Provide path to temporary JSON file."""
        return os.path.join(temp_dir, "test_results.json")

    @pytest.fixture
    def csv_file(self, temp_dir):
        """Provide path to temporary CSV file."""
        return os.path.join(temp_dir, "test_results.csv")

    @pytest.fixture
    def save_results(self, json_file, csv_file):
        """Create SaveResults instance with temporary files."""
        return SaveResults(json_results_file=json_file, csv_results_file=csv_file)

    @pytest.fixture
    def sample_result_dict(self):
        """Sample result dictionary for testing."""
        return {
            "paper_metadata": {
                "doi": "10.1000/test1",
                "title": "Test Article",
                "year": "2023",
            },
            "extraction_data": {
                "composition": "Li2O",
                "property": "conductivity",
                "value": "0.5",
            },
        }

    @pytest.fixture
    def sample_nested_result(self):
        """Sample nested result for JSON testing."""
        return {
            "doi": "10.1000/test1",
            "title": "Test Article",
            "data": {"composition": "Li2O", "property": "conductivity"},
        }

    def test_initialization_creates_empty_results(self, json_file, csv_file):
        """Test initialization creates empty results dictionary."""
        save_results = SaveResults(
            json_results_file=json_file, csv_results_file=csv_file
        )

        assert save_results.results == {}
        assert save_results.json_results_file == json_file
        assert save_results.csv_results_file == csv_file

    def test_initialization_loads_existing_json(self, json_file, csv_file):
        """Test initialization loads existing JSON file."""
        existing_data = {"10.1000/existing": {"data": "value"}}

        with open(json_file, "w") as f:
            json.dump(existing_data, f)

        save_results = SaveResults(
            json_results_file=json_file, csv_results_file=csv_file
        )

        assert save_results.results == existing_data

    def test_initialization_handles_corrupted_json(self, json_file, csv_file):
        """Test initialization handles corrupted JSON file."""
        with open(json_file, "w") as f:
            f.write("invalid json content {")

        save_results = SaveResults(
            json_results_file=json_file, csv_results_file=csv_file
        )

        assert save_results.results == {}

    def test_initialization_handles_empty_json(self, json_file, csv_file):
        """Test initialization handles empty JSON file."""
        # Create empty file
        open(json_file, "w").close()

        save_results = SaveResults(
            json_results_file=json_file, csv_results_file=csv_file
        )

        assert save_results.results == {}

    def test_update_in_json_single_result(
        self, save_results, json_file, sample_nested_result
    ):
        """Test updating JSON with a single result."""
        save_results.update_in_json("10.1000/test1", sample_nested_result)

        assert "10.1000/test1" in save_results.results
        assert save_results.results["10.1000/test1"] == sample_nested_result

        # Verify file was written
        with open(json_file, "r") as f:
            saved_data = json.load(f)
        assert saved_data["10.1000/test1"] == sample_nested_result

    def test_update_in_json_multiple_results(
        self, save_results, json_file, sample_nested_result
    ):
        """Test updating JSON with multiple results."""
        result1 = sample_nested_result
        result2 = {
            "doi": "10.1000/test2",
            "title": "Another Article",
        }

        save_results.update_in_json("10.1000/test1", result1)
        save_results.update_in_json("10.1000/test2", result2)

        assert len(save_results.results) == 2
        assert save_results.results["10.1000/test1"] == result1
        assert save_results.results["10.1000/test2"] == result2

        # Verify file contains both
        with open(json_file, "r") as f:
            saved_data = json.load(f)
        assert len(saved_data) == 2

    def test_update_in_json_overwrites_existing(
        self, save_results, json_file, sample_nested_result
    ):
        """Test that updating same DOI overwrites previous data."""
        result1 = {"data": "original"}
        result2 = {"data": "updated"}

        save_results.update_in_json("10.1000/test1", result1)
        save_results.update_in_json("10.1000/test1", result2)

        assert save_results.results["10.1000/test1"] == result2

    def test_update_in_json_creates_directory(self, temp_dir):
        """Test that update_in_json creates directory if it doesn't exist."""
        nested_dir = os.path.join(temp_dir, "subdir", "nested")
        json_file = os.path.join(nested_dir, "results.json")
        csv_file = os.path.join(temp_dir, "results.csv")

        save_results = SaveResults(
            json_results_file=json_file, csv_results_file=csv_file
        )
        save_results.update_in_json("10.1000/test", {"data": "value"})

        assert os.path.exists(nested_dir)
        assert os.path.exists(json_file)

    def test_update_in_json_handles_non_serializable(
        self, save_results, json_file, capsys
    ):
        """Test handling of non-serializable objects."""
        # Create a non-serializable object
        non_serializable = {"date": pd.Timestamp("2023-01-01")}

        save_results.update_in_json("10.1000/test", non_serializable)

        # Should still save using string conversion
        assert os.path.exists(json_file)
        with open(json_file, "r") as f:
            saved_data = json.load(f)
        assert "10.1000/test" in saved_data

    def test_update_in_csv_new_file(self, save_results, csv_file, sample_result_dict):
        """Test creating new CSV file."""
        save_results.update_in_csv(sample_result_dict)

        assert os.path.exists(csv_file)

        df = pd.read_csv(csv_file)
        assert len(df) == 1
        assert "doi" in df.columns
        assert "title" in df.columns
        assert "composition" in df.columns

    def test_update_in_csv_appends_to_existing(
        self, save_results, csv_file, sample_result_dict
    ):
        """Test appending to existing CSV file."""
        # First write
        save_results.update_in_csv(sample_result_dict)

        # Second write with different data
        result_dict2 = {
            "paper_metadata": {
                "doi": "10.1000/test2",
                "title": "Second Article",
                "year": "2024",
            },
            "extraction_data": {
                "composition": "Na2O",
                "property": "resistance",
                "value": "1.0",
            },
        }

        save_results.update_in_csv(result_dict2)

        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert df.iloc[0]["doi"] == "10.1000/test1"
        assert df.iloc[1]["doi"] == "10.1000/test2"

    def test_update_in_csv_extracts_nested_keys(
        self, save_results, csv_file, sample_result_dict
    ):
        """Test that nested dictionary keys are properly extracted."""
        save_results.update_in_csv(sample_result_dict)

        df = pd.read_csv(csv_file)

        # Should have flattened keys from both nested dicts
        expected_columns = ["doi", "title", "year", "composition", "property", "value"]
        for col in expected_columns:
            assert col in df.columns

    def test_update_in_csv_handles_empty_csv(self, save_results, csv_file):
        """Test handling of existing but empty CSV file."""
        # Create empty CSV file
        open(csv_file, "w").close()

        result_dict = {
            "data": {"doi": "10.1000/test", "title": "Test"},
        }

        save_results.update_in_csv(result_dict)

        df = pd.read_csv(csv_file)
        assert len(df) == 1
        assert "doi" in df.columns

    def test_update_in_csv_with_none_values(self, save_results, csv_file):
        """Test CSV update with None values."""
        result_dict = {
            "data1": {"field1": "value1", "field2": None},
            "data2": {"field3": "value3"},
        }

        save_results.update_in_csv(result_dict)

        df = pd.read_csv(csv_file)
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["field2"])

    def test_update_in_csv_consistent_header_order(self, save_results, csv_file):
        """Test that CSV headers maintain consistent order."""
        result1 = {"data": {"z_field": "1", "a_field": "2", "m_field": "3"}}

        save_results.update_in_csv(result1)

        df = pd.read_csv(csv_file)
        columns = list(df.columns)

        # Should be sorted alphabetically
        assert columns == sorted(columns)

    def test_update_in_csv_with_non_dict_values(self, save_results, csv_file):
        """Test CSV update when main dict values are not dictionaries."""
        result_dict = {
            "data": {"field1": "value1"},
            "non_dict": "string_value",
            "another": 123,
        }

        save_results.update_in_csv(result_dict)

        df = pd.read_csv(csv_file)
        assert len(df) == 1
        assert "field1" in df.columns

    def test_load_existing_results_with_unicode(self, json_file, csv_file):
        """Test loading JSON with Unicode characters."""
        unicode_data = {
            "10.1000/test": {
                "title": "Test with Unicode: 中文 • ñ • €",
                "data": "value",
            }
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(unicode_data, f, ensure_ascii=False)

        save_results = SaveResults(
            json_results_file=json_file, csv_results_file=csv_file
        )

        assert save_results.results == unicode_data

    def test_update_in_json_with_special_characters(self, save_results, json_file):
        """Test JSON update with special characters."""
        result = {
            "title": "Test • ® © ™",
            "formula": "Li₂O",
            "temperature": "300°C",
        }

        save_results.update_in_json("10.1000/test", result)

        with open(json_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data["10.1000/test"]["title"] == "Test • ® © ™"

    def test_update_in_json_preserves_formatting(self, save_results, json_file):
        """Test that JSON is saved with proper indentation."""
        result = {"data": "value"}

        save_results.update_in_json("10.1000/test", result)

        with open(json_file, "r") as f:
            content = f.read()

        # Check for indentation (should have spaces from indent=2)
        assert "  " in content

    def test_concurrent_updates(self, save_results, json_file, csv_file):
        """Test multiple rapid updates."""
        for i in range(5):
            result = {"data": f"value_{i}"}
            save_results.update_in_json(f"10.1000/test{i}", result)

            csv_data = {
                "info": {"doi": f"10.1000/test{i}", "index": i},
            }
            save_results.update_in_csv(csv_data)

        assert len(save_results.results) == 5

        df = pd.read_csv(csv_file)
        assert len(df) == 5

    def test_update_in_csv_with_mixed_data_types(self, save_results, csv_file):
        """Test CSV update with mixed data types."""
        result_dict = {
            "data": {
                "string_field": "text",
                "int_field": 42,
                "float_field": 3.14,
                "bool_field": True,
            },
        }

        save_results.update_in_csv(result_dict)

        df = pd.read_csv(csv_file)
        assert df.iloc[0]["string_field"] == "text"
        assert df.iloc[0]["int_field"] == 42
        assert df.iloc[0]["float_field"] == 3.14
        assert df.iloc[0]["bool_field"] == True

    def test_empty_nested_dictionary(self, save_results, csv_file):
        """Test handling of empty nested dictionaries."""
        result_dict = {"data": {}, "info": {"field": "value"}}

        save_results.update_in_csv(result_dict)

        df = pd.read_csv(csv_file)
        assert "field" in df.columns
        assert len(df) == 1


class TestSaveResultsEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    def test_json_file_with_no_directory_path(self, temp_dir):
        """Test JSON file without directory path."""
        original_dir = os.getcwd()
        try:
            os.chdir(temp_dir)
            save_results = SaveResults(
                json_results_file="simple.json", csv_results_file="simple.csv"
            )

            save_results.update_in_json("10.1000/test", {"data": "value"})

            assert os.path.exists(os.path.join(temp_dir, "simple.json"))
        finally:
            os.chdir(original_dir)
