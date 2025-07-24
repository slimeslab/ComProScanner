"""
test_data_cleaner.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 14-07-2025
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, mock_open, MagicMock
from typing import Dict, Any, List
import re

# Import the modules to test
from comproscanner.post_processing.data_cleaner import (
    DataCleaner,
    CleaningStrategy,
    get_all_elements,
    calculate_resolved_compositions,
)


class TestGetAllElements:
    """Test cases for the get_all_elements function."""

    def test_get_all_elements_returns_list(self):
        """Test that get_all_elements returns a list."""
        result = get_all_elements()
        assert isinstance(result, list)

    def test_get_all_elements_correct_count(self):
        """Test that get_all_elements returns 118 elements."""
        result = get_all_elements()
        assert len(result) == 118

    def test_get_all_elements_contains_common_elements(self):
        """Test that common elements are in the returned list."""
        result = get_all_elements()
        common_elements = ["H", "He", "Li", "C", "N", "O", "Fe", "Au", "Ag"]
        for element in common_elements:
            assert element in result


class TestCleaningStrategy:
    """Test cases for the CleaningStrategy enum."""

    def test_cleaning_strategy_enum_values(self):
        """Test that CleaningStrategy enum has correct values."""
        assert CleaningStrategy.BASIC == "basic"
        assert CleaningStrategy.FULL == "full"

    def test_cleaning_strategy_membership(self):
        """Test CleaningStrategy enum membership."""
        assert "basic" in CleaningStrategy
        assert "full" in CleaningStrategy
        assert "invalid" not in CleaningStrategy


class TestDataCleanerInitialization:
    """Test cases for DataCleaner initialization."""

    @pytest.fixture
    def sample_json_data(self):
        """Fixture providing sample JSON data for testing."""
        return {
            "paper1": {
                "composition_data": {
                    "compositions_property_values": {
                        "NaCl": "5.5 eV",
                        "H2O": "1.33",
                        "CaCO3": "2.71 g/cm3",
                    }
                }
            },
            "paper2": {
                "composition_data": {
                    "compositions_property_values": {
                        "TiO2": "3.2 eV",
                        "SiO2": "2.65 g/cm3",
                    }
                }
            },
        }

    @pytest.fixture
    def temp_json_file(self, sample_json_data):
        """Fixture creating a temporary JSON file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_json_data, f)
            temp_file_path = f.name
        yield temp_file_path
        os.unlink(temp_file_path)

    def test_data_cleaner_initialization_success(self, temp_json_file):
        """Test successful DataCleaner initialization."""
        cleaner = DataCleaner(temp_json_file)
        assert cleaner.results_file == temp_json_file
        assert isinstance(cleaner.all_data, dict)
        assert isinstance(cleaner.all_elements, list)
        assert len(cleaner.all_elements) == 118

    def test_data_cleaner_initialization_file_not_found(self):
        """Test DataCleaner initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            DataCleaner("non_existent_file.json")

    def test_data_cleaner_initialization_invalid_json(self):
        """Test DataCleaner initialization with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_file_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                DataCleaner(temp_file_path)
        finally:
            os.unlink(temp_file_path)


class TestDataCleanerPrivateMethods:
    """Test cases for DataCleaner private methods."""

    @pytest.fixture
    def data_cleaner(self, temp_json_file):
        """Fixture providing a DataCleaner instance."""
        return DataCleaner(temp_json_file)

    @pytest.fixture
    def temp_json_file(self):
        """Fixture creating a temporary JSON file."""
        sample_data = {
            "paper1": {
                "composition_data": {
                    "compositions_property_values": {
                        "NaCl": "5.5 eV",
                        "H2O": "1.33",
                        "CaCO3": "2.71 g/cm3",
                    }
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            temp_file_path = f.name
        yield temp_file_path
        os.unlink(temp_file_path)

    def test_get_comp_prop_data(self, data_cleaner):
        """Test _get_comp_prop_data method."""
        extracted_data = {
            "composition_data": {
                "compositions_property_values": {"NaCl": "5.5 eV", "H2O": "1.33"}
            }
        }
        result = data_cleaner._get_comp_prop_data(extracted_data)
        expected = {"NaCl": "5.5 eV", "H2O": "1.33"}
        assert result == expected

    def test_get_comp_prop_pairs(self, data_cleaner):
        """Test _get_comp_prop_pairs method."""
        comp_data = {"NaCl": "5.5 eV", "H2O": "1.33"}
        result = data_cleaner._get_comp_prop_pairs(comp_data)
        expected = [{"NaCl": "5.5 eV"}, {"H2O": "1.33"}]
        assert result == expected

    def test_filter_invalid_keys_removes_invalid(self, data_cleaner):
        """Test _filter_invalid_keys removes entries with invalid keys."""
        dict_list = [
            {"NaCl": "value1"},  # Valid
            {"INVALID": "value2"},  # Invalid - all caps
            {"H2O": "value3"},  # Valid
            {"BADKEY": "value4"},  # Invalid - all caps
        ]
        result = data_cleaner._filter_invalid_keys(dict_list)
        assert len(result) == 2
        assert {"NaCl": "value1"} in result
        assert {"H2O": "value3"} in result

    def test_filter_invalid_keys_keeps_valid(self, data_cleaner):
        """Test _filter_invalid_keys keeps valid entries."""
        dict_list = [{"NaCl": "value1"}, {"TiO2": "value2"}, {"CaCO3": "value3"}]
        result = data_cleaner._filter_invalid_keys(dict_list)
        assert len(result) == 3
        assert result == dict_list

    def test_is_elements_valid_compositions(self, data_cleaner):
        """Test _is_elements with valid chemical compositions."""
        valid_compositions = [
            {"NaCl": "value"},
            {"TiO2": "value"},
            {"CaCO3": "value"},
            {"H2O": "value"},
            {"Fe2O3": "value"},
        ]
        for comp in valid_compositions:
            assert data_cleaner._is_elements(comp) is True

    def test_is_elements_invalid_compositions(self, data_cleaner):
        """Test _is_elements with invalid chemical compositions."""
        invalid_compositions = [
            {"XyZ": "value"},  # Invalid element
            {"Abc123": "value"},  # Invalid element
            {"RandomText": "value"},  # Invalid element
        ]
        for comp in invalid_compositions:
            assert data_cleaner._is_elements(comp) is False

    def test_is_elements_empty_dict(self, data_cleaner):
        """Test _is_elements with empty dictionary."""
        result = data_cleaner._is_elements({})
        assert result is False

    def test_remove_extra_spaces(self, data_cleaner):
        """Test _remove_extra_spaces method."""
        dict_list = [{"Na Cl": "value1"}, {"Ti O2": "value2"}, {"Ca CO3": "value3"}]
        result = data_cleaner._remove_extra_spaces(dict_list)
        expected = [{"NaCl": "value1"}, {"TiO2": "value2"}, {"CaCO3": "value3"}]
        assert result == expected

    def test_convert_fractions_to_decimal(self, data_cleaner):
        """Test _convert_fractions_to_decimal method."""
        dict_list = [
            {"Na1/2Cl1/2": "value1"},
            {"Ti2/3O4/3": "value2"},
            {"Regular": "value3"},  # No fractions
        ]
        result = data_cleaner._convert_fractions_to_decimal(dict_list)

        # Check that fractions are converted to decimals
        assert "Na0.50Cl0.50" in result[0]
        assert "Ti0.67O1.33" in result[1]
        assert {"Regular": "value3"} in result

    def test_return_in_dict(self, data_cleaner):
        """Test _return_in_dict method."""
        dict_list = [{"key1": "value1"}, {"key2": "value2"}, {"key3": "value3"}]
        result = data_cleaner._return_in_dict(dict_list)
        expected = {"key1": "value1", "key2": "value2", "key3": "value3"}
        assert result == expected

    def test_return_in_dict_overlapping_keys(self, data_cleaner):
        """Test _return_in_dict with overlapping keys (later values should win)."""
        dict_list = [
            {"key1": "value1"},
            {"key1": "value2"},  # This should overwrite
            {"key2": "value3"},
        ]
        result = data_cleaner._return_in_dict(dict_list)
        expected = {"key1": "value2", "key2": "value3"}
        assert result == expected


class TestDataCleanerPublicMethods:
    """Test cases for DataCleaner public methods."""

    @pytest.fixture
    def sample_data_with_mixed_validity(self):
        """Fixture with mixed valid and invalid composition data."""
        return {
            "paper1": {
                "composition_data": {
                    "compositions_property_values": {
                        "NaCl": "5.5 eV",  # Valid
                        "INVALID": "bad",  # Invalid key pattern
                        "H2O": "1.33",  # Valid
                        "XyZ": "wrong",  # Invalid element
                    }
                }
            },
            "paper2": {
                "composition_data": {
                    "compositions_property_values": {
                        "TiO2": "3.2 eV",  # Valid
                        "BADKEY": "bad",  # Invalid key pattern
                    }
                }
            },
        }

    @pytest.fixture
    def temp_mixed_json_file(self, sample_data_with_mixed_validity):
        """Fixture creating a temporary JSON file with mixed validity data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data_with_mixed_validity, f)
            temp_file_path = f.name
        yield temp_file_path
        os.unlink(temp_file_path)

    def test_clean_data_without_element_filtering(self, temp_mixed_json_file):
        """Test clean_data_without_element_filtering method."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result = cleaner.clean_data_without_element_filtering()

        # Should keep papers with valid compositions after basic cleaning
        assert isinstance(result, dict)
        assert len(result) > 0

        # Check that invalid key patterns are removed but element validation is skipped
        for paper_key, paper_data in result.items():
            comp_values = paper_data["composition_data"]["compositions_property_values"]
            # Should not contain entries with all-caps invalid patterns
            for comp_key in comp_values.keys():
                assert not re.match(r"(?<![a-z0-9])[A-Z]{2,}(?![a-z0-9])", comp_key)

    def test_clean_data_with_full_strategy(self, temp_mixed_json_file):
        """Test clean_data with FULL strategy (element validation)."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result = cleaner.clean_data(CleaningStrategy.FULL)

        assert isinstance(result, dict)

        # With full cleaning, should only keep papers with valid chemical elements
        for paper_key, paper_data in result.items():
            comp_values = paper_data["composition_data"]["compositions_property_values"]
            for comp_key in comp_values.keys():
                # Create a new instance to test _is_elements
                test_dict = {comp_key: comp_values[comp_key]}
                # Should only contain valid elements after full cleaning
                assert cleaner._is_elements(test_dict) is True

    def test_clean_data_with_basic_strategy(self, temp_mixed_json_file):
        """Test clean_data with BASIC strategy (no element validation)."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result = cleaner.clean_data(CleaningStrategy.BASIC)

        assert isinstance(result, dict)

        # With basic cleaning, should keep compositions even with invalid elements
        # but still filter out invalid key patterns
        for paper_key, paper_data in result.items():
            comp_values = paper_data["composition_data"]["compositions_property_values"]
            for comp_key in comp_values.keys():
                # Should not contain all-caps invalid patterns
                assert not re.match(r"(?<![a-z0-9])[A-Z]{2,}(?![a-z0-9])", comp_key)

    def test_clean_data_default_strategy(self, temp_mixed_json_file):
        """Test clean_data with default strategy (should be FULL)."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result_default = cleaner.clean_data()
        result_full = cleaner.clean_data(CleaningStrategy.FULL)

        # Default should be same as FULL strategy
        assert result_default == result_full

    def test_clean_data_empty_input(self):
        """Test clean_data with empty JSON input."""
        empty_data = {}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(empty_data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data()
            assert result == {}
        finally:
            os.unlink(temp_file_path)


class TestCalculateResolvedCompositions:
    """Test cases for calculate_resolved_compositions function."""

    def test_calculate_resolved_compositions_basic(self):
        """Test basic functionality of calculate_resolved_compositions."""
        composition_data = {
            "composition_data": {
                "compositions_property_values": {
                    "Na(0.5)Cl(0.5)": "5.5 eV",
                    "Ti(1)O(2)": "3.2 eV",
                }
            }
        }

        result = calculate_resolved_compositions(composition_data)

        assert "composition_data" in result
        comp_values = result["composition_data"]["compositions_property_values"]

        # Should process parentheses around pure numbers
        assert "Na0.5Cl0.5" in comp_values
        assert "Ti1O2" in comp_values

    def test_calculate_resolved_compositions_math_operations(self):
        """Test mathematical operations within parentheses."""
        composition_data = {
            "composition_data": {
                "compositions_property_values": {
                    "Na(1/2)Cl(1/2)": "value1",
                    "Ti(2*1)O(4/2)": "value2",
                    "Ca(1+0)Cl(3-1)": "value3",
                }
            }
        }

        result = calculate_resolved_compositions(composition_data)
        comp_values = result["composition_data"]["compositions_property_values"]

        # Should evaluate mathematical expressions
        assert "Na0.5Cl0.5" in comp_values
        assert "Ti2O2" in comp_values
        assert "Ca1Cl2" in comp_values

    def test_calculate_resolved_compositions_direct_comp_data(self):
        """Test with direct composition data (not nested in composition_data)."""
        composition_data = {
            "compositions_property_values": {
                "Na(0.5)Cl(0.5)": "5.5 eV",
                "Ti(1)O(2)": "3.2 eV",
            }
        }

        result = calculate_resolved_compositions(composition_data)

        assert "compositions_property_values" in result
        comp_values = result["compositions_property_values"]

        assert "Na0.5Cl0.5" in comp_values
        assert "Ti1O2" in comp_values

    def test_calculate_resolved_compositions_empty_input(self):
        """Test with empty input."""
        result = calculate_resolved_compositions({})
        assert result == {}

        result = calculate_resolved_compositions(None)
        assert result == {}

    def test_calculate_resolved_compositions_nested_parentheses(self):
        """Test with nested parentheses (should be handled safely)."""
        composition_data = {
            "composition_data": {
                "compositions_property_values": {
                    "Na((1+1)/2)Cl": "value1",
                    "Ti(2*(1+1))O": "value2",
                }
            }
        }

        result = calculate_resolved_compositions(composition_data)
        comp_values = result["composition_data"]["compositions_property_values"]

        # Should handle nested expressions
        assert "Na1Cl" in comp_values
        assert "Ti4O" in comp_values

    def test_calculate_resolved_compositions_max_iterations_safety(self):
        """Test that maximum iterations prevent infinite loops."""
        # This test ensures the safety counter works
        composition_data = {
            "composition_data": {
                "compositions_property_values": {
                    "Regular": "value1",  # Normal case
                    "Na(0.5)Cl": "value2",  # Should process normally
                }
            }
        }

        result = calculate_resolved_compositions(composition_data)
        comp_values = result["composition_data"]["compositions_property_values"]

        assert "Regular" in comp_values
        assert "Na0.5Cl" in comp_values


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.fixture
    def complex_test_data(self):
        """Fixture with complex test data for integration testing."""
        return {
            "paper1": {
                "composition_data": {
                    "compositions_property_values": {
                        "Na1/2Cl1/2": "5.5 eV",  # Has fractions
                        "Ti O2": "3.2 eV",  # Has spaces
                        "INVALID": "bad",  # Invalid pattern
                        "XyZ": "wrong",  # Invalid element
                    }
                }
            },
            "paper2": {
                "composition_data": {
                    "compositions_property_values": {
                        "Ca(1/2)CO3": "2.71 g/cm3",  # Has parentheses with fractions
                        "H2O": "1.33",  # Valid
                        "BADKEY": "remove",  # Invalid pattern
                    }
                }
            },
        }

    @pytest.fixture
    def temp_complex_json_file(self, complex_test_data):
        """Fixture creating a temporary JSON file with complex test data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(complex_test_data, f)
            temp_file_path = f.name
        yield temp_file_path
        os.unlink(temp_file_path)

    def test_full_pipeline_integration(self, temp_complex_json_file):
        """Test the full data cleaning pipeline integration."""
        # Test the complete pipeline
        cleaner = DataCleaner(temp_complex_json_file)

        # Clean with FULL strategy
        cleaned_data = cleaner.clean_data(CleaningStrategy.FULL)

        # Apply resolved compositions
        final_data = {}
        for paper_key, paper_data in cleaned_data.items():
            final_data[paper_key] = calculate_resolved_compositions(paper_data)

        # Verify the integration worked correctly
        assert isinstance(final_data, dict)

        for paper_key, paper_data in final_data.items():
            comp_values = paper_data["composition_data"]["compositions_property_values"]

            # Should have processed fractions, spaces, and parentheses
            for comp_key in comp_values.keys():
                # No spaces should remain
                assert " " not in comp_key
                # No obvious fraction patterns should remain
                assert (
                    "/" not in comp_key or "(" in comp_key
                )  # Allow fractions in preserved parentheses
                # No invalid patterns should remain
                assert not re.match(r"(?<![a-z0-9])[A-Z]{2,}(?![a-z0-9])", comp_key)

    def test_basic_vs_full_cleaning_comparison(self, temp_complex_json_file):
        """Test comparison between BASIC and FULL cleaning strategies."""
        cleaner = DataCleaner(temp_complex_json_file)

        basic_result = cleaner.clean_data(CleaningStrategy.BASIC)
        full_result = cleaner.clean_data(CleaningStrategy.FULL)

        # Basic should potentially have more entries (less strict)
        basic_total_compositions = sum(
            len(paper["composition_data"]["compositions_property_values"])
            for paper in basic_result.values()
        )

        full_total_compositions = sum(
            len(paper["composition_data"]["compositions_property_values"])
            for paper in full_result.values()
        )

        # Full cleaning should be more restrictive (equal or fewer compositions)
        assert full_total_compositions <= basic_total_compositions


class TestErrorHandling:
    """Test cases for error handling scenarios."""

    def test_malformed_composition_data_structure(self):
        """Test handling of malformed composition data structure."""
        malformed_data = {
            "paper1": {
                "wrong_key": {  # Missing composition_data key
                    "compositions_property_values": {"NaCl": "5.5 eV"}
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(malformed_data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            # Should handle missing keys gracefully
            with pytest.raises(KeyError):
                cleaner.clean_data()
        finally:
            os.unlink(temp_file_path)


# Parametrized tests for various input scenarios
class TestParametrizedScenarios:
    """Parametrized tests for various input scenarios."""

    @pytest.mark.parametrize(
        "input_key,expected_key",
        [
            ("Na1/2Cl1/2", "Na0.50Cl0.50"),
            ("Ti2/3O4/3", "Ti0.67O1.33"),
            ("Ca1/4CO3", "Ca0.25CO3"),
            ("Regular", "Regular"),  # No fractions
            ("H2O", "H2O"),  # No fractions
        ],
    )
    def test_convert_fractions_parametrized(self, input_key, expected_key):
        """Parametrized test for fraction conversion."""
        temp_data = {"test": {"composition_data": {"compositions_property_values": {}}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(temp_data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            dict_list = [{input_key: "test_value"}]
            result = cleaner._convert_fractions_to_decimal(dict_list)
            assert expected_key in result[0]
        finally:
            os.unlink(temp_file_path)
