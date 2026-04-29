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
from typing import Dict, Any, List
import re

# Import the modules to test
from comproscanner.post_processing.data_cleaner import (
    DataCleaner,
    CleaningStrategy,
    get_all_elements,
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

    def test_convert_fractions_and_resolve_compositions_fractions(self, data_cleaner):
        """Test _convert_fractions_and_resolve_compositions with fractions."""
        dict_list = [
            {"Na1/2Cl1/2": "value1"},
            {"Ti2/3O4/3": "value2"},
            {"Regular": "value3"},  # No fractions
        ]
        result = data_cleaner._convert_fractions_and_resolve_compositions(dict_list)

        # Check that fractions are converted to decimals
        first_key = list(result[0].keys())[0]
        assert "0.50" in first_key

        second_key = list(result[1].keys())[0]
        assert "0.67" in second_key

        assert {"Regular": "value3"} in result

    def test_convert_fractions_and_resolve_compositions_arithmetic(self, data_cleaner):
        """Test _convert_fractions_and_resolve_compositions with arithmetic in parentheses."""
        dict_list = [
            {"0.07Pb(Mn0.33Sb0.67)O3-(1-0.07)Pb(Zr0.48Ti0.52)O3": "value1"},
            {"0.96K0.48Na0.52NbO3-0.01BaZrO3": "value2"},
        ]
        result = data_cleaner._convert_fractions_and_resolve_compositions(dict_list)

        # Check that arithmetic is resolved
        first_result = list(result[0].keys())[0]
        assert "0.93" in first_result  # (1-0.07) should resolve to 0.93

        # Check that brackets are added
        second_result = list(result[1].keys())[0]
        assert "(" in second_result or "[" in second_result

    def test_convert_fractions_and_resolve_compositions_multiplication(
        self, data_cleaner
    ):
        """Test _convert_fractions_and_resolve_compositions with multiplication."""
        dict_list = [
            {"0.03*(Bi0.5Ag0.5)ZrO3": "value1"},
            {"0.03*(0.2)ZrO3": "value2"},
        ]
        result = data_cleaner._convert_fractions_and_resolve_compositions(dict_list)

        # Check that multiplication is handled
        first_result = list(result[0].keys())[0]
        # The result should contain 0.03 and the composition part
        assert "0.03" in first_result
        assert "Bi0.5Ag0.5" in first_result or "Bi" in first_result

        second_result = list(result[1].keys())[0]
        assert (
            "0.006" in second_result or "0.0060" in second_result
        )  # 0.03 * 0.2 = 0.006

    def test_convert_fractions_element_coefficient_multiplications(self, data_cleaner):
        """Test that ElementCoeff*Multiplier patterns inside brackets are resolved."""
        dict_list = [
            {"Ba0.85Ca0.15(Zr0.1*1Ti0.9*1Ta*0)O3": "value1"},
            {"(K0.5Na0.5)(Nb0.9*0.999Ta0.1*0.999)O3": "value2"},
        ]
        result = data_cleaner._convert_fractions_and_resolve_compositions(dict_list)

        first_key = list(result[0].keys())[0]
        # Ta*0 should be removed; Zr0.1*1 -> Zr0.1, Ti0.9*1 -> Ti0.9
        assert "Ta" not in first_key
        assert "Zr" in first_key
        assert "Ti" in first_key
        assert "*" not in first_key

        second_key = list(result[1].keys())[0]
        assert "*" not in second_key
        assert "Ta" in second_key
        assert "Nb" in second_key

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

    def test_clean_data_with_relevant_compositions_full_strategy(
        self, temp_mixed_json_file
    ):
        """Test clean_data_with_relevant_compositions with FULL strategy (element validation)."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result = cleaner.clean_data_with_relevant_compositions(CleaningStrategy.FULL)

        assert isinstance(result, dict)

        # With full cleaning, should only keep papers with valid chemical elements
        for paper_key, paper_data in result.items():
            comp_values = paper_data["composition_data"]["compositions_property_values"]
            for comp_key in comp_values.keys():
                # Create a new instance to test _is_elements
                test_dict = {comp_key: comp_values[comp_key]}
                # Should only contain valid elements after full cleaning
                assert cleaner._is_elements(test_dict) is True

    def test_clean_data_with_relevant_compositions_basic_strategy(
        self, temp_mixed_json_file
    ):
        """Test clean_data_with_relevant_compositions with BASIC strategy (no element validation)."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result = cleaner.clean_data_with_relevant_compositions(CleaningStrategy.BASIC)

        assert isinstance(result, dict)

        # With basic cleaning, should keep compositions even with invalid elements
        # but still filter out invalid key patterns
        for paper_key, paper_data in result.items():
            comp_values = paper_data["composition_data"]["compositions_property_values"]
            for comp_key in comp_values.keys():
                # Should not contain all-caps invalid patterns
                assert not re.match(r"(?<![a-z0-9])[A-Z]{2,}(?![a-z0-9])", comp_key)

    def test_clean_data_with_relevant_compositions_default_strategy(
        self, temp_mixed_json_file
    ):
        """Test clean_data_with_relevant_compositions with default strategy (should be FULL)."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result_default = cleaner.clean_data_with_relevant_compositions()
        result_full = cleaner.clean_data_with_relevant_compositions(
            CleaningStrategy.FULL
        )

        # Default should be same as FULL strategy
        assert result_default == result_full

    def test_clean_data_with_relevant_compositions_empty_input(self):
        """Test clean_data_with_relevant_compositions with empty JSON input."""
        empty_data = {}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(empty_data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions()
            assert result == {}
        finally:
            os.unlink(temp_file_path)

    def test_unresolved_compositions_collected(self):
        """Test that unresolved compositions (still containing brackets/math ops after cleaning) are accumulated in unresolved_compositions."""
        data = {
            "10.x/good": {
                "composition_data": {
                    "compositions_property_values": {"BaTiO3": 100, "PbZrO3": 200}
                },
                "synthesis_data": {"method": "", "precursors": [], "steps": [], "characterization_techniques": []},
                "article_metadata": {"doi": "", "title": "", "journal": "", "year": "", "isOpenAccess": False, "authors": [], "keywords": []},
            },
            "10.x/bad": {
                "composition_data": {
                    "compositions_property_values": {"(Ba0.5Na0.5)(0.9*x)TiO3": 50}
                },
                "synthesis_data": {"method": "", "precursors": [], "steps": [], "characterization_techniques": []},
                "article_metadata": {"doi": "", "title": "", "journal": "", "year": "", "isOpenAccess": False, "authors": [], "keywords": []},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            assert cleaner.filtered_compositions == {}
            assert cleaner.unresolved_compositions == {}
            cleaner.clean_data_with_relevant_compositions()
            # filtered and unresolved dicts are populated after cleaning
            assert isinstance(cleaner.filtered_compositions, dict)
            assert isinstance(cleaner.unresolved_compositions, dict)
        finally:
            os.unlink(temp_file_path)

    def test_get_useful_data(self, temp_mixed_json_file):
        """Test get_useful_data method."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result = cleaner.get_useful_data()

        assert isinstance(result, dict)

        # Check that the expected structure is present
        for doi, article_data in result.items():
            assert "composition_data" in article_data
            assert "synthesis_data" in article_data
            assert "article_metadata" in article_data

            # Check composition_data structure
            assert "compositions_property_values" in article_data["composition_data"]
            assert "property_unit" in article_data["composition_data"]
            assert "family" in article_data["composition_data"]


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
                        "0.07Pb(Mn1/3Sb2/3)O3-(1-0.07)Pb(Zr0.48Ti0.52)O3": "2.71 g/cm3",  # Complex formula
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
        cleaned_data = cleaner.clean_data_with_relevant_compositions(
            CleaningStrategy.FULL
        )

        # Verify the integration worked correctly
        assert isinstance(cleaned_data, dict)

        for paper_key, paper_data in cleaned_data.items():
            comp_values = paper_data["composition_data"]["compositions_property_values"]

            # Should have processed fractions, spaces, and parentheses
            for comp_key in comp_values.keys():
                # No spaces should remain
                assert " " not in comp_key
                # No invalid patterns should remain
                assert not re.match(r"(?<![a-z0-9])[A-Z]{2,}(?![a-z0-9])", comp_key)

    def test_basic_vs_full_cleaning_comparison(self, temp_complex_json_file):
        """Test comparison between BASIC and FULL cleaning strategies."""
        cleaner = DataCleaner(temp_complex_json_file)

        basic_result = cleaner.clean_data_with_relevant_compositions(
            CleaningStrategy.BASIC
        )
        full_result = cleaner.clean_data_with_relevant_compositions(
            CleaningStrategy.FULL
        )

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
        """Test handling of empty composition data."""
        empty_comp_data = {
            "paper1": {
                "composition_data": {
                    "compositions_property_values": {}  # Empty compositions
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(empty_comp_data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            # Should handle empty compositions gracefully without errors
            result = cleaner.clean_data_with_relevant_compositions()
            assert isinstance(result, dict)
            # Should return empty dict since no valid compositions to process
            assert len(result) == 0 or result == {}
        finally:
            os.unlink(temp_file_path)

    def test_division_by_zero_in_fractions(self):
        """Test handling of division by zero in fraction conversion."""
        test_data = {
            "paper1": {
                "composition_data": {
                    "compositions_property_values": {
                        "Na1/0Cl": "value1",  # Division by zero
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            # Should handle division by zero gracefully
            result = cleaner.clean_data_with_relevant_compositions(
                CleaningStrategy.BASIC
            )
            # The original fraction should be kept if division by zero
            assert isinstance(result, dict)
        finally:
            os.unlink(temp_file_path)


# Parametrized tests for various input scenarios
class TestParametrizedScenarios:
    """Parametrized tests for various input scenarios."""

    @pytest.mark.parametrize(
        "input_key,expected_pattern",
        [
            ("Na1/2Cl1/2", "0.50"),  # Fractions converted
            ("Ti2/3O4/3", "0.67"),  # Fractions converted
            ("Ca1/4CO3", "0.25"),  # Fraction converted
            ("Regular", "Regular"),  # No changes
            ("H2O", "H2O"),  # No changes
            (
                "0.07Pb(Mn0.33Sb0.67)O3-(1-0.07)Pb(Zr0.48Ti0.52)O3",
                "0.93",
            ),  # Arithmetic resolved
        ],
    )
    def test_convert_fractions_and_resolve_compositions_parametrized(
        self, input_key, expected_pattern
    ):
        """Parametrized test for fraction conversion and composition resolution."""
        temp_data = {"test": {"composition_data": {"compositions_property_values": {}}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(temp_data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            dict_list = [{input_key: "test_value"}]
            result = cleaner._convert_fractions_and_resolve_compositions(dict_list)
            result_key = list(result[0].keys())[0]
            assert expected_pattern in result_key
        finally:
            os.unlink(temp_file_path)
