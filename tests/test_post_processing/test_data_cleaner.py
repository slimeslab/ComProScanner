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
    CleaningStep,
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


class TestCleaningStep:
    """Test cases for the CleaningStep enum."""

    def test_cleaning_step_enum_values(self):
        """Test that CleaningStep enum has the expected seven values."""
        assert CleaningStep.ABBREVIATION_FILTERING == "abbreviation_filtering"
        assert CleaningStep.ELEMENT_VALIDATION_STRICT == "element_validation_strict"
        assert CleaningStep.ELEMENT_VALIDATION_LENIENT == "element_validation_lenient"
        assert CleaningStep.TEXT_NORMALIZATION == "text_normalization"
        assert CleaningStep.MILLER_INDICES == "miller_indices"
        assert CleaningStep.COEFFICIENT_EXPANSION_STRICT == "coefficient_expansion_strict"
        assert (
            CleaningStep.COEFFICIENT_EXPANSION_LENIENT == "coefficient_expansion_lenient"
        )

    def test_cleaning_step_all(self):
        """Test that CleaningStep.all() returns exactly the seven expected step names."""
        assert set(CleaningStep.all()) == {
            "abbreviation_filtering",
            "element_validation_strict",
            "element_validation_lenient",
            "text_normalization",
            "miller_indices",
            "coefficient_expansion_strict",
            "coefficient_expansion_lenient",
        }

    def test_cleaning_step_membership(self):
        """Test CleaningStep enum membership."""
        assert "element_validation_strict" in CleaningStep
        assert "element_validation_lenient" in CleaningStep
        assert "coefficient_expansion_lenient" in CleaningStep
        assert "miller_indices" in CleaningStep
        assert "invalid" not in CleaningStep
        assert "normalization" not in CleaningStep
        assert "zero_coefficient" not in CleaningStep


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

    def test_contains_element_token_finds_embedded_formula(self, data_cleaner):
        """Test _contains_element_token detects a formula fragment embedded in text."""
        pair = {
            "Cellulose nanofibers/BaTiO3@TiO2/Polyvinylidene fluoride-(%)": "value"
        }
        assert data_cleaner._contains_element_token(pair) is True

    def test_contains_element_token_pure_formula(self, data_cleaner):
        """Test _contains_element_token detects a composition that is already pure elements."""
        assert data_cleaner._contains_element_token({"BaTiO3": "value"}) is True

    def test_contains_element_token_no_element_anywhere(self, data_cleaner):
        """Test _contains_element_token returns False when no letter-run parses as elements."""
        assert data_cleaner._contains_element_token({"RandomTextNoElement": "value"}) is False

    def test_contains_element_token_empty_dict(self, data_cleaner):
        """Test _contains_element_token with an empty dict."""
        assert data_cleaner._contains_element_token({}) is False

    def test_has_balanced_annotated_brackets_true_for_text_in_parens(self, data_cleaner):
        """Test _has_balanced_annotated_brackets True for balanced brackets with text."""
        assert data_cleaner._has_balanced_annotated_brackets("Bi0.5Ag0.5ZrO3-(as-sintered)") is True

    def test_has_balanced_annotated_brackets_false_for_stray_asterisk(self, data_cleaner):
        """Test _has_balanced_annotated_brackets False when a stray '*' remains."""
        assert data_cleaner._has_balanced_annotated_brackets("0.03*(Bi0.5Ag0.5)ZrO3") is False

    def test_has_balanced_annotated_brackets_false_for_unbalanced_brackets(self, data_cleaner):
        """Test _has_balanced_annotated_brackets False when brackets are unmatched."""
        assert data_cleaner._has_balanced_annotated_brackets("BaTiO3-(unbalanced") is False

    def test_has_balanced_annotated_brackets_false_for_pure_arithmetic_content(self, data_cleaner):
        """Test _has_balanced_annotated_brackets False when bracket content is purely numeric/arithmetic."""
        assert data_cleaner._has_balanced_annotated_brackets("BaTiO3(0.04-0.03)") is False

    def test_normalize_text_title_cases_descriptive_words(self, data_cleaner):
        """Test _normalize_text title-cases descriptive word tokens and preserves spaces."""
        dict_list = [
            {"Bi4Ti3O12 ultrathin with oxygen vacancies": "value1"},
        ]
        result = data_cleaner._normalize_text(dict_list)
        assert result == [{"Bi4Ti3O12 Ultrathin with Oxygen Vacancies": "value1"}]

    def test_normalize_text_leaves_glued_words_untouched(self, data_cleaner):
        """Test _normalize_text does not insert spaces where none exist in the source."""
        dict_list = [{"K0.5Na0.5Nb0.9Ta0.1O3-Milling15h": "value1"}]
        result = data_cleaner._normalize_text(dict_list)
        assert result == [{"K0.5Na0.5Nb0.9Ta0.1O3-Milling15h": "value1"}]

    def test_normalize_text_preserves_all_caps_abbreviations(self, data_cleaner):
        """Test _normalize_text leaves all-caps abbreviation tokens (e.g. XRD) unchanged."""
        dict_list = [{"BaTiO3 XRD pattern": "value1"}]
        result = data_cleaner._normalize_text(dict_list)
        assert result == [{"BaTiO3 XRD Pattern": "value1"}]

    def test_normalize_text_digit_tokens_untouched(self, data_cleaner):
        """Test _normalize_text does not modify tokens containing digits."""
        dict_list = [{"Ti O2": "value1"}]
        result = data_cleaner._normalize_text(dict_list)
        # "Ti" is title-cased (already correct), "O2" contains a digit so is untouched
        assert result == [{"Ti O2": "value1"}]

    def test_normalize_text_collapses_multiple_internal_spaces(self, data_cleaner):
        """Test _normalize_text collapses runs of multiple spaces down to one."""
        dict_list = [{"Bi4Ti3O12   ultrathin  with oxygen vacancies": "value1"}]
        result = data_cleaner._normalize_text(dict_list)
        assert result == [{"Bi4Ti3O12 Ultrathin with Oxygen Vacancies": "value1"}]

    def test_normalize_text_strips_leading_and_trailing_whitespace(self, data_cleaner):
        """Test _normalize_text strips leading/trailing whitespace from composition keys."""
        dict_list = [{"  Bi4Ti3O12 ultrathin with oxygen vacancies  ": "value1"}]
        result = data_cleaner._normalize_text(dict_list)
        assert result == [{"Bi4Ti3O12 Ultrathin with Oxygen Vacancies": "value1"}]

    def test_normalize_text_does_not_capitalize_element_lookalike_units(
        self, data_cleaner
    ):
        """Regression test: capitalizing a short unit-abbreviation token like
        "h" (hours) would create "H" — a genuine periodic-table element
        (Hydrogen) — which a later coefficient-expansion pass could then
        scale as if it were real stoichiometry. Such tokens must be left in
        their original (lowercase) form instead of being title-cased."""
        dict_list = [{"BaTiO3 sintered for 20 h": "value1"}]
        result = data_cleaner._normalize_text(dict_list)
        assert result == [{"BaTiO3 Sintered for 20 h": "value1"}]

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

    def test_distribute_multiterm_brackets_user_example(self, data_cleaner):
        """Regression test: an outer coefficient multiplying a multi-term
        bracket (terms separated by +/- inside, each with its own inner
        coefficient) must be distributed correctly instead of shredding the
        bracket structure. 0.89 must multiply with (Bi0.5Na0.5)TiO3, 0.11
        with BaTiO3, both further scaled by the outer 0.75, and similarly
        for the second half."""
        formula = (
            "0.75*(0.89(Bi0.5Na0.5)TiO3-0.11BaTiO3) + "
            "0.25*(0.87(Bi0.5Na0.5)TiO3-0.11BaTiO3-0.02(Sm0.5K0.5)TiO3)"
        )
        result = data_cleaner._distribute_multiterm_brackets(formula)
        assert result == (
            "0.6675(Bi0.5Na0.5)TiO3-0.0825BaTiO3 + "
            "0.2175(Bi0.5Na0.5)TiO3-0.0275BaTiO3-0.005(Sm0.5K0.5)TiO3"
        )

    def test_distribute_multiterm_brackets_two_level_nesting(self, data_cleaner):
        """A doubly-nested multi-term bracket must resolve across multiple
        passes without explicit recursion."""
        result = data_cleaner._distribute_multiterm_brackets(
            "0.5*(0.5*(0.5A-0.5B)-0.5C)"
        )
        assert result == "0.125A-0.125B-0.25C"

    def test_distribute_multiterm_brackets_negated_top_level_term(
        self, data_cleaner
    ):
        """A multi-term bracket subtracted at the top level must have its
        internal signs flipped correctly (distributing the negation), not
        just have the outer coefficient applied blindly."""
        result = data_cleaner._distribute_multiterm_brackets("1-0.5*(0.3A-0.2B)")
        assert result == "1-0.15A+0.1B"

    def test_distribute_multiterm_brackets_leaves_single_term_bracket_untouched(
        self, data_cleaner
    ):
        """A bracket with no top-level +/- inside it (single term) is left
        for the existing coefficient_expansion pipeline to handle."""
        formula = "0.03*(Bi0.5Ag0.5)ZrO3"
        assert data_cleaner._distribute_multiterm_brackets(formula) == formula

    def test_distribute_multiterm_brackets_defers_pure_numeric_arithmetic(
        self, data_cleaner
    ):
        """A bracket containing only numbers/operators (e.g. "0.5*(0.2+0.3)")
        must be left for the existing arithmetic-evaluation machinery, not
        shredded into dangling additive numeric terms."""
        formula = "0.5*(0.2+0.3)"
        assert data_cleaner._distribute_multiterm_brackets(formula) == formula

    def test_formula_prefix_end_stops_at_trailing_annotation(self, data_cleaner):
        """A real formula followed by a space-separated descriptive
        annotation must have its boundary end right after the formula,
        even when the annotation contains a real single-letter element
        symbol (e.g. "C" for Celsius) that must never be scaled."""
        text = "PbTiO3 (calcined at 660C)"
        assert data_cleaner._formula_prefix_end(text) == len("PbTiO3")

    def test_formula_prefix_end_rejects_word_prefixed_by_real_element(
        self, data_cleaner
    ):
        """An apparent element match that's actually the start of a longer
        descriptive word (e.g. "Re" in "Reoxidized") must not be included
        in the valid prefix at all."""
        assert data_cleaner._formula_prefix_end("Reoxidized") == 0

    def test_formula_prefix_end_rejects_fake_element(self, data_cleaner):
        """A capitalized 1-2 letter run that isn't a real periodic-table
        symbol (e.g. "Bo" in "Bottom") must not be included in the prefix."""
        assert data_cleaner._formula_prefix_end("Bottom") == 0

    def test_formula_prefix_end_exempts_percent_placeholder(self, data_cleaner):
        """A genuine element immediately followed by the internal
        percent-annotation placeholder (itself lowercase) must still be
        included in the valid prefix, since the placeholder is not real
        corrupting text."""
        text = "Li0.5Bi0.5TiO3zzzpctannot0zzz"
        assert data_cleaner._formula_prefix_end(text) == len("Li0.5Bi0.5TiO3")

    def test_formula_prefix_end_recurses_into_valid_bracket(self, data_cleaner):
        """A legitimate nested formula bracket (e.g. site-occupancy
        notation) must be treated as part of the formula, not a break
        point, as long as its own content is fully valid."""
        text = "K0.48Na0.52NbO2.7SnO2"
        assert data_cleaner._formula_prefix_end(text) == len(text)
        text2 = "(Bi0.5Na0.5)TiO3"
        assert data_cleaner._formula_prefix_end(text2) == len(text2)

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

    def test_clean_data_without_element_validation_strict(self, temp_mixed_json_file):
        """Test clean_data_with_relevant_compositions without the element_validation_strict step."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result = cleaner.clean_data_with_relevant_compositions(
            cleaning_steps=["abbreviation_filtering"]
        )

        # Should keep papers with valid compositions after basic cleaning
        assert isinstance(result, dict)
        assert len(result) > 0

        # Check that invalid key patterns are removed but element validation is skipped
        for paper_key, paper_data in result.items():
            comp_values = paper_data["composition_data"]["compositions_property_values"]
            # Should not contain entries with all-caps invalid patterns
            for comp_key in comp_values.keys():
                assert not re.match(r"(?<![a-z0-9])[A-Z]{2,}(?![a-z0-9])", comp_key)

    def test_abbreviation_filtering_optional_keeps_invalid_keys_when_omitted(
        self, temp_mixed_json_file
    ):
        """Test that 2+-consecutive-capital-letter keys survive when abbreviation_filtering is omitted."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result = cleaner.clean_data_with_relevant_compositions(cleaning_steps=[])

        all_comp_keys = [
            comp_key
            for paper_data in result.values()
            for comp_key in paper_data["composition_data"][
                "compositions_property_values"
            ].keys()
        ]
        assert any(
            re.search(r"(?<![a-z0-9])[A-Z]{2,}(?![a-z0-9])", comp_key)
            for comp_key in all_comp_keys
        )

    def test_clean_data_with_relevant_compositions_with_element_validation_strict(
        self, temp_mixed_json_file
    ):
        """Test clean_data_with_relevant_compositions with element_validation_strict (cleaning_steps='all')."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result = cleaner.clean_data_with_relevant_compositions(cleaning_steps="all")

        assert isinstance(result, dict)

        # With element validation, should only keep papers with valid chemical elements
        for paper_key, paper_data in result.items():
            comp_values = paper_data["composition_data"]["compositions_property_values"]
            for comp_key in comp_values.keys():
                # Create a new instance to test _is_elements
                test_dict = {comp_key: comp_values[comp_key]}
                # Should only contain valid elements after full cleaning
                assert cleaner._is_elements(test_dict) is True

    def test_clean_data_with_relevant_compositions_without_element_validation_strict(
        self, temp_mixed_json_file
    ):
        """Test clean_data_with_relevant_compositions without element_validation_strict selected."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result = cleaner.clean_data_with_relevant_compositions(
            cleaning_steps=[
                "abbreviation_filtering",
                "miller_indices",
                "coefficient_expansion_strict",
            ]
        )

        assert isinstance(result, dict)

        # Without element validation, should keep compositions even with invalid elements
        # but still filter out invalid key patterns
        for paper_key, paper_data in result.items():
            comp_values = paper_data["composition_data"]["compositions_property_values"]
            for comp_key in comp_values.keys():
                # Should not contain all-caps invalid patterns
                assert not re.match(r"(?<![a-z0-9])[A-Z]{2,}(?![a-z0-9])", comp_key)

    def test_clean_data_with_relevant_compositions_default_is_all(
        self, temp_mixed_json_file
    ):
        """Test clean_data_with_relevant_compositions default equals cleaning_steps='all'."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result_default = cleaner.clean_data_with_relevant_compositions()
        result_all = cleaner.clean_data_with_relevant_compositions(cleaning_steps="all")

        # Default should be same as "all"
        assert result_default == result_all

    def test_clean_data_with_relevant_compositions_rejects_unknown_step(
        self, temp_mixed_json_file
    ):
        """Test clean_data_with_relevant_compositions raises ValueError for unknown step names."""
        cleaner = DataCleaner(temp_mixed_json_file)
        with pytest.raises(ValueError):
            cleaner.clean_data_with_relevant_compositions(cleaning_steps=["bogus_step"])

    def test_empty_cleaning_steps_still_runs_mandatory_operations(
        self, temp_mixed_json_file
    ):
        """Test that unicode conversion and arithmetic resolution still run when cleaning_steps=[]."""
        cleaner = DataCleaner(temp_mixed_json_file)
        result = cleaner.clean_data_with_relevant_compositions(cleaning_steps=[])
        # No optional steps selected — invalid-pattern keys survive (abbreviation_filtering skipped)
        all_comp_keys = [
            comp_key
            for paper_data in result.values()
            for comp_key in paper_data["composition_data"][
                "compositions_property_values"
            ].keys()
        ]
        assert len(all_comp_keys) > 0

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
                "synthesis_data": {
                    "method": "",
                    "precursors": [],
                    "steps": [],
                    "characterization_techniques": [],
                },
                "article_metadata": {
                    "doi": "",
                    "title": "",
                    "journal": "",
                    "year": "",
                    "isOpenAccess": False,
                    "authors": [],
                    "keywords": [],
                },
            },
            "10.x/bad": {
                "composition_data": {
                    "compositions_property_values": {"(Ba0.5Na0.5)(0.9*x)TiO3": 50}
                },
                "synthesis_data": {
                    "method": "",
                    "precursors": [],
                    "steps": [],
                    "characterization_techniques": [],
                },
                "article_metadata": {
                    "doi": "",
                    "title": "",
                    "journal": "",
                    "year": "",
                    "isOpenAccess": False,
                    "authors": [],
                    "keywords": [],
                },
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

        # Clean with all steps enabled
        cleaned_data = cleaner.clean_data_with_relevant_compositions(
            cleaning_steps="all"
        )

        # Verify the integration worked correctly
        assert isinstance(cleaned_data, dict)

        for paper_key, paper_data in cleaned_data.items():
            comp_values = paper_data["composition_data"]["compositions_property_values"]

            # Should have processed fractions and parentheses, and dropped invalid patterns
            for comp_key in comp_values.keys():
                # Fractions should be resolved (no bare "/" left)
                assert "/" not in comp_key
                # No invalid patterns should remain
                assert not re.match(r"(?<![a-z0-9])[A-Z]{2,}(?![a-z0-9])", comp_key)

    def test_basic_vs_full_cleaning_comparison(self, temp_complex_json_file):
        """Test comparison between cleaning with and without element_validation_strict."""
        cleaner = DataCleaner(temp_complex_json_file)

        basic_result = cleaner.clean_data_with_relevant_compositions(
            cleaning_steps=[
                "abbreviation_filtering",
                "miller_indices",
                "coefficient_expansion_strict",
            ]
        )
        full_result = cleaner.clean_data_with_relevant_compositions(
            cleaning_steps="all"
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

    def test_miller_indices_selected_drops_entries_entirely(self):
        """Regression test: when miller_indices IS selected, compositions carrying a
        crystal-plane notation must be dropped entirely, not resolved to the bare
        formula. Stripping the notation and keeping "AlN" would collapse distinct
        surface-orientation entries for the same material into the same dict key
        (e.g. "AlN (002)" and "AlN (110)" both -> "AlN"), silently overwriting one
        value with the other when merged."""
        data = {
            "10.x/miller": {
                "composition_data": {
                    "compositions_property_values": {
                        "AlN (002)": 1,
                        "AlN (110)": 2,
                        "ZnO (101)": 3,
                        "BaTiO3": 4,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(cleaning_steps="all")
            comp_values = result["10.x/miller"]["composition_data"][
                "compositions_property_values"
            ]
            # Only the composition without a Miller index survives
            assert comp_values == {"BaTiO3": 4}
            filtered_keys = {
                entry["composition"]
                for entry in cleaner.filtered_compositions.get("10.x/miller", [])
            }
            assert filtered_keys == {"AlN (002)", "AlN (110)", "ZnO (101)"}
        finally:
            os.unlink(temp_file_path)

    def test_miller_index_shaped_bracket_flagged_unresolved_when_not_selected(self):
        """Regression test: when miller_indices is NOT selected, a Miller-index-shaped
        bracket like "(002)" must NOT be silently merged into the preceding element by
        coefficient_expansion_strict's "remove brackets without coefficients" step (which would
        otherwise turn AlN (002) into AlN 002/AlN2). Instead it should be left untouched
        and dropped as unresolved, so a stray bracket is never silently misinterpreted.
        """
        data = {
            "10.x/miller2": {
                "composition_data": {"compositions_property_values": {"AlN (002)": 1}},
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(
                cleaning_steps=["coefficient_expansion_strict"]
            )
            comp_values = (
                result.get("10.x/miller2", {})
                .get("composition_data", {})
                .get("compositions_property_values", {})
            )
            # Must not silently resolve to a wrong formula like "AlN2" or "AlN 002"
            assert comp_values == {}
            assert cleaner.unresolved_compositions.get("10.x/miller2") == [
                {"composition": "AlN (002)", "reason": "unresolved_brackets_or_operators"}
            ]
        finally:
            os.unlink(temp_file_path)

    def test_coefficient_expansion_lenient_keeps_annotated_brackets(self):
        """coefficient_expansion_lenient should expand coefficients like coefficient_expansion_strict,
        but keep compositions with balanced brackets containing genuine text instead of
        dropping them as unresolved."""
        data = {
            "10.x/lenient": {
                "composition_data": {
                    "compositions_property_values": {
                        "(Bi0.5Ag0.5)ZrO3-(as-sintered)": 1,
                        "0.7(K0.48Na0.52NbO3)": 2,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(
                cleaning_steps=["coefficient_expansion_lenient"]
            )
            comp_values = result["10.x/lenient"]["composition_data"][
                "compositions_property_values"
            ]
            assert comp_values == {
                "Bi0.5Ag0.5ZrO3-(as-sintered)": 1,
                "K0.336Na0.364Nb0.7O2.1": 2,
            }
            assert cleaner.unresolved_compositions == {}
        finally:
            os.unlink(temp_file_path)

    def test_coefficient_expansion_does_not_mangle_non_element_words(self):
        """Regression test: a descriptive word/annotation trailing a formula
        term (e.g. "(001) bottom", left attached because miller_indices
        wasn't selected) must not have fragments of it misread as element
        symbols and scaled — e.g. "Bottom" must not become "Bo0.31ttom" just
        because "Bo" happens to match the [A-Z][a-z]? pattern; "Bo" isn't a
        real periodic-table symbol. Since miller_indices isn't selected here,
        the leftover Miller-index-shaped bracket "(001)" still makes the
        composition land in unresolved_compositions (established, documented
        behavior) — but its recorded text must show the formula correctly
        scaled and the annotation completely untouched, not corrupted."""
        data = {
            "10.x/nonelement": {
                "composition_data": {
                    "compositions_property_values": {
                        "0.25Pb(In1/2Nb1/2)O3-0.44Pb(Mg1/3Nb2/3)O3-0.31PbTiO3 (001) bottom": 1,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(
                cleaning_steps=["text_normalization", "coefficient_expansion_lenient"]
            )
            comp_values = result["10.x/nonelement"]["composition_data"][
                "compositions_property_values"
            ]
            assert comp_values == {}
            (unresolved_entry,) = cleaner.unresolved_compositions["10.x/nonelement"]
            resolved_key = unresolved_entry["composition"]
            assert "Bo0.31ttom" not in resolved_key
            assert "Bottom" in resolved_key
            assert "Pb0.31Ti0.31O0.93" in resolved_key
        finally:
            os.unlink(temp_file_path)

    def test_coefficient_expansion_does_not_mangle_real_element_prefixed_word(self):
        """Regression test: unlike "Bo" in "Bottom" ("Bo" isn't a real element so the
        all_elements check alone rejects it), "Re" in "Reoxidized" IS a real periodic-table
        symbol (Rhenium), so it must instead be rejected by the "not immediately followed by
        more lowercase letters" check. Without that check, "reoxidized" (title-cased to
        "Reoxidized" by text_normalization) sitting near a coefficient like "10^-10" would be
        corrupted into "Re10oxidized" instead of staying intact."""
        data = {
            "10.x/reoxidized": {
                "composition_data": {
                    "compositions_property_values": {
                        "0.1(K0.5Na0.5)NbO3 reoxidized at 850C": 1,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(
                cleaning_steps=["text_normalization", "coefficient_expansion_lenient"]
            )
            comp_values = result["10.x/reoxidized"]["composition_data"][
                "compositions_property_values"
            ]
            (resolved_key,) = comp_values.keys()
            assert "Re10oxidized" not in resolved_key
            assert "Reoxidized" in resolved_key
        finally:
            os.unlink(temp_file_path)

    def test_percent_annotation_preceded_by_real_element_still_scales(self):
        """Regression test: the placeholder used to protect percent-dopant annotations
        (e.g. "0.1%MgO") from being misread as coefficients is itself lowercase-letters-only
        ("zzzpctannotNzzz"), which could false-trigger the "reject match followed by more
        lowercase letters" anti-corruption heuristic (added for the "Reoxidized" bug) on a
        genuine element sitting immediately before the placeholder. A real element like the
        "O3" in "...TiO3:0.1%MgO" must still be scaled by its outer coefficient."""
        data = {
            "10.x/pctelement": {
                "composition_data": {
                    "compositions_property_values": {
                        "0.1Li0.5Bi0.5TiO3:0.1%MgO": 1,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(cleaning_steps="all")
            comp_values = result["10.x/pctelement"]["composition_data"][
                "compositions_property_values"
            ]
            assert comp_values == {
                "Li0.05Bi0.05Ti0.1O0.3:0.1%MgO": 1,
            }
        finally:
            os.unlink(temp_file_path)

    def test_percent_annotation_with_bracketed_target_not_distributed(self):
        """Regression test: a weight-percent dopant annotation whose target is
        a bracketed multi-term expression (e.g. "1.25 wt% (0.78PbO-0.22CuO)")
        must be protected in full, not just the leading number — otherwise the
        1.25 is treated as a genuine stoichiometric coefficient and
        distributed across the bracket by coefficient expansion."""
        composition = (
            "0.645Pb(Zr0.59Ti0.41)O3-0.355Pb(Ni1/3Nb2/3)O3 + "
            "1.25 wt% (0.78PbO-0.22CuO) sintered at 1000C"
        )
        data = {
            "10.x/wtbracket": {
                "composition_data": {
                    "compositions_property_values": {
                        composition: 1,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(
                cleaning_steps=["text_normalization", "coefficient_expansion_lenient"]
            )
            comp_values = result["10.x/wtbracket"]["composition_data"][
                "compositions_property_values"
            ]
            assert comp_values == {
                "Pb0.645Zr0.38055Ti0.26445O1.935-Pb0.355Ni0.11715Nb0.23785O1.065"
                "+1.25 wt% (0.78PbO-0.22CuO) Sintered at 1000C": 1,
            }
        finally:
            os.unlink(temp_file_path)

    def test_unit_abbreviation_not_scaled_as_element(self):
        """Regression test: a duration unit like "20 h" must not be
        title-cased to "20 H" and then have the disguised element "H"
        (Hydrogen) scaled by a nearby coefficient."""
        composition = "0.5Ba(Zr0.2Ti0.8)O3-0.5(Ba0.7Ca0.3)TiO3 sintered for 20 h"
        data = {
            "10.x/hourunit": {
                "composition_data": {
                    "compositions_property_values": {
                        composition: 1,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(
                cleaning_steps=["text_normalization", "coefficient_expansion_lenient"]
            )
            comp_values = result["10.x/hourunit"]["composition_data"][
                "compositions_property_values"
            ]
            (resolved_key,) = comp_values.keys()
            assert "20 H" not in resolved_key
            assert "20 h" in resolved_key
        finally:
            os.unlink(temp_file_path)

    def test_annotation_with_real_element_symbol_not_scaled(self):
        """Regression test: a trailing descriptive annotation containing a
        real single-letter element symbol immediately after a number (e.g.
        "C" for Celsius in "660C") must never be scaled by a coefficient
        meant for the preceding formula — "0.64PbTiO3 (calcined at 660C)"
        must not become "[Pb0.64Ti0.64O1.92 (calcined at 660C0.64)]"."""
        composition = "0.36BiScO3-0.64PbTiO3 (calcined at 660C)"
        data = {
            "10.x/calcined": {
                "composition_data": {
                    "compositions_property_values": {
                        composition: 1,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(
                cleaning_steps=["text_normalization", "coefficient_expansion_lenient"]
            )
            comp_values = result["10.x/calcined"]["composition_data"][
                "compositions_property_values"
            ]
            assert comp_values == {
                "Bi0.36Sc0.36O1.08-Pb0.64Ti0.64O1.92 (calcined at 660C)": 1,
            }
            assert cleaner.unresolved_compositions == {}
        finally:
            os.unlink(temp_file_path)

    def test_nested_multiterm_coefficient_distribution(self):
        """Regression test: an outer coefficient multiplying a multi-term
        bracket (e.g. "0.75*(0.89(Bi0.5Na0.5)TiO3-0.11BaTiO3)") must be fully
        distributed and expanded to elements, not corrupted into mismatched
        brackets with un-distributed coefficients sitting in front of them."""
        composition = (
            "0.75*(0.89(Bi0.5Na0.5)TiO3-0.11BaTiO3) + "
            "0.25*(0.87(Bi0.5Na0.5)TiO3-0.11BaTiO3-0.02(Sm0.5K0.5)TiO3)"
        )
        data = {
            "10.x/nested": {
                "composition_data": {
                    "compositions_property_values": {
                        composition: 1,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(cleaning_steps="all")
            comp_values = result["10.x/nested"]["composition_data"][
                "compositions_property_values"
            ]
            assert comp_values == {
                "Bi0.33375Na0.33375Ti0.6675O2.0025-Ba0.0825Ti0.0825O0.2475"
                "+Bi0.10875Na0.10875Ti0.2175O0.6525-Ba0.0275Ti0.0275O0.0825"
                "-Sm0.0025K0.0025Ti0.005O0.015": 1,
            }
            assert cleaner.unresolved_compositions == {}
        finally:
            os.unlink(temp_file_path)

    def test_comma_separated_site_notation_not_corrupted(self):
        """Comma-separated site-occupancy notation (e.g. "(K,Na,Li)(Nb,Ta)O3")
        is not specially parsed and may legitimately end up unresolved, but it
        must never be actively corrupted into mismatched/stray brackets."""
        composition = (
            "(K,Na,Li)(Nb,Ta)O3 (sintered at 1000C, pO2=10^-10 atm, "
            "reoxidized at 850C)"
        )
        data = {
            "10.x/comma": {
                "composition_data": {
                    "compositions_property_values": {
                        composition: 1,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(
                cleaning_steps=["text_normalization", "coefficient_expansion_lenient"]
            )
            comp_values = result["10.x/comma"]["composition_data"][
                "compositions_property_values"
            ]
            for resolved_key in list(comp_values.keys()) + [
                entry["composition"]
                for entries in cleaner.unresolved_compositions.values()
                for entry in entries
            ]:
                assert resolved_key.count("(") == resolved_key.count(")")
                assert resolved_key.count("[") == resolved_key.count("]")
                assert "[" not in resolved_key
        finally:
            os.unlink(temp_file_path)

    def test_coefficient_expansion_and_lenient_together_reverts_to_strict(self):
        """Selecting coefficient_expansion_strict alongside coefficient_expansion_lenient should
        revert to strict behavior: annotated brackets are dropped as unresolved, same as
        coefficient_expansion_strict alone."""
        data = {
            "10.x/strict": {
                "composition_data": {
                    "compositions_property_values": {
                        "(Bi0.5Ag0.5)ZrO3-(as-sintered)": 1,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(
                cleaning_steps=["coefficient_expansion_strict", "coefficient_expansion_lenient"]
            )
            comp_values = result["10.x/strict"]["composition_data"][
                "compositions_property_values"
            ]
            assert comp_values == {}
            assert cleaner.unresolved_compositions.get("10.x/strict") == [
                {
                    "composition": "Bi0.5Ag0.5ZrO3-(as-sintered)",
                    "reason": "unresolved_brackets_or_operators",
                }
            ]
        finally:
            os.unlink(temp_file_path)

    def test_element_validation_lenient_keeps_text_with_embedded_formula(self):
        """element_validation_lenient should keep compositions containing at least one
        embedded formula fragment, and drop compositions with no element anywhere."""
        data = {
            "10.x/elem_lenient": {
                "composition_data": {
                    "compositions_property_values": {
                        "Cellulose nanofibers/BaTiO3@TiO2/Polyvinylidene fluoride-(%)": 1,
                        "RandomTextNoElement": 2,
                        "BaTiO3": 3,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(
                cleaning_steps=["element_validation_lenient", "coefficient_expansion_lenient"]
            )
            comp_values = result["10.x/elem_lenient"]["composition_data"][
                "compositions_property_values"
            ]
            assert comp_values == {
                "Cellulose nanofibers/BaTiO3@TiO2/Polyvinylidene fluoride-(%)": 1,
                "BaTiO3": 3,
            }
            filtered_keys = {
                entry["composition"]
                for entry in cleaner.filtered_compositions.get("10.x/elem_lenient", [])
            }
            assert filtered_keys == {"RandomTextNoElement"}
        finally:
            os.unlink(temp_file_path)

    def test_element_validation_and_lenient_together_reverts_to_strict(self):
        """Selecting element_validation_strict alongside element_validation_lenient should revert
        to strict behavior: text+formula mixtures are dropped, only pure-element
        compositions survive."""
        data = {
            "10.x/elem_strict": {
                "composition_data": {
                    "compositions_property_values": {
                        "Cellulose nanofibers/BaTiO3@TiO2/Polyvinylidene fluoride-(%)": 1,
                        "BaTiO3": 2,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(
                cleaning_steps=["element_validation_strict", "element_validation_lenient"]
            )
            comp_values = result["10.x/elem_strict"]["composition_data"][
                "compositions_property_values"
            ]
            assert comp_values == {"BaTiO3": 2}
        finally:
            os.unlink(temp_file_path)

    def test_no_coefficient_expansion_selected_passes_composition_through_unfiltered(self):
        """Regression test: when neither coefficient_expansion_strict nor
        coefficient_expansion_lenient is selected, a composition should never be
        dropped as "unresolved" just because the mandatory arithmetic step added
        brackets it didn't ask to have expanded. It should pass through as-is."""
        data = {
            "10.x/no_expansion": {
                "composition_data": {
                    "compositions_property_values": {
                        "0.7K0.48Na0.52NbO3": 1,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(cleaning_steps=[])
            comp_values = result["10.x/no_expansion"]["composition_data"][
                "compositions_property_values"
            ]
            # Not dropped, and not expanded either — passes through with the
            # mandatory bracket standardization only.
            assert comp_values == {"0.7(K0.48Na0.52NbO3)": 1}
            assert cleaner.unresolved_compositions == {}
        finally:
            os.unlink(temp_file_path)

    def test_percent_dopant_annotation_not_treated_as_coefficient(self):
        """Regression test: a weight/mole-percent dopant annotation like "7 wt% NiO"
        or "0.1%MgO" must not have its number distributed as a stoichiometric
        coefficient across the following formula (e.g. NiO -> Ni7O7)."""
        data = {
            "10.x/percent": {
                "composition_data": {
                    "compositions_property_values": {
                        "PVDF + 7 wt% NiO + 0.1 wt% ZnO": 1,
                        "0.845Na0.5Bi0.5TiO3-0.055BaTiO3-0.1Li0.5Bi0.5TiO3:0.1%MgO": 2,
                    }
                },
                "synthesis_data": {},
                "article_metadata": {},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_file_path = f.name

        try:
            cleaner = DataCleaner(temp_file_path)
            result = cleaner.clean_data_with_relevant_compositions(cleaning_steps="all")
            comp_values = result["10.x/percent"]["composition_data"][
                "compositions_property_values"
            ]
            assert comp_values == {
                "Na0.4225Bi0.4225Ti0.845O2.535-Ba0.055Ti0.055O0.165-Li0.05Bi0.05Ti0.1O0.3:0.1%MgO": 2,
            }
        finally:
            os.unlink(temp_file_path)


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
                cleaning_steps=[
                    "abbreviation_filtering",
                    "miller_indices",
                    "coefficient_expansion_strict",
                ]
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
