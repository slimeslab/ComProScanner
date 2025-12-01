"""
data_cleaner.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 08-04-2025
"""

# Standard library imports
import json
from typing import List, Dict, Any
import re
from enum import Enum
import copy

# Third-party imports
from pymatgen.core.periodic_table import Element


@staticmethod
def get_all_elements() -> List[str]:
    """Get list of all element symbols."""
    return [Element.from_Z(i).symbol for i in range(1, 119)]


class CleaningStrategy(str, Enum):
    """Cleaning strategies for data cleaning."""

    BASIC = "basic"  # Without element validation
    FULL = "full"  # With element validation


class DataCleaner:
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.all_data = self._load_results()
        self.all_elements = get_all_elements()

    def _load_results(self) -> Dict[str, Any]:
        """Load results from JSON file."""
        with open(self.results_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_comp_prop_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract composition property data from all results."""
        return extracted_data["composition_data"]["compositions_property_values"]

    def _get_comp_prop_pairs(self, comp_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all composition-property pairs."""
        return [{comp: prop} for comp, prop in comp_data.items()]

    def _filter_invalid_keys(self, dict_list):
        """Filter dictionaries with invalid keys (more than 2 consecutive capital letters)."""
        pattern = r"(?<![a-z0-9])[A-Z]{2,}(?![a-z0-9])"

        return [
            d for d in dict_list if not any(re.search(pattern, key) for key in d.keys())
        ]

    def _is_elements(self, comp_pro_pair: Dict[str, Any]) -> bool:
        def _remove_special_chars(string: str) -> str:
            return re.sub(r"[^a-zA-Z]+", "", string)

        try:
            key = next(iter(comp_pro_pair))  # Get the key
            key = _remove_special_chars(str(key))
            key_set = set(re.findall(r"[A-Z][^A-Z]*", key))
            for element in key_set:
                if element not in self.all_elements:
                    return False
            return True
        except Exception:
            return False

    def _remove_extra_spaces(self, dict_list):
        # remove any spaces in the key
        return [
            {key.replace(" ", ""): value for key, value in d.items()} for d in dict_list
        ]

    def _clean_comp_prop_data_with_element_check(
        self, comp_prop_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Clean composition-property data with element validation from periodic table."""
        comp_prop_data = self._get_comp_prop_pairs(comp_prop_data)
        comp_prop_data = self._filter_invalid_keys(comp_prop_data)
        valid_comp_prop_pairs = []
        for single_data in comp_prop_data:
            if self._is_elements(single_data):
                valid_comp_prop_pairs.append(single_data)
        valid_comp_prop_pairs = self._remove_extra_spaces(valid_comp_prop_pairs)
        valid_comp_prop_pairs = self._convert_fractions_and_resolve_compositions(
            valid_comp_prop_pairs
        )
        return valid_comp_prop_pairs

    def _clean_comp_prop_data_without_element_check(
        self, comp_prop_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Clean composition-property data without element validation."""
        comp_prop_data = self._get_comp_prop_pairs(comp_prop_data)
        comp_prop_data = self._filter_invalid_keys(comp_prop_data)
        valid_comp_prop_pairs = comp_prop_data
        valid_comp_prop_pairs = self._remove_extra_spaces(valid_comp_prop_pairs)
        valid_comp_prop_pairs = self._convert_fractions_and_resolve_compositions(
            valid_comp_prop_pairs
        )
        return valid_comp_prop_pairs

    def _convert_fractions_and_resolve_compositions(self, dict_list):
        """
        Convert fractions to decimal format and resolve composition formulas by ahndling mathematical operations and normalizing bracket notation.
        """

        def _replace_fraction(match):
            """Convert fraction to decimal format."""
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if denominator == 0:
                return match.group(0)
            return f"{numerator/denominator:.2f}"

        def _evaluate_all_parenthetical_expressions(formula):
            """
            - Evaluate ALL arithmetic operations within parentheses following BODMAS rules.
            - Process from innermost to outermost parentheses.
            """
            max_iterations = 100
            iteration_count = 0
            changed = True

            while changed and iteration_count < max_iterations:
                changed = False

                # Find ALL parentheses in the formula
                matches = list(re.finditer(r"\(([^()]+)\)", formula))

                # Process each match, but only evaluate arithmetic ones
                for match in matches:
                    expression = match.group(1).strip()

                    # Check if it's a purely arithmetic expression (only numbers and operators)
                    if re.match(r"^[0-9.\s+\-*/]+$", expression) and any(
                        op in expression for op in ["+", "-", "*", "/"]
                    ):
                        try:
                            # Evaluate the expression using BODMAS rule and format result
                            result = eval(expression)
                            if isinstance(result, float):
                                if result.is_integer():
                                    evaluated_value = str(int(result))
                                else:
                                    evaluated_value = str(round(result, 4))
                            else:
                                evaluated_value = str(result)

                            # Replace in formula (remove the parentheses entirely)
                            formula = (
                                formula[: match.start()]
                                + evaluated_value
                                + formula[match.end() :]
                            )
                            changed = True
                            break  # Start over after making a change
                        except Exception:
                            # If evaluation fails, skip this match
                            continue

                iteration_count += 1

            return formula

        def _multiply_pure_number_coefficients(formula):
            """
            Handle patterns like: 0.03*(0.2) -> 0.006
            Only when the content in parentheses is a pure number.
            """
            max_iterations = 50
            iteration_count = 0

            while iteration_count < max_iterations:
                # Match: digit*(digit)
                match = re.search(r"(\d+\.?\d*)\s*\*\s*\(([0-9.]+)\)", formula)

                if match:
                    coeff1 = float(match.group(1))
                    coeff2 = float(match.group(2))

                    result = coeff1 * coeff2

                    # Format result
                    if result == int(result):
                        result_str = str(int(result))
                    else:
                        result_str = str(round(result, 4))

                    # Replace the pattern
                    formula = (
                        formula[: match.start()] + result_str + formula[match.end() :]
                    )
                else:
                    break

                iteration_count += 1

            return formula

        def _resolve_coefficient_multiplication(formula):
            """
            Resolve patterns like:
            - (0.04-0.03)*CaZrO3 -> 0.01*CaZrO3 -> 0.01CaZrO3
            - 0.03*(Bi0.5Ag0.5)ZrO3 -> 0.03(Bi0.5Ag0.5)ZrO3
            - (0.03)*(0.2)ZrO3 -> 0.006*ZrO3 -> 0.006ZrO3

            Note: By the time this runs, pure arithmetic parentheses should already be evaluated.
            """
            max_iterations = 50
            iteration_count = 0

            while iteration_count < max_iterations:
                match = re.search(r"\(([0-9.\s+\-*/]+)\)\s*\*\s*", formula)

                if match:
                    expression = match.group(1).strip()
                    try:
                        # Evaluate the arithmetic expression
                        result = eval(expression)

                        # Format result
                        if isinstance(result, float):
                            if result.is_integer():
                                evaluated_value = str(int(result))
                            else:
                                evaluated_value = str(round(result, 4))
                        else:
                            evaluated_value = str(result)

                        # Replace (expression)* with the evaluated value followed by *
                        formula = (
                            formula[: match.start()]
                            + evaluated_value
                            + "*"
                            + formula[match.end() :]
                        )
                    except Exception:
                        iteration_count += 1
                        continue
                else:
                    break

                iteration_count += 1

            # Now handle patterns like: coefficient*(pure_number) -> multiply them
            formula = _multiply_pure_number_coefficients(formula)

            return formula

        def _remove_redundant_multiply_signs(formula):
            """
            Remove * signs between:
            - digit*letter: 2*Ca -> 2Ca
            - digit*(non-pure-number): 0.03*(Bi0.5Ag0.5) -> 0.03(Bi0.5Ag0.5)
            """
            # Remove * between digit and letter
            formula = re.sub(r"(\d)\s*\*\s*([A-Z])", r"\1\2", formula)

            # Remove * between digit and opening parenthesis (if parenthesis contains letters)
            def _replace_multiply_before_paren(match):
                digit = match.group(1)
                paren_content = match.group(2)

                # If parenthesis contains only digits and operators, keep the *
                if re.match(r"^[0-9.\s+\-*/]+$", paren_content):
                    return match.group(0)  # Keep original

                # Otherwise remove the *
                return digit + "(" + paren_content

            formula = re.sub(
                r"(\d+\.?\d*)\s*\*\s*\(([^)]+)\)",
                _replace_multiply_before_paren,
                formula,
            )

            return formula

        def _resolve_arithmetic_and_multiply(formula):
            """
            Resolve arithmetic expressions and handle multiplication operations.
            This must be done BEFORE adding composition brackets.
            """
            if not formula or not isinstance(formula, str):
                return str(formula) if formula is not None else ""

            # Step 1: Evaluate ALL parenthetical expressions with arithmetic first
            formula = _evaluate_all_parenthetical_expressions(formula)

            # Step 2: Handle patterns like (arithmetic)*composition or digit*(composition)
            formula = _resolve_coefficient_multiplication(formula)

            # Step 3: Remove * between digit and letter/parenthesis (if parenthesis contains non-digits)
            formula = _remove_redundant_multiply_signs(formula)

            return formula

        def _add_composition_brackets(formula):
            """
            Add brackets around composition parts after numerical coefficients.
            Uses () if no parentheses exist in the composition part.
            Uses [] if parentheses already exist in the composition part.

            Rules:
            - Only add brackets if there's a digit coefficient before the composition
            - Don't add brackets at the very beginning if no coefficient
            - Don't add brackets after - if no coefficient follows the -
            """
            # Split formula by +/- operators while preserving them
            parts = re.split(r"(?=[+\-])", formula)

            processed_parts = []
            for _, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue

                # Check if part starts with a sign
                sign = ""
                if part.startswith("-") or part.startswith("+"):
                    sign = part[0]
                    part = part[1:].strip()

                if not part:
                    continue

                # Match coefficient at the beginning (number with optional decimal)
                coeff_match = re.match(r"^(\d+\.?\d*)", part)

                if coeff_match:
                    coefficient = coeff_match.group(1)
                    composition_part = part[len(coefficient) :].strip()

                    if not composition_part:
                        # Only coefficient, no composition part
                        processed_parts.append(sign + coefficient)
                        continue

                    # Check if composition part already has proper brackets at the outermost level
                    if (
                        composition_part.startswith("(")
                        and composition_part.endswith(")")
                    ) or (
                        composition_part.startswith("[")
                        and composition_part.endswith("]")
                    ):
                        # Already properly bracketed
                        processed_parts.append(sign + coefficient + composition_part)
                    else:
                        # Determine bracket type based on whether parentheses exist in composition
                        if "(" in composition_part or ")" in composition_part:
                            # Use square brackets
                            processed_parts.append(
                                sign + coefficient + "[" + composition_part + "]"
                            )
                        else:
                            # Use round brackets
                            processed_parts.append(
                                sign + coefficient + "(" + composition_part + ")"
                            )
                else:
                    # No coefficient found at the beginning
                    # Don't add brackets - keep as is
                    processed_parts.append(sign + part)

            return "".join(processed_parts)

        def _resolve_composition(formula):
            """
            Process chemical formulas with the following operations:
            1. Evaluate arithmetic operations and multiplications FIRST
            2. Add brackets around composition parts after coefficients
            """
            if not formula or not isinstance(formula, str):
                return str(formula) if formula is not None else ""

            # Step 1: Resolve arithmetic and multiplication COMPLETELY FIRST
            formula = _resolve_arithmetic_and_multiply(formula)

            # Step 2: Add brackets around composition parts
            formula = _add_composition_brackets(formula)

            return formula

        # Main processing logic
        result = []
        for d in dict_list:
            new_dict = {}
            for key, value in d.items():
                # Step 1: Convert simple fractions to decimals
                processed_key = re.sub(r"(\d+)/(\d+)", _replace_fraction, key)

                # Step 2: Resolve compositions
                processed_key = _resolve_composition(processed_key)

                new_dict[processed_key] = value
            result.append(new_dict)
        return result

    def _return_in_dict(self, dict_list):
        final_dict = {}
        for d in dict_list:
            final_dict.update(d)
        return final_dict

    def get_useful_data(self) -> Dict[str, Any]:
        """Get only the useful information from all the data passed by the extraction agents based on key searching."""
        result = {}

        # Define the expected structure with default empty values
        expected_structure = {
            "composition_data": {
                "compositions_property_values": {},
                "property_unit": "",
                "family": "",
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
        }

        for doi, article_data in self.all_data.items():
            cleaned_article = {}

            for main_key, sub_keys_defaults in expected_structure.items():
                cleaned_section = {}

                for sub_key, default_value in sub_keys_defaults.items():
                    if main_key in article_data and sub_key in article_data[main_key]:
                        value = article_data[main_key][sub_key]

                        # If the value is a dict with 'default' key, extract only the default value
                        if isinstance(value, dict) and "default" in value:
                            cleaned_section[sub_key] = value["default"]
                        else:
                            cleaned_section[sub_key] = value
                    else:
                        # Use the default empty value if key doesn't exist
                        cleaned_section[sub_key] = default_value

                cleaned_article[main_key] = cleaned_section

            result[doi] = cleaned_article

        return result

    def clean_data_based_on_elements(self) -> Dict[str, Any]:
        """Run complete composition analysis with element validation."""
        result = {}
        for key, value in self.all_data.items():
            comp_prop_data = self._get_comp_prop_data(value)
            cleaned_data = self._clean_comp_prop_data_with_element_check(comp_prop_data)
            # Only include entries with valid compositions
            if cleaned_data:
                result[key] = value.copy()
                result[key]["composition_data"]["compositions_property_values"] = (
                    self._return_in_dict(cleaned_data)
                )
        return result

    def clean_data_without_element_filtering(self) -> Dict[str, Any]:
        """Run composition analysis without element validation."""
        result = {}
        for key, value in self.all_data.items():
            comp_prop_data = self._get_comp_prop_data(value)
            cleaned_data = self._clean_comp_prop_data_without_element_check(
                comp_prop_data
            )
            # Include all entries that passed other cleaning steps
            if cleaned_data:
                result[key] = value.copy()
                result[key]["composition_data"]["compositions_property_values"] = (
                    self._return_in_dict(cleaned_data)
                )
        return result

    def clean_data_with_relevant_compositions(
        self, strategy: CleaningStrategy = CleaningStrategy.FULL
    ) -> Dict[str, Any]:
        """
        Clean data using the specified strategy.

        Args:
            strategy: CleaningStrategy enum value determining the cleaning approach
                - BASIC: Basic cleaning without element validation
                - FULL: Complete cleaning with element validation (default)

        Returns:
            Dict[str, Any]: Cleaned data based on selected strategy
        """
        self.all_data = self.get_useful_data()
        if strategy == CleaningStrategy.BASIC:
            # Clean without element validation
            return self.clean_data_without_element_filtering()
        else:
            # Full cleaning with element validation (default)
            return self.clean_data_based_on_elements()
