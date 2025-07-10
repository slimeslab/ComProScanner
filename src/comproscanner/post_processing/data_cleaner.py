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
        with open(self.results_file, "r") as f:
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

    def _clean_comp_prop_data(self, comp_prop_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean composition-property data."""
        comp_prop_data = self._get_comp_prop_pairs(comp_prop_data)
        comp_prop_data = self._filter_invalid_keys(comp_prop_data)
        valid_comp_prop_pairs = []
        for single_data in comp_prop_data:
            if self._is_elements(single_data):
                valid_comp_prop_pairs.append(single_data)
        valid_comp_prop_pairs = self._remove_extra_spaces(valid_comp_prop_pairs)
        valid_comp_prop_pairs = self._convert_fractions_to_decimal(
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
        valid_comp_prop_pairs = self._convert_fractions_to_decimal(
            valid_comp_prop_pairs
        )
        return valid_comp_prop_pairs

    def _convert_fractions_to_decimal(self, dict_list):
        """Convert fractions like 1/3 to decimal format (0.33)."""
        result = []
        for d in dict_list:
            new_dict = {}
            for key, value in d.items():
                # Find all fractions in the format of x/y
                new_key = re.sub(
                    r"(\d+)/(\d+)",
                    lambda m: f"{float(m.group(1))/float(m.group(2)):.2f}",
                    key,
                )
                new_dict[new_key] = value
            result.append(new_dict)
        return result

    def _return_in_dict(self, dict_list):
        final_dict = {}
        for d in dict_list:
            final_dict.update(d)
        return final_dict

    def _clean_data_based_on_elements(self) -> List[Any]:
        """Run complete composition analysis with element validation."""
        result = {}
        for key, value in self.all_data.items():
            comp_prop_data = self._get_comp_prop_data(value)
            cleaned_data = self._clean_comp_prop_data(comp_prop_data)
            # Only include entries with valid compositions
            if cleaned_data:
                result[key] = value.copy()  # Make a copy of the original data
                # Update just the compositions_property_values with cleaned data
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
                result[key] = value.copy()  # Make a copy of the original data
                # Update just the compositions_property_values with cleaned data
                result[key]["composition_data"]["compositions_property_values"] = (
                    self._return_in_dict(cleaned_data)
                )
        return result

    def clean_data(
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
        if strategy == CleaningStrategy.BASIC:
            # Clean without element validation
            return self.clean_data_without_element_filtering()
        else:
            # Full cleaning with element validation (default)
            return self._clean_data_based_on_elements()


def calculate_resolved_compositions(composition_data):
    """
    Process and normalize material composition data with complex chemical formulas.
    Handles mathematical operations (+, -, *, /) inside parentheses.
    Removes parentheses around pure numbers but preserves chemical formulas.

    Args:
        composition_data (dict): Dictionary containing composition data with 'compositions_property_values' key or the entire result dictionary with composition_data as a key

    Returns:
        dict: Processed composition data with normalized formulas as keys
    """

    def _process_composition_data(comp_data):
        """
        Process the composition data part of the results.

        Args:
            comp_data (dict): Dictionary with 'compositions_property_values' key

        Returns:
            dict: Processed composition data
        """
        if (
            not isinstance(comp_data, dict)
            or "compositions_property_values" not in comp_data
        ):
            return comp_data

        result = comp_data.copy()
        original_compositions = result.get("compositions_property_values", {})

        if not isinstance(original_compositions, dict):
            return result

        processed_compositions = {}

        for formula, value in original_compositions.items():
            processed_formula = _process_formula(formula)
            processed_compositions[processed_formula] = value

        result["compositions_property_values"] = processed_compositions
        return result

    def _process_formula(formula):
        """
        Process chemical formulas to handle any mathematical calculations within brackets.
        Also removes parentheses around pure numbers like (0.75) but keeps (Na0.25)

        Args:
            formula (str): The chemical formula string

        Returns:
            str: Normalized formula
        """
        if not formula or not isinstance(formula, str):
            return str(formula) if formula is not None else ""

        def _evaluate_expression(expr):
            """
            Safely evaluate a mathematical expression if it contains operators.
            If no operators, check if it's a pure number or contains chemical symbols.
            """
            # Check if there are mathematical operators
            if not any(op in expr for op in ["+", "-", "*", "/"]):
                # No operators - check if it's a pure number or contains chemical symbols
                if re.match(r"^[0-9.\s]+$", expr.strip()):
                    # Pure number - remove parentheses
                    return expr.strip()
                else:
                    # Contains letters/symbols - keep with parentheses
                    return f"({expr})"

            # Allow numbers, floating points, and basic math operators.
            # This is a security measure to restrict what eval() can process.
            allowed_chars = r"^[0-9\.\s\+\-\*\/\(\)]*$"
            if not re.match(allowed_chars, expr.strip()):
                # If invalid characters, return with parentheses to preserve structure
                return f"({expr})"

            try:
                # Eval is used here after checking the expression contains only allowed characters.
                result = eval(expr)
                # Format the result - keep decimal places for non-integers
                if isinstance(result, float):
                    if result.is_integer():
                        return str(int(result))
                    else:
                        # Keep up to 4 decimal places
                        return str(round(result, 4))
                else:
                    return str(result)
            except (SyntaxError, ZeroDivisionError, TypeError, NameError):
                # If evaluation fails, return with parentheses to preserve structure
                return f"({expr})"

        # Process parentheses - both mathematical expressions and pure numbers
        # Add a safety counter to prevent infinite loops
        max_iterations = 100
        iteration_count = 0

        while iteration_count < max_iterations:
            match = re.search(r"\(([^()]+)\)", formula)
            if not match:
                break

            expression_inside_parentheses = match.group(1)
            evaluated_value = _evaluate_expression(expression_inside_parentheses)

            # For avoiding infinite loops
            if evaluated_value == f"({expression_inside_parentheses})":
                break

            # Replace the matched part
            formula = (
                formula[: match.start()] + str(evaluated_value) + formula[match.end() :]
            )

            iteration_count += 1

        if iteration_count >= max_iterations:
            print(f"Warning: Maximum iterations reached for formula: {formula}")

        return formula

    if not composition_data:
        return {}

    # Extract the composition data if it's nested
    if isinstance(composition_data, dict) and "composition_data" in composition_data:
        result = composition_data.copy()
        result["composition_data"] = _process_composition_data(
            composition_data["composition_data"]
        )
        return result

    # Otherwise, process the composition data directly
    return _process_composition_data(composition_data)
