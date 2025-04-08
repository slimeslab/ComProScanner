"""
data_cleaner.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 02-04-2025
"""

import re


def calculate_resolved_compositions(composition_data):
    """
    Process and normalize material composition data with complex chemical formulas.
    Only handles subtraction calculations inside brackets (float-float) → calculated value with result in brackets.

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
        Process chemical formulas to handle any subtraction calculations within brackets:
        (float-float) → calculated value with result in brackets
        e.g., (0.7-0.2) → (0.5) or (1-0.2-0.1) → (0.7)

        Args:
            formula (str): The chemical formula string

        Returns:
            str: Normalized formula
        """
        if not formula or not isinstance(formula, str):
            return str(formula) if formula is not None else ""

        # Handle any subtraction inside parentheses: (0.7-0.2) → (0.5) or (1-0.2-0.1) → (0.7)
        # Matches any expression with numbers and hyphens inside parentheses
        normalized = re.sub(
            r"\(([0-9.-]+)\)", lambda m: _format_subtraction(m), formula
        )

        return normalized

    def _format_subtraction(match):
        """
        Calculate the result of subtractions inside parentheses.
        Keep the calculated value inside brackets.

        Args:
            match: Regex match object

        Returns:
            str: Result of the subtraction in brackets
        """
        # Split the string by hyphens and convert each part to float
        parts = match.group(1).split("-")

        # If the first character is empty, it means the expression starts with a minus
        if parts[0] == "":
            parts[1] = "-" + parts[1]
            parts.pop(0)

        numbers = [float(num) for num in parts]

        # Calculate the result by starting with the first number and subtracting all others
        result = numbers[0]
        for num in numbers[1:]:
            result -= num
        result = round(result, 3)  # Round to 3 decimal places

        # Always keep the parentheses for the result
        return f"({result})"

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
