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
    """Cleans and normalises composition-property data extracted from research articles."""

    def __init__(self, results_file: str):
        """
        Args:
            results_file (str): Path to the JSON file containing raw extraction results.
        """
        self.results_file = results_file
        self.all_data = self._load_results()
        self.all_elements = get_all_elements()
        self.filtered_compositions: Dict[str, List[str]] = {}
        self.unresolved_compositions: Dict[str, List[str]] = {}

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

    def _filter_invalid_keys(self, dict_list, doi: str = ""):
        """Filter dictionaries with invalid keys (more than 2 consecutive capital letters)."""
        pattern = r"(?<![a-z0-9])[A-Z]{2,}(?![a-z0-9])"
        valid = []
        for d in dict_list:
            if any(re.search(pattern, key) for key in d.keys()):
                if doi:
                    self.filtered_compositions.setdefault(doi, []).extend(d.keys())
            else:
                valid.append(d)
        return valid

    def _is_elements(self, comp_pro_pair: Dict[str, Any]) -> bool:
        """Check whether the composition key in a pair can be fully parsed as a sequence of valid element symbols.

        Args:
            comp_pro_pair (Dict[str, Any]): A single {composition: value} dictionary.

        Returns:
            bool: True if the key resolves to a valid chemical composition, False otherwise.
        """
        def _convert_subscript_unicode(string: str) -> str:
            """Convert Unicode subscript digits to regular digits."""
            subscript_unicode = {
                0: "\u2080",
                1: "\u2081",
                2: "\u2082",
                3: "\u2083",
                4: "\u2084",
                5: "\u2085",
                6: "\u2086",
                7: "\u2087",
                8: "\u2088",
                9: "\u2089",
            }
            # Create reverse mapping: Unicode -> digit
            unicode_to_digit = {v: str(k) for k, v in subscript_unicode.items()}

            # Replace each Unicode subscript with its corresponding digit
            for unicode_char, digit in unicode_to_digit.items():
                string = string.replace(unicode_char, digit)

            return string

        def _remove_special_chars(string: str) -> str:
            # First convert Unicode subscripts to regular digits
            string = _convert_subscript_unicode(string)
            # Then remove all non-alphabetic characters
            return re.sub(r"[^a-zA-Z]+", "", string)

        def _can_parse_as_elements(s: str, elements: List[str]) -> bool:
            """
            Check if string can be completely parsed as a sequence of valid element symbols.
            Uses greedy matching, preferring 2-letter elements over 1-letter.
            """
            if not s:
                return True

            # Try to match 2-letter element first (e.g., "Pb", "Sr")
            if len(s) >= 2:
                two_letter = s[:2]
                if two_letter in elements:
                    # Recursively check the rest
                    if _can_parse_as_elements(s[2:], elements):
                        return True

            # Try to match 1-letter element (e.g., "P", "O")
            if len(s) >= 1:
                one_letter = s[:1]
                if one_letter in elements:
                    # Recursively check the rest
                    if _can_parse_as_elements(s[1:], elements):
                        return True

            # Cannot parse this string
            return False

        try:
            key = next(iter(comp_pro_pair))  # Get the key
            key = _remove_special_chars(str(key))

            # If no capital letters found, it's not a valid chemical composition
            if not key or not re.search(r"[A-Z]", key):
                return False

            # CRITICAL FIX: Verify that the entire string can be parsed as valid elements
            return _can_parse_as_elements(key, self.all_elements)

        except Exception:
            return False

    def _remove_extra_spaces(self, dict_list):
        """Remove all spaces from composition keys in a list of {composition: value} dicts.

        Args:
            dict_list (list): List of single-entry dicts mapping composition strings to values.

        Returns:
            list: Same structure with spaces stripped from every key.
        """
        # remove any spaces in the key
        return [
            {key.replace(" ", ""): value for key, value in d.items()} for d in dict_list
        ]

    def _clean_comp_prop_data_with_element_check(
        self, comp_prop_data: Dict[str, Any], doi: str = ""
    ) -> Dict[str, Any]:
        """Clean composition-property data with element validation from periodic table."""
        comp_prop_data = self._get_comp_prop_pairs(comp_prop_data)
        comp_prop_data = self._filter_invalid_keys(comp_prop_data, doi)
        valid_comp_prop_pairs = []
        for single_data in comp_prop_data:
            if self._is_elements(single_data):
                valid_comp_prop_pairs.append(single_data)
            else:
                if doi:
                    self.filtered_compositions.setdefault(doi, []).extend(single_data.keys())
        valid_comp_prop_pairs = self._remove_extra_spaces(valid_comp_prop_pairs)
        valid_comp_prop_pairs = self._convert_fractions_and_resolve_compositions(
            valid_comp_prop_pairs
        )
        return valid_comp_prop_pairs

    def _clean_comp_prop_data_without_element_check(
        self, comp_prop_data: Dict[str, Any], doi: str = ""
    ) -> Dict[str, Any]:
        """Clean composition-property data without element validation."""
        comp_prop_data = self._get_comp_prop_pairs(comp_prop_data)
        comp_prop_data = self._filter_invalid_keys(comp_prop_data, doi)
        valid_comp_prop_pairs = comp_prop_data
        valid_comp_prop_pairs = self._remove_extra_spaces(valid_comp_prop_pairs)
        valid_comp_prop_pairs = self._convert_fractions_and_resolve_compositions(
            valid_comp_prop_pairs
        )
        return valid_comp_prop_pairs

    def _convert_fractions_and_resolve_compositions(self, dict_list):
        """
        Convert fractions to decimal format and resolve composition formulas by handling mathematical operations and normalizing bracket notation.
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

                    # Check if it's a purely numeric/arithmetic expression (only numbers and operators)
                    if re.match(r"^[0-9.\s+\-*/]+$", expression):
                        if any(op in expression for op in ["+", "-", "*", "/"]):
                            try:
                                # Evaluate the expression using BODMAS rule and format result
                                result = eval(expression)
                                if isinstance(result, float):
                                    if result.is_integer():
                                        evaluated_value = str(int(result))
                                    else:
                                        evaluated_value = str(round(result, 5))
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
                        else:
                            # Bare number in brackets — strip brackets without evaluating.
                            # e.g. Gd(0) → Gd0, (0.00) blocking (0.972-(0.00)) → unblocks outer.
                            # Skip if preceded by * so _multiply_pure_number_coefficients can handle it.
                            if match.start() > 0 and formula[match.start() - 1] == "*":
                                continue
                            formula = (
                                formula[: match.start()]
                                + expression
                                + formula[match.end() :]
                            )
                            changed = True
                            break

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
                # Match: digit*(digit) - only valid decimal numbers
                match = re.search(
                    r"(\d+(?:\.\d+)?)\s*\*\s*\((\d+(?:\.\d+)?)\)", formula
                )

                if match:
                    try:
                        coeff1 = float(match.group(1))
                        coeff2 = float(match.group(2))
                    except ValueError:
                        # Skip invalid numbers
                        iteration_count += 1
                        continue

                    result = coeff1 * coeff2

                    # Format result
                    if result == int(result):
                        result_str = str(int(result))
                    else:
                        result_str = str(round(result, 5))

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
                                evaluated_value = str(round(result, 5))
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

        def _resolve_element_coefficient_multiplications(formula):
            """
            Resolve ElementCoeff*Multiplier patterns such as Zr0.1*1, Ti0.9*0.999, Ta*0.
            Each match is replaced by the element with its base coefficient multiplied by
            the trailing multiplier. Elements whose result is zero are removed entirely.
            """
            element_mult_pattern = r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?\*(\d+(?:\.\d+)?)"

            def _replace_element_mult(match):
                element = match.group(1)
                base_coeff_str = match.group(2)
                multiplier_str = match.group(3)
                base_coeff = float(base_coeff_str) if base_coeff_str else 1.0
                result_coeff = round(base_coeff * float(multiplier_str), 8)
                if result_coeff == 0:
                    return ""
                elif result_coeff == 1.0:
                    return element
                elif result_coeff == int(result_coeff):
                    return f"{element}{int(result_coeff)}"
                else:
                    formatted = f"{result_coeff:.8f}".rstrip("0").rstrip(".")
                    return f"{element}{formatted}" if formatted not in ("0", "0.") else ""

            return re.sub(element_mult_pattern, _replace_element_mult, formula)

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

            # Step 4: Resolve ElementCoeff*Multiplier patterns (e.g. Zr0.1*1, Ta*0)
            formula = _resolve_element_coefficient_multiplications(formula)

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

        def _convert_subscript_unicode_to_digits(string: str) -> str:
            """Convert Unicode subscript digits to regular digits."""
            subscript_unicode = {
                0: "\u2080",
                1: "\u2081",
                2: "\u2082",
                3: "\u2083",
                4: "\u2084",
                5: "\u2085",
                6: "\u2086",
                7: "\u2087",
                8: "\u2088",
                9: "\u2089",
            }
            # Create reverse mapping: Unicode -> digit
            unicode_to_digit = {v: str(k) for k, v in subscript_unicode.items()}

            # Replace each Unicode subscript with its corresponding digit
            for unicode_char, digit in unicode_to_digit.items():
                string = string.replace(unicode_char, digit)

            return string

        # Main processing logic
        result = []
        for d in dict_list:
            new_dict = {}
            for key, value in d.items():
                # Step 0: Convert Unicode subscripts to regular digits
                processed_key = _convert_subscript_unicode_to_digits(str(key))

                # Step 1: Convert simple fractions to decimals (handles both integer and decimal numerators/denominators)
                processed_key = re.sub(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", _replace_fraction, processed_key)

                # Step 2: Resolve compositions
                processed_key = _resolve_composition(processed_key)

                new_dict[processed_key] = value
            result.append(new_dict)
        return result

    def _remove_miller_indices(self, formula: str) -> str:
        """
        Remove Miller indices (crystal plane notations) from chemical formulas.
        Miller indices are represented as three integers in parentheses.

        Examples:
        - AlN (002) -> AlN
        - ZnO (111) -> ZnO
        - BaTiO3 (100) -> BaTiO3

        Does NOT remove:
        - (0.02) - contains decimal point
        - (12) - only two digits
        - (K0.5Na0.5) - contains letters
        """
        if not formula or not isinstance(formula, str):
            return formula

        # Match: space + opening parenthesis + exactly 3 digits + closing parenthesis
        # The \s* matches optional whitespace before the parenthesis
        pattern = r"\s*\([0-9]{3}\)"

        return re.sub(pattern, "", formula).strip()

    def _remove_zero_coefficient_elements(self, formula: str) -> str:
        """
        Remove elements with coefficient 0 or 0.0 from chemical formulas.

        Examples:
        - BaTiZr0O3 -> BaTiO3
        - Na0.5K0.5Nb0O3 -> Na0.5K0.5O3
        - Pb0La0.1ZrTiO3 -> La0.1ZrTiO3
        """
        if not formula or not isinstance(formula, str):
            return formula

        # Match element symbol followed by coefficient
        pattern = r"([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)?"

        def keep_element(match):
            element = match.group(1)
            coeff_str = match.group(2)

            if not coeff_str:
                # No coefficient means 1, keep it
                return match.group(0)

            try:
                coeff = float(coeff_str)
                # Remove elements with coefficient 0
                if coeff == 0 or coeff == 0.0:
                    return ""
                else:
                    return match.group(0)
            except ValueError:
                # If conversion fails, keep original
                return match.group(0)

        return re.sub(pattern, keep_element, formula)

    def _normalize_coefficients(self, formula: str) -> str:
        """
        Normalize element coefficients by removing trailing zeros.

        Examples:
        - Pb0.90 -> Pb0.9
        - La0.10 -> La0.1
        - Ti2.00 -> Ti2
        - O3 -> O3 (unchanged)
        """
        if not formula or not isinstance(formula, str):
            return formula

        def format_coefficient(match):
            element = match.group(1)
            coeff_str = match.group(2)

            if not coeff_str:
                # No coefficient means 1
                return element

            try:
                coeff = float(coeff_str)

                # Check if it's effectively 1
                if coeff == 1.0:
                    return element
                # Check if it's an integer
                elif coeff == int(coeff):
                    return f"{element}{int(coeff)}"
                else:
                    # Format and remove trailing zeros
                    formatted = f"{coeff:.8f}".rstrip("0").rstrip(".")
                    return f"{element}{formatted}"
            except ValueError:
                # If conversion fails, keep original
                return match.group(0)

        # Match element symbol followed by optional decimal number
        pattern = r"([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)?"
        return re.sub(pattern, format_coefficient, formula)

    def _expand_leading_and_trailing_coefficients(self, formula: str) -> str:
        """
        Expand leading and trailing coefficient patterns before bracket expansion.

        Handles:
        1. Leading coefficients: 0.7(composition) or (0.15)composition
        2. Trailing coefficients after parentheses: (K0.5Na0.5)(0.97)

        Examples:
        - 0.7(K0.48Na0.52NbO2.7SnO2) -> (K0.336Na0.364Nb1.89Sn0.49O1.4)
        - (0.15)Dy2O3 -> (Dy0.15O0.45)
        - (K0.5Na0.5)(0.97)Ag0.03NbO3 -> (K0.485Na0.485)Ag0.03NbO3
        """
        if not formula or not isinstance(formula, str):
            return formula

        def multiply_element_coefficients(composition: str, multiplier: float) -> str:
            """Multiply all element coefficients in a composition by a multiplier."""
            # Match element symbol followed by optional valid decimal number
            element_pattern = r"([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)?"

            def replace_coeff(match):
                element = match.group(1)
                coeff_str = match.group(2)
                coeff = float(coeff_str) if coeff_str else 1.0
                new_coeff = round(
                    coeff * multiplier, 8
                )  # Higher precision for very small numbers

                # Only remove if exactly zero
                if new_coeff == 0:
                    return ""
                elif new_coeff == 1.0:
                    return element
                elif new_coeff == int(new_coeff):
                    return f"{element}{int(new_coeff)}"
                else:
                    # Format with up to 8 decimal places, remove trailing zeros
                    formatted_coeff = f"{new_coeff:.8f}".rstrip("0").rstrip(".")
                    # Double-check: if formatting resulted in "0", remove element
                    if formatted_coeff == "0" or formatted_coeff == "0.":
                        return ""
                    return f"{element}{formatted_coeff}"

            return re.sub(element_pattern, replace_coeff, composition)

        max_iterations = 50
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            changed = False

            # Handle pattern 2: (composition)(coefficient) or (composition)coefficient
            # Match closing bracket followed by optional bracket with number or just number
            match = re.search(
                r"\)([\[\(]?)(\d+(?:\.\d+)?)([\]\)]?)(?=[+\-\u2013A-Z]|$)", formula
            )
            if match:
                open_inner = match.group(1)
                coefficient_str = match.group(2)
                close_inner = match.group(3)

                # Verify it's a valid coefficient pattern
                if (not open_inner and not close_inner) or (open_inner and close_inner):
                    try:
                        coefficient = float(coefficient_str)
                    except ValueError:
                        iteration += 1
                        continue
                    close_pos = match.start()

                    # Find matching opening bracket
                    bracket_depth = 0
                    open_pos = -1
                    for i in range(close_pos, -1, -1):
                        if formula[i] in ")]":
                            bracket_depth += 1
                        elif formula[i] in "([":
                            bracket_depth -= 1
                            if bracket_depth == 0:
                                open_pos = i
                                break

                    if open_pos != -1:
                        bracket_type = formula[open_pos]
                        close_bracket = ")" if bracket_type == "(" else "]"
                        composition = formula[open_pos + 1 : close_pos]

                        # If coefficient is zero, remove the entire bracketed section
                        if coefficient == 0:
                            formula = formula[:open_pos] + formula[match.end() :]
                        else:
                            expanded = multiply_element_coefficients(
                                composition, coefficient
                            )
                            # If expansion results in empty string, remove the brackets entirely
                            if not expanded or expanded.strip() == "":
                                formula = formula[:open_pos] + formula[match.end() :]
                            else:
                                formula = (
                                    formula[:open_pos]
                                    + bracket_type
                                    + expanded
                                    + close_bracket
                                    + formula[match.end() :]
                                )
                        changed = True
                        continue

            # Handle pattern 1a: Leading coefficient before bracket
            # Match: 0.7(composition) or 0.7[composition] at start or after +/-/−
            match = re.search(r"(?:^|([+\-\u2013]))(\d+(?:\.\d+)?)([\[\(])", formula)
            if match:
                operator = match.group(1) or ""
                try:
                    coefficient = float(match.group(2))
                except ValueError:
                    iteration += 1
                    continue
                open_bracket = match.group(3)
                close_bracket = "]" if open_bracket == "[" else ")"

                # Find the matching closing bracket
                bracket_depth = 1
                start_pos = match.end()
                close_pos = -1

                for i in range(start_pos, len(formula)):
                    if formula[i] in "([":
                        bracket_depth += 1
                    elif formula[i] in ")]":
                        bracket_depth -= 1
                        if bracket_depth == 0:
                            close_pos = i
                            break

                if close_pos != -1:
                    composition = formula[start_pos:close_pos]

                    # If coefficient is zero, remove the entire bracketed section including operator
                    if coefficient == 0:
                        formula = formula[: match.start()] + formula[close_pos + 1 :]
                    else:
                        expanded = multiply_element_coefficients(composition, coefficient)
                        # If expansion results in empty string, remove the brackets entirely
                        if not expanded or expanded.strip() == "":
                            formula = formula[: match.start()] + formula[close_pos + 1 :]
                        else:
                            formula = (
                                formula[: match.start()]
                                + operator
                                + open_bracket
                                + expanded
                                + close_bracket
                                + formula[close_pos + 1 :]
                            )
                    changed = True
                    continue

            # Handle pattern 1b: (coefficient)composition
            # Match: (0.15)Dy2O3 where parenthesis contains only a number
            match = re.search(r"(?:^|([+\-\u2013]))\((\d+(?:\.\d+)?)\)([A-Z])", formula)
            if match:
                operator = match.group(1) or ""
                try:
                    coefficient = float(match.group(2))
                except ValueError:
                    iteration += 1
                    continue

                # Extract the composition part (until next operator or end)
                composition_start = match.start(3)
                composition_end = len(formula)

                for i in range(composition_start, len(formula)):
                    if formula[i] in "+-\u2013":
                        composition_end = i
                        break

                composition = formula[composition_start:composition_end]

                # If coefficient is zero, remove the entire section including operator
                if coefficient == 0:
                    formula = formula[: match.start()] + formula[composition_end:]
                else:
                    expanded = multiply_element_coefficients(composition, coefficient)
                    # If expansion results in empty string, remove the section entirely
                    if not expanded or expanded.strip() == "":
                        formula = formula[: match.start()] + formula[composition_end:]
                    else:
                        formula = (
                            formula[: match.start()]
                            + operator
                            + "("
                            + expanded
                            + ")"
                            + formula[composition_end:]
                        )
                changed = True
                continue

            if not changed:
                break

        # Final cleanup: remove empty brackets and fix operators
        # Repeat cleanup multiple times to handle nested cases
        for _ in range(5):
            formula = re.sub(
                r"[\[\(]\s*[\]\)]", "", formula
            )  # Remove empty brackets ()  or []
            formula = re.sub(
                r"[-+\u2013]\s*[\[\(]\s*[\]\)]", "", formula
            )  # Remove operator followed by empty brackets
            formula = re.sub(
                r"[-+\u2013]{2,}", "-", formula
            )  # Replace multiple operators with single -

        formula = re.sub(r"^[-+\u2013]+", "", formula)  # Remove leading operators
        formula = re.sub(r"[-+\u2013]+$", "", formula)  # Remove trailing operators
        formula = formula.strip()

        # Normalize coefficients to remove trailing zeros
        formula = self._normalize_coefficients(formula)

        # Remove elements with zero coefficients
        formula = self._remove_zero_coefficient_elements(formula)

        return formula

    def _expand_parenthetical_coefficients(
        self, formula: str, total_expansion: int = 0
    ) -> tuple:
        """
        Expand parenthetical expressions in chemical formulas by multiplying
        the coefficients inside with the coefficient outside.

        Handles:
        1. Nested brackets: [(Na0.84K0.16)0.5Bi0.5]0.99 -> Na0.4158K0.0792Bi0.495
        2. Removes elements with coefficient 0
        3. Removes brackets without coefficients: (Na0.5K0.5) -> Na0.5K0.5
        4. Supports both () and [] brackets

        Args:
            formula: Chemical formula string
            total_expansion: Counter for expansions

        Returns:
            Tuple of (expanded formula, updated expansion counter)
        """
        if not formula or not isinstance(formula, str):
            return formula, total_expansion

        max_iterations = 20
        iteration = 0
        prev_formula = formula

        while iteration < max_iterations:
            iteration += 1
            changed = False

            try:
                # Step 1: Handle brackets with coefficients (both () and [])
                # Match innermost brackets with coefficient (handles both )0.93 and )*0.010)
                bracket_match = re.search(
                    r"[\[\(]([A-Za-z0-9.]+)[\]\)]\*?([\d.]+)", formula
                )
                if bracket_match:
                    inner_content = bracket_match.group(1)
                    outer_coefficient = float(bracket_match.group(2))

                    element_pattern = r"([A-Z][a-z]?)([\d.]*)"
                    elements = re.findall(element_pattern, inner_content)

                    expanded = ""
                    for element, coefficient_str in elements:
                        if not element:
                            continue

                        inner_coefficient = (
                            float(coefficient_str) if coefficient_str else 1.0
                        )
                        new_coefficient = round(inner_coefficient * outer_coefficient, 5)

                        # Skip elements with coefficient 0
                        if new_coefficient == 0:
                            continue

                        if new_coefficient == 1.0:
                            expanded += element
                        elif new_coefficient == int(new_coefficient):
                            expanded += f"{element}{int(new_coefficient)}"
                        else:
                            formatted_coeff = f"{new_coefficient:.4f}".rstrip("0").rstrip(
                                "."
                            )
                            expanded += f"{element}{formatted_coeff}"

                    formula = (
                        formula[: bracket_match.start()]
                        + expanded
                        + formula[bracket_match.end() :]
                    )
                    changed = True
                    continue

                # Step 2: Remove brackets without coefficients
                # Make sure we don't match if there's a * followed by a number
                no_coeff_match = re.search(
                    r"[\[\(]([A-Za-z0-9.]+)[\]\)](?![\*\d.])", formula
                )
                if no_coeff_match:
                    inner_content = no_coeff_match.group(1)
                    formula = (
                        formula[: no_coeff_match.start()]
                        + inner_content
                        + formula[no_coeff_match.end() :]
                    )
                    changed = True
                    continue

                if not changed:
                    break
            except Exception:
                # If evaluation fails, skip this iteration
                break

        if formula != prev_formula:
            total_expansion += 1

        # Normalize coefficients to remove trailing zeros
        formula = self._normalize_coefficients(formula)

        # Remove elements with zero coefficients
        formula = self._remove_zero_coefficient_elements(formula)

        return formula, total_expansion

    def _apply_advanced_composition_cleaning(
        self, comp_prop_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply advanced composition cleaning including Miller indices removal,
        coefficient expansion, normalization, and zero-coefficient removal.

        Uses local cleaning methods to process compositions.
        """
        cleaned_dict = {}
        for composition, property_value in comp_prop_dict.items():
            # Step 1: Remove Miller indices (crystal plane notations)
            cleaned_comp = self._remove_miller_indices(composition)

            # Step 2: Expand leading and trailing coefficients
            cleaned_comp = self._expand_leading_and_trailing_coefficients(cleaned_comp)

            # Step 3: Expand parenthetical coefficients
            cleaned_comp, _ = self._expand_parenthetical_coefficients(cleaned_comp, 0)

            # Step 4: Normalize coefficients (remove trailing zeros)
            cleaned_comp = self._normalize_coefficients(cleaned_comp)

            # Step 5: Remove elements with zero coefficients
            cleaned_comp = self._remove_zero_coefficient_elements(cleaned_comp)

            cleaned_dict[cleaned_comp] = property_value

        return cleaned_dict

    def _filter_unresolved_compositions(self, comp_prop_dict: Dict[str, Any], doi: str = "") -> Dict[str, Any]:
        """Remove individual compositions that still contain unresolved parentheses, brackets, or multiplication operators."""
        resolved = {}
        for comp, val in comp_prop_dict.items():
            if re.search(r"[()[\]*]", comp):
                if doi:
                    self.unresolved_compositions.setdefault(doi, []).append(comp)
            else:
                resolved[comp] = val
        return resolved

    def _return_in_dict(self, dict_list):
        """Merge a list of single-entry dicts into one flat dict.

        Args:
            dict_list (list): List of {composition: value} dicts.

        Returns:
            dict: Single merged dictionary of all composition-value pairs.
        """
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
            # Skip entries where article_data is not a dictionary
            if not isinstance(article_data, dict):
                continue

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

    def clean_data_based_on_elements(self, apply_advanced_cleaning: bool = True) -> Dict[str, Any]:
        """
        Run complete composition analysis with element validation.

        Args:
            apply_advanced_cleaning: If True (default), applies advanced composition cleaning
                                    (Miller indices removal, coefficient expansion, normalization).
                                    If False, returns basic cleaned compositions only.
        """
        result = {}
        for key, value in self.all_data.items():
            comp_prop_data = self._get_comp_prop_data(value)
            cleaned_data = self._clean_comp_prop_data_with_element_check(comp_prop_data, doi=key)
            # Only include entries with valid compositions
            if cleaned_data:
                result[key] = value.copy()
                comp_prop_dict = self._return_in_dict(cleaned_data)
                # Apply advanced composition cleaning if requested
                if apply_advanced_cleaning:
                    comp_prop_dict = self._apply_advanced_composition_cleaning(comp_prop_dict)
                # Remove compositions that still have unresolved brackets or math ops
                comp_prop_dict = self._filter_unresolved_compositions(comp_prop_dict, doi=key)
                result[key]["composition_data"]["compositions_property_values"] = comp_prop_dict
        return result

    def clean_data_without_element_filtering(self, apply_advanced_cleaning: bool = True) -> Dict[str, Any]:
        """
        Run composition analysis without element validation.

        Args:
            apply_advanced_cleaning: If True (default), applies advanced composition cleaning
                                    (Miller indices removal, coefficient expansion, normalization).
                                    If False, returns basic cleaned compositions only.
        """
        result = {}
        for key, value in self.all_data.items():
            comp_prop_data = self._get_comp_prop_data(value)
            cleaned_data = self._clean_comp_prop_data_without_element_check(
                comp_prop_data, doi=key
            )
            # Include all entries that passed other cleaning steps
            if cleaned_data:
                result[key] = value.copy()
                comp_prop_dict = self._return_in_dict(cleaned_data)
                # Apply advanced composition cleaning if requested
                if apply_advanced_cleaning:
                    comp_prop_dict = self._apply_advanced_composition_cleaning(comp_prop_dict)
                # Remove compositions that still have unresolved brackets or math ops
                comp_prop_dict = self._filter_unresolved_compositions(comp_prop_dict, doi=key)
                result[key]["composition_data"]["compositions_property_values"] = comp_prop_dict
        return result

    def clean_data_with_relevant_compositions(
        self, strategy: CleaningStrategy = CleaningStrategy.FULL,
        apply_advanced_cleaning: bool = True
    ) -> Dict[str, Any]:
        """
        Clean data using the specified strategy.

        Args:
            strategy: CleaningStrategy enum value determining the cleaning approach
                - BASIC: Basic cleaning without element validation
                - FULL: Complete cleaning with element validation (default)
            apply_advanced_cleaning: If True (default), applies advanced composition cleaning
                                    (Miller indices removal, coefficient expansion, normalization).
                                    If False, returns basic cleaned compositions only.

        Returns:
            Dict[str, Any]: Cleaned data based on selected strategy
        """
        self.all_data = self.get_useful_data()
        if strategy == CleaningStrategy.BASIC:
            # Clean without element validation
            return self.clean_data_without_element_filtering(apply_advanced_cleaning=apply_advanced_cleaning)
        else:
            # Full cleaning with element validation (default)
            return self.clean_data_based_on_elements(apply_advanced_cleaning=apply_advanced_cleaning)

    def get_all_composition_property_pairs(
        self,
        strategy: CleaningStrategy = CleaningStrategy.FULL,
        is_return_doi: bool = False,
        apply_advanced_cleaning: bool = True,
    ) -> Dict[str, Any]:
        """
        Get all composition-property pairs from cleaned data after applying cleaning strategy
        and resolving composition calculations.

        Args:
            strategy: CleaningStrategy enum value determining the cleaning approach
                - BASIC: Basic cleaning without element validation
                - FULL: Complete cleaning with element validation (default)
            is_return_doi: If True, returns nested dictionary with DOI as keys.
                If False (default), returns flat composition-property dictionary.
            apply_advanced_cleaning: If True (default), applies advanced composition cleaning
                                    (Miller indices removal, coefficient expansion, normalization).
                                    If False, returns basic cleaned compositions only.

        Returns:
            Dict[str, Any]: If is_return_doi is False (default):
                Dictionary mapping compositions to property values.
                Values are either float (single value) or list of floats (multiple values).
                If multiple occurrences of the same composition exist, all property values
                are collected in a list.

                If is_return_doi is True:
                Nested dictionary where keys are DOIs and values are dictionaries
                of composition-property pairs for that DOI.

                Null values are filtered out in both cases.
        """
        # Clean the data using the specified strategy
        cleaned_data = self.clean_data_with_relevant_compositions(
            strategy=strategy,
            apply_advanced_cleaning=apply_advanced_cleaning
        )

        if is_return_doi:
            # Return nested dictionary with DOI as keys
            doi_composition_map = {}

            for doi, article_data in cleaned_data.items():
                if "composition_data" in article_data:
                    comp_prop_values = article_data["composition_data"].get(
                        "compositions_property_values", {}
                    )

                    # Filter out null values
                    filtered_comp_prop = {
                        comp: prop
                        for comp, prop in comp_prop_values.items()
                        if prop is not None
                    }

                    if filtered_comp_prop:
                        doi_composition_map[doi] = filtered_comp_prop

            return doi_composition_map

        else:
            # Return flat composition -> property values dictionary
            composition_property_map = {}

            # Iterate through all DOIs and extract composition-property pairs
            for doi, article_data in cleaned_data.items():
                if "composition_data" in article_data:
                    comp_prop_values = article_data["composition_data"].get(
                        "compositions_property_values", {}
                    )

                    for composition, property_value in comp_prop_values.items():
                        # Skip null values
                        if property_value is None:
                            continue

                        if composition in composition_property_map:
                            # Composition already exists
                            existing_value = composition_property_map[composition]

                            if isinstance(existing_value, list):
                                # Already a list, append new value
                                existing_value.append(property_value)
                            else:
                                # Convert to list with both values
                                composition_property_map[composition] = [
                                    existing_value,
                                    property_value,
                                ]
                        else:
                            # First occurrence of this composition
                            composition_property_map[composition] = property_value

            return composition_property_map
