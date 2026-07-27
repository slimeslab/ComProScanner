"""
data_cleaner.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 08-04-2025
"""

# Standard library imports
import json
from typing import List, Dict, Any, Union, Set
import re
from enum import Enum
import copy

# Third-party imports
from pymatgen.core.periodic_table import Element


@staticmethod
def get_all_elements() -> List[str]:
    """Get list of all element symbols."""
    return [Element.from_Z(i).symbol for i in range(1, 119)]


class CleaningStep(str, Enum):
    """Individually selectable optional data-cleaning steps.

    Unicode subscript conversion and arithmetic/fraction resolution are not
    part of this enum — they always run, since coefficient_expansion_strict/
    _lenient, miller_indices, and element_validation_strict/_lenient all
    assume their output.
    """

    ABBREVIATION_FILTERING = "abbreviation_filtering"
    ELEMENT_VALIDATION_STRICT = "element_validation_strict"
    ELEMENT_VALIDATION_LENIENT = "element_validation_lenient"
    TEXT_NORMALIZATION = "text_normalization"
    MILLER_INDICES = "miller_indices"
    COEFFICIENT_EXPANSION_STRICT = "coefficient_expansion_strict"
    COEFFICIENT_EXPANSION_LENIENT = "coefficient_expansion_lenient"

    @classmethod
    def all(cls) -> List[str]:
        """Return every valid step name."""
        return [step.value for step in cls]


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
        self.filtered_compositions: Dict[str, List[Dict[str, str]]] = {}
        self.unresolved_compositions: Dict[str, List[Dict[str, str]]] = {}
        # Placeholder -> original-text map for protected percent annotations
        # (e.g. "7 wt% NiO", "0.1%MgO"), reset at the start of every clean run.
        self._pct_annotation_map: Dict[str, str] = {}

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
                    self.filtered_compositions.setdefault(doi, []).extend(
                        {
                            "composition": key,
                            "reason": CleaningStep.ABBREVIATION_FILTERING.value,
                        }
                        for key in d.keys()
                    )
            else:
                valid.append(d)
        return valid

    def _can_parse_as_elements(self, s: str) -> bool:
        """
        Check if a purely-alphabetic string can be completely parsed as a sequence
        of valid element symbols. Uses greedy matching, preferring 2-letter
        elements over 1-letter (e.g. "Pb", "Sr" before falling back to "P", "O").
        """
        if not s:
            return True

        if len(s) >= 2:
            two_letter = s[:2]
            if two_letter in self.all_elements:
                if self._can_parse_as_elements(s[2:]):
                    return True

        if len(s) >= 1:
            one_letter = s[:1]
            if one_letter in self.all_elements:
                if self._can_parse_as_elements(s[1:]):
                    return True

        return False

    # Matches the numeric+percent-sign prefix of weight/mole/atomic-percent
    # dopant annotations such as "7 wt% NiO", "0.1 mol% Fe2O3", or "0.1%MgO".
    # What follows this prefix (a plain token or a bracketed expression) is
    # captured separately in _protect_percent_annotations, since a bracket's
    # length is variable and can't be matched by a fixed-width regex group.
    _PERCENT_ANNOTATION_PATTERN = re.compile(
        r"(?::\s*)?(\d+(?:\.\d+)?)\s*(?:wt\.?|mol\.?|at\.?)?\s*%\s*"
    )
    _PERCENT_ANNOTATION_TARGET_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*")

    # Shared prefix for the inert placeholders substituted in for percent-annotations.
    # Lowercase-letters-only is intentional (keeps bracket-content char-class checks elsewhere
    # treating it as plain text), but that same lowercase-ness means the "reject an element match
    # immediately followed by more lowercase word text" anti-corruption checks below must special-case
    # it explicitly, otherwise a genuine element right before a protected annotation gets skipped.
    _PCT_PLACEHOLDER_PREFIX = "zzzpctannot"

    def _protect_percent_annotations(self, text: str) -> str:
        """Replace percent-based dopant annotations with inert placeholders.

        The target being annotated (what follows the "%") may be a plain
        token (e.g. "NiO") or a bracketed multi-term expression (e.g.
        "(0.78PbO-0.22CuO)") — the latter's length is variable, so it can't
        be captured by _PERCENT_ANNOTATION_PATTERN itself; it's located here
        via _find_matching_close_bracket instead.

        Args:
            text (str): Raw composition key, before arithmetic/coefficient processing.

        Returns:
            str: Same text with each percent annotation replaced by a placeholder.
        """
        result = []
        pos = 0
        for match in self._PERCENT_ANNOTATION_PATTERN.finditer(text):
            if match.start() < pos:
                continue  # already consumed by a previous annotation's span

            end = match.end()
            if end < len(text) and text[end] in "([":
                close_pos = self._find_matching_close_bracket(text, end)
                if close_pos == -1:
                    continue  # unbalanced bracket — leave unprotected
                end = close_pos + 1
            else:
                token_match = self._PERCENT_ANNOTATION_TARGET_TOKEN.match(text, end)
                if not token_match:
                    continue  # nothing sensible to protect
                end = token_match.end()

            result.append(text[pos : match.start()])
            placeholder = (
                f"{self._PCT_PLACEHOLDER_PREFIX}{len(self._pct_annotation_map)}zzz"
            )
            self._pct_annotation_map[placeholder] = text[match.start() : end]
            result.append(placeholder)
            pos = end

        result.append(text[pos:])
        return "".join(result)

    def _restore_percent_annotations_in_dict(
        self, comp_prop_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Restore any protected percent annotations in composition keys.

        Args:
            comp_prop_dict (Dict[str, Any]): Composition -> property-value mapping, potentially still containing placeholders from `_protect_percent_annotations`.

        Returns:
            Dict[str, Any]: Same mapping with placeholders replaced by their original percent-annotation text.
        """
        if not self._pct_annotation_map:
            return comp_prop_dict
        restored = {}
        for comp, val in comp_prop_dict.items():
            restored_comp = comp
            for placeholder, original in self._pct_annotation_map.items():
                if placeholder in restored_comp:
                    restored_comp = restored_comp.replace(placeholder, original)
            restored[restored_comp] = val
        return restored

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

        try:
            key = next(iter(comp_pro_pair))  # Get the key
            key = _remove_special_chars(str(key))

            # If no capital letters found, it's not a valid chemical composition
            if not key or not re.search(r"[A-Z]", key):
                return False

            # CRITICAL FIX: Verify that the entire string can be parsed as valid elements
            return self._can_parse_as_elements(key)

        except Exception:
            return False

    def _contains_element_token(self, comp_pro_pair: Dict[str, Any]) -> bool:
        """Check whether the composition key contains at least one embedded formula fragment, even if the key as a whole is not purely elements.

        Args:
            comp_pro_pair (Dict[str, Any]): A single {composition: value} dictionary.

        Returns:
            bool: True if any letter-run in the key parses as element symbols.
        """
        try:
            key = str(next(iter(comp_pro_pair)))
        except Exception:
            return False

        for token in re.findall(r"[A-Za-z]+", key):
            if re.search(r"[A-Z]", token) and self._can_parse_as_elements(token):
                return True
        return False

    def _has_balanced_annotated_brackets(self, comp: str) -> bool:
        """Check whether a composition's leftover brackets are balanced and
        contain genuine text (an annotation), rather than a failed/partial
        arithmetic or coefficient expression.

        Used by the coefficient_expansion_lenient step to spare compositions
        like "...-(%)" or "...(as-sintered)" from being dropped as
        unresolved, while still treating unmatched brackets, a stray "*", or
        brackets containing only leftover numbers/operators as genuine
        cleaning failures.

        Args:
            comp (str): Composition key to check.

        Returns:
            bool: True if brackets are balanced and at least one bracketed
                span contains non-arithmetic content.
        """
        if "*" in comp:
            return False
        if comp.count("(") != comp.count(")"):
            return False
        if comp.count("[") != comp.count("]"):
            return False

        for open_content, bracket_content in re.findall(
            r"\(([^()]*)\)|\[([^\[\]]*)\]", comp
        ):
            content = open_content or bracket_content
            if content and not re.match(r"^[0-9.\s+\-]*$", content):
                return True
        return False

    def _find_matching_close_bracket(self, text: str, open_pos: int) -> int:
        """Given text[open_pos] is '(' or '[', return the index of its
        depth-aware matching close bracket (handles mixed ()/[] nesting), or
        -1 if unbalanced."""
        depth = 0
        for i in range(open_pos, len(text)):
            if text[i] in "([":
                depth += 1
            elif text[i] in ")]":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _split_top_level_terms(self, text: str) -> List[str]:
        """Split text on +/-/– that occur at bracket depth 0, keeping each
        sign attached to the term that follows it. Unlike a plain regex
        split, this does not split on a +/- hidden inside a bracket's own
        multi-term content (e.g. the "-" inside "(0.89A-0.11B)")."""
        parts: List[str] = []
        depth = 0
        start = 0
        for i, ch in enumerate(text):
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch in "+-–" and depth == 0 and i > start:
                parts.append(text[start:i])
                start = i
        parts.append(text[start:])
        return parts

    def _formula_prefix_end(self, text: str) -> int:
        """Length of the longest prefix of `text` that is valid chemical-
        formula content (Element[coefficient] tokens, +/-/– joins, and
        brackets whose content is itself fully valid by this same rule).

        Everything from the first point that breaks this — a non-element
        token, an element that's actually the start of a longer descriptive
        word, a bracket containing free text, plain whitespace, etc. — is
        not formula content and must never be scaled by a coefficient that
        appeared earlier in the string. This is the single shared rule
        behind why e.g. "Bottom", "Reoxidized", and "PbTiO3 (calcined at
        660C)" must each stop being touched at a specific point rather than
        needing a bespoke check for every new shape of non-formula text.
        """
        element_pattern = re.compile(r"[A-Z][a-z]?")
        number_pattern = re.compile(r"[0-9]+(?:\.[0-9]+)?")
        pos = 0
        n = len(text)
        while pos < n:
            ch = text[pos]
            if ch in "([":
                close = self._find_matching_close_bracket(text, pos)
                if close == -1:
                    break
                inner = text[pos + 1 : close]
                if self._formula_prefix_end(inner) != len(inner):
                    break
                pos = close + 1
                continue
            if ch in "+-–":
                pos += 1
                continue
            m = element_pattern.match(text, pos)
            if not m or m.group(0) not in self.all_elements:
                break
            end = m.end()
            num_m = number_pattern.match(text, end)
            if num_m:
                end = num_m.end()
            next_text = text[end:]
            if (
                next_text[:1].isalpha()
                and next_text[:1].islower()
                and not next_text.startswith(self._PCT_PLACEHOLDER_PREFIX)
            ):
                break
            pos = end
        return pos

    def _distribute_multiterm_brackets(self, formula: str) -> str:
        """Resolve nested multi-term coefficient*(term1±term2±...) expressions
        (e.g. "0.75*(0.89(Bi0.5Na0.5)TiO3-0.11BaTiO3)") by distributing the
        outer coefficient into each of the bracket's own top-level +/-
        separated sub-terms, multiplying it into that sub-term's own leading
        coefficient. Only the numeric coefficients are combined here —
        element-level scaling (e.g. expanding "Bi0.5Na0.5") is left to the
        existing coefficient_expansion pipeline, which already scales
        elements correctly through a bracket wrapper regardless of what's
        nested inside it. This purely eliminates +/- signs that would
        otherwise be hidden inside a bracket, which the (bracket-depth-
        unaware) top-level formula splitting elsewhere cannot handle safely.

        No explicit recursion is needed for deeper nesting: each successful
        splice restarts the scan over the updated formula, so a spliced-in
        sub-term that is itself a further-nested coeff*(multi-term) shape
        gets picked up and resolved on a later pass.
        """
        if not formula or not isinstance(formula, str):
            return formula

        # Anchored to term-start (start-of-string or right after a top-level
        # +/-/–), mirroring the leading-coefficient-before-bracket pattern in
        # _expand_leading_and_trailing_coefficients — not just "any digit run
        # before a bracket", which would false-positive-match an element's
        # own trailing coefficient sitting next to an unrelated bracket (e.g.
        # the "0.15" in "Ca0.15(Zr0.1-Ti0.9)O3" belongs to Ca, not a
        # multiplier for the bracket).
        pattern = re.compile(
            r"(?:^|(?P<sign>[+\-–]))\s*(?P<coeff>\d+(?:\.\d+)?)\*?\s*(?P<open>[\(\[])"
        )
        iterations = 0
        while iterations < 20:
            iterations += 1
            progressed = False
            for m in pattern.finditer(formula):
                open_pos = m.start("open")
                close_pos = self._find_matching_close_bracket(formula, open_pos)
                if close_pos == -1:
                    continue
                inner = formula[open_pos + 1 : close_pos]

                # Defer pure-numeric arithmetic parens (e.g. "0.5*(0.2+0.3)")
                # to the existing arithmetic-evaluation machinery — this
                # function only handles brackets containing actual
                # composition text.
                if re.match(r"^[0-9.\s+\-*/]+$", inner):
                    continue

                sub_terms = self._split_top_level_terms(inner)
                if len(sub_terms) <= 1:
                    # Single-term bracket — leave to the existing
                    # coefficient_expansion pipeline.
                    continue

                outer_coeff = float(m.group("coeff"))
                negate = m.group("sign") in ("-", "–")
                # When negating, consume the leading "-" into the splice so
                # it can be folded into each sub-term's flipped sign;
                # otherwise leave any "+" (or start-of-string) untouched,
                # matching how the rest of the formula is joined.
                splice_start = m.start("sign") if negate else m.start("coeff")

                resolved_terms = []
                for term in sub_terms:
                    sign = ""
                    body = term
                    if body[:1] in ("+", "-", "–"):
                        sign = "-" if body[0] in ("-", "–") else "+"
                        body = body[1:]
                    body = body.strip()

                    inner_match = re.match(r"^(\d+(?:\.\d+)?)\*?", body)
                    if inner_match:
                        inner_coeff = float(inner_match.group(1))
                        remainder = body[inner_match.end() :].strip()
                    else:
                        inner_coeff = 1.0
                        remainder = body

                    if negate:
                        sign = "+" if sign == "-" else "-"

                    new_coeff = round(outer_coeff * inner_coeff, 8)
                    new_coeff_str = (
                        str(int(new_coeff))
                        if new_coeff == int(new_coeff)
                        else f"{new_coeff:.8f}".rstrip("0").rstrip(".")
                    )
                    resolved_terms.append(f"{sign}{new_coeff_str}{remainder}")

                replacement = "".join(resolved_terms)
                if replacement.startswith("+"):
                    replacement = replacement[1:]

                formula = (
                    formula[:splice_start] + replacement + formula[close_pos + 1 :]
                )
                progressed = True
                break

            if not progressed:
                break

        return formula

    _TITLE_CASE_STOPWORDS = {"with", "of", "at", "for", "the", "and"}

    def _normalize_text(self, dict_list):
        """Normalize whitespace and title-case descriptive word tokens in composition keys.

        Args:
            dict_list (list): List of single-entry dicts mapping composition strings to values.

        Returns:
            list: Same structure with whitespace normalized and descriptive
                word tokens title-cased.
        """

        def format_key(key: str) -> str:
            tokens = key.strip().split()
            formatted = []
            for i, tok in enumerate(tokens):
                if (
                    tok.isalpha()
                    and not tok.isupper()
                    and not self._can_parse_as_elements(tok)
                ):
                    lower = tok.lower()
                    if i > 0 and lower in self._TITLE_CASE_STOPWORDS:
                        formatted.append(lower)
                    elif self._can_parse_as_elements(tok.capitalize()):
                        formatted.append(tok)
                    else:
                        formatted.append(tok.capitalize())
                else:
                    formatted.append(tok)
            return " ".join(formatted)

        return [
            {format_key(str(key)): value for key, value in d.items()} for d in dict_list
        ]

    def _clean_comp_prop_pairs(
        self,
        comp_prop_data: Dict[str, Any],
        steps: Set[str],
        doi: str = "",
    ) -> List[Dict[str, Any]]:
        """Clean composition-property pairs according to the selected optional steps.

        Args:
            comp_prop_data (Dict[str, Any]): Raw composition -> property-value mapping.
            steps (Set[str]): Selected optional CleaningStep values.
            doi (str, optional): DOI used to track filtered compositions.

        Returns:
            List[Dict[str, Any]]: List of single-entry {composition: value} dicts after
                cleaning. Unicode conversion and arithmetic resolution always run,
                regardless of which optional steps are selected.
        """
        comp_prop_pairs = self._get_comp_prop_pairs(comp_prop_data)

        if CleaningStep.ABBREVIATION_FILTERING.value in steps:
            comp_prop_pairs = self._filter_invalid_keys(comp_prop_pairs, doi)

        if CleaningStep.ELEMENT_VALIDATION_STRICT.value in steps:
            valid_pairs = []
            for pair in comp_prop_pairs:
                if self._is_elements(pair):
                    valid_pairs.append(pair)
                elif doi:
                    self.filtered_compositions.setdefault(doi, []).extend(
                        {
                            "composition": key,
                            "reason": CleaningStep.ELEMENT_VALIDATION_STRICT.value,
                        }
                        for key in pair.keys()
                    )
            comp_prop_pairs = valid_pairs

        if CleaningStep.ELEMENT_VALIDATION_LENIENT.value in steps:
            # Weaker than element_validation_strict: keeps a composition as long as
            # it contains at least one embedded formula fragment (e.g. "BaTiO3" inside
            # "Cellulose nanofibers/BaTiO3@TiO2/..."), instead of requiring the
            # entire key to be pure elements. Applied as an additional sequential
            # filter, so if element_validation_strict is also selected, its stricter
            # result already excludes everything this step alone would allow —
            # this step gives no extra ground back when both are selected.
            valid_pairs = []
            for pair in comp_prop_pairs:
                if self._contains_element_token(pair):
                    valid_pairs.append(pair)
                elif doi:
                    self.filtered_compositions.setdefault(doi, []).extend(
                        {
                            "composition": key,
                            "reason": CleaningStep.ELEMENT_VALIDATION_LENIENT.value,
                        }
                        for key in pair.keys()
                    )
            comp_prop_pairs = valid_pairs

        if CleaningStep.TEXT_NORMALIZATION.value in steps:
            comp_prop_pairs = self._normalize_text(comp_prop_pairs)

        if CleaningStep.MILLER_INDICES.value in steps:
            # Drop (not transform) compositions carrying a crystal-plane notation.
            # Stripping "(002)"/"(110)" and keeping the bare formula would collapse
            # distinct surface-orientation entries for the same material down to
            # the same dict key — e.g. "AlN (002)" and "AlN (110)" would both
            # become "AlN", silently overwriting one another when merged. Must
            # run before arithmetic/bracket resolution below: the mandatory
            # bracket resolver would otherwise treat the bare, purely-numeric
            # parenthetical as a coefficient to fold into the formula.
            kept_pairs = []
            for d in comp_prop_pairs:
                key = next(iter(d))
                if self._remove_miller_indices(key) != key:
                    if doi:
                        self.filtered_compositions.setdefault(doi, []).append(
                            {
                                "composition": key,
                                "reason": CleaningStep.MILLER_INDICES.value,
                            }
                        )
                else:
                    kept_pairs.append(d)
            comp_prop_pairs = kept_pairs

        # Mandatory, always runs regardless of `steps` — later steps depend on this output.
        comp_prop_pairs = self._convert_fractions_and_resolve_compositions(
            comp_prop_pairs
        )

        return comp_prop_pairs

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
                            # Skip bare 3-digit integers (no decimal point) — these are
                            # Miller-index-shaped, e.g. "(002)". If the miller_indices
                            # step is selected it already removed these earlier in the
                            # pipeline; if not, silently merging the digits into the
                            # preceding element (AlN (002) -> AlN2) would be wrong
                            # regardless, so leave the bracket untouched here — it then
                            # gets caught as "unresolved" downstream.
                            if re.match(r"^[0-9]{3}$", expression):
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
                    return (
                        f"{element}{formatted}" if formatted not in ("0", "0.") else ""
                    )

            return re.sub(element_mult_pattern, _replace_element_mult, formula)

        def _resolve_arithmetic_and_multiply(formula):
            """
            Resolve arithmetic expressions and handle multiplication operations.
            This must be done BEFORE adding composition brackets.
            """
            if not formula or not isinstance(formula, str):
                return str(formula) if formula is not None else ""

            # Step 0: Distribute nested multi-term coefficient*(term1±term2±...)
            # expressions before anything else touches the bracket structure.
            formula = self._distribute_multiterm_brackets(formula)

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
            # Split formula by +/- operators while preserving them, without
            # splitting on a +/- hidden inside a bracket's own multi-term
            # content (depth-aware — see _split_top_level_terms).
            parts = self._split_top_level_terms(formula)

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

                    # Split off any trailing content that isn't valid formula
                    # text (per _formula_prefix_end), e.g. a descriptive
                    # annotation like "PbTiO3 (calcined at 660C)" — the
                    # annotation must never share the composition's
                    # coefficient/bracket scope, since it can incidentally
                    # contain single-letter unit symbols that are also valid
                    # periodic-table elements (e.g. "C" for Celsius, "K" for
                    # Kelvin), which coefficient_expansion would otherwise
                    # scale as if they were real stoichiometry.
                    boundary = self._formula_prefix_end(composition_part)
                    formula_part = composition_part[:boundary]
                    trailing_annotation = composition_part[boundary:]

                    if not formula_part:
                        # Nothing formula-shaped follows the coefficient at
                        # all — leave everything untouched.
                        processed_parts.append(
                            sign + coefficient + " " + composition_part
                        )
                        continue

                    # Check if the formula part already has proper brackets at the outermost level
                    if (
                        formula_part.startswith("(") and formula_part.endswith(")")
                    ) or (formula_part.startswith("[") and formula_part.endswith("]")):
                        # Already properly bracketed
                        processed_parts.append(
                            sign + coefficient + formula_part + trailing_annotation
                        )
                    else:
                        # Determine bracket type based on whether parentheses exist in the formula part
                        if "(" in formula_part or ")" in formula_part:
                            # Use square brackets
                            processed_parts.append(
                                sign
                                + coefficient
                                + "["
                                + formula_part
                                + "]"
                                + trailing_annotation
                            )
                        else:
                            # Use round brackets
                            processed_parts.append(
                                sign
                                + coefficient
                                + "("
                                + formula_part
                                + ")"
                                + trailing_annotation
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
                # Step 0: Convert Unicode subscripts to regular digits and protect percent annotations before any arithmetic or bracket processing
                processed_key = _convert_subscript_unicode_to_digits(str(key))

                processed_key = self._protect_percent_annotations(processed_key)

                # Step 1: Convert simple fractions to decimals (handles both integer and decimal numerators/denominators)
                processed_key = re.sub(
                    r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", _replace_fraction, processed_key
                )

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
            """Multiply all element coefficients within composition's valid
            formula prefix by multiplier; everything from the first
            non-formula point onward (e.g. a trailing descriptive
            annotation like " (calcined at 660C)") is left completely
            untouched, per _formula_prefix_end."""
            element_pattern = r"([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)?"
            boundary = self._formula_prefix_end(composition)
            scannable, trailing = composition[:boundary], composition[boundary:]

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

            return re.sub(element_pattern, replace_coeff, scannable) + trailing

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
            # Match: 0.7(composition), 0.7[composition], or the same with a
            # space before the bracket (e.g. "0.7 (composition)") - the space
            # is purely a formatting artifact and must not change the result.
            #
            # A candidate is only acted on if its bracket contents are at
            # least partly genuine formula content (per _formula_prefix_end),
            # or the coefficient is exactly 0 (always removable regardless of
            # content). A bracket that's actually free text - e.g.
            # "0.1 (Sandwich-structured)" or "20 (0 Phr hydroxyapatite, ...)"
            # - must be left completely untouched, coefficient included,
            # rather than having its digits silently dropped as if they'd
            # been consumed into a scaling that never happened. Candidates
            # are scanned left to right so an earlier non-formula bracket
            # doesn't block a later, genuinely actionable one.
            match = None
            match_close_pos = -1
            for candidate in re.finditer(
                r"(?:^|([+\-–]))(\d+(?:\.\d+)?)\s*([\[\(])", formula
            ):
                open_bracket = candidate.group(3)
                bracket_depth = 1
                start_pos = candidate.end()
                close_pos = -1
                for i in range(start_pos, len(formula)):
                    if formula[i] in "([":
                        bracket_depth += 1
                    elif formula[i] in ")]":
                        bracket_depth -= 1
                        if bracket_depth == 0:
                            close_pos = i
                            break
                if close_pos == -1:
                    continue
                try:
                    coefficient = float(candidate.group(2))
                except ValueError:
                    continue
                composition = formula[start_pos:close_pos]
                if coefficient != 0 and self._formula_prefix_end(composition) == 0:
                    continue
                match = candidate
                match_close_pos = close_pos
                break

            if match:
                operator = match.group(1) or ""
                coefficient = float(match.group(2))
                open_bracket = match.group(3)
                close_bracket = "]" if open_bracket == "[" else ")"
                start_pos = match.end()
                close_pos = match_close_pos
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

                    # Only scale within the bracket content's valid formula
                    # prefix (per _formula_prefix_end) — anything from the
                    # first non-formula point onward is appended untouched
                    # rather than scanned for element-like matches at all.
                    boundary = self._formula_prefix_end(inner_content)
                    scannable = inner_content[:boundary]
                    trailing = inner_content[boundary:]

                    expanded = ""
                    for m in re.finditer(element_pattern, scannable):
                        element, coefficient_str = m.group(1), m.group(2)
                        if not element:
                            continue

                        inner_coefficient = (
                            float(coefficient_str) if coefficient_str else 1.0
                        )
                        new_coefficient = round(
                            inner_coefficient * outer_coefficient, 5
                        )

                        # Skip elements with coefficient 0
                        if new_coefficient == 0:
                            continue

                        if new_coefficient == 1.0:
                            expanded += element
                        elif new_coefficient == int(new_coefficient):
                            expanded += f"{element}{int(new_coefficient)}"
                        else:
                            formatted_coeff = f"{new_coefficient:.4f}".rstrip(
                                "0"
                            ).rstrip(".")
                            expanded += f"{element}{formatted_coeff}"

                    expanded += trailing

                    formula = (
                        formula[: bracket_match.start()]
                        + expanded
                        + formula[bracket_match.end() :]
                    )
                    changed = True
                    continue

                # Step 2: Remove brackets without coefficients
                # Make sure we don't match if there's a * followed by a number
                # Skip bare 3-digit integers (Miller-index-shaped, e.g. "(002)")
                # — those must be handled by the miller_indices step, not
                # silently stripped here.
                no_coeff_match = None
                for candidate in re.finditer(
                    r"[\[\(]([A-Za-z0-9.]+)[\]\)](?![\*\d.])", formula
                ):
                    if re.match(r"^[0-9]{3}$", candidate.group(1)):
                        continue
                    no_coeff_match = candidate
                    break
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
        self, comp_prop_dict: Dict[str, Any], steps: Set[str]
    ) -> Dict[str, Any]:
        """
        Apply coefficient expansion, gated by `steps`.

        Miller indices removal is handled earlier in `_clean_comp_prop_pairs`,
        before the mandatory arithmetic/bracket resolution — see the comment
        there for why the ordering matters.

        Coefficient expansion internally normalizes trailing zeros and removes
        zero-coefficient elements as part of expanding leading/trailing/nested
        bracket coefficients — there are no separate steps for those.
        """
        cleaned_dict = {}
        for composition, property_value in comp_prop_dict.items():
            cleaned_comp = composition

            if steps & {
                CleaningStep.COEFFICIENT_EXPANSION_STRICT.value,
                CleaningStep.COEFFICIENT_EXPANSION_LENIENT.value,
            }:
                cleaned_comp = self._expand_leading_and_trailing_coefficients(
                    cleaned_comp
                )
                cleaned_comp, _ = self._expand_parenthetical_coefficients(
                    cleaned_comp, 0
                )

            cleaned_dict[cleaned_comp] = property_value

        return cleaned_dict

    def _filter_unresolved_compositions(
        self, comp_prop_dict: Dict[str, Any], doi: str = "", steps: Set[str] = None
    ) -> Dict[str, Any]:
        """Remove individual compositions that still contain unresolved parentheses, brackets, or multiplication operators.

        If coefficient_expansion_lenient is selected without
        coefficient_expansion_strict, compositions with balanced
        brackets/braces containing genuine text (not a stray "*" or leftover
        arithmetic) are spared instead of being dropped — see
        `_has_balanced_annotated_brackets`. Selecting both steps together
        reverts to the strict behavior below.

        Args:
            steps (Set[str], optional): Resolved set of selected CleaningStep
                values, used to decide whether the lenient carve-out applies.
        """
        steps = steps or set()
        lenient_active = (
            CleaningStep.COEFFICIENT_EXPANSION_LENIENT.value in steps
            and CleaningStep.COEFFICIENT_EXPANSION_STRICT.value not in steps
        )
        resolved = {}
        for comp, val in comp_prop_dict.items():
            if lenient_active and self._has_balanced_annotated_brackets(comp):
                resolved[comp] = val
                continue
            if re.search(r"[()[\]*]", comp):
                if doi:
                    self.unresolved_compositions.setdefault(doi, []).append(
                        {
                            "composition": comp,
                            "reason": "unresolved_brackets_or_operators",
                        }
                    )
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

    def _resolve_cleaning_steps(
        self, cleaning_steps: Union[str, List[str]]
    ) -> Set[str]:
        """Validate and resolve the public `cleaning_steps` argument into a concrete set.

        Args:
            cleaning_steps: Either the string "all" or a list of CleaningStep values.

        Returns:
            Set[str]: The resolved set of selected optional step names.

        Raises:
            ValueError: If `cleaning_steps` is a non-"all" string, or contains
                unknown step names.
        """
        valid_steps = set(CleaningStep.all())
        if cleaning_steps == "all":
            return valid_steps
        if isinstance(cleaning_steps, str):
            raise ValueError(
                "Invalid cleaning_steps value. Must be 'all' or a list of step names: "
                f"{CleaningStep.all()}."
            )
        unknown = set(cleaning_steps) - valid_steps
        if unknown:
            raise ValueError(
                f"Invalid cleaning step(s): {sorted(unknown)}. Valid options: {CleaningStep.all()}."
            )
        return set(cleaning_steps)

    def clean_data_with_selected_steps(self, steps: Set[str]) -> Dict[str, Any]:
        """
        Run composition cleaning using exactly the given set of optional steps.

        Unicode conversion and arithmetic/fraction resolution always run
        regardless of `steps`, since later steps depend on their output.

        Args:
            steps: Resolved set of CleaningStep values to apply.
        """
        self._pct_annotation_map = {}
        result = {}
        for key, value in self.all_data.items():
            comp_prop_data = self._get_comp_prop_data(value)
            cleaned_data = self._clean_comp_prop_pairs(comp_prop_data, steps, doi=key)
            if cleaned_data:
                result[key] = value.copy()
                comp_prop_dict = self._return_in_dict(cleaned_data)
                if steps & {
                    CleaningStep.COEFFICIENT_EXPANSION_STRICT.value,
                    CleaningStep.COEFFICIENT_EXPANSION_LENIENT.value,
                }:
                    comp_prop_dict = self._apply_advanced_composition_cleaning(
                        comp_prop_dict, steps
                    )
                    # Only filter out unresolved brackets/operators when coefficient
                    # expansion was actually attempted. Without it selected, the
                    # brackets the mandatory arithmetic step adds around coefficient
                    # segments are expected and unexpanded on purpose — they are not
                    # "failures", so the composition should pass through untouched
                    # rather than being dropped.
                    comp_prop_dict = self._filter_unresolved_compositions(
                        comp_prop_dict, doi=key, steps=steps
                    )
                comp_prop_dict = self._restore_percent_annotations_in_dict(
                    comp_prop_dict
                )
                result[key]["composition_data"][
                    "compositions_property_values"
                ] = comp_prop_dict
        return result

    def clean_data_with_relevant_compositions(
        self,
        cleaning_steps: Union[str, List[str]] = "all",
    ) -> Dict[str, Any]:
        """
        Clean data using the given set of optional cleaning steps.

        Args:
            cleaning_steps: Either "all" (default, every optional step enabled) or a
                list of step names selecting exactly which optional steps run:
                abbreviation_filtering, element_validation_strict, element_validation_lenient,
                text_normalization, miller_indices, coefficient_expansion_strict,
                coefficient_expansion_lenient. Unicode conversion and
                arithmetic/fraction resolution always run regardless of this
                parameter. element_validation_lenient/coefficient_expansion_lenient
                are weaker companions of element_validation_strict/coefficient_expansion_strict —
                selecting both a step and its lenient companion together yields the
                stricter step's result.

        Returns:
            Dict[str, Any]: Cleaned data based on selected steps.
        """
        steps = self._resolve_cleaning_steps(cleaning_steps)
        self.all_data = self.get_useful_data()
        return self.clean_data_with_selected_steps(steps)

    def get_all_composition_property_pairs(
        self,
        cleaning_steps: Union[str, List[str]] = "all",
        is_return_doi: bool = False,
    ) -> Dict[str, Any]:
        """
        Get all composition-property pairs from cleaned data after applying the
        selected cleaning steps and resolving composition calculations.

        Args:
            cleaning_steps: Either "all" (default, every optional step enabled) or a
                list of step names selecting exactly which optional steps run.
                See `clean_data_with_relevant_compositions` for the full list.
            is_return_doi: If True, returns nested dictionary with DOI as keys.
                If False (default), returns flat composition-property dictionary.

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
        # Clean the data using the specified steps
        cleaned_data = self.clean_data_with_relevant_compositions(
            cleaning_steps=cleaning_steps,
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
