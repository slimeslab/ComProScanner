"""
semantic_evaluator.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 16-04-2025
"""

# Standard library imports
import json
import string
from difflib import SequenceMatcher

# Custom imports
from ...utils.configs import CustomDictionary
from ...utils.error_handler import ValueErrorHandler


class MaterialsDataSemanticEvaluator:
    """
    A class to evaluate materials science data against reference data using semantic matching and exact matching (for specific items).

    Args:
        use_semantic_model (bool, optional): Whether to attempt to use semantic models for similarity (default: True)
        primary_model_name (str, optional): Name of the primary model to use (default: "thellert/physbert_cased")
        fallback_model_name (str, optional): Name of the fallback sentence transformer model (default: "all-mpnet-base-v2")
        similarity_thresholds (dict, optional): Custom thresholds for similarity scoring
    """

    def __init__(
        self,
        use_semantic_model=True,
        primary_model_name="thellert/physbert_cased",
        fallback_model_name="all-mpnet-base-v2",
        similarity_thresholds=None,
    ):
        """
        Initialize the evaluator with optional semantic models.

        Args:
            use_semantic_model (bool, optional): Whether to attempt to use semantic models
            primary_model_name (str, optional): Name of the primary model to use
            fallback_model_name (str, optional): Name of the fallback sentence transformer model
            similarity_thresholds (dict, optional): Custom thresholds for similarity scoring
        """
        self.use_semantic_model = use_semantic_model
        self.primary_model_name = primary_model_name
        self.fallback_model_name = fallback_model_name
        self.physbert_available = False
        self.model_available = False

        # Load models if requested
        if self.use_semantic_model:
            self._load_models()

        # Setting similarity thresholds for different components (default or custom)
        self.default_similarity_thresholds = {
            "composition_overall_match": 0.80,
            "synthesis_overall_match": 0.80,
            "composition_key_match": 0.80,
            "property_unit_match": 0.80,
            "family_match": 0.80,
            "method_match": 0.80,
            "precursors_match": 0.80,
            "characterization_match": 0.80,
            "steps_match": 0.80,
        }
        self.similarity_thresholds = self.default_similarity_thresholds.copy()
        if similarity_thresholds:
            self.similarity_thresholds.update(similarity_thresholds)

        # Default weights for different components
        self.default_weights = {
            "compositions_property_values": 0.3,
            "property_unit": 0.1,
            "family": 0.1,
            "method": 0.1,
            "precursors": 0.15,
            "characterization_techniques": 0.15,
            "steps": 0.1,
        }

    def _load_models(self):
        """
        Attempt to load the semantic models in order of preference.
        First tries to load PhysBERT, then falls back to sentence-transformers.
        """
        # Try to load PhysBERT first
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            self.physbert_tokenizer = AutoTokenizer.from_pretrained(
                self.primary_model_name
            )
            self.physbert_model = AutoModel.from_pretrained(self.primary_model_name)
            self.torch = torch
            self.physbert_available = True
        except Exception as e:
            print(f"PhysBERT model not available: {e}, trying sentence-transformers")
            self.physbert_available = False

            # Try sentence-transformers as backup
            if not self.physbert_available:
                try:
                    from sentence_transformers import SentenceTransformer
                    import torch

                    self.model = SentenceTransformer(self.fallback_model_name)
                    self.torch = torch
                    self.model_available = True
                except Exception as e:
                    print(
                        f"Sentence-transformer model not available: {e}, falling back to sequence matching"
                    )
                    self.model_available = False

    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts.
        Uses PhysBERT if available, falls back to transformer model,
        otherwise uses sequence matching.
        """
        if self.physbert_available:
            # Use PhysBERT model for domain-specific similarity calculation
            inputs1 = self.physbert_tokenizer(
                text1, return_tensors="pt", padding=True, truncation=True
            )
            inputs2 = self.physbert_tokenizer(
                text2, return_tensors="pt", padding=True, truncation=True
            )

            with self.torch.no_grad():
                outputs1 = self.physbert_model(**inputs1)
                outputs2 = self.physbert_model(**inputs2)

            # Use CLS token embeddings for sentence representation
            embedding1 = outputs1.last_hidden_state[:, 0, :]
            embedding2 = outputs2.last_hidden_state[:, 0, :]

            # Normalize embeddings
            embedding1 = embedding1 / embedding1.norm(dim=1, keepdim=True)
            embedding2 = embedding2 / embedding2.norm(dim=1, keepdim=True)

            # Calculate cosine similarity
            similarity = self.torch.nn.functional.cosine_similarity(
                embedding1, embedding2, dim=1
            ).item()

            return similarity

        elif self.model_available:
            # Fall back to sentence-transformer model
            embedding1 = self.model.encode(text1, convert_to_tensor=True)
            embedding2 = self.model.encode(text2, convert_to_tensor=True)
            similarity = self.torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0), embedding2.unsqueeze(0), dim=1
            ).item()

            return similarity
        else:
            # Last resort: sequence matcher fallback
            text1_proc = self._simple_preprocess(text1)
            text2_proc = self._simple_preprocess(text2)
            return SequenceMatcher(None, text1_proc, text2_proc).ratio()

    def _simple_preprocess(self, text):
        """
        Simple text preprocessing without external dependencies

        Args:
            text (str): Text to preprocess

        Returns:
            str: Preprocessed text
        """
        text = str(text).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        common_stops = CustomDictionary.get_common_stops

        words = text.split()
        filtered_words = [word for word in words if word not in common_stops]

        return " ".join(filtered_words)

    def _is_value_in_range(self, ref_val, test_val):
        """
        Check if test_val is within the range specified by ref_val.

        Args:
            ref_val: Reference value, which can be a number or a list [min, max]
            test_val: Test value to check against the reference

        Returns:
            bool: True if the test value matches or falls within the reference range, False otherwise
        """
        # Handle case where either value is None
        if ref_val is None or test_val is None:
            return ref_val == test_val

        # Case 1: ref_val is a list defining a range [min, max]
        if isinstance(ref_val, list) and len(ref_val) == 2:
            try:
                if not isinstance(test_val, (int, float)):
                    test_val = float(test_val)
                return float(ref_val[0]) <= test_val <= float(ref_val[1])
            except (ValueError, TypeError):
                # If conversion fails, do exact comparison
                return ref_val == test_val

        # Case 2: ref_val is not a range, do regular comparison
        try:
            if isinstance(test_val, (int, float)) and isinstance(ref_val, (int, float)):
                return abs(float(ref_val) - float(test_val)) < 1e-6
            else:
                return ref_val == test_val
        except (ValueError, TypeError):
            # If conversion fails, fall back to exact comparison
            return ref_val == test_val

    def _evaluate_composition_data(self, reference_comp, test_comp, weights=None):
        """
        Evaluate composition data section with semantic matching for keys and exact matching for values.

        Args:
            reference_comp (dict): Reference composition data
            test_comp (dict): Test composition data
            weights (dict, optional): Weights for different components

        Returns:
            tuple: (score, details, absolute_metrics, component_weights)
        """
        # Check if weights is None or if sum of weights is not 1.0 (with small tolerance for floating point)
        if weights is None:
            weights = self.default_weights
        else:
            # Check if the sum of weights is approximately 1.0
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 1e-6:  # Allow tiny floating point differences
                print(
                    f"Warning: Provided weights sum to {weight_sum}, not 1.0. Using default weights."
                )
                weights = self.default_weights

        # Initialize component weights to track relative importance for normalization
        component_weights = {
            "property_unit": weights["property_unit"],
            "family": weights["family"],
            "compositions_property_values": weights["compositions_property_values"],
        }

        # Normalize component weights to sum to 0.5 (composition portion)
        total_weight = sum(component_weights.values())
        if total_weight > 0:
            for key in component_weights:
                component_weights[key] = component_weights[key] / total_weight * 0.5

        score = 1.0
        details = {}
        absolute_metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        # Check property_unit (exact match)
        ref_unit = reference_comp.get("property_unit")
        test_unit = test_comp.get("property_unit")
        unit_match = ref_unit == test_unit
        details["property_unit"] = {
            "match": unit_match,
            "reference": ref_unit,
            "test": test_unit,
        }

        unit_weight = component_weights["property_unit"]

        if unit_match and ref_unit is not None:
            absolute_metrics["true_positives"] += 1
        elif ref_unit is not None and test_unit is not None:
            absolute_metrics["false_negatives"] += 1
            absolute_metrics["false_positives"] += 1
            score -= unit_weight
        elif ref_unit is not None:
            absolute_metrics["false_negatives"] += 1
            score -= unit_weight
        elif test_unit is not None:
            absolute_metrics["false_positives"] += 1

        # Check family (exact match)
        ref_family = reference_comp.get("family")
        test_family = test_comp.get("family")
        family_match = ref_family == test_family
        details["family"] = {
            "match": family_match,
            "reference": ref_family,
            "test": test_family,
        }

        family_weight = component_weights["family"]

        if family_match and ref_family is not None:
            absolute_metrics["true_positives"] += 1
        elif ref_family is not None and test_family is not None:
            absolute_metrics["false_negatives"] += 1
            absolute_metrics["false_positives"] += 1
            score -= family_weight
        elif ref_family is not None:
            absolute_metrics["false_negatives"] += 1
            score -= family_weight
        elif test_family is not None:
            absolute_metrics["false_positives"] += 1

        # Check compositions_property_values with semantic matching for keys
        ref_values = reference_comp.get("compositions_property_values", {})
        test_values = test_comp.get("compositions_property_values", {})

        comp_values_weight = component_weights["compositions_property_values"]

        # Case 1: Both empty
        if not ref_values and not test_values:
            details["compositions_property_values"] = {
                "match": True,
                "details": {"message": "Both compositions_property_values are empty"},
            }
            return score, details, absolute_metrics, component_weights

        # Case 2: One empty, one not
        if not ref_values:
            details["compositions_property_values"] = {
                "match": False,
                "details": {
                    "message": "Reference compositions_property_values is empty"
                },
            }
            absolute_metrics["false_positives"] += len(test_values) * 2
            return score, details, absolute_metrics, component_weights

        if not test_values:
            details["compositions_property_values"] = {
                "match": False,
                "details": {"message": "Test compositions_property_values is empty"},
            }
            absolute_metrics["false_negatives"] += len(ref_values) * 2
            score -= comp_values_weight
            return score, details, absolute_metrics, component_weights

        # Case 3: Both non-empty, first find exact matches using sets, then apply similarity matching
        ref_keys_set = set(ref_values.keys())
        test_keys_set = set(test_values.keys())

        # Step 1: Find exact matches using set operations
        common_keys = ref_keys_set.intersection(test_keys_set)
        remaining_ref_keys = ref_keys_set - common_keys
        remaining_test_keys = test_keys_set - common_keys

        # Track key matches and value matches
        key_matches = {}
        value_matches = {}
        pair_matches = []

        # Process exact matches
        for key in common_keys:
            # The key exists in both sets - count as true positive for key
            absolute_metrics["true_positives"] += 1

            key_matches[key] = {
                "matched_with": key,
                "similarity": 1.0,
                "match_type": "exact",
            }

            # Check if values match exactly
            ref_val = ref_values[key]
            test_val = test_values[key]
            value_match = self._is_value_in_range(ref_val, test_val)
            value_matches[key] = value_match

            if value_match:
                # Value matches - count as true positive for value
                absolute_metrics["true_positives"] += 1

                pair_matches.append(
                    {
                        "match": True,
                        "reference": {key: ref_val},
                        "test": {key: test_val},
                        "similarity": 1.0,
                        "match_type": "exact",
                    }
                )
            else:
                # Values don't match - split false positive/negative for value
                absolute_metrics["false_positives"] += 1
                absolute_metrics["false_negatives"] += 1

                pair_matches.append(
                    {
                        "match": False,
                        "reference": {key: ref_val},
                        "test": {key: test_val},
                        "similarity": 1.0,
                        "match_type": "exact_key_different_value",
                    }
                )

        # Step 2: Apply similarity matching for remaining keys
        # Convert remaining keys to lists for indexing
        remaining_ref_keys_list = list(remaining_ref_keys)
        remaining_test_keys_list = list(remaining_test_keys)

        # Calculate similarity matrix between remaining ref keys and test keys
        similarity_matrix = []
        for ref_key in remaining_ref_keys_list:
            row = []
            for test_key in remaining_test_keys_list:
                similarity = self._calculate_text_similarity(ref_key, test_key)
                row.append(similarity)
            similarity_matrix.append(row)

        # Track which keys have been matched already
        matched_ref_idxs = set()
        matched_test_idxs = set()

        # Find high similarity matches (similarity > 0.9)
        for i, ref_key in enumerate(remaining_ref_keys_list):
            if i in matched_ref_idxs:
                continue  # Skip already matched reference keys

            best_match_idx = -1
            best_similarity = 0.9  # Only consider matches with similarity > 0.9

            # Find best match for this reference key
            for j, test_key in enumerate(remaining_test_keys_list):
                if j in matched_test_idxs:
                    continue  # Skip already matched test keys

                if similarity_matrix[i][j] > best_similarity:
                    best_similarity = similarity_matrix[i][j]
                    best_match_idx = j

            # If a high similarity match is found
            if best_match_idx >= 0:
                test_key = remaining_test_keys_list[best_match_idx]
                matched_ref_idxs.add(i)
                matched_test_idxs.add(best_match_idx)

                # Store match information
                key_matches[ref_key] = {
                    "matched_with": test_key,
                    "similarity": best_similarity,
                    "match_type": "high_similarity",
                }

                # For high similarity matches, give partial credit for the key
                absolute_metrics["true_positives"] += 1

                ref_val = ref_values[ref_key]
                test_val = test_values[test_key]
                value_match = self._is_value_in_range(ref_val, test_val)
                value_matches[ref_key] = value_match

                if value_match:
                    # Value matches - count as true positive for value
                    absolute_metrics["true_positives"] += 1

                    pair_matches.append(
                        {
                            "match": False,  # Not a true match because key isn't exact
                            "reference": {ref_key: ref_val},
                            "test": {test_key: test_val},
                            "similarity": best_similarity,
                            "match_type": "high_similarity_same_value",
                        }
                    )
                else:
                    # Value doesn't match - split false positive/negative for value
                    absolute_metrics["false_positives"] += 1
                    absolute_metrics["false_negatives"] += 1

                    pair_matches.append(
                        {
                            "match": False,
                            "reference": {ref_key: ref_val},
                            "test": {test_key: test_val},
                            "similarity": best_similarity,
                            "match_type": "high_similarity_different_value",
                        }
                    )

        # Process unmatched reference keys (truly missing keys)
        unmatched_ref_keys = [
            remaining_ref_keys_list[i]
            for i in range(len(remaining_ref_keys_list))
            if i not in matched_ref_idxs
        ]

        if unmatched_ref_keys:
            # Missing keys - count as false negatives for both key and value components
            absolute_metrics["false_negatives"] += len(unmatched_ref_keys) * 2

            for key in unmatched_ref_keys:
                pair_matches.append(
                    {
                        "match": False,
                        "reference": {key: ref_values[key]},
                        "test": None,
                        "similarity": 0.0,
                        "match_type": "missing",
                    }
                )

        # Process unmatched test keys (truly extra keys)
        unmatched_test_keys = [
            remaining_test_keys_list[i]
            for i in range(len(remaining_test_keys_list))
            if i not in matched_test_idxs
        ]

        if unmatched_test_keys:
            # Extra keys - count as false positives for both key and value
            absolute_metrics["false_positives"] += len(unmatched_test_keys) * 2

            for key in unmatched_test_keys:
                pair_matches.append(
                    {
                        "match": False,
                        "reference": None,
                        "test": {key: test_values[key]},
                        "similarity": 0.0,
                        "match_type": "extra",
                    }
                )

        # Calculate match ratios
        total_ref_keys = len(ref_keys_set)

        # Exact matches are from common_keys
        exact_matches = len(common_keys)

        # High similarity matches
        high_similarity_matches = len(matched_ref_idxs)

        # Key match ratio considers both exact and high similarity matches
        matched_keys_count = exact_matches + high_similarity_matches
        key_match_ratio = (
            matched_keys_count / total_ref_keys if total_ref_keys > 0 else 0
        )

        # Only consider exact matched keys for value match ratio
        value_match_count = sum(1 for k in common_keys if value_matches.get(k, False))
        value_match_ratio = (
            value_match_count / total_ref_keys if total_ref_keys > 0 else 0
        )

        # Pair matches - count only exact key matches with matching values
        pair_match_count = sum(1 for pair in pair_matches if pair.get("match", False))
        pair_match_ratio = (
            pair_match_count / total_ref_keys if total_ref_keys > 0 else 0
        )

        # Overall match calculation
        overall_match_ratio = 0.4 * key_match_ratio + 0.6 * pair_match_ratio
        composition_match = (
            overall_match_ratio > 0.85
        )  # Using 85% threshold for overall match

        # Apply penalty if not a good match
        if not composition_match:
            penalty = comp_values_weight * (1 - overall_match_ratio)
            score -= penalty

        # Build detailed results
        value_details = {
            "key_matches": key_matches,
            "value_matches": value_matches,
            "pair_matches": pair_matches,
            "key_match_ratio": key_match_ratio,
            "value_match_ratio": value_match_ratio,
            "pair_match_ratio": pair_match_ratio,
            "overall_match_ratio": overall_match_ratio,
            "exact_matches": exact_matches,
            "high_similarity_matches": high_similarity_matches,
            "missing_keys": unmatched_ref_keys,
            "extra_keys": unmatched_test_keys,
            "similarity_threshold_used": 0.9,
            "fractional_value": 1.0 / total_ref_keys if total_ref_keys > 0 else 0,
            "total_reference_items": total_ref_keys,
        }

        details["compositions_property_values"] = {
            "match": composition_match,
            "similarity_score": overall_match_ratio,
            "details": value_details,
        }

        return max(0, score), details, absolute_metrics, component_weights

    def _evaluate_synthesis_data(self, reference_synth, test_synth, weights=None):
        """
        Evaluate synthesis data section with semantic understanding.

        Args:
            reference_synth (dict): Reference synthesis data
            test_synth (dict): Test synthesis data
            weights (dict, optional): Weights for different components

        Returns:
            tuple: (score, details, absolute_metrics, component_weights)
        """
        # Check if weights is None or if sum of weights is not 1.0 (with small tolerance for floating point)
        if weights is None:
            weights = self.default_weights
        else:
            # Check if the sum of weights is approximately 1.0
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 1e-6:  # Allow tiny floating point differences
                print(
                    f"Warning: Provided weights sum to {weight_sum}, not 1.0. Using default weights."
                )
                weights = self.default_weights

        # Initialize component weights to track relative importance for normalization
        component_weights = {
            "method": weights["method"],
            "precursors": weights["precursors"],
            "characterization_techniques": weights["characterization_techniques"],
            "steps": weights["steps"],
        }

        # Normalize component weights to sum to 0.5 (synthesis portion)
        total_weight = sum(component_weights.values())
        if total_weight > 0:
            for key in component_weights:
                component_weights[key] = component_weights[key] / total_weight * 0.5

        score = 1.0
        details = {}
        absolute_metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        # Check method (semantic similarity with high threshold)
        ref_method = reference_synth.get("method")
        test_method = test_synth.get("method")

        method_weight = component_weights["method"]

        if ref_method is not None and test_method is not None:
            method_similarity = self._calculate_text_similarity(ref_method, test_method)
            method_match = (
                method_similarity >= 0.9
            )  # High threshold for semantic similarity
            details["method"] = {
                "match": method_match,
                "reference": ref_method,
                "test": test_method,
                "similarity": method_similarity,
            }

            if method_match:
                absolute_metrics["true_positives"] += 1
            else:
                # Split the weight between false positive and false negative
                absolute_metrics["false_negatives"] += 1
                absolute_metrics["false_positives"] += 1
                score -= method_weight
        else:
            method_match = (
                ref_method == test_method
            )  # Both None is a match, one None is not
            details["method"] = {
                "match": method_match,
                "reference": ref_method,
                "test": test_method,
                "similarity": 1.0 if method_match else 0.0,
            }

            if method_match and ref_method is not None:
                absolute_metrics["true_positives"] += 1
            elif ref_method is not None:
                absolute_metrics["false_negatives"] += 1
                score -= method_weight
            elif test_method is not None:
                absolute_metrics["false_positives"] += 1

        # Check precursors using semantic similarity
        ref_precursors = reference_synth.get("precursors", [])
        test_precursors = test_synth.get("precursors", [])

        precursors_weight = component_weights["precursors"]

        # Case 1: Both empty
        if not ref_precursors and not test_precursors:
            details["precursors"] = {
                "match": True,
                "similarity": 1.0,
                "reference": [],
                "test": [],
                "very_high_similarity_matches": [],
                "high_similarity_matches": [],
                "missing_items": [],
                "extra_items": [],
            }
        # Case 2: One empty, one not
        elif not ref_precursors:
            details["precursors"] = {
                "match": False,
                "similarity": 0.0,
                "reference": [],
                "test": test_precursors,
                "very_high_similarity_matches": [],
                "high_similarity_matches": [],
                "missing_items": [],
                "extra_items": test_precursors,
            }

            # All weight goes to false positives
            absolute_metrics["false_positives"] += len(test_precursors) * 2

        elif not test_precursors:
            details["precursors"] = {
                "match": False,
                "similarity": 0.0,
                "reference": ref_precursors,
                "test": [],
                "very_high_similarity_matches": [],
                "high_similarity_matches": [],
                "missing_items": ref_precursors,
                "extra_items": [],
            }

            # All weight goes to false negatives
            absolute_metrics["false_negatives"] += len(ref_precursors) * 2
            score -= precursors_weight

        else:
            # Case 3: Both non-empty, evaluate with item list comparison
            # Define similarity thresholds
            very_high_threshold = 0.90
            high_threshold = 0.70

            # Calculate similarity matrix
            similarity_matrix = []
            for ref_item in ref_precursors:
                row = []
                for test_item in test_precursors:
                    similarity = self._calculate_text_similarity(ref_item, test_item)
                    row.append(similarity)
                similarity_matrix.append(row)

            # Track matches
            very_high_matches = []
            high_matches = []
            matched_ref_idxs = set()
            matched_test_idxs = set()

            # First pass: Find very high similarity matches (>0.90)
            for i, ref_item in enumerate(ref_precursors):
                if i in matched_ref_idxs:
                    continue  # Skip already matched reference items

                best_match_idx = -1
                best_similarity = very_high_threshold

                for j, test_item in enumerate(test_precursors):
                    if j in matched_test_idxs:
                        continue  # Skip already matched test items

                    if similarity_matrix[i][j] >= best_similarity:
                        best_similarity = similarity_matrix[i][j]
                        best_match_idx = j

                if best_match_idx >= 0:
                    test_item = test_precursors[best_match_idx]
                    matched_ref_idxs.add(i)
                    matched_test_idxs.add(best_match_idx)

                    match_info = {
                        "reference": ref_item,
                        "test": test_item,
                        "similarity": best_similarity,
                    }
                    very_high_matches.append(match_info)

                    # Add to metrics - count as true positive
                    absolute_metrics["true_positives"] += 2  # Count as 2 true positives

            # Second pass: Find high similarity matches (>0.70)
            for i, ref_item in enumerate(ref_precursors):
                if i in matched_ref_idxs:
                    continue  # Skip already matched reference items

                best_match_idx = -1
                best_similarity = high_threshold

                for j, test_item in enumerate(test_precursors):
                    if j in matched_test_idxs:
                        continue  # Skip already matched test items

                    if similarity_matrix[i][j] >= best_similarity:
                        best_similarity = similarity_matrix[i][j]
                        best_match_idx = j

                if best_match_idx >= 0:
                    test_item = test_precursors[best_match_idx]
                    matched_ref_idxs.add(i)
                    matched_test_idxs.add(best_match_idx)

                    match_info = {
                        "reference": ref_item,
                        "test": test_item,
                        "similarity": best_similarity,
                    }
                    high_matches.append(match_info)

                    # Only partial credit for high similarity matches
                    absolute_metrics["true_positives"] += 1
                    absolute_metrics["false_positives"] += 1

            # Process unmatched items
            missing_items = [
                ref_precursors[i]
                for i in range(len(ref_precursors))
                if i not in matched_ref_idxs
            ]
            extra_items = [
                test_precursors[i]
                for i in range(len(test_precursors))
                if i not in matched_test_idxs
            ]

            # Add metrics for missing items
            if missing_items:
                absolute_metrics["false_negatives"] += len(missing_items) * 2

            # Add metrics for extra items
            if extra_items:
                absolute_metrics["false_positives"] += len(extra_items) * 2

            # Calculate overall similarity
            total_matches = len(very_high_matches) + 0.7 * len(high_matches)
            similarity_score = (
                total_matches / len(ref_precursors) if ref_precursors else 0.0
            )
            precursors_match = similarity_score >= 0.80  # Threshold for overall match

            # Apply score penalty if not a good match
            if not precursors_match:
                penalty = precursors_weight * (1 - similarity_score)
                score -= penalty

            # Store details
            details["precursors"] = {
                "match": precursors_match,
                "similarity": similarity_score,
                "reference": ref_precursors,
                "test": test_precursors,
                "very_high_similarity_matches": very_high_matches,
                "high_similarity_matches": high_matches,
                "missing_items": missing_items,
                "extra_items": extra_items,
                "fractional_value": 1.0 / len(ref_precursors) if ref_precursors else 0,
            }

        # Check characterization_techniques using semantic similarity
        ref_tech = reference_synth.get("characterization_techniques", [])
        test_tech = test_synth.get("characterization_techniques", [])

        tech_weight = component_weights["characterization_techniques"]

        # Similar approach as precursors, but for characterization techniques
        # Case 1: Both empty
        if not ref_tech and not test_tech:
            details["characterization_techniques"] = {
                "match": True,
                "similarity": 1.0,
                "reference": [],
                "test": [],
                "very_high_similarity_matches": [],
                "high_similarity_matches": [],
                "missing_items": [],
                "extra_items": [],
            }
        # Case 2: One empty, one not
        elif not ref_tech:
            details["characterization_techniques"] = {
                "match": False,
                "similarity": 0.0,
                "reference": [],
                "test": test_tech,
                "very_high_similarity_matches": [],
                "high_similarity_matches": [],
                "missing_items": [],
                "extra_items": test_tech,
            }

            # All weight goes to false positives
            absolute_metrics["false_positives"] += len(test_tech) * 2

        elif not test_tech:
            details["characterization_techniques"] = {
                "match": False,
                "similarity": 0.0,
                "reference": ref_tech,
                "test": [],
                "very_high_similarity_matches": [],
                "high_similarity_matches": [],
                "missing_items": ref_tech,
                "extra_items": [],
            }

            # All weight goes to false negatives
            absolute_metrics["false_negatives"] += len(ref_tech) * 2
            score -= tech_weight

        else:
            # Case 3: Both non-empty, evaluate with item list comparison
            # Define similarity thresholds
            very_high_threshold = 0.90
            high_threshold = 0.70

            # Calculate similarity matrix
            similarity_matrix = []
            for ref_item in ref_tech:
                row = []
                for test_item in test_tech:
                    similarity = self._calculate_text_similarity(ref_item, test_item)
                    row.append(similarity)
                similarity_matrix.append(row)

            # Track matches
            very_high_matches = []
            high_matches = []
            matched_ref_idxs = set()
            matched_test_idxs = set()

            # First pass: Find very high similarity matches (>0.90)
            for i, ref_item in enumerate(ref_tech):
                if i in matched_ref_idxs:
                    continue  # Skip already matched reference items

                best_match_idx = -1
                best_similarity = very_high_threshold

                for j, test_item in enumerate(test_tech):
                    if j in matched_test_idxs:
                        continue  # Skip already matched test items

                    if similarity_matrix[i][j] >= best_similarity:
                        best_similarity = similarity_matrix[i][j]
                        best_match_idx = j

                if best_match_idx >= 0:
                    test_item = test_tech[best_match_idx]
                    matched_ref_idxs.add(i)
                    matched_test_idxs.add(best_match_idx)

                    match_info = {
                        "reference": ref_item,
                        "test": test_item,
                        "similarity": best_similarity,
                    }
                    very_high_matches.append(match_info)

                    # Add to metrics - full weight per matched item
                    absolute_metrics["true_positives"] += 2

            # Second pass: Find high similarity matches (>0.70)
            for i, ref_item in enumerate(ref_tech):
                if i in matched_ref_idxs:
                    continue  # Skip already matched reference items

                best_match_idx = -1
                best_similarity = high_threshold

                for j, test_item in enumerate(test_tech):
                    if j in matched_test_idxs:
                        continue  # Skip already matched test items

                    if similarity_matrix[i][j] >= best_similarity:
                        best_similarity = similarity_matrix[i][j]
                        best_match_idx = j

                if best_match_idx >= 0:
                    test_item = test_tech[best_match_idx]
                    matched_ref_idxs.add(i)
                    matched_test_idxs.add(best_match_idx)

                    match_info = {
                        "reference": ref_item,
                        "test": test_item,
                        "similarity": best_similarity,
                    }
                    high_matches.append(match_info)

                    # Partial match - count as some true positive and some false
                    absolute_metrics["true_positives"] += 1
                    absolute_metrics["false_positives"] += 1

            # Process unmatched items
            missing_items = [
                ref_tech[i] for i in range(len(ref_tech)) if i not in matched_ref_idxs
            ]
            extra_items = [
                test_tech[i]
                for i in range(len(test_tech))
                if i not in matched_test_idxs
            ]

            # Add metrics for missing items
            if missing_items:
                absolute_metrics["false_negatives"] += len(missing_items) * 2

            # Add metrics for extra items
            if extra_items:
                absolute_metrics["false_positives"] += len(extra_items) * 2

            # Calculate overall similarity
            total_matches = len(very_high_matches) + 0.7 * len(high_matches)
            similarity_score = total_matches / len(ref_tech) if ref_tech else 0.0
            tech_match = similarity_score >= 0.80  # Threshold for overall match

            # Apply score penalty if not a good match
            if not tech_match:
                penalty = tech_weight * (1 - similarity_score)
                score -= penalty

            # Store details
            details["characterization_techniques"] = {
                "match": tech_match,
                "similarity": similarity_score,
                "reference": ref_tech,
                "test": test_tech,
                "very_high_similarity_matches": very_high_matches,
                "high_similarity_matches": high_matches,
                "missing_items": missing_items,
                "extra_items": extra_items,
                "fractional_value": 1.0 / len(ref_tech) if ref_tech else 0,
            }

        # Check steps using semantic paragraph comparison
        ref_steps = reference_synth.get("steps", [])
        test_steps = test_synth.get("steps", [])

        steps_weight = component_weights["steps"]

        try:
            # Case 1: Both empty
            if not ref_steps and not test_steps:
                steps_details = {
                    "steps_match": True,
                    "reference_steps": [],
                    "test_steps": [],
                }
            # Case 2: One empty, one not
            elif not ref_steps:
                steps_details = {
                    "steps_match": False,
                    "message": "Reference steps are empty",
                    "reference_steps": [],
                    "test_steps": test_steps,
                }

                # All weight goes to false positives
                absolute_metrics["false_positives"] += 2  # Count as 2 false positives

            elif not test_steps:
                steps_details = {
                    "steps_match": False,
                    "message": "Test steps are empty",
                    "reference_steps": ref_steps,
                    "test_steps": [],
                }

                # All weight goes to false negatives
                absolute_metrics["false_negatives"] += 2  # Count as 2 false negatives
                score -= steps_weight

            else:
                # Case 3: Both non-empty, use semantic paragraph comparison
                ref_paragraph = " ".join(ref_steps)
                test_paragraph = " ".join(test_steps)
                paragraph_similarity = self._calculate_text_similarity(
                    ref_paragraph, test_paragraph
                )

                steps_match = (
                    paragraph_similarity >= self.similarity_thresholds["steps_match"]
                )

                steps_details = {
                    "reference_steps": ref_steps,
                    "test_steps": test_steps,
                    "paragraph_similarity": paragraph_similarity,
                    "steps_match": steps_match,
                    "paragraph_comparison": {
                        "similarity_score": paragraph_similarity,
                        "high_similarity": paragraph_similarity
                        >= self.similarity_thresholds["steps_match"],
                        "method": (
                            "semantic_model"
                            if self.model_available
                            else "sequence_matcher"
                        ),
                    },
                }

                # Add metrics based on similarity
                if steps_match:
                    # High similarity - all weight to true positives
                    absolute_metrics["true_positives"] += 2
                else:
                    # Low similarity - split between false positive and negative
                    absolute_metrics["false_positives"] += 1
                    absolute_metrics["false_negatives"] += 1
                    score -= steps_weight  # Full penalty
        except Exception as e:
            # Handle any errors in steps evaluation
            print(f"Error in step evaluation: {str(e)}")
            steps_details = {
                "steps_match": False,
                "error": str(e),
                "reference_steps": ref_steps,
                "test_steps": test_steps,
            }

            # Count as false positive and negative
            absolute_metrics["false_positives"] += 1
            absolute_metrics["false_negatives"] += 1
            score -= steps_weight

        details["steps"] = steps_details

        return max(0, score), details, absolute_metrics, component_weights

    def _count_all_items(self, item):
        """Count all individual items in a data entry"""
        count = 0

        # Count composition items
        comp_data = item.get("composition_data", {})
        if comp_data is not None:  # Add this check to handle None values
            if "property_unit" in comp_data:
                count += 1
            if "family" in comp_data:
                count += 1
            count += (
                len(comp_data.get("compositions_property_values", {})) * 2
            )  # Keys and values

        # Count synthesis items
        synth_data = item.get("synthesis_data", {})
        if synth_data is not None:  # Add this check to handle None values
            if "method" in synth_data:
                count += 1
            count += len(synth_data.get("precursors", []))
            count += len(synth_data.get("characterization_techniques", []))
            if synth_data.get("steps"):
                count += 2  # Count steps as 2 items (presence + content)

        return count

    def evaluate(
        self,
        ground_truth_file=None,
        test_data_file=None,
        used_used_agent_model_name=None,
        weights=None,
        output_file="detailed_evaluation.json",
        is_synthesis_evaluation=True,
    ):
        """
        Evaluate materials science data using normalized weights to ensure fair comparison
        across DOIs with different numbers of items.

        Args:
            ground_truth_file (str): Path to the ground truth JSON file
            test_data_file (str): Path to the test data JSON file
            used_used_agent_model_name (str): Name of the agent model used in data extraction
            weights (dict, optional): Custom weights for different components
            output_file (str, optional): Path to save the detailed evaluation results
            is_synthesis_evaluation (bool, optional): Whether to evaluate synthesis data

        Returns:
            dict: Evaluation results with scores and details including F1 metrics
        """
        if not ground_truth_file or not test_data_file:
            raise ValueErrorHandler(
                "Both ground truth and test data files are required"
            )

        if not used_used_agent_model_name:
            raise ValueErrorHandler("Used agent model name is required")

        if weights is None:
            weights = self.default_weights

        # Load the JSON files
        with open(ground_truth_file, "r", encoding="utf-8") as gt_file:
            reference_data_full = json.load(gt_file)

        with open(test_data_file, "r", encoding="utf-8") as test_file:
            test_data_full = json.load(test_file)

        # Initialize results
        accumulated_results = {
            "used_used_agent_model_name": used_used_agent_model_name,
            "overall_accuracy": 0.0,
            "overall_composition_accuracy": 0.0,
            "overall_synthesis_accuracy": 0.0,
            "total_items": 0,
            "absolute_classification_metrics": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "normalized_classification_metrics": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "item_results": {},
        }

        all_dois = set(reference_data_full.keys()).union(set(test_data_full.keys()))
        total_dois = len(all_dois)  # Total number of unique DOIs

        if total_dois == 0:
            # No data to evaluate
            print("No DOIs found in either reference or test data")
            return accumulated_results

        accumulated_results["total_items"] = total_dois

        # Track scores for calculating overall accuracy
        total_scores = 0.0
        total_composition_score = 0.0
        total_synthesis_score = 0.0

        # Process all DOIs
        for doi in all_dois:
            # Initialize DOI-specific metrics
            item_results = {
                "field_scores": {},
                "absolute_classification_metrics": {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                },
                "normalized_classification_metrics": {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                },
                "details": {},
            }

            # Case 1: DOI exists in both datasets
            if doi in reference_data_full and doi in test_data_full:
                ref_item = reference_data_full[doi]
                test_item = test_data_full[doi]

                # Evaluate composition data
                comp_score, comp_details, comp_abs_metrics, comp_weights = (
                    self._evaluate_composition_data(
                        ref_item.get("composition_data", {}),
                        test_item.get("composition_data", {}),
                        weights,
                    )
                )

                item_results["field_scores"]["composition_data"] = comp_score
                item_results["details"]["composition_data"] = comp_details
                total_composition_score += comp_score

                # Add composition metrics to article metrics
                for metric in ["true_positives", "false_positives", "false_negatives"]:
                    item_results["absolute_classification_metrics"][
                        metric
                    ] += comp_abs_metrics[metric]

                # Evaluate synthesis data if requested
                if is_synthesis_evaluation:
                    synth_score, synth_details, synth_abs_metrics, synth_weights = (
                        self._evaluate_synthesis_data(
                            ref_item.get("synthesis_data", {}),
                            test_item.get("synthesis_data", {}),
                            weights,
                        )
                    )

                    item_results["field_scores"]["synthesis_data"] = synth_score
                    item_results["details"]["synthesis_data"] = synth_details
                    total_synthesis_score += synth_score

                    # Add synthesis metrics to article metrics
                    for metric in [
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                    ]:
                        item_results["absolute_classification_metrics"][
                            metric
                        ] += synth_abs_metrics[metric]

                    # Calculate overall score as weighted average
                    overall_score = 0.5 * comp_score + 0.5 * synth_score
                else:
                    # If only composition evaluation, that's the overall score
                    overall_score = comp_score

                total_scores += overall_score
                item_results["overall_score"] = overall_score
                item_results["overall_match"] = (
                    overall_score >= 0.8
                )  # 80% threshold for match

                # Calculate normalized classification metrics where each DOI contributes exactly 1.0 to the total
                tp_abs = item_results["absolute_classification_metrics"][
                    "true_positives"
                ]
                fp_abs = item_results["absolute_classification_metrics"][
                    "false_positives"
                ]
                fn_abs = item_results["absolute_classification_metrics"][
                    "false_negatives"
                ]

                # Total items for this DOI
                total_items = tp_abs + fp_abs + fn_abs

                if total_items > 0:
                    # Normalize to 1.0 contribution per DOI
                    item_results["normalized_classification_metrics"][
                        "true_positives"
                    ] = (tp_abs / total_items)
                    item_results["normalized_classification_metrics"][
                        "false_positives"
                    ] = (fp_abs / total_items)
                    item_results["normalized_classification_metrics"][
                        "false_negatives"
                    ] = (fn_abs / total_items)

                # Calculate absolute metrics
                if tp_abs + fp_abs > 0:
                    item_results["absolute_classification_metrics"]["precision"] = (
                        tp_abs / (tp_abs + fp_abs)
                    )
                if tp_abs + fn_abs > 0:
                    item_results["absolute_classification_metrics"]["recall"] = (
                        tp_abs / (tp_abs + fn_abs)
                    )

                precision_abs = item_results["absolute_classification_metrics"][
                    "precision"
                ]
                recall_abs = item_results["absolute_classification_metrics"]["recall"]

                if precision_abs + recall_abs > 0:
                    item_results["absolute_classification_metrics"]["f1_score"] = (
                        2 * precision_abs * recall_abs / (precision_abs + recall_abs)
                    )

                # Calculate normalized metrics
                tp_norm = item_results["normalized_classification_metrics"][
                    "true_positives"
                ]
                fp_norm = item_results["normalized_classification_metrics"][
                    "false_positives"
                ]
                fn_norm = item_results["normalized_classification_metrics"][
                    "false_negatives"
                ]

                if tp_norm + fp_norm > 0:
                    item_results["normalized_classification_metrics"]["precision"] = (
                        tp_norm / (tp_norm + fp_norm)
                    )
                if tp_norm + fn_norm > 0:
                    item_results["normalized_classification_metrics"]["recall"] = (
                        tp_norm / (tp_norm + fn_norm)
                    )

                precision_norm = item_results["normalized_classification_metrics"][
                    "precision"
                ]
                recall_norm = item_results["normalized_classification_metrics"][
                    "recall"
                ]

                if precision_norm + recall_norm > 0:
                    item_results["normalized_classification_metrics"]["f1_score"] = (
                        2
                        * precision_norm
                        * recall_norm
                        / (precision_norm + recall_norm)
                    )

            # Case 2: DOI exists in reference but not in test (missing)
            elif doi in reference_data_full:
                ref_item = reference_data_full[doi]
                item_count = self._count_all_items(ref_item)

                # All items are false negatives
                item_results["absolute_classification_metrics"][
                    "false_negatives"
                ] = item_count
                item_results["normalized_classification_metrics"][
                    "false_negatives"
                ] = 1.0  # Full DOI weight
                item_results["overall_score"] = 0.0
                item_results["overall_match"] = False

                # Zero contribution to total scores
                if is_synthesis_evaluation:
                    item_results["field_scores"]["composition_data"] = 0.0
                    item_results["field_scores"]["synthesis_data"] = 0.0
                else:
                    item_results["field_scores"]["composition_data"] = 0.0

            # Case 3: DOI exists in test but not in reference (extra)
            elif doi in test_data_full:
                test_item = test_data_full[doi]
                item_count = self._count_all_items(test_item)

                # All items are false positives
                item_results["absolute_classification_metrics"][
                    "false_positives"
                ] = item_count
                item_results["normalized_classification_metrics"][
                    "false_positives"
                ] = 1.0  # Full DOI weight
                item_results["overall_score"] = 0.0
                item_results["overall_match"] = False

                # No contribution to total scores

            # Add this DOI's results to the accumulated results
            accumulated_results["item_results"][doi] = item_results

            # Add to accumulated metrics
            for metric in ["true_positives", "false_positives", "false_negatives"]:
                accumulated_results["absolute_classification_metrics"][
                    metric
                ] += item_results["absolute_classification_metrics"][metric]
                accumulated_results["normalized_classification_metrics"][
                    metric
                ] += item_results["normalized_classification_metrics"][metric]

        # Calculate overall metrics
        tp_abs = accumulated_results["absolute_classification_metrics"][
            "true_positives"
        ]
        fp_abs = accumulated_results["absolute_classification_metrics"][
            "false_positives"
        ]
        fn_abs = accumulated_results["absolute_classification_metrics"][
            "false_negatives"
        ]

        if tp_abs + fp_abs > 0:
            accumulated_results["absolute_classification_metrics"]["precision"] = (
                tp_abs / (tp_abs + fp_abs)
            )
        if tp_abs + fn_abs > 0:
            accumulated_results["absolute_classification_metrics"]["recall"] = (
                tp_abs / (tp_abs + fn_abs)
            )

        precision_abs = accumulated_results["absolute_classification_metrics"][
            "precision"
        ]
        recall_abs = accumulated_results["absolute_classification_metrics"]["recall"]

        if precision_abs + recall_abs > 0:
            accumulated_results["absolute_classification_metrics"]["f1_score"] = (
                2 * precision_abs * recall_abs / (precision_abs + recall_abs)
            )

        # Calculate normalized metrics
        tp_norm = accumulated_results["normalized_classification_metrics"][
            "true_positives"
        ]
        fp_norm = accumulated_results["normalized_classification_metrics"][
            "false_positives"
        ]
        fn_norm = accumulated_results["normalized_classification_metrics"][
            "false_negatives"
        ]

        if tp_norm + fp_norm > 0:
            accumulated_results["normalized_classification_metrics"]["precision"] = (
                tp_norm / (tp_norm + fp_norm)
            )
        if tp_norm + fn_norm > 0:
            accumulated_results["normalized_classification_metrics"]["recall"] = (
                tp_norm / (tp_norm + fn_norm)
            )

        precision_norm = accumulated_results["normalized_classification_metrics"][
            "precision"
        ]
        recall_norm = accumulated_results["normalized_classification_metrics"]["recall"]

        if precision_norm + recall_norm > 0:
            accumulated_results["normalized_classification_metrics"]["f1_score"] = (
                2 * precision_norm * recall_norm / (precision_norm + recall_norm)
            )

        # Calculate overall accuracy
        if total_dois > 0:
            accumulated_results["overall_accuracy"] = total_scores / total_dois
            accumulated_results["overall_composition_accuracy"] = (
                total_composition_score / total_dois
            )
            if is_synthesis_evaluation:
                accumulated_results["overall_synthesis_accuracy"] = (
                    total_synthesis_score / total_dois
                )

        # Save results to file
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(accumulated_results, file, indent=2)

        return accumulated_results
