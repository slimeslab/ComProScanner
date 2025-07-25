"""
eval_flow.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 20-04-2025
"""

# Standard library imports
from typing import Dict, Optional, Any, List
import json
import os
import time
from pathlib import Path

# Third party imports
from tqdm import tqdm
from pydantic import BaseModel, ConfigDict
from crewai.flow.flow import Flow, listen, start, router, or_
from crewai import LLM

from comproscanner.utils.error_handler import ValueErrorHandler
from comproscanner.utils.logger import setup_logger
from .crews.composition_evaluation_crew.composition_evaluation_crew import (
    CompositionEvaluationCrew,
)
from .crews.synthesis_evaluation_crew.synthesis_evaluation_crew import (
    SynthesisEvaluationCrew,
)

# Logger Configuration
logger = setup_logger("comproscanner.log", module_name="eval_flow")


class AgentEvaluationState(BaseModel):
    """State model for the AgentMaterialsEvaluationFlow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input files and paths
    ground_truth_file: str = ""
    test_data_file: str = ""
    output_file: str = "agentic_evaluation_result.json"

    # Evaluation parameters
    extraction_agent_model_name: str = "gpt-4o-mini"
    is_synthesis_evaluation: bool = True

    # Data storage
    ground_truth_data: Dict = {}
    test_data: Dict = {}
    existing_results: Dict = {}

    # LLM configuration
    llm: Optional[LLM] = None

    # Weights for evaluation components
    weights: Dict[str, float] = {}

    # Results storage
    evaluation_details: Dict = {}
    item_results: Dict = {}
    combined_evaluation_results: Dict = {}

    # Progress tracking
    processed_count: int = 0
    total_count: int = 0
    remaining_dois: List[str] = []


class MaterialsDataAgenticEvaluatorFlow(Flow[AgentEvaluationState]):
    """
    CrewAI Flow for evaluating materials data extraction using AI agents that make
    binary (yes/no) decisions about whether items match, rather than using semantic
    similarity or exact matching.

    Args:
        ground_truth_file (str): Path to the ground truth JSON file
        test_data_file (str): Path to the test data JSON file
        output_file (str, optional): Path to save the evaluation results. Default: "agentic_evaluation_result.json"
        extraction_agent_model_name (str, optional): Name of the agent model used in data extraction (default: "gpt-4o-mini")
        is_synthesis_evaluation (bool, optional): Whether to evaluate synthesis data. Default: True
        weights (Dict[str, float], optional): Custom weights for different components. Default: None
        llm (LLM, optional): LLM instance for the agents. Default: None

    Returns:
        Dict: Detailed evaluation results with binary match decisions and metrics
    """

    def __init__(
        self,
        ground_truth_file: str = None,
        test_data_file: str = None,
        extraction_agent_model_name: str = None,
        output_file: str = "agentic_evaluation_result.json",
        is_synthesis_evaluation: bool = True,
        weights: Dict[str, float] = None,
        llm: Optional[LLM] = None,
    ):
        super().__init__()

        # Validate required inputs
        if not ground_truth_file:
            raise ValueErrorHandler("Ground truth file path is required")
        if not test_data_file:
            raise ValueErrorHandler("Test data file path is required")
        if not extraction_agent_model_name:
            raise ValueErrorHandler("Used agent model name is required")

        # Validate file existence
        if not os.path.exists(ground_truth_file):
            raise ValueErrorHandler(f"Ground truth file not found: {ground_truth_file}")
        if not os.path.exists(test_data_file):
            raise ValueErrorHandler(f"Test data file not found: {test_data_file}")

        # Set state parameters
        self.state.ground_truth_file = ground_truth_file
        self.state.test_data_file = test_data_file
        self.state.output_file = output_file
        self.state.extraction_agent_model_name = extraction_agent_model_name
        self.state.is_synthesis_evaluation = is_synthesis_evaluation
        self.state.llm = llm

        # Set up weights with defaults if not provided
        default_weights = {
            "compositions_property_values": 0.3,
            "property_unit": 0.1,
            "family": 0.1,
            "method": 0.1,
            "precursors": 0.15,
            "characterization_techniques": 0.15,
            "steps": 0.1,
        }
        self.state.weights = default_weights.copy()
        if weights:
            self.state.weights.update(weights)

    def _calculate_tp_fp_fn(self, details, section):
        """
        Calculate true positives, false positives, and false negatives for a given section.

        Args:
            details (dict): The details section of evaluation results
            section (str): The section to calculate metrics for (composition_data or synthesis_data)

        Returns:
            dict: Dictionary with true_positives, false_positives, and false_negatives
        """
        metrics = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}

        if section not in details or not details[section]:
            return metrics

        section_data = details[section]

        # Process composition data
        if section == "composition_data":
            # Property unit
            if "property_unit" in section_data:
                match_value = section_data["property_unit"].get("match_value", 0)
                if match_value == 1:
                    metrics["true_positives"] += 1
                else:
                    # Count as false positive if test exists, false negative if reference exists
                    if section_data["property_unit"].get("reference") is not None:
                        metrics["false_negatives"] += 1
                    if section_data["property_unit"].get("test") is not None:
                        metrics["false_positives"] += 1

            # Family
            if "family" in section_data:
                match_value = section_data["family"].get("match_value", 0)
                if match_value == 1:
                    metrics["true_positives"] += 1
                else:
                    # Count as false positive if test exists, false negative if reference exists
                    if section_data["family"].get("reference") is not None:
                        metrics["false_negatives"] += 1
                    if section_data["family"].get("test") is not None:
                        metrics["false_positives"] += 1

            # Compositions property values
            if "compositions_property_values" in section_data:
                cpv = section_data["compositions_property_values"]

                # Process key matches
                if "key_matches" in cpv:
                    for key_match in cpv["key_matches"]:
                        if key_match.get("match_value", 0) == 1:
                            metrics["true_positives"] += 1
                        else:
                            if key_match.get("reference_key"):
                                metrics["false_negatives"] += 1
                            if key_match.get("test_key"):
                                metrics["false_positives"] += 1

                # Process value matches
                if "value_matches" in cpv:
                    for value_match in cpv["value_matches"]:
                        if value_match.get("match_value", 0) == 1:
                            metrics["true_positives"] += 1
                        else:
                            if value_match.get("reference_value") is not None:
                                metrics["false_negatives"] += 1
                            if value_match.get("test_value") is not None:
                                metrics["false_positives"] += 1

                # Process missing keys (count as false negatives)
                metrics["false_negatives"] += len(cpv.get("missing_keys", []))

                # Process extra keys (count as false positives)
                metrics["false_positives"] += len(cpv.get("extra_keys", []))

        # Process synthesis data
        elif section == "synthesis_data":
            # Method
            if "method" in section_data:
                match_value = section_data["method"].get("match_value", 0)
                if match_value == 1:
                    metrics["true_positives"] += 1
                else:
                    # Count as false positive if test exists, false negative if reference exists
                    if section_data["method"].get("reference") is not None:
                        metrics["false_negatives"] += 1
                    if section_data["method"].get("test") is not None:
                        metrics["false_positives"] += 1

            # Precursors
            if "precursors" in section_data:
                pre = section_data["precursors"]

                # Process matches
                if "matches" in pre:
                    for match in pre["matches"]:
                        if match.get("match_value", 0) == 1:
                            metrics["true_positives"] += 1
                        else:
                            if match.get("reference_item") is not None:
                                metrics["false_negatives"] += 1
                            if match.get("test_item") is not None:
                                metrics["false_positives"] += 1

                # Count missing items as false negatives
                metrics["false_negatives"] += len(pre.get("missing_items", []))

                # Count extra items as false positives
                metrics["false_positives"] += len(pre.get("extra_items", []))

            # Characterization techniques
            if "characterization_techniques" in section_data:
                tech = section_data["characterization_techniques"]

                # Process matches
                if "matches" in tech:
                    for match in tech["matches"]:
                        if match.get("match_value", 0) == 1:
                            metrics["true_positives"] += 1
                        else:
                            if match.get("reference_item") is not None:
                                metrics["false_negatives"] += 1
                            if match.get("test_item") is not None:
                                metrics["false_positives"] += 1

                # Count missing items as false negatives
                metrics["false_negatives"] += len(tech.get("missing_items", []))

                # Count extra items as false positives
                metrics["false_positives"] += len(tech.get("extra_items", []))

            # Steps
            if "steps" in section_data:
                steps_match = section_data["steps"].get("match_value", 0)

                # If steps_match is a float between 0 and 1
                if isinstance(steps_match, float):
                    if steps_match > 0.5:  # Partial match above threshold
                        metrics["true_positives"] += 1
                    else:
                        if section_data["steps"].get("reference_steps"):
                            metrics["false_negatives"] += 1
                        if section_data["steps"].get("test_steps"):
                            metrics["false_positives"] += 1
                # If steps_match is binary (0 or 1)
                elif steps_match == 1:
                    metrics["true_positives"] += 1
                else:
                    if section_data["steps"].get("reference_steps"):
                        metrics["false_negatives"] += 1
                    if section_data["steps"].get("test_steps"):
                        metrics["false_positives"] += 1

        return metrics

    def _calculate_classification_metrics(self, metrics_dict):
        """
        Calculate precision, recall, and F1 score from classification metrics.

        Args:
            metrics_dict (dict): Dictionary containing true_positives, false_positives, and false_negatives

        Returns:
            None: Updates the metrics_dict in place with precision, recall, and f1_score
        """
        tp = metrics_dict.get("true_positives", 0)
        fp = metrics_dict.get("false_positives", 0)
        fn = metrics_dict.get("false_negatives", 0)

        # Calculate precision
        if tp + fp > 0:
            metrics_dict["precision"] = tp / (tp + fp)
        else:
            metrics_dict["precision"] = 0.0

        # Calculate recall
        if tp + fn > 0:
            metrics_dict["recall"] = tp / (tp + fn)
        else:
            metrics_dict["recall"] = 0.0

        # Calculate F1 score
        precision = metrics_dict["precision"]
        recall = metrics_dict["recall"]

        if precision + recall > 0:
            metrics_dict["f1_score"] = 2 * precision * recall / (precision + recall)
        else:
            metrics_dict["f1_score"] = 0.0

    def _calculate_score(self, details, section, weights):
        """
        Calculate a weighted score for a section based on matches.

        Args:
            details (dict): The details section of evaluation results
            section (str): The section to calculate score for
            weights (dict): Weights for different components

        Returns:
            float: Score between 0.0 and 1.0
        """
        score = 1.0

        if section not in details:
            return 0.0

        section_data = details[section]

        if section == "composition_data":
            # Property unit
            if (
                "property_unit" in section_data
                and section_data["property_unit"]["match_value"] == 0
            ):
                score -= weights["property_unit"]

            # Family
            if "family" in section_data and section_data["family"]["match_value"] == 0:
                score -= weights["family"]

            # Compositions property values - calculate based on total_match and total_ground_truth_keys
            if "compositions_property_values" in section_data:
                cpv = section_data["compositions_property_values"]
                total_match = cpv.get("total_match", 0)
                total_ground_truth_keys = cpv.get("total_ground_truth_keys", 0)

                # Calculate match ratio if there are test keys
                if total_ground_truth_keys > 0:
                    match_ratio = total_match / total_ground_truth_keys
                    # Apply penalty proportional to match ratio
                    if match_ratio < 1.0:
                        score -= weights["compositions_property_values"] * (
                            1 - match_ratio
                        )
                # If no match info or no test keys, apply full penalty
                elif not total_match and not total_ground_truth_keys:
                    score -= weights["compositions_property_values"]

        elif section == "synthesis_data":
            # Method
            if "method" in section_data and section_data["method"]["match_value"] == 0:
                score -= weights["method"]

            # Precursors - calculate based on total_match and total_ground_truth_items
            if "precursors" in section_data:
                pre = section_data["precursors"]
                total_match = pre.get("total_match", 0)
                total_ground_truth_items = pre.get("total_ground_truth_items", 0)

                # Calculate match ratio if there are test items
                if total_ground_truth_items > 0:
                    match_ratio = total_match / total_ground_truth_items
                    # Apply penalty proportional to match ratio
                    if match_ratio < 1.0:
                        score -= weights["precursors"] * (1 - match_ratio)
                # If no match info or no test items, apply full penalty
                elif not total_match and not total_ground_truth_items:
                    score -= weights["precursors"]

            # Characterization techniques - calculate based on total_match and total_ground_truth_items
            if "characterization_techniques" in section_data:
                tech = section_data["characterization_techniques"]
                total_match = tech.get("total_match", 0)
                total_ground_truth_items = tech.get("total_ground_truth_items", 0)

                # Calculate match ratio if there are test items
                if total_ground_truth_items > 0:
                    match_ratio = total_match / total_ground_truth_items
                    # Apply penalty proportional to match ratio
                    if match_ratio < 1.0:
                        score -= weights["characterization_techniques"] * (
                            1 - match_ratio
                        )
                # If no match info or no test items, apply full penalty
                elif not total_match and not total_ground_truth_items:
                    score -= weights["characterization_techniques"]

            # Steps - now using float value between 0 and 1
            if "steps" in section_data:
                steps_match = section_data["steps"]["match_value"]
                # If steps_match is a float, use it directly
                if isinstance(steps_match, float):
                    if steps_match < 1.0:
                        score -= weights["steps"] * (1 - steps_match)
                # If it's binary (0 or 1)
                elif steps_match == 0:
                    score -= weights["steps"]

        return max(0.0, score)

    def _load_existing_results(self):
        """
        Load existing results from the output file if it exists.

        Returns:
            Dict: Existing evaluation results or empty dict if file doesn't exist
        """
        if os.path.exists(self.state.output_file):
            try:
                with open(self.state.output_file, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                logger.info(f"Loaded existing results from {self.state.output_file}")
                return existing_results
            except Exception as e:
                logger.warning(f"Error loading existing results: {str(e)}")
                return {}
        return {}

    def _update_results_file(self, combined_results):
        """
        Update the results file with the latest evaluation results.

        Args:
            combined_results (dict): The current combined results to write

        Returns:
            bool: True if successful, False otherwise
        """
        output_path = Path(self.state.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(combined_results, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to write evaluation results: {str(e)}")
            return False

    def _normalize_weights(self):
        """
        Normalize weights for each category (composition and synthesis) to sum to 0.5.
        This ensures that the weights are balanced while calculating normalized classification scores.
        """
        composition_weights = {
            "property_unit": self.state.weights["property_unit"],
            "family": self.state.weights["family"],
            "compositions_property_values": self.state.weights[
                "compositions_property_values"
            ],
        }

        synthesis_weights = {
            "method": self.state.weights["method"],
            "precursors": self.state.weights["precursors"],
            "characterization_techniques": self.state.weights[
                "characterization_techniques"
            ],
            "steps": self.state.weights["steps"],
        }

        # Calculate total weights for each category
        comp_total = sum(composition_weights.values())
        synth_total = sum(synthesis_weights.values())

        # Normalize weights to sum to 0.5 for each category
        normalized_weights = {}
        for key, value in composition_weights.items():
            normalized_weights[key] = (value / comp_total) * 0.5

        for key, value in synthesis_weights.items():
            normalized_weights[key] = (value / synth_total) * 0.5

        return normalized_weights

    def _calculate_normalized_metrics(self, details, section, normalized_weights):
        """
        Calculate normalized metrics for a given section based on agent evaluation results.

        Args:
            details (dict): The details section of evaluation results
            section (str): The section to calculate metrics for (composition_data or synthesis_data)
            normalized_weights (dict): Normalized weights for different components

        Returns:
            dict: Dictionary with normalized true_positives, false_positives, and false_negatives
        """
        metrics = {
            "true_positives": 0.0,
            "false_positives": 0.0,
            "false_negatives": 0.0,
        }

        if section not in details or not details[section]:
            return metrics

        section_data = details[section]

        if section == "composition_data":
            # Property unit
            if "property_unit" in section_data:
                match_value = section_data["property_unit"].get("match_value", 0)
                if match_value == 1:
                    metrics["true_positives"] += normalized_weights["property_unit"]
                else:
                    # Split the weight between false positives and false negatives
                    if section_data["property_unit"].get("reference") is not None:
                        metrics["false_negatives"] += (
                            normalized_weights["property_unit"] / 2
                        )
                    if section_data["property_unit"].get("test") is not None:
                        metrics["false_positives"] += (
                            normalized_weights["property_unit"] / 2
                        )

            # Family
            if "family" in section_data:
                match_value = section_data["family"].get("match_value", 0)
                if match_value == 1:
                    metrics["true_positives"] += normalized_weights["family"]
                else:
                    # Split the weight between false positives and false negatives
                    if section_data["family"].get("reference") is not None:
                        metrics["false_negatives"] += normalized_weights["family"] / 2
                    if section_data["family"].get("test") is not None:
                        metrics["false_positives"] += normalized_weights["family"] / 2

            # Compositions property values
            if "compositions_property_values" in section_data:
                cpv = section_data["compositions_property_values"]
                total_ground_truth_keys = cpv.get("total_ground_truth_keys", 0)

                if total_ground_truth_keys > 0:
                    # Split the component weight between keys and values
                    key_weight = normalized_weights["compositions_property_values"] / 2
                    value_weight = (
                        normalized_weights["compositions_property_values"] / 2
                    )

                    # Calculate per-item weight
                    key_item_weight = key_weight / total_ground_truth_keys
                    value_item_weight = value_weight / total_ground_truth_keys

                    # Process key matches
                    if "key_matches" in cpv:
                        for key_match in cpv["key_matches"]:
                            if key_match.get("match_value", 0) == 1:
                                metrics["true_positives"] += key_item_weight
                            else:
                                if key_match.get("reference_key"):
                                    metrics["false_negatives"] += key_item_weight / 2
                                if key_match.get("test_key"):
                                    metrics["false_positives"] += key_item_weight / 2

                    # Process value matches
                    if "value_matches" in cpv:
                        for value_match in cpv["value_matches"]:
                            if value_match.get("match_value", 0) == 1:
                                metrics["true_positives"] += value_item_weight
                            else:
                                if value_match.get("reference_value") is not None:
                                    metrics["false_negatives"] += value_item_weight / 2
                                if value_match.get("test_value") is not None:
                                    metrics["false_positives"] += value_item_weight / 2

                    # Missing keys (just add as false negatives)
                    missing_keys_count = len(cpv.get("missing_keys", []))
                    if missing_keys_count > 0:
                        metrics["false_negatives"] += missing_keys_count * (
                            key_item_weight + value_item_weight
                        )

                    # Extra keys (just add as false positives)
                    extra_keys_count = len(cpv.get("extra_keys", []))
                    if extra_keys_count > 0:
                        metrics["false_positives"] += extra_keys_count * (
                            key_item_weight + value_item_weight
                        )

        elif section == "synthesis_data":
            # Method
            if "method" in section_data:
                match_value = section_data["method"].get("match_value", 0)
                if match_value == 1:
                    metrics["true_positives"] += normalized_weights["method"]
                else:
                    # Split the weight between false positives and false negatives
                    if section_data["method"].get("reference") is not None:
                        metrics["false_negatives"] += normalized_weights["method"] / 2
                    if section_data["method"].get("test") is not None:
                        metrics["false_positives"] += normalized_weights["method"] / 2

            # Precursors
            if "precursors" in section_data:
                pre = section_data["precursors"]
                total_ground_truth_items = pre.get("total_ground_truth_items", 0)

                if total_ground_truth_items > 0:
                    # Calculate per-item weight
                    item_weight = (
                        normalized_weights["precursors"] / total_ground_truth_items
                    )

                    # Process matches
                    if "matches" in pre:
                        for match in pre["matches"]:
                            if match.get("match_value", 0) == 1:
                                metrics["true_positives"] += item_weight
                            else:
                                if match.get("reference_item") is not None:
                                    metrics["false_negatives"] += item_weight / 2
                                if match.get("test_item") is not None:
                                    metrics["false_positives"] += item_weight / 2

                    # Missing items
                    missing_items_count = len(pre.get("missing_items", []))
                    if missing_items_count > 0:
                        metrics["false_negatives"] += missing_items_count * item_weight

                    # Extra items
                    extra_items_count = len(pre.get("extra_items", []))
                    if extra_items_count > 0:
                        metrics["false_positives"] += extra_items_count * item_weight

            # Characterization techniques
            if "characterization_techniques" in section_data:
                tech = section_data["characterization_techniques"]
                total_ground_truth_items = tech.get("total_ground_truth_items", 0)

                if total_ground_truth_items > 0:
                    # Calculate per-item weight
                    item_weight = (
                        normalized_weights["characterization_techniques"]
                        / total_ground_truth_items
                    )

                    # Process matches
                    if "matches" in tech:
                        for match in tech["matches"]:
                            if match.get("match_value", 0) == 1:
                                metrics["true_positives"] += item_weight
                            else:
                                if match.get("reference_item") is not None:
                                    metrics["false_negatives"] += item_weight / 2
                                if match.get("test_item") is not None:
                                    metrics["false_positives"] += item_weight / 2

                    # Missing items
                    missing_items_count = len(tech.get("missing_items", []))
                    if missing_items_count > 0:
                        metrics["false_negatives"] += missing_items_count * item_weight

                    # Extra items
                    extra_items_count = len(tech.get("extra_items", []))
                    if extra_items_count > 0:
                        metrics["false_positives"] += extra_items_count * item_weight

            # Steps
            if "steps" in section_data:
                steps_match = section_data["steps"].get("match_value", 0)

                # If steps_match is a float (0.0 to 1.0)
                if isinstance(steps_match, float):
                    metrics["true_positives"] += (
                        normalized_weights["steps"] * steps_match
                    )
                    remaining = normalized_weights["steps"] * (1 - steps_match)
                    metrics["false_negatives"] += remaining / 2
                    metrics["false_positives"] += remaining / 2
                # If it's binary (0 or 1)
                elif steps_match == 1:
                    metrics["true_positives"] += normalized_weights["steps"]
                else:
                    # Split weight between false positives and negatives
                    metrics["false_negatives"] += normalized_weights["steps"] / 2
                    metrics["false_positives"] += normalized_weights["steps"] / 2

        return metrics

    def _calculate_key_value_match_ratios(self, details):
        """
        Calculate key-value pair match ratios for composition data similar to the semantic evaluator.

        Args:
            details (dict): The evaluation details for a DOI

        Returns:
            dict: Updated details with match ratios
        """
        if "composition_data" not in details or not details["composition_data"]:
            return details

        comp_data = details["composition_data"]
        if "compositions_property_values" not in comp_data:
            return details

        cpv = comp_data["compositions_property_values"]

        # Initialize counters
        key_matches_count = 0
        value_matches_count = 0
        pair_matches_count = 0
        total_ref_keys = cpv.get("total_ground_truth_keys", 0)

        # Count key matches from key_matches section
        if "key_matches" in cpv:
            for key_match in cpv["key_matches"]:
                if key_match.get("match_value", 0) == 1:
                    key_matches_count += 1

        # Count value matches from value_matches section
        if "value_matches" in cpv:
            for value_match in cpv["value_matches"]:
                if value_match.get("match_value", 0) == 1:
                    value_matches_count += 1

        # Count pair matches from pair_matches section (keys that match AND values that match)
        if "pair_matches" in cpv:
            for pair_match in cpv["pair_matches"]:
                if pair_match.get("match_value", 0) == 1:
                    pair_matches_count += 1

        # Calculate ratios if there are reference keys
        if total_ref_keys > 0:
            key_match_ratio = key_matches_count / total_ref_keys
            value_match_ratio = value_matches_count / total_ref_keys
            pair_match_ratio = pair_matches_count / total_ref_keys

            # Calculate overall match ratio (weighted combination of key and pair match)
            overall_match_ratio = 0.4 * key_match_ratio + 0.6 * pair_match_ratio
        else:
            key_match_ratio = 0.0
            value_match_ratio = 0.0
            pair_match_ratio = 0.0
            overall_match_ratio = 0.0

        # Add match ratios to the details
        cpv.update(
            {
                "key_match_ratio": key_match_ratio,
                "value_match_ratio": value_match_ratio,
                "pair_match_ratio": pair_match_ratio,
                "overall_match_ratio": overall_match_ratio,
            }
        )

        # Add a similarity_score for compatibility with the semantic evaluator format
        comp_data["compositions_property_values"][
            "similarity_score"
        ] = overall_match_ratio

        # Update the match flag based on overall_match_ratio with a threshold of 0.85
        comp_data["compositions_property_values"]["match"] = overall_match_ratio > 0.85

        return details

    def _enhance_evaluation_details(self, details):
        """
        Enhance the evaluation details to include match ratios and similarity scores.

        Args:
            details (dict): The original evaluation details

        Returns:
            dict: Enhanced details with additional metrics
        """
        # If details is empty or not a dictionary, return as is
        if not details or not isinstance(details, dict):
            return details

        # Enhanced the composition data with key-value match ratios
        details = self._calculate_key_value_match_ratios(details)

        # Add similarity scores and match flags to synthesis data if needed
        if "synthesis_data" in details and details["synthesis_data"]:
            synth_data = details["synthesis_data"]

            # Method
            if "method" in synth_data:
                match_value = synth_data["method"].get("match_value", 0)
                synth_data["method"]["similarity"] = 1.0 if match_value == 1 else 0.0
                synth_data["method"]["match"] = match_value == 1

            # Precursors
            if "precursors" in synth_data:
                pre = synth_data["precursors"]
                total_ground_truth_items = pre.get("total_ground_truth_items", 0)
                total_match = pre.get("total_match", 0)

                if total_ground_truth_items > 0:
                    pre["similarity"] = total_match / total_ground_truth_items
                    pre["match"] = pre["similarity"] >= 0.8
                else:
                    # If both reference and test are empty, it's a perfect match
                    if (
                        not pre.get("reference") or len(pre.get("reference", [])) == 0
                    ) and (not pre.get("test") or len(pre.get("test", [])) == 0):
                        pre["similarity"] = 1.0
                        pre["match"] = True
                    else:
                        pre["similarity"] = 0.0
                        pre["match"] = False

            # Characterization techniques
            if "characterization_techniques" in synth_data:
                tech = synth_data["characterization_techniques"]
                total_ground_truth_items = tech.get("total_ground_truth_items", 0)
                total_match = tech.get("total_match", 0)

                if total_ground_truth_items > 0:
                    tech["similarity"] = total_match / total_ground_truth_items
                    tech["match"] = tech["similarity"] >= 0.8
                else:
                    # If both reference and test are empty, it's a perfect match
                    if (
                        not tech.get("reference") or len(tech.get("reference", [])) == 0
                    ) and (not tech.get("test") or len(tech.get("test", [])) == 0):
                        tech["similarity"] = 1.0
                        tech["match"] = True
                    else:
                        tech["similarity"] = 0.0
                        tech["match"] = False

            # Steps - rename match_value to steps_match for compatibility
            if "steps" in synth_data:
                steps_match = synth_data["steps"].get("match_value", 0)

                if isinstance(steps_match, float):
                    synth_data["steps"]["steps_match"] = (
                        "true" if steps_match > 0.6 else "false"
                    )
                else:
                    synth_data["steps"]["steps_match"] = (
                        "true" if steps_match == 1 else "false"
                    )

        return details

    def _is_section_empty(self, section_data):
        """
        Helper method to determine if a section is effectively empty

        Args:
            section_data (dict): The section data to check

        Returns:
            bool: True if the section is empty, False otherwise
        """
        if not section_data or not isinstance(section_data, dict):
            return True

        # Check if all values are None, empty dicts, or empty lists
        for key, value in section_data.items():
            if value is not None and value != {} and value != []:
                if isinstance(value, dict) and len(value) > 0:
                    return False
                elif isinstance(value, list) and len(value) > 0:
                    return False
                elif not isinstance(value, (dict, list)) and value:
                    return False

        return True

    def _count_all_items(self, item):
        """Count all individual items in a data entry"""
        count = 0

        # Count composition items
        comp_data = item.get("composition_data", {})
        if comp_data is not None:  # handles None values
            if "property_unit" in comp_data:
                count += 1
            if "family" in comp_data:
                count += 1
            count += (
                len(comp_data.get("compositions_property_values", {})) * 2
            )  # Keys and values

        # Count synthesis items
        synth_data = item.get("synthesis_data", {})
        if synth_data is not None:  # handles None values
            if "method" in synth_data:
                count += 1
            count += len(synth_data.get("precursors", []))
            count += len(synth_data.get("characterization_techniques", []))
            if synth_data.get("steps"):
                count += 2  # Count steps as 2 items (presence + content)

        return count

    @start()
    def load_data(self):
        """Load ground truth and test data files, and check for existing results."""
        logger.info(f"Loading ground truth data from {self.state.ground_truth_file}")
        logger.info(f"Loading test data from {self.state.test_data_file}")

        try:
            # Load ground truth and test data
            with open(self.state.ground_truth_file, "r", encoding="utf-8") as gt_file:
                self.state.ground_truth_data = json.load(gt_file)

            with open(self.state.test_data_file, "r", encoding="utf-8") as test_file:
                self.state.test_data = json.load(test_file)

            # Get all DOIs from ground truth
            all_dois = list(self.state.ground_truth_data.keys())
            self.state.total_count = len(all_dois)

            # Load existing results if available
            self.state.existing_results = self._load_existing_results()

            # Extract already processed DOIs
            already_processed_dois = []
            if (
                self.state.existing_results
                and "item_results" in self.state.existing_results
            ):
                already_processed_dois = list(
                    self.state.existing_results["item_results"].keys()
                )
                self.state.processed_count = len(already_processed_dois)
                logger.info(
                    f"Found {self.state.processed_count} already processed items"
                )

            # Determine remaining DOIs to process
            self.state.remaining_dois = [
                doi for doi in all_dois if doi not in already_processed_dois
            ]

            # Initialize evaluation details and item results from existing data
            if self.state.existing_results:
                if "item_results" in self.state.existing_results:
                    self.state.item_results = self.state.existing_results[
                        "item_results"
                    ]

                # Extract existing evaluation details
                for doi, item_result in self.state.item_results.items():
                    if "details" in item_result:
                        self.state.evaluation_details[doi] = item_result["details"]

            logger.info(
                f"Successfully loaded data: {self.state.total_count} ground truth items, "
                f"{len(self.state.test_data)} test items, "
                f"{len(self.state.remaining_dois)} remaining to process"
            )

            return {
                "ground_truth_count": self.state.total_count,
                "test_data_count": len(self.state.test_data),
                "processed_count": self.state.processed_count,
                "remaining_dois": self.state.remaining_dois,
            }
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueErrorHandler(f"Failed to load data files: {str(e)}")

    @listen(load_data)
    def evaluate_items(self, data_info):
        """Evaluate remaining items in the dataset."""
        # Get all DOIs from the ground truth data
        all_dois = set(self.state.ground_truth_data.keys())

        # Filter to only process DOIs that aren't already in existing results
        if (
            self.state.existing_results
            and "item_results" in self.state.existing_results
        ):
            already_processed_dois = set(
                self.state.existing_results["item_results"].keys()
            )
        else:
            already_processed_dois = set()

        remaining_dois = [doi for doi in all_dois if doi not in already_processed_dois]

        remaining_count = len(remaining_dois)
        logger.info(
            f"Starting evaluation of {remaining_count} remaining items (including missing DOIs)"
        )

        if remaining_count == 0:
            logger.info("No remaining items to process, using existing results")
            return self.state.evaluation_details

        # Get normalized weights for metric calculations
        normalized_weights = self._normalize_weights()

        # Set up the crews
        composition_crew = CompositionEvaluationCrew(llm=self.state.llm).crew()
        if self.state.is_synthesis_evaluation:
            synthesis_crew = SynthesisEvaluationCrew(llm=self.state.llm).crew()

        # Initialize the combined results with existing data
        combined_results = {
            "extraction_agent_model_name": self.state.extraction_agent_model_name,
            "overall_accuracy": 0.0,
            "overall_composition_accuracy": 0.0,
            "overall_synthesis_accuracy": 0.0,
            "total_items": len(all_dois),  # Use all DOIs count
            "absolute_classification_metrics": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "normalized_classification_metrics": {
                "true_positives": 0.0,
                "false_positives": 0.0,
                "false_negatives": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "item_results": self.state.item_results,
        }

        if not self.state.is_synthesis_evaluation:
            combined_results.pop("overall_synthesis_accuracy", None)

        # Process each remaining DOI one at a time
        for doi_index, doi in enumerate(
            tqdm(remaining_dois, desc="Processing DOIs", unit="item")
        ):
            logger.info(
                f"Evaluating item {doi_index+1}/{remaining_count} with DOI: {doi}"
            )

            # Get ground truth and test data for this DOI
            ground_truth_item = self.state.ground_truth_data.get(doi, {})
            test_item = self.state.test_data.get(doi, {})

            # Initialize results for this DOI
            item_result = {
                "overall_match": False,
                "field_scores": {},
                "overall_score": 0.0,
                "absolute_classification_metrics": {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                },
                "normalized_classification_metrics": {
                    "true_positives": 0.0,
                    "false_positives": 0.0,
                    "false_negatives": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                },
                "details": {},
            }

            # Case 1: DOI exists in both datasets
            if ground_truth_item and test_item:
                doi_details = {}
                has_valid_data = False

                # Check if composition data exists in both ground truth and test items
                has_composition_data = (
                    "composition_data" in ground_truth_item
                    and ground_truth_item["composition_data"]
                    and "composition_data" in test_item
                    and test_item["composition_data"]
                )

                # Evaluate composition data if it exists
                if has_composition_data:
                    composition_result = composition_crew.kickoff(
                        inputs={
                            "ground_truth_item": json.dumps(
                                ground_truth_item.get("composition_data", {})
                            ),
                            "test_item": json.dumps(
                                test_item.get("composition_data", {})
                            ),
                        }
                    )

                    # Store composition evaluation details
                    try:
                        if isinstance(composition_result.raw, str):
                            composition_details = json.loads(composition_result.raw)
                        else:
                            composition_details = composition_result.raw

                        # Check if composition_data is in the result
                        if "composition_data" in composition_details:
                            doi_details["composition_data"] = composition_details[
                                "composition_data"
                            ]
                            if not self._is_section_empty(
                                doi_details["composition_data"]
                            ):
                                has_valid_data = True
                        else:
                            doi_details["composition_data"] = composition_details
                            if not self._is_section_empty(
                                doi_details["composition_data"]
                            ):
                                has_valid_data = True
                    except Exception as e:
                        logger.error(
                            f"Error processing composition results for DOI {doi}: {str(e)}"
                        )
                        doi_details["composition_data"] = {}

                # Check if synthesis data exists in both ground truth and test items
                has_synthesis_data = (
                    self.state.is_synthesis_evaluation
                    and "synthesis_data" in ground_truth_item
                    and ground_truth_item["synthesis_data"]
                    and "synthesis_data" in test_item
                    and test_item["synthesis_data"]
                )

                # Evaluate synthesis data if it exists
                if has_synthesis_data:
                    synthesis_result = synthesis_crew.kickoff(
                        inputs={
                            "ground_truth_item": json.dumps(
                                ground_truth_item.get("synthesis_data", {})
                            ),
                            "test_item": json.dumps(
                                test_item.get("synthesis_data", {})
                            ),
                        }
                    )

                    # Store synthesis evaluation details
                    try:
                        if isinstance(synthesis_result.raw, str):
                            synthesis_details = json.loads(synthesis_result.raw)
                        else:
                            synthesis_details = synthesis_result.raw

                        # Check if synthesis_data is in the result
                        if "synthesis_data" in synthesis_details:
                            doi_details["synthesis_data"] = synthesis_details[
                                "synthesis_data"
                            ]
                            if not self._is_section_empty(
                                doi_details["synthesis_data"]
                            ):
                                has_valid_data = True
                        else:
                            doi_details["synthesis_data"] = synthesis_details
                            if not self._is_section_empty(
                                doi_details["synthesis_data"]
                            ):
                                has_valid_data = True
                    except Exception as e:
                        logger.error(
                            f"Error processing synthesis results for DOI {doi}: {str(e)}"
                        )
                        doi_details["synthesis_data"] = {}

                # Process valid data if found
                if has_valid_data:
                    # Store evaluation details for this DOI
                    self.state.evaluation_details[doi] = doi_details

                    # Enhance details with match ratios and similarity scores
                    enhanced_details = self._enhance_evaluation_details(doi_details)
                    item_result["details"] = enhanced_details

                    # Calculate composition score
                    if (
                        "composition_data" in enhanced_details
                        and not self._is_section_empty(
                            enhanced_details["composition_data"]
                        )
                    ):
                        composition_score = self._calculate_score(
                            enhanced_details, "composition_data", self.state.weights
                        )
                        item_result["field_scores"][
                            "composition_data"
                        ] = composition_score

                        # Calculate composition metrics
                        comp_metrics = self._calculate_tp_fp_fn(
                            enhanced_details, "composition_data"
                        )
                        for metric in [
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                        ]:
                            item_result["absolute_classification_metrics"][
                                metric
                            ] += comp_metrics[metric]

                        # Calculate normalized metrics for composition
                        comp_normalized_metrics = self._calculate_normalized_metrics(
                            enhanced_details, "composition_data", normalized_weights
                        )
                        for metric in [
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                        ]:
                            item_result["normalized_classification_metrics"][
                                metric
                            ] += comp_normalized_metrics[metric]

                    # Calculate synthesis score if applicable
                    if (
                        self.state.is_synthesis_evaluation
                        and "synthesis_data" in enhanced_details
                        and not self._is_section_empty(
                            enhanced_details["synthesis_data"]
                        )
                    ):
                        synthesis_score = self._calculate_score(
                            enhanced_details, "synthesis_data", self.state.weights
                        )
                        item_result["field_scores"]["synthesis_data"] = synthesis_score

                        # Calculate synthesis metrics
                        synth_metrics = self._calculate_tp_fp_fn(
                            enhanced_details, "synthesis_data"
                        )
                        for metric in [
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                        ]:
                            item_result["absolute_classification_metrics"][
                                metric
                            ] += synth_metrics[metric]

                        # Calculate normalized metrics for synthesis
                        synth_normalized_metrics = self._calculate_normalized_metrics(
                            enhanced_details, "synthesis_data", normalized_weights
                        )
                        for metric in [
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                        ]:
                            item_result["normalized_classification_metrics"][
                                metric
                            ] += synth_normalized_metrics[metric]

                    # Calculate overall score for the item
                    if (
                        self.state.is_synthesis_evaluation
                        and "composition_data" in item_result["field_scores"]
                        and "synthesis_data" in item_result["field_scores"]
                    ):
                        item_result["overall_score"] = 0.5 * item_result[
                            "field_scores"
                        ].get("composition_data", 0.0) + 0.5 * item_result[
                            "field_scores"
                        ].get(
                            "synthesis_data", 0.0
                        )
                    elif "composition_data" in item_result["field_scores"]:
                        item_result["overall_score"] = item_result["field_scores"].get(
                            "composition_data", 0.0
                        )
                    elif "synthesis_data" in item_result["field_scores"]:
                        item_result["overall_score"] = item_result["field_scores"].get(
                            "synthesis_data", 0.0
                        )

                    # Determine if overall match (threshold: 0.8)
                    item_result["overall_match"] = item_result["overall_score"] > 0.8
                else:
                    # No valid data found - treat as no match
                    item_result["overall_score"] = 0.0
                    item_result["overall_match"] = False
                    logger.warning(f"DOI {doi}: No valid evaluation data found")

            # Case 2: DOI exists in ground truth but not in test (missing)
            elif ground_truth_item and not test_item:
                item_count = self._count_all_items(ground_truth_item)

                # All items are false negatives
                item_result["absolute_classification_metrics"][
                    "false_negatives"
                ] = item_count
                item_result["normalized_classification_metrics"][
                    "false_negatives"
                ] = 1.0  # Full DOI weight
                item_result["overall_score"] = 0.0
                item_result["overall_match"] = False

                # Set field scores to zero
                item_result["field_scores"]["composition_data"] = 0.0
                if self.state.is_synthesis_evaluation:
                    item_result["field_scores"]["synthesis_data"] = 0.0

                logger.info(
                    f"DOI {doi}: Missing from test data (false negatives: {item_count})"
                )

            # Calculate precision, recall, F1 for absolute metrics
            self._calculate_classification_metrics(
                item_result["absolute_classification_metrics"]
            )

            # Calculate precision, recall, F1 for normalized metrics
            self._calculate_classification_metrics(
                item_result["normalized_classification_metrics"]
            )

            # Store item result
            self.state.item_results[doi] = item_result
            combined_results["item_results"][doi] = item_result

            # Update combined metrics after each item
            self._update_combined_metrics(combined_results)

            # Write updated results to file after each item
            self._update_results_file(combined_results)

            logger.info(
                f"Item {doi} processed. "
                f"Score: {item_result['overall_score']:.4f}, "
                f"Absolute F1: {item_result['absolute_classification_metrics']['f1_score']:.4f}, "
                f"Normalized F1: {item_result['normalized_classification_metrics']['f1_score']:.4f}"
            )

        logger.info(f"Completed evaluation of {remaining_count} items")
        time.sleep(5)
        return self.state.evaluation_details

    def _update_combined_metrics(self, combined_results):
        """
        Update the combined metrics based on current item results.

        Args:
            combined_results (dict): The combined results to update
        """
        # Reset accumulator metrics
        total_absolute_metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        total_normalized_metrics = {
            "true_positives": 0.0,
            "false_positives": 0.0,
            "false_negatives": 0.0,
        }

        # Reset scores
        total_composition_score = 0.0
        total_synthesis_score = 0.0
        composition_count = 0
        synthesis_count = 0

        # Sum up all metrics
        for doi, item in combined_results["item_results"].items():
            # Update absolute metrics
            for metric in ["true_positives", "false_positives", "false_negatives"]:
                total_absolute_metrics[metric] += item[
                    "absolute_classification_metrics"
                ].get(metric, 0)

                # Also update normalized metrics
                total_normalized_metrics[metric] += item[
                    "normalized_classification_metrics"
                ].get(metric, 0.0)

            # Update scores
            if "composition_data" in item["field_scores"]:
                total_composition_score += item["field_scores"]["composition_data"]
                composition_count += 1

            if "synthesis_data" in item["field_scores"]:
                total_synthesis_score += item["field_scores"]["synthesis_data"]
                synthesis_count += 1

        # Update combined absolute metrics
        for metric in ["true_positives", "false_positives", "false_negatives"]:
            combined_results["absolute_classification_metrics"][metric] = (
                total_absolute_metrics[metric]
            )

            # Also update normalized metrics
            combined_results["normalized_classification_metrics"][metric] = (
                total_normalized_metrics[metric]
            )

        # Calculate precision, recall, F1 for absolute metrics
        self._calculate_classification_metrics(
            combined_results["absolute_classification_metrics"]
        )

        # Calculate precision, recall, F1 for normalized metrics
        self._calculate_classification_metrics(
            combined_results["normalized_classification_metrics"]
        )

        # Calculate overall accuracies
        total_items = len(combined_results["item_results"])

        # Calculate composition accuracy if composition data is available
        if composition_count > 0:
            combined_results["overall_composition_accuracy"] = (
                total_composition_score / composition_count
            )
        else:
            combined_results["overall_composition_accuracy"] = 0.0

        # Calculate synthesis accuracy if synthesis data is available and it's requested
        if self.state.is_synthesis_evaluation:
            if synthesis_count > 0:
                combined_results["overall_synthesis_accuracy"] = (
                    total_synthesis_score / synthesis_count
                )
            else:
                combined_results["overall_synthesis_accuracy"] = 0.0

            # Calculate overall accuracy as average of composition and synthesis
            if composition_count > 0 and synthesis_count > 0:
                combined_results["overall_accuracy"] = (
                    combined_results["overall_composition_accuracy"]
                    + combined_results["overall_synthesis_accuracy"]
                ) / 2
            elif composition_count > 0:
                combined_results["overall_accuracy"] = combined_results[
                    "overall_composition_accuracy"
                ]
            elif synthesis_count > 0:
                combined_results["overall_accuracy"] = combined_results[
                    "overall_synthesis_accuracy"
                ]
            else:
                combined_results["overall_accuracy"] = 0.0
        else:
            # If synthesis evaluation is not enabled, overall is just composition
            combined_results["overall_accuracy"] = combined_results[
                "overall_composition_accuracy"
            ]

    @listen(evaluate_items)
    def finalize_evaluation(self, evaluation_details):
        """
        Finalize the evaluation process and return the final results.

        Args:
            evaluation_details (dict): Combined evaluation details by DOI

        Returns:
            dict: Final evaluation results with metrics and scores
        """
        logger.info("Finalizing agent-based material data evaluation")

        # Get normalized weights
        normalized_weights = self._normalize_weights()

        # Calculate final combined metrics
        combined_results = {
            "extraction_agent_model_name": self.state.extraction_agent_model_name,
            "overall_accuracy": 0.0,
            "overall_composition_accuracy": 0.0,
            "overall_synthesis_accuracy": 0.0,
            "total_items": len(self.state.item_results),
            "absolute_classification_metrics": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "normalized_classification_metrics": {
                "true_positives": 0.0,
                "false_positives": 0.0,
                "false_negatives": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "item_results": {},
        }

        if not self.state.is_synthesis_evaluation:
            combined_results.pop("overall_synthesis_accuracy", None)

        # Reset accumulator metrics
        total_absolute_metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        total_normalized_metrics = {
            "true_positives": 0.0,
            "false_positives": 0.0,
            "false_negatives": 0.0,
        }

        # Reset scores
        total_composition_score = 0.0
        total_synthesis_score = 0.0

        # Process each item's results
        for doi, item in self.state.item_results.items():
            # Enhance the evaluation details with match ratios and similarity scores
            item_details = self._enhance_evaluation_details(item.get("details", {}))

            # Create enhanced item result
            enhanced_item = {
                "overall_match": item.get("overall_match", False),
                "field_scores": item.get("field_scores", {}),
                "overall_score": item.get("overall_score", 0.0),
                "absolute_classification_metrics": item.get(
                    "absolute_classification_metrics",
                    {
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                    },
                ),
                "normalized_classification_metrics": {
                    "true_positives": 0.0,
                    "false_positives": 0.0,
                    "false_negatives": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                },
                "details": item_details,
            }

            # Calculate normalized metrics for this item
            comp_normalized_metrics = self._calculate_normalized_metrics(
                item_details, "composition_data", normalized_weights
            )

            # Add composition normalized metrics to item metrics
            for metric in ["true_positives", "false_positives", "false_negatives"]:
                enhanced_item["normalized_classification_metrics"][
                    metric
                ] += comp_normalized_metrics[metric]

            # Add synthesis normalized metrics if applicable
            if self.state.is_synthesis_evaluation:
                synth_normalized_metrics = self._calculate_normalized_metrics(
                    item_details, "synthesis_data", normalized_weights
                )

                for metric in ["true_positives", "false_positives", "false_negatives"]:
                    enhanced_item["normalized_classification_metrics"][
                        metric
                    ] += synth_normalized_metrics[metric]

            # Calculate precision, recall, F1 for normalized metrics
            self._calculate_classification_metrics(
                enhanced_item["normalized_classification_metrics"]
            )

            # Update the combined results item
            combined_results["item_results"][doi] = enhanced_item

            # Update absolute metrics totals
            for metric in ["true_positives", "false_positives", "false_negatives"]:
                total_absolute_metrics[metric] += enhanced_item[
                    "absolute_classification_metrics"
                ].get(metric, 0)
                total_normalized_metrics[metric] += enhanced_item[
                    "normalized_classification_metrics"
                ].get(metric, 0)

            # Update score totals
            total_composition_score += enhanced_item["field_scores"].get(
                "composition_data", 0.0
            )
            if self.state.is_synthesis_evaluation:
                total_synthesis_score += enhanced_item["field_scores"].get(
                    "synthesis_data", 0.0
                )

        # Update combined metrics
        for metric in ["true_positives", "false_positives", "false_negatives"]:
            combined_results["absolute_classification_metrics"][metric] = (
                total_absolute_metrics[metric]
            )
            combined_results["normalized_classification_metrics"][metric] = (
                total_normalized_metrics[metric]
            )

        # Calculate precision, recall, F1 for absolute metrics
        self._calculate_classification_metrics(
            combined_results["absolute_classification_metrics"]
        )

        # Calculate precision, recall, F1 for normalized metrics
        self._calculate_classification_metrics(
            combined_results["normalized_classification_metrics"]
        )

        # Calculate overall accuracies
        total_items = len(combined_results["item_results"])
        if total_items > 0:
            combined_results["overall_composition_accuracy"] = (
                total_composition_score / total_items
            )

            if self.state.is_synthesis_evaluation:
                combined_results["overall_synthesis_accuracy"] = (
                    total_synthesis_score / total_items
                )
                combined_results["overall_accuracy"] = (
                    0.5 * combined_results["overall_composition_accuracy"]
                    + 0.5 * combined_results["overall_synthesis_accuracy"]
                )
            else:
                combined_results["overall_accuracy"] = combined_results[
                    "overall_composition_accuracy"
                ]

        # Write final results to file
        self._update_results_file(combined_results)

        # Log summary statistics
        logger.info(f"Overall accuracy: {combined_results['overall_accuracy']:.4f}")
        logger.info(
            f"Composition accuracy: {combined_results['overall_composition_accuracy']:.4f}"
        )

        if (
            self.state.is_synthesis_evaluation
            and "overall_synthesis_accuracy" in combined_results
        ):
            logger.info(
                f"Synthesis accuracy: {combined_results['overall_synthesis_accuracy']:.4f}"
            )

        # Log F1 scores
        abs_f1 = combined_results["absolute_classification_metrics"]["f1_score"]
        norm_f1 = combined_results["normalized_classification_metrics"]["f1_score"]
        logger.info(f"Absolute F1 score: {abs_f1:.4f}")
        logger.info(f"Normalized F1 score: {norm_f1:.4f}")

        logger.info(f"Evaluation complete. Results saved to {self.state.output_file}")

        # Store in state and return
        self.state.combined_evaluation_results = combined_results
        return combined_results
