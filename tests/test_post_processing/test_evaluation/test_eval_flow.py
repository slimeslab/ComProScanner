"""
test_eval_flow.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 20-07-2025
"""

import json
import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from comproscanner.post_processing.evaluation.eval_flow.eval_flow import (
    MaterialsDataAgenticEvaluatorFlow,
    AgentEvaluationState,
)
from comproscanner.utils.error_handler import ValueErrorHandler, BaseError
from crewai import LLM


class TestAgentEvaluationState:
    """Test suite for AgentEvaluationState model"""

    def test_agent_evaluation_state_default_values(self):
        """Test default values of AgentEvaluationState"""
        state = AgentEvaluationState()

        assert state.ground_truth_file == ""
        assert state.test_data_file == ""
        assert state.output_file == "agentic_evaluation_result.json"
        assert state.used_agent_model_name == "gpt-4o-mini"
        assert state.is_synthesis_evaluation is True
        assert state.ground_truth_data == {}
        assert state.test_data == {}
        assert state.llm is None
        assert state.processed_count == 0
        assert state.total_count == 0
        assert state.remaining_dois == []

    def test_agent_evaluation_state_custom_values(self):
        """Test AgentEvaluationState with custom values"""
        mock_llm = Mock(spec=LLM)
        weights = {"composition": 0.5, "synthesis": 0.5}

        state = AgentEvaluationState(
            ground_truth_file="test_gt.json",
            test_data_file="test_data.json",
            used_agent_model_name="test-model",
            llm=mock_llm,
            weights=weights,
            processed_count=5,
        )

        assert state.ground_truth_file == "test_gt.json"
        assert state.test_data_file == "test_data.json"
        assert state.used_agent_model_name == "test-model"
        assert state.llm == mock_llm
        assert state.weights == weights
        assert state.processed_count == 5


class TestMaterialsDataAgenticEvaluatorFlow:
    """Test suite for MaterialsDataAgenticEvaluatorFlow class"""

    @pytest.fixture
    def sample_ground_truth_data(self):
        """Sample ground truth data for testing"""
        return {
            "10.1000/test1": {
                "composition_data": {
                    "property_unit": "MPa",
                    "family": "ceramics",
                    "compositions_property_values": {"Al2O3": 150.5, "SiO2": 100.0},
                },
                "synthesis_data": {
                    "method": "sol-gel synthesis",
                    "precursors": ["aluminum nitrate", "silicon oxide"],
                    "characterization_techniques": ["XRD", "SEM"],
                    "steps": ["Mix precursors", "Heat at 500C"],
                },
            },
            "10.1000/test2": {
                "composition_data": {
                    "property_unit": "GPa",
                    "family": "polymers",
                    "compositions_property_values": {"PE": 2.5, "PP": 3.0},
                },
                "synthesis_data": {
                    "method": "polymerization",
                    "precursors": ["ethylene", "propylene"],
                    "characterization_techniques": ["FTIR"],
                    "steps": ["Polymerize", "Purify"],
                },
            },
        }

    @pytest.fixture
    def sample_test_data(self):
        """Sample test data for testing"""
        return {
            "10.1000/test1": {
                "composition_data": {
                    "property_unit": "MPa",
                    "family": "ceramics",
                    "compositions_property_values": {
                        "Al2O3": 150.0,  # Slightly different
                        "SiO2": 100.0,
                    },
                },
                "synthesis_data": {
                    "method": "sol-gel method",  # Similar but different
                    "precursors": ["aluminum salt", "silicon dioxide"],  # Similar
                    "characterization_techniques": ["XRD", "SEM"],
                    "steps": ["Mix materials", "Heat sample"],  # Similar
                },
            },
            "10.1000/test2": {
                "composition_data": {
                    "property_unit": "GPa",
                    "family": "polymers",
                    "compositions_property_values": {
                        "polyethylene": 2.4,  # Different key, similar value
                        "polypropylene": 3.1,  # Different key, similar value
                    },
                },
                "synthesis_data": {
                    "method": "radical polymerization",  # Different method
                    "precursors": ["ethylene monomer"],  # Missing one precursor
                    "characterization_techniques": [
                        "FTIR",
                        "NMR",
                    ],  # One same, one different
                    "steps": ["Initiate reaction", "Clean product"],
                },
            },
        }

    @pytest.fixture
    def temp_files(self, sample_ground_truth_data, sample_test_data):
        """Create temporary files for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_file = os.path.join(tmpdir, "ground_truth.json")
            test_file = os.path.join(tmpdir, "test_data.json")
            output_file = os.path.join(tmpdir, "output.json")

            with open(gt_file, "w") as f:
                json.dump(sample_ground_truth_data, f)

            with open(test_file, "w") as f:
                json.dump(sample_test_data, f)

            yield gt_file, test_file, output_file

    def test_init_missing_ground_truth_file(self):
        """Test initialization with missing ground truth file"""
        with patch.object(BaseError, "exit_program", lambda self: None):
            with pytest.raises(
                ValueErrorHandler, match="Ground truth file path is required"
            ):
                MaterialsDataAgenticEvaluatorFlow(
                    ground_truth_file=None,
                    test_data_file="test.json",
                    used_agent_model_name="test-model",
                )

    def test_init_missing_test_data_file(self):
        """Test initialization with missing test data file"""
        with patch.object(BaseError, "exit_program", lambda self: None):
            with pytest.raises(
                ValueErrorHandler, match="Test data file path is required"
            ):
                MaterialsDataAgenticEvaluatorFlow(
                    ground_truth_file="gt.json",
                    test_data_file=None,
                    used_agent_model_name="test-model",
                )

    def test_init_missing_agent_model_name(self):
        """Test initialization with missing agent model name"""
        with patch.object(BaseError, "exit_program", lambda self: None):
            with pytest.raises(
                ValueErrorHandler, match="Used agent model name is required"
            ):
                MaterialsDataAgenticEvaluatorFlow(
                    ground_truth_file="gt.json",
                    test_data_file="test.json",
                    used_agent_model_name=None,
                )

    def test_init_nonexistent_ground_truth_file(self, temp_files):
        """Test initialization with non-existent ground truth file"""
        _, test_file, _ = temp_files

        with patch.object(BaseError, "exit_program", lambda self: None):
            with pytest.raises(ValueErrorHandler, match="Ground truth file not found"):
                MaterialsDataAgenticEvaluatorFlow(
                    ground_truth_file="nonexistent.json",
                    test_data_file=test_file,
                    used_agent_model_name="test-model",
                )

    def test_init_nonexistent_test_data_file(self, temp_files):
        """Test initialization with non-existent test data file"""
        gt_file, _, _ = temp_files

        with patch.object(BaseError, "exit_program", lambda self: None):
            with pytest.raises(ValueErrorHandler, match="Test data file not found"):
                MaterialsDataAgenticEvaluatorFlow(
                    ground_truth_file=gt_file,
                    test_data_file="nonexistent.json",
                    used_agent_model_name="test-model",
                )

    def test_init_successful(self, temp_files):
        """Test successful initialization"""
        gt_file, test_file, output_file = temp_files
        mock_llm = Mock(spec=LLM)
        custom_weights = {"compositions_property_values": 0.4}

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            output_file=output_file,
            used_agent_model_name="test-model",
            is_synthesis_evaluation=False,
            weights=custom_weights,
            llm=mock_llm,
        )

        assert flow.state.ground_truth_file == gt_file
        assert flow.state.test_data_file == test_file
        assert flow.state.output_file == output_file
        assert flow.state.used_agent_model_name == "test-model"
        assert flow.state.is_synthesis_evaluation is False
        assert flow.state.llm == mock_llm
        assert flow.state.weights["compositions_property_values"] == 0.4

    def test_init_default_weights(self, temp_files):
        """Test initialization with default weights"""
        gt_file, test_file, output_file = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        expected_weights = {
            "compositions_property_values": 0.3,
            "property_unit": 0.1,
            "family": 0.1,
            "method": 0.1,
            "precursors": 0.15,
            "characterization_techniques": 0.15,
            "steps": 0.1,
        }

        assert flow.state.weights == expected_weights

    def test_calculate_tp_fp_fn_composition_data(self, temp_files):
        """Test TP/FP/FN calculation for composition data"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        details = {
            "composition_data": {
                "property_unit": {"match_value": 1, "reference": "MPa", "test": "MPa"},
                "family": {"match_value": 0, "reference": "ceramics", "test": "metals"},
                "compositions_property_values": {
                    "key_matches": [
                        {
                            "match_value": 1,
                            "reference_key": "Al2O3",
                            "test_key": "Al2O3",
                        },
                        {"match_value": 0, "reference_key": "SiO2", "test_key": "SiO"},
                    ],
                    "value_matches": [
                        {"match_value": 1, "reference_value": 100, "test_value": 100},
                        {"match_value": 0, "reference_value": 200, "test_value": 180},
                    ],
                    "missing_keys": ["TiO2"],
                    "extra_keys": ["Fe2O3"],
                },
            }
        }

        metrics = flow._calculate_tp_fp_fn(details, "composition_data")

        assert (
            metrics["true_positives"] == 3
        )  # property_unit(1) + key_match(1) + value_match(1)
        assert metrics["false_positives"] >= 2  # family mismatch + extra keys
        assert metrics["false_negatives"] >= 2  # family mismatch + missing keys

    def test_calculate_tp_fp_fn_synthesis_data(self, temp_files):
        """Test TP/FP/FN calculation for synthesis data"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        details = {
            "synthesis_data": {
                "method": {"match_value": 1, "reference": "sol-gel", "test": "sol-gel"},
                "precursors": {
                    "matches": [
                        {
                            "match_value": 1,
                            "reference_item": "Al(NO3)3",
                            "test_item": "aluminum nitrate",
                        },
                        {
                            "match_value": 0,
                            "reference_item": "SiO2",
                            "test_item": "silicon",
                        },
                    ],
                    "missing_items": ["TiO2"],
                    "extra_items": ["Fe2O3"],
                },
                "characterization_techniques": {
                    "matches": [
                        {"match_value": 1, "reference_item": "XRD", "test_item": "XRD"}
                    ],
                    "missing_items": [],
                    "extra_items": ["SEM"],
                },
                "steps": {
                    "match_value": 0.8,  # Float value between 0 and 1
                    "reference_steps": ["Mix", "Heat"],
                    "test_steps": ["Combine", "Heat"],
                },
            }
        }

        metrics = flow._calculate_tp_fp_fn(details, "synthesis_data")

        assert (
            metrics["true_positives"] >= 3
        )  # method(1) + precursor match(1) + technique match(1) + steps partial(1)
        assert metrics["false_positives"] >= 1  # extra items
        assert metrics["false_negatives"] >= 1  # missing items

    def test_calculate_classification_metrics(self, temp_files):
        """Test precision, recall, F1 calculation"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        metrics = {"true_positives": 8, "false_positives": 2, "false_negatives": 3}

        flow._calculate_classification_metrics(metrics)

        expected_precision = 8 / (8 + 2)  # 0.8
        expected_recall = 8 / (8 + 3)  # ~0.727
        expected_f1 = (
            2
            * expected_precision
            * expected_recall
            / (expected_precision + expected_recall)
        )

        assert abs(metrics["precision"] - expected_precision) < 1e-6
        assert abs(metrics["recall"] - expected_recall) < 1e-6
        assert abs(metrics["f1_score"] - expected_f1) < 1e-6

    def test_calculate_classification_metrics_zero_division(self, temp_files):
        """Test classification metrics with zero division cases"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        # Test case: no predictions made
        metrics = {"true_positives": 0, "false_positives": 0, "false_negatives": 5}
        flow._calculate_classification_metrics(metrics)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0

    def test_calculate_score_composition_data(self, temp_files):
        """Test score calculation for composition data"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        weights = flow.state.weights

        # Perfect match scenario
        details = {
            "composition_data": {
                "property_unit": {"match_value": 1},
                "family": {"match_value": 1},
                "compositions_property_values": {
                    "total_match": 5,
                    "total_ground_truth_keys": 5,
                },
            }
        }

        score = flow._calculate_score(details, "composition_data", weights)
        assert score == 1.0  # Perfect score

        # Partial match scenario
        details["composition_data"]["property_unit"]["match_value"] = 0
        details["composition_data"]["compositions_property_values"]["total_match"] = 3

        score = flow._calculate_score(details, "composition_data", weights)
        expected_score = (
            1.0
            - weights["property_unit"]
            - weights["compositions_property_values"] * (1 - 3 / 5)
        )
        assert abs(score - expected_score) < 1e-6

    def test_calculate_score_synthesis_data(self, temp_files):
        """Test score calculation for synthesis data"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        weights = flow.state.weights

        # Perfect match scenario
        details = {
            "synthesis_data": {
                "method": {"match_value": 1},
                "precursors": {"total_match": 3, "total_ground_truth_items": 3},
                "characterization_techniques": {
                    "total_match": 2,
                    "total_ground_truth_items": 2,
                },
                "steps": {"match_value": 1.0},
            }
        }

        score = flow._calculate_score(details, "synthesis_data", weights)
        assert score == 1.0  # Perfect score

        # Partial match scenario with float steps value
        details["synthesis_data"]["steps"]["match_value"] = 0.7
        details["synthesis_data"]["precursors"]["total_match"] = 2

        score = flow._calculate_score(details, "synthesis_data", weights)
        expected_penalty = weights["precursors"] * (1 - 2 / 3) + weights["steps"] * (
            1 - 0.7
        )
        expected_score = 1.0 - expected_penalty
        assert abs(score - expected_score) < 1e-6

    def test_load_existing_results_file_exists(self, temp_files):
        """Test loading existing results when file exists"""
        gt_file, test_file, output_file = temp_files

        # Create existing results file
        existing_results = {"test": "data", "item_results": {"10.1000/test": {}}}
        with open(output_file, "w") as f:
            json.dump(existing_results, f)

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            output_file=output_file,
            used_agent_model_name="test-model",
        )

        result = flow._load_existing_results()
        assert result == existing_results

    def test_load_existing_results_file_not_exists(self, temp_files):
        """Test loading existing results when file doesn't exist"""
        gt_file, test_file, output_file = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            output_file=output_file,
            used_agent_model_name="test-model",
        )

        result = flow._load_existing_results()
        assert result == {}

    def test_update_results_file(self, temp_files):
        """Test updating results file"""
        gt_file, test_file, output_file = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            output_file=output_file,
            used_agent_model_name="test-model",
        )

        test_results = {"test": "data", "metrics": {"accuracy": 0.85}}
        success = flow._update_results_file(test_results)

        assert success is True
        assert os.path.exists(output_file)

        # Verify file contents
        with open(output_file, "r") as f:
            saved_results = json.load(f)
        assert saved_results == test_results

    def test_normalize_weights(self, temp_files):
        """Test weight normalization"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        normalized = flow._normalize_weights()

        # Check that composition weights sum to 0.5
        comp_weights = ["property_unit", "family", "compositions_property_values"]
        comp_sum = sum(normalized[key] for key in comp_weights)
        assert abs(comp_sum - 0.5) < 1e-6

        # Check that synthesis weights sum to 0.5
        synth_weights = ["method", "precursors", "characterization_techniques", "steps"]
        synth_sum = sum(normalized[key] for key in synth_weights)
        assert abs(synth_sum - 0.5) < 1e-6

    def test_is_section_empty(self, temp_files):
        """Test section emptiness detection"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        # Test empty cases
        assert flow._is_section_empty(None) is True
        assert flow._is_section_empty({}) is True
        assert flow._is_section_empty({"key": None}) is True
        assert flow._is_section_empty({"key": {}}) is True
        assert flow._is_section_empty({"key": []}) is True

        # Test non-empty cases
        assert flow._is_section_empty({"key": "value"}) is False
        assert flow._is_section_empty({"key": {"nested": "value"}}) is False
        assert flow._is_section_empty({"key": ["item"]}) is False

    def test_count_all_items(self, temp_files):
        """Test counting all items in data entry"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        item = {
            "composition_data": {
                "property_unit": "MPa",
                "family": "ceramics",
                "compositions_property_values": {"Al2O3": 100, "SiO2": 200},
            },
            "synthesis_data": {
                "method": "sol-gel",
                "precursors": ["Al(NO3)3", "SiO2"],
                "characterization_techniques": ["XRD"],
                "steps": ["Mix", "Heat"],
            },
        }

        count = flow._count_all_items(item)
        # Expected: property_unit(1) + family(1) + comp_values*2(4) + method(1) + precursors(2) + techniques(1) + steps(2) = 12
        expected_count = 1 + 1 + 4 + 1 + 2 + 1 + 2
        assert count == expected_count

    def test_count_all_items_none_data(self, temp_files):
        """Test counting items with None data"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        item = {"composition_data": None, "synthesis_data": None}

        count = flow._count_all_items(item)
        assert count == 0

    @patch(
        "comproscanner.post_processing.evaluation.eval_flow.eval_flow.CompositionEvaluationCrew"
    )
    @patch(
        "comproscanner.post_processing.evaluation.eval_flow.eval_flow.SynthesisEvaluationCrew"
    )
    def test_load_data_method(self, mock_synth_crew, mock_comp_crew, temp_files):
        """Test the load_data method"""
        gt_file, test_file, output_file = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            output_file=output_file,
            used_agent_model_name="test-model",
        )

        result = flow.load_data()

        assert isinstance(result, dict)
        assert "ground_truth_count" in result
        assert "test_data_count" in result
        assert "processed_count" in result
        assert "remaining_dois" in result

        assert result["ground_truth_count"] == 2  # Based on sample data
        assert result["test_data_count"] == 2
        assert result["processed_count"] == 0  # No existing results
        assert len(result["remaining_dois"]) == 2

    def test_load_data_invalid_json(self, temp_files):
        """Test load_data with invalid JSON files"""
        gt_file, test_file, output_file = temp_files

        # Write invalid JSON to ground truth file
        with open(gt_file, "w") as f:
            f.write("invalid json content")

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            output_file=output_file,
            used_agent_model_name="test-model",
        )

        with patch.object(BaseError, "exit_program", lambda self: None):
            with pytest.raises(ValueErrorHandler, match="Failed to load data files"):
                flow.load_data()

    @patch("comproscanner.post_processing.evaluation.eval_flow.eval_flow.tqdm")
    @patch(
        "comproscanner.post_processing.evaluation.eval_flow.eval_flow.CompositionEvaluationCrew"
    )
    @patch(
        "comproscanner.post_processing.evaluation.eval_flow.eval_flow.SynthesisEvaluationCrew"
    )
    def test_evaluate_items_no_remaining_dois(
        self, mock_synth_crew, mock_comp_crew, mock_tqdm, temp_files
    ):
        """Test evaluate_items when no DOIs remain to process"""
        gt_file, test_file, output_file = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            output_file=output_file,
            used_agent_model_name="test-model",
        )

        # Set up state as if all items are already processed
        flow.state.ground_truth_data = {"10.1000/test1": {}}
        flow.state.existing_results = {"item_results": {"10.1000/test1": {}}}
        flow.state.evaluation_details = {}

        data_info = {"remaining_dois": []}
        result = flow.evaluate_items(data_info)

        # Should return existing evaluation details without processing
        assert result == flow.state.evaluation_details
        mock_comp_crew.assert_not_called()
        mock_synth_crew.assert_not_called()

    def test_enhance_evaluation_details(self, temp_files):
        """Test enhancement of evaluation details"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        details = {
            "composition_data": {
                "compositions_property_values": {
                    "key_matches": [{"match_value": 1}],
                    "value_matches": [{"match_value": 1}],
                    "pair_matches": [{"match_value": 1}],
                    "total_ground_truth_keys": 1,
                }
            },
            "synthesis_data": {
                "method": {"match_value": 1},
                "precursors": {"total_ground_truth_items": 2, "total_match": 2},
                "steps": {"match_value": 0.8},
            },
        }

        enhanced = flow._enhance_evaluation_details(details)

        # Check that match ratios are added
        cpv = enhanced["composition_data"]["compositions_property_values"]
        assert "key_match_ratio" in cpv
        assert "overall_match_ratio" in cpv
        assert "similarity_score" in cpv

        # Check synthesis enhancements
        assert enhanced["synthesis_data"]["method"]["similarity"] == 1.0
        assert enhanced["synthesis_data"]["precursors"]["similarity"] == 1.0
        assert enhanced["synthesis_data"]["steps"]["steps_match"] == "true"

    def test_enhance_evaluation_details_empty_input(self, temp_files):
        """Test enhancement with empty or invalid input"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        # Test with empty dict
        result = flow._enhance_evaluation_details({})
        assert result == {}

        # Test with None
        result = flow._enhance_evaluation_details(None)
        assert result is None

        # Test with non-dict
        result = flow._enhance_evaluation_details("not a dict")
        assert result == "not a dict"

    def test_calculate_key_value_match_ratios(self, temp_files):
        """Test calculation of key-value match ratios"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        details = {
            "composition_data": {
                "compositions_property_values": {
                    "key_matches": [{"match_value": 1}, {"match_value": 0}],
                    "value_matches": [{"match_value": 1}, {"match_value": 1}],
                    "pair_matches": [{"match_value": 1}, {"match_value": 0}],
                    "total_ground_truth_keys": 2,
                }
            }
        }

        result = flow._calculate_key_value_match_ratios(details)

        cpv = result["composition_data"]["compositions_property_values"]
        assert cpv["key_match_ratio"] == 0.5  # 1/2
        assert cpv["value_match_ratio"] == 1.0  # 2/2
        assert cpv["pair_match_ratio"] == 0.5  # 1/2
        assert (
            cpv["overall_match_ratio"] == 0.4 * 0.5 + 0.6 * 0.5
        )  # weighted combination
        assert cpv["match"] == (cpv["overall_match_ratio"] > 0.85)

    @patch(
        "comproscanner.post_processing.evaluation.eval_flow.eval_flow.CompositionEvaluationCrew"
    )
    @patch(
        "comproscanner.post_processing.evaluation.eval_flow.eval_flow.SynthesisEvaluationCrew"
    )
    def test_full_evaluation_flow_mocked(
        self, mock_synth_crew, mock_comp_crew, temp_files
    ):
        """Test full evaluation flow with mocked crews"""
        gt_file, test_file, output_file = temp_files

        # Mock crew responses
        mock_comp_result = Mock()
        mock_comp_result.raw = json.dumps(
            {
                "composition_data": {
                    "property_unit": {
                        "match_value": 1,
                        "reference": "MPa",
                        "test": "MPa",
                    },
                    "family": {
                        "match_value": 1,
                        "reference": "ceramics",
                        "test": "ceramics",
                    },
                    "compositions_property_values": {
                        "key_matches": [{"match_value": 1}],
                        "value_matches": [{"match_value": 1}],
                        "pair_matches": [{"match_value": 1}],
                        "total_ground_truth_keys": 1,
                        "total_match": 1,
                        "missing_keys": [],
                        "extra_keys": [],
                    },
                }
            }
        )

        mock_synth_result = Mock()
        mock_synth_result.raw = json.dumps(
            {
                "synthesis_data": {
                    "method": {
                        "match_value": 1,
                        "reference": "sol-gel",
                        "test": "sol-gel",
                    },
                    "precursors": {
                        "matches": [{"match_value": 1}],
                        "total_ground_truth_items": 1,
                        "total_match": 1,
                        "missing_items": [],
                        "extra_items": [],
                    },
                    "characterization_techniques": {
                        "matches": [{"match_value": 1}],
                        "total_ground_truth_items": 1,
                        "total_match": 1,
                        "missing_items": [],
                        "extra_items": [],
                    },
                    "steps": {"match_value": 1.0},
                }
            }
        )

        mock_comp_crew_instance = Mock()
        mock_comp_crew_instance.kickoff.return_value = mock_comp_result
        mock_comp_crew.return_value.crew.return_value = mock_comp_crew_instance

        mock_synth_crew_instance = Mock()
        mock_synth_crew_instance.kickoff.return_value = mock_synth_result
        mock_synth_crew.return_value.crew.return_value = mock_synth_crew_instance

        # Create flow and run evaluation
        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            output_file=output_file,
            used_agent_model_name="test-model",
        )

        # Manually trigger the flow steps
        with patch(
            "comproscanner.post_processing.evaluation.eval_flow.eval_flow.tqdm"
        ) as mock_tqdm:
            mock_tqdm.return_value = [
                "10.1000/test1",
                "10.1000/test2",
            ]  # Mock tqdm iteration

            # Load data
            flow.load_data()

            # Evaluate items
            data_info = {"remaining_dois": list(flow.state.ground_truth_data.keys())}
            flow.evaluate_items(data_info)

            # Finalize evaluation
            result = flow.finalize_evaluation(flow.state.evaluation_details)

        # Verify results structure
        assert isinstance(result, dict)
        assert "overall_accuracy" in result
        assert "overall_composition_accuracy" in result
        assert "overall_synthesis_accuracy" in result
        assert "item_results" in result
        assert "absolute_classification_metrics" in result
        assert "normalized_classification_metrics" in result

        # Verify that output file was created
        assert os.path.exists(output_file)

    def test_custom_weights_validation(self, temp_files):
        """Test custom weights are properly applied"""
        gt_file, test_file, _ = temp_files

        custom_weights = {
            "compositions_property_values": 0.4,
            "property_unit": 0.05,
            "family": 0.05,
            "method": 0.2,
            "precursors": 0.1,
            "characterization_techniques": 0.1,
            "steps": 0.1,
        }

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
            weights=custom_weights,
        )

        # Verify custom weights are applied while preserving defaults for missing keys
        for key, value in custom_weights.items():
            assert flow.state.weights[key] == value

    @pytest.mark.parametrize("is_synthesis_evaluation", [True, False])
    def test_synthesis_evaluation_flag(self, is_synthesis_evaluation, temp_files):
        """Test behavior with synthesis evaluation enabled/disabled"""
        gt_file, test_file, output_file = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            output_file=output_file,
            used_agent_model_name="test-model",
            is_synthesis_evaluation=is_synthesis_evaluation,
        )

        assert flow.state.is_synthesis_evaluation == is_synthesis_evaluation

    def test_update_combined_metrics(self, temp_files):
        """Test updating combined metrics from item results"""
        gt_file, test_file, _ = temp_files

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            used_agent_model_name="test-model",
        )

        # Create mock combined results with some item results
        combined_results = {
            "absolute_classification_metrics": {},
            "normalized_classification_metrics": {},
            "item_results": {
                "10.1000/test1": {
                    "field_scores": {"composition_data": 0.8, "synthesis_data": 0.9},
                    "absolute_classification_metrics": {
                        "true_positives": 5,
                        "false_positives": 1,
                        "false_negatives": 2,
                    },
                    "normalized_classification_metrics": {
                        "true_positives": 0.4,
                        "false_positives": 0.1,
                        "false_negatives": 0.1,
                    },
                },
                "10.1000/test2": {
                    "field_scores": {"composition_data": 0.7, "synthesis_data": 0.6},
                    "absolute_classification_metrics": {
                        "true_positives": 3,
                        "false_positives": 2,
                        "false_negatives": 1,
                    },
                    "normalized_classification_metrics": {
                        "true_positives": 0.3,
                        "false_positives": 0.15,
                        "false_negatives": 0.05,
                    },
                },
            },
        }

        flow._update_combined_metrics(combined_results)

        # Check absolute metrics are summed correctly
        assert (
            combined_results["absolute_classification_metrics"]["true_positives"] == 8
        )
        assert (
            combined_results["absolute_classification_metrics"]["false_positives"] == 3
        )
        assert (
            combined_results["absolute_classification_metrics"]["false_negatives"] == 3
        )

        # Check normalized metrics are summed correctly
        assert (
            abs(
                combined_results["normalized_classification_metrics"]["true_positives"]
                - 0.7
            )
            < 1e-6
        )
        assert (
            abs(
                combined_results["normalized_classification_metrics"]["false_positives"]
                - 0.25
            )
            < 1e-6
        )

        # Check accuracy calculations
        assert (
            abs(combined_results["overall_composition_accuracy"] - 0.75) < 1e-6
        )  # (0.8 + 0.7) / 2
        assert (
            abs(combined_results["overall_synthesis_accuracy"] - 0.75) < 1e-6
        )  # (0.9 + 0.6) / 2
        assert (
            abs(combined_results["overall_accuracy"] - 0.75) < 1e-6
        )  # average of comp and synth

    def test_file_path_creation(self, temp_files):
        """Test that output directory is created if it doesn't exist"""
        gt_file, test_file, _ = temp_files

        # Create a nested output path
        nested_output = os.path.join(
            os.path.dirname(gt_file), "nested", "deep", "output.json"
        )

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            output_file=nested_output,
            used_agent_model_name="test-model",
        )

        test_results = {"test": "data"}
        success = flow._update_results_file(test_results)

        assert success is True
        assert os.path.exists(nested_output)
        assert os.path.exists(os.path.dirname(nested_output))

    def test_missing_dois_handling(self, temp_files):
        """Test handling of DOIs missing from test data"""
        gt_file, test_file, output_file = temp_files

        # Modify test data to remove one DOI
        with open(test_file, "r") as f:
            test_data = json.load(f)

        # Remove one DOI from test data
        del test_data["10.1000/test2"]

        with open(test_file, "w") as f:
            json.dump(test_data, f)

        flow = MaterialsDataAgenticEvaluatorFlow(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            output_file=output_file,
            used_agent_model_name="test-model",
        )

        # Load data to set up state
        flow.load_data()

        # Check that all ground truth DOIs are considered for processing
        assert len(flow.state.remaining_dois) == 2  # All ground truth DOIs
        assert "10.1000/test1" in flow.state.remaining_dois
        assert "10.1000/test2" in flow.state.remaining_dois
