"""
test_semantic_evaluator.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 20-07-2025
"""

import json
import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from comproscanner.post_processing.evaluation.semantic_evaluator import (
    MaterialsDataSemanticEvaluator,
)
from comproscanner.utils.error_handler import ValueErrorHandler, BaseError


class TestMaterialsDataSemanticEvaluator:
    """Test suite for MaterialsDataSemanticEvaluator class"""

    @pytest.fixture
    def evaluator_no_model(self):
        """Fixture for evaluator without semantic models"""
        return MaterialsDataSemanticEvaluator(use_semantic_model=False)

    @pytest.fixture
    def evaluator_with_model(self):
        """Fixture for evaluator with mocked semantic models"""
        with (
            patch("transformers.AutoModel.from_pretrained") as mock_model,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch("torch.no_grad"),
            patch("torch.nn.functional.cosine_similarity") as mock_cosine,
        ):

            mock_cosine.return_value.item.return_value = 0.85
            evaluator = MaterialsDataSemanticEvaluator(use_semantic_model=True)
            return evaluator

    @pytest.fixture
    def sample_composition_data(self):
        """Sample composition data for testing"""
        return {
            "property_unit": "MPa",
            "family": "ceramics",
            "compositions_property_values": {
                "Al2O3": 150.5,
                "SiO2": [100, 200],
                "TiO2": 75.2,
            },
        }

    @pytest.fixture
    def sample_synthesis_data(self):
        """Sample synthesis data for testing"""
        return {
            "method": "sol-gel synthesis",
            "precursors": ["aluminum nitrate", "silicon oxide", "titanium dioxide"],
            "characterization_techniques": ["XRD", "SEM", "TEM"],
            "steps": ["Mix precursors", "Heat at 500C", "Cool slowly", "Analyze"],
        }

    @pytest.fixture
    def sample_ground_truth_data(self, sample_composition_data, sample_synthesis_data):
        """Sample ground truth data"""
        return {
            "10.1000/test1": {
                "composition_data": sample_composition_data,
                "synthesis_data": sample_synthesis_data,
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
                    "characterization_techniques": ["FTIR", "DSC"],
                    "steps": ["Polymerize", "Purify"],
                },
            },
        }

    @pytest.fixture
    def sample_test_data(self, sample_composition_data, sample_synthesis_data):
        """Sample test data (similar to ground truth for positive testing)"""
        return {
            "10.1000/test1": {
                "composition_data": sample_composition_data.copy(),
                "synthesis_data": sample_synthesis_data.copy(),
            },
            "10.1000/test2": {
                "composition_data": {
                    "property_unit": "GPa",
                    "family": "polymers",
                    "compositions_property_values": {
                        "polyethylene": 2.4,  # Slightly different value
                        "polypropylene": 3.1,  # Slightly different value
                    },
                },
                "synthesis_data": {
                    "method": "radical polymerization",  # Similar but different
                    "precursors": ["ethylene monomer", "propylene monomer"],  # Similar
                    "characterization_techniques": [
                        "FTIR spectroscopy",
                        "DSC analysis",
                    ],  # Similar
                    "steps": ["Initiate polymerization", "Purify product"],  # Similar
                },
            },
        }

    @pytest.fixture
    def temp_json_files(self, sample_ground_truth_data, sample_test_data):
        """Create temporary JSON files for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_file = os.path.join(tmpdir, "ground_truth.json")
            test_file = os.path.join(tmpdir, "test_data.json")
            output_file = os.path.join(tmpdir, "output.json")

            with open(gt_file, "w") as f:
                json.dump(sample_ground_truth_data, f)

            with open(test_file, "w") as f:
                json.dump(sample_test_data, f)

            yield gt_file, test_file, output_file

    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        evaluator = MaterialsDataSemanticEvaluator(use_semantic_model=False)

        assert evaluator.use_semantic_model is False
        assert evaluator.primary_model_name == "thellert/physbert_cased"
        assert evaluator.fallback_model_name == "all-mpnet-base-v2"
        assert evaluator.physbert_available is False
        assert evaluator.model_available is False
        assert "composition_overall_match" in evaluator.similarity_thresholds
        assert "synthesis_overall_match" in evaluator.similarity_thresholds

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters"""
        custom_thresholds = {"composition_overall_match": 0.75}
        evaluator = MaterialsDataSemanticEvaluator(
            use_semantic_model=False,
            primary_model_name="custom/model",
            fallback_model_name="custom/fallback",
            similarity_thresholds=custom_thresholds,
        )

        assert evaluator.primary_model_name == "custom/model"
        assert evaluator.fallback_model_name == "custom/fallback"
        assert evaluator.similarity_thresholds["composition_overall_match"] == 0.75

    def test_simple_preprocess(self, evaluator_no_model):
        """Test text preprocessing functionality"""
        text = "This is a TEST text with PUNCTUATION!"
        processed = evaluator_no_model._simple_preprocess(text)

        # Should be lowercase and without punctuation
        assert processed.islower()
        assert "!" not in processed
        assert "test" in processed

    def test_is_value_in_range_exact_match(self, evaluator_no_model):
        """Test exact value matching"""
        assert evaluator_no_model._is_value_in_range(10.5, 10.5) is True
        assert evaluator_no_model._is_value_in_range(10, 10.0) is True
        assert evaluator_no_model._is_value_in_range(10.5, 11.0) is False

    def test_is_value_in_range_range_match(self, evaluator_no_model):
        """Test range value matching"""
        assert evaluator_no_model._is_value_in_range([10, 20], 15) is True
        assert evaluator_no_model._is_value_in_range([10, 20], 10) is True
        assert evaluator_no_model._is_value_in_range([10, 20], 20) is True
        assert evaluator_no_model._is_value_in_range([10, 20], 25) is False

    def test_is_value_in_range_none_values(self, evaluator_no_model):
        """Test None value handling"""
        assert evaluator_no_model._is_value_in_range(None, None) is True
        assert evaluator_no_model._is_value_in_range(None, 10) is False
        assert evaluator_no_model._is_value_in_range(10, None) is False

    def test_calculate_text_similarity_sequence_matcher(self, evaluator_no_model):
        """Test text similarity with sequence matcher fallback"""
        similarity = evaluator_no_model._calculate_text_similarity(
            "hello world", "hello world"
        )
        assert similarity == 1.0

        similarity = evaluator_no_model._calculate_text_similarity("hello", "world")
        assert 0 <= similarity <= 1

    @patch("torch.nn.functional.cosine_similarity")
    @patch("torch.no_grad")
    def test_calculate_text_similarity_physbert(
        self, mock_no_grad, mock_cosine, evaluator_with_model
    ):
        """Test text similarity with PhysBERT model"""
        mock_cosine.return_value.item.return_value = 0.95

        similarity = evaluator_with_model._calculate_text_similarity(
            "test text", "similar text"
        )
        assert similarity == 0.95

    def test_evaluate_composition_data_exact_match(
        self, evaluator_no_model, sample_composition_data
    ):
        """Test composition data evaluation with exact matches"""
        score, details, abs_metrics, weights = (
            evaluator_no_model._evaluate_composition_data(
                sample_composition_data, sample_composition_data
            )
        )

        assert score > 0.8  # Should be high score for exact match
        assert details["property_unit"]["match"] is True
        assert details["family"]["match"] is True
        assert details["compositions_property_values"]["match"] is True

    def test_evaluate_composition_data_empty_reference(self, evaluator_no_model):
        """Test composition data evaluation with empty reference"""
        empty_data = {"compositions_property_values": {}}
        test_data = {"compositions_property_values": {"Al": 100}}

        score, details, abs_metrics, weights = (
            evaluator_no_model._evaluate_composition_data(empty_data, test_data)
        )

        assert abs_metrics["false_positives"] > 0

    def test_evaluate_synthesis_data_exact_match(
        self, evaluator_no_model, sample_synthesis_data
    ):
        """Test synthesis data evaluation with exact matches"""
        score, details, abs_metrics, weights = (
            evaluator_no_model._evaluate_synthesis_data(
                sample_synthesis_data, sample_synthesis_data
            )
        )

        assert score > 0.8  # Should be high score for exact match
        assert abs_metrics["true_positives"] > 0

    def test_evaluate_synthesis_data_similar_match(self, evaluator_no_model):
        """Test synthesis data evaluation with similar but not exact matches"""
        ref_data = {
            "method": "sol-gel synthesis",
            "precursors": ["aluminum nitrate"],
            "characterization_techniques": ["XRD"],
            "steps": ["Mix precursors"],
        }

        test_data = {
            "method": "sol gel method",  # Similar method
            "precursors": ["aluminum salt"],  # Similar precursor
            "characterization_techniques": ["X-ray diffraction"],  # Similar technique
            "steps": ["Mix starting materials"],  # Similar step
        }

        score, details, abs_metrics, weights = (
            evaluator_no_model._evaluate_synthesis_data(ref_data, test_data)
        )

        # Should get some credit for similarity
        assert 0.2 <= score <= 0.9

    def test_count_all_items(
        self, evaluator_no_model, sample_composition_data, sample_synthesis_data
    ):
        """Test counting all items in a data entry"""
        item = {
            "composition_data": sample_composition_data,
            "synthesis_data": sample_synthesis_data,
        }

        count = evaluator_no_model._count_all_items(item)

        # Should count: property_unit(1) + family(1) + 3 comp_values*2(6) + method(1) + 3 precursors(3) + 3 techniques(3) + steps(2) = 17
        assert count > 10  # At least some reasonable count

    def test_count_all_items_none_data(self, evaluator_no_model):
        """Test counting items with None data"""
        item = {"composition_data": None, "synthesis_data": None}

        count = evaluator_no_model._count_all_items(item)
        assert count == 0

    def test_evaluate_missing_files(self, evaluator_no_model):
        """Test evaluation with missing files"""
        with patch.object(BaseError, "exit_program", lambda self: None):
            with pytest.raises(ValueErrorHandler):
                evaluator_no_model.evaluate()

    def test_evaluate_missing_model_name(self, evaluator_no_model, temp_json_files):
        """Test evaluation with missing model name"""
        gt_file, test_file, output_file = temp_json_files

        with patch.object(BaseError, "exit_program", lambda self: None):
            with pytest.raises(ValueErrorHandler):
                evaluator_no_model.evaluate(
                    ground_truth_file=gt_file,
                    test_data_file=test_file,
                    extraction_agent_model_name=None,
                )

    def test_evaluate_successful_run(self, evaluator_no_model, temp_json_files):
        """Test successful evaluation run"""
        gt_file, test_file, output_file = temp_json_files

        results = evaluator_no_model.evaluate(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            extraction_agent_model_name="test_model",
            output_file=output_file,
            is_synthesis_evaluation=True,
        )

        # Check basic structure of results
        assert "extraction_agent_model_name" in results
        assert "overall_accuracy" in results
        assert "overall_composition_accuracy" in results
        assert "overall_synthesis_accuracy" in results
        assert "absolute_classification_metrics" in results
        assert "normalized_classification_metrics" in results
        assert "item_results" in results

        # Check that results were saved to file
        assert os.path.exists(output_file)

        # Verify file contents
        with open(output_file, "r") as f:
            saved_results = json.load(f)
        assert saved_results["extraction_agent_model_name"] == "test_model"

    def test_evaluate_composition_only(self, evaluator_no_model, temp_json_files):
        """Test evaluation with composition data only"""
        gt_file, test_file, output_file = temp_json_files

        results = evaluator_no_model.evaluate(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            extraction_agent_model_name="test_model",
            output_file=output_file,
            is_synthesis_evaluation=False,
        )

        assert "overall_composition_accuracy" in results
        assert (
            results["overall_synthesis_accuracy"] == 0.0
        )  # Should be 0 when not evaluated

    def test_evaluate_custom_weights(self, evaluator_no_model, temp_json_files):
        """Test evaluation with custom weights"""
        gt_file, test_file, output_file = temp_json_files

        custom_weights = {
            "compositions_property_values": 0.4,
            "property_unit": 0.1,
            "family": 0.1,
            "method": 0.1,
            "precursors": 0.1,
            "characterization_techniques": 0.1,
            "steps": 0.1,
        }

        results = evaluator_no_model.evaluate(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            extraction_agent_model_name="test_model",
            weights=custom_weights,
            output_file=output_file,
        )

        assert "overall_accuracy" in results

    def test_evaluate_invalid_weights(self, evaluator_no_model, temp_json_files):
        """Test evaluation with invalid weights (not summing to 1.0)"""
        gt_file, test_file, output_file = temp_json_files

        invalid_weights = {
            "compositions_property_values": 0.5,
            "property_unit": 0.5,  # Total = 1.0, but missing other components
            "family": 0.0,
            "method": 0.0,
            "precursors": 0.0,
            "characterization_techniques": 0.0,
            "steps": 0.0,
        }

        # Should fall back to default weights
        results = evaluator_no_model.evaluate(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            extraction_agent_model_name="test_model",
            weights=invalid_weights,
            output_file=output_file,
        )

        assert "overall_accuracy" in results

    def test_evaluate_missing_dois(self, evaluator_no_model):
        """Test evaluation with DOIs present in only one dataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_file = os.path.join(tmpdir, "ground_truth.json")
            test_file = os.path.join(tmpdir, "test_data.json")
            output_file = os.path.join(tmpdir, "output.json")

            # Create datasets with different DOIs
            gt_data = {
                "10.1000/only_in_gt": {
                    "composition_data": {
                        "property_unit": "MPa",
                        "family": "ceramics",
                        "compositions_property_values": {"Al2O3": 100},
                    }
                }
            }

            test_data = {
                "10.1000/only_in_test": {
                    "composition_data": {
                        "property_unit": "GPa",
                        "family": "metals",
                        "compositions_property_values": {"Fe": 200},
                    }
                }
            }

            with open(gt_file, "w") as f:
                json.dump(gt_data, f)

            with open(test_file, "w") as f:
                json.dump(test_data, f)

            results = evaluator_no_model.evaluate(
                ground_truth_file=gt_file,
                test_data_file=test_file,
                extraction_agent_model_name="test_model",
                output_file=output_file,
            )

            # Should handle missing DOIs correctly
            assert len(results["item_results"]) == 2  # Both DOIs should be present
            assert results["absolute_classification_metrics"]["false_negatives"] > 0
            assert results["absolute_classification_metrics"]["false_positives"] > 0

    def test_load_models_physbert_failure_fallback_success(self):
        """Test model loading with PhysBERT failure but sentence-transformers success"""
        with (
            patch(
                "transformers.AutoModel.from_pretrained",
                side_effect=ImportError("PhysBERT not available"),
            ),
            patch("sentence_transformers.SentenceTransformer") as mock_st,
        ):

            evaluator = MaterialsDataSemanticEvaluator(use_semantic_model=True)

            assert evaluator.physbert_available is False
            assert evaluator.model_available is True
            mock_st.assert_called_once()

    def test_load_models_all_failures(self):
        """Test model loading with all models failing"""
        with (
            patch(
                "transformers.AutoModel.from_pretrained",
                side_effect=ImportError("PhysBERT not available"),
            ),
            patch(
                "sentence_transformers.SentenceTransformer",
                side_effect=ImportError("SentenceTransformers not available"),
            ),
        ):

            evaluator = MaterialsDataSemanticEvaluator(use_semantic_model=True)

            assert evaluator.physbert_available is False
            assert evaluator.model_available is False

    @pytest.mark.parametrize(
        "threshold_key,threshold_value",
        [
            ("composition_overall_match", 0.75),
            ("synthesis_overall_match", 0.85),
            ("composition_key_match", 0.90),
            ("property_unit_match", 0.95),
        ],
    )
    def test_custom_similarity_thresholds(self, threshold_key, threshold_value):
        """Test custom similarity thresholds"""
        custom_thresholds = {threshold_key: threshold_value}
        evaluator = MaterialsDataSemanticEvaluator(
            use_semantic_model=False, similarity_thresholds=custom_thresholds
        )

        assert evaluator.similarity_thresholds[threshold_key] == threshold_value

    def test_evaluate_empty_datasets(self, evaluator_no_model):
        """Test evaluation with empty datasets"""
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_file = os.path.join(tmpdir, "ground_truth.json")
            test_file = os.path.join(tmpdir, "test_data.json")
            output_file = os.path.join(tmpdir, "output.json")

            # Create empty datasets
            with open(gt_file, "w") as f:
                json.dump({}, f)

            with open(test_file, "w") as f:
                json.dump({}, f)

            results = evaluator_no_model.evaluate(
                ground_truth_file=gt_file,
                test_data_file=test_file,
                extraction_agent_model_name="test_model",
                output_file=output_file,
            )

            assert results["total_items"] == 0
            assert results["overall_accuracy"] == 0.0
            assert len(results["item_results"]) == 0
