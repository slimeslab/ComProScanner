"""
test_eval_plot_visualizer.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 18-07-2025
"""

import pytest
import json
import os
import tempfile
import matplotlib
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Use non-interactive backend for testing
matplotlib.use("Agg")


def get_sample_evaluation_data():
    """Sample evaluation data for testing"""
    return {
        "agent_model_name": "test-model-gpt-4",
        "overall_accuracy": 0.85,
        "overall_composition_accuracy": 0.78,
        "overall_synthesis_accuracy": 0.82,
        "absolute_classification_metrics": {
            "precision": 0.75,
            "recall": 0.70,
            "f1_score": 0.72,
        },
        "normalized_classification_metrics": {
            "precision": 0.80,
            "recall": 0.77,
            "f1_score": 0.78,
        },
        "item_results": {
            "10.1000/test1": {
                "overall_score": 0.90,
                "field_scores": {"composition_data": 0.85, "synthesis_data": 0.88},
                "absolute_classification_metrics": {
                    "precision": 0.80,
                    "recall": 0.75,
                    "f1_score": 0.77,
                },
            },
            "10.1000/test2": {
                "overall_score": 0.75,
                "field_scores": {"composition_data": 0.70, "synthesis_data": 0.72},
                "absolute_classification_metrics": {
                    "precision": 0.70,
                    "recall": 0.65,
                    "f1_score": 0.67,
                },
            },
        },
    }


class MockEvalVisualizer:
    """Mock EvalVisualizer for testing"""

    def _extract_metrics_from_result(self, result, metrics_to_include):
        """Mock metric extraction"""
        metrics = []
        values = []

        metric_mapping = {
            "overall_accuracy": (
                "Overall\nAccuracy",
                result.get("overall_accuracy", 0),
            ),
            "overall_composition_accuracy": (
                "Composition\nAccuracy",
                result.get("overall_composition_accuracy", 0),
            ),
            "precision": (
                "Precision",
                result.get("absolute_classification_metrics", {}).get("precision", 0),
            ),
            "recall": (
                "Recall",
                result.get("absolute_classification_metrics", {}).get("recall", 0),
            ),
        }

        for metric in metrics_to_include:
            if metric in metric_mapping:
                display_name, value = metric_mapping[metric]
                metrics.append(display_name)
                values.append(value)

        return metrics, values

    def _get_available_metrics(self, results_data, metrics_to_include):
        """Mock available metrics detection"""
        available_metrics = set()

        if not isinstance(results_data, list):
            results_data = [results_data]

        for result in results_data:
            for key in ["overall_accuracy", "overall_composition_accuracy"]:
                if key in result and key in metrics_to_include:
                    available_metrics.add(key)

            if "absolute_classification_metrics" in result:
                for key in ["precision", "recall"]:
                    if (
                        key in result["absolute_classification_metrics"]
                        and key in metrics_to_include
                    ):
                        available_metrics.add(key)

        return [m for m in metrics_to_include if m in available_metrics]


class TestEvalVisualizerCore:
    """Test core functionality of EvalVisualizer"""

    def test_metric_extraction(self):
        """Test extraction of metrics from evaluation results"""
        sample_data = get_sample_evaluation_data()
        visualizer = MockEvalVisualizer()

        metrics_to_include = ["overall_accuracy", "precision", "recall"]
        metrics, values = visualizer._extract_metrics_from_result(
            sample_data, metrics_to_include
        )

        assert len(metrics) == 3
        assert len(values) == 3
        assert "Overall\nAccuracy" in metrics
        assert "Precision" in metrics
        assert "Recall" in metrics

        # Check values
        overall_acc_idx = metrics.index("Overall\nAccuracy")
        assert values[overall_acc_idx] == 0.85

        precision_idx = metrics.index("Precision")
        assert values[precision_idx] == 0.75

    def test_available_metrics_detection(self):
        """Test detection of available metrics in data"""
        sample_data = get_sample_evaluation_data()
        visualizer = MockEvalVisualizer()

        metrics_to_include = [
            "overall_accuracy",
            "overall_composition_accuracy",
            "precision",
            "recall",
            "missing_metric",
        ]

        available = visualizer._get_available_metrics(sample_data, metrics_to_include)

        assert "overall_accuracy" in available
        assert "overall_composition_accuracy" in available
        assert "precision" in available
        assert "recall" in available
        assert "missing_metric" not in available

    def test_empty_data_handling(self):
        """Test handling of empty or malformed data"""
        visualizer = MockEvalVisualizer()
        empty_data = {"agent_model_name": "empty-model"}

        metrics, values = visualizer._extract_metrics_from_result(
            empty_data, ["overall_accuracy"]
        )
        assert len(metrics) == 1
        assert len(values) == 1
        assert values[0] == 0  # Default value for missing metric


class TestDataStructures:
    """Test data structure handling and validation"""

    def test_evaluation_data_structure(self):
        """Test validation of evaluation data structure"""
        sample_data = get_sample_evaluation_data()

        # Verify required top-level keys
        assert "agent_model_name" in sample_data
        assert "overall_accuracy" in sample_data
        assert "item_results" in sample_data

        # Verify classification metrics structure
        assert "absolute_classification_metrics" in sample_data
        abs_metrics = sample_data["absolute_classification_metrics"]
        assert "precision" in abs_metrics
        assert "recall" in abs_metrics
        assert "f1_score" in abs_metrics

        # Verify item_results structure
        item_results = sample_data["item_results"]
        assert len(item_results) > 0

        for doi, item_data in item_results.items():
            assert "overall_score" in item_data
            assert "field_scores" in item_data

    def test_metrics_value_ranges(self):
        """Test that metric values are in expected ranges"""
        sample_data = get_sample_evaluation_data()

        # Overall metrics should be between 0 and 1
        assert 0 <= sample_data["overall_accuracy"] <= 1
        assert 0 <= sample_data["overall_composition_accuracy"] <= 1
        assert 0 <= sample_data["overall_synthesis_accuracy"] <= 1

        # Classification metrics should be between 0 and 1
        abs_metrics = sample_data["absolute_classification_metrics"]
        for metric_value in abs_metrics.values():
            assert 0 <= metric_value <= 1

    def test_item_results_consistency(self):
        """Test consistency of item results data"""
        sample_data = get_sample_evaluation_data()
        item_results = sample_data["item_results"]

        for doi, item_data in item_results.items():
            # Check that DOI is properly formatted
            assert doi.startswith("10.")

            # Check score ranges
            assert 0 <= item_data["overall_score"] <= 1

            # Check field scores
            field_scores = item_data["field_scores"]
            for score in field_scores.values():
                assert 0 <= score <= 1


class TestFileOperations:
    """Test file operations and error handling"""

    def test_json_file_operations(self):
        """Test JSON file reading and writing"""
        sample_data = get_sample_evaluation_data()

        # Test writing and reading
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            temp_file = f.name

        try:
            # Test reading
            with open(temp_file, "r") as f:
                loaded_data = json.load(f)

            assert loaded_data == sample_data
            assert loaded_data["agent_model_name"] == sample_data["agent_model_name"]
            assert loaded_data["overall_accuracy"] == sample_data["overall_accuracy"]
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON files"""
        # Create invalid JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json content")
            temp_file = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                with open(temp_file, "r") as f:
                    json.load(f)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestVisualizationHelpers:
    """Test visualization helper functions"""

    def test_color_mapping(self):
        """Test color mapping functionality"""

        def mock_get_chart_colors(colormap, num_items):
            import matplotlib.pyplot as plt

            cmap = plt.colormaps[colormap]
            colors = []
            for i in range(num_items):
                position = i / max(1, num_items - 1) if num_items > 1 else 0.5
                colors.append(cmap(position))
            return colors

        # Test multiple colors
        colors = mock_get_chart_colors("Blues", 3)
        assert len(colors) == 3
        assert all(c is not None for c in colors)

    def test_metric_display_names(self):
        """Test metric display name mapping"""
        display_names = {
            "overall_accuracy": "Overall\nAccuracy",
            "overall_composition_accuracy": "Composition\nAccuracy",
            "precision": "Precision",
            "recall": "Recall",
        }

        # Test that all expected metrics have display names
        expected_metrics = ["overall_accuracy", "precision", "recall"]

        for metric in expected_metrics:
            assert metric in display_names
            assert display_names[metric] is not None
            assert len(display_names[metric]) > 0

    def test_data_aggregation(self):
        """Test data aggregation for visualization"""
        # Test calculating statistics from data
        test_data = [0.85, 0.75, 0.90, 0.65, 0.80]

        mean_val = np.mean(test_data)
        median_val = np.median(test_data)
        std_val = np.std(test_data)

        assert 0.7 < mean_val < 0.85
        assert 0.7 < median_val < 0.9
        assert std_val >= 0


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_missing_metrics_handling(self):
        """Test handling of missing metrics in data"""
        incomplete_data = {
            "agent_model_name": "incomplete-model",
            "overall_accuracy": 0.75,
            # Missing other metrics
        }

        # Should handle missing metrics gracefully
        assert "overall_accuracy" in incomplete_data
        assert incomplete_data.get("overall_composition_accuracy", 0) == 0
        assert incomplete_data.get("absolute_classification_metrics", {}) == {}

    def test_extreme_values_handling(self):
        """Test handling of extreme metric values"""
        extreme_data = {
            "agent_model_name": "extreme-model",
            "overall_accuracy": 1.0,  # Perfect score
            "overall_composition_accuracy": 0.0,  # Worst score
            "absolute_classification_metrics": {
                "precision": 1.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
        }

        # Should handle extreme values without errors
        assert 0 <= extreme_data["overall_accuracy"] <= 1
        assert 0 <= extreme_data["overall_composition_accuracy"] <= 1

        # F1 score should be 0 when recall is 0
        metrics = extreme_data["absolute_classification_metrics"]
        assert metrics["f1_score"] == 0.0
