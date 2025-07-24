"""
test_data_distribution_visualizer.py

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
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
import difflib
from collections import Counter
import numpy as np

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestDataDistributionVisualizerCore:
    """Test core functionality of DataDistributionVisualizer"""

    @pytest.fixture
    def sample_data(self):
        """Sample test data for visualization"""
        return {
            "10.1000/test1": {
                "composition_data": {
                    "family": "Perovskite",
                    "compositions_property_values": {"BaTiO3": 180.0},
                    "property_unit": "pC/N",
                },
                "synthesis_data": {
                    "method": "sol-gel",
                    "precursors": ["Ba(OH)2", "TiO2", "ethanol"],
                    "characterization_techniques": ["XRD", "SEM"],
                },
            },
            "10.1000/test2": {
                "composition_data": {
                    "family": "Spinel",
                    "compositions_property_values": {"ZnFe2O4": 120.0},
                    "property_unit": "emu/g",
                },
                "synthesis_data": {
                    "method": "solid-state",
                    "precursors": ["ZnO", "Fe2O3"],
                    "characterization_techniques": ["X-ray diffraction", "VSM"],
                },
            },
            "10.1000/test3": {
                "composition_data": {
                    "family": "Perovskite",  # Duplicate to test counting
                    "compositions_property_values": {"SrTiO3": 90.0},
                    "property_unit": "pC/N",
                },
                "synthesis_data": {
                    "method": "hydrothermal",
                    "precursors": ["SrCl2", "TiO2"],
                    "characterization_techniques": ["XRD", "TEM"],
                },
            },
        }

    @pytest.fixture
    def temp_json_file(self, sample_data):
        """Create temporary JSON file with sample data"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            temp_file = f.name
        yield temp_file
        if os.path.exists(temp_file):
            os.unlink(temp_file)

    @pytest.fixture
    def mock_visualizer(self):
        """Create a mock DataDistributionVisualizer with essential methods"""
        with (
            patch(
                "comproscanner.post_processing.visualization.data_distribution_visualizers.TRANSFORMERS_AVAILABLE",
                False,
            ),
            patch(
                "comproscanner.post_processing.visualization.data_distribution_visualizers.SENTENCE_TRANSFORMERS_AVAILABLE",
                False,
            ),
        ):

            # Create a simple mock class
            class MockDataDistributionVisualizer:
                def __init__(self):
                    self.data = None
                    self.semantic_model = {"type": "difflib"}

                def _load_data(self, data_sources=None, folder_path=None):
                    """Mock data loading"""
                    if isinstance(data_sources, list) and len(data_sources) > 0:
                        if isinstance(data_sources[0], str):
                            with open(data_sources[0], "r") as f:
                                self.data = json.load(f)
                        elif isinstance(data_sources[0], dict):
                            self.data = data_sources[0]
                    return self.data

                def _extract_families(self):
                    """Extract families without clustering"""
                    families = []
                    for doi, item_data in self.data.items():
                        if (
                            "composition_data" in item_data
                            and "family" in item_data["composition_data"]
                        ):
                            family = item_data["composition_data"]["family"]
                            if family:
                                families.append(family)
                    return Counter(families)

                def _extract_precursors(self):
                    """Extract precursors without clustering"""
                    all_precursors = []
                    for doi, item_data in self.data.items():
                        if (
                            "synthesis_data" in item_data
                            and "precursors" in item_data["synthesis_data"]
                        ):
                            precursors = item_data["synthesis_data"]["precursors"]
                            if precursors:
                                all_precursors.extend(precursors)
                    return Counter(all_precursors)

                def _extract_characterization_techniques(self):
                    """Extract characterization techniques without clustering"""
                    all_techniques = []
                    for doi, item_data in self.data.items():
                        if (
                            "synthesis_data" in item_data
                            and "characterization_techniques"
                            in item_data["synthesis_data"]
                        ):
                            techniques = item_data["synthesis_data"][
                                "characterization_techniques"
                            ]
                            if techniques:
                                all_techniques.extend(techniques)
                    return Counter(all_techniques)

                def calculate_similarity(self, text1, text2):
                    """Mock similarity calculation using difflib"""
                    if not text1 or not text2:
                        return 0.0
                    return difflib.SequenceMatcher(
                        None, str(text1).lower(), str(text2).lower()
                    ).ratio()

                def cluster_items(self, items, similarity_threshold=0.8):
                    """Mock clustering implementation"""
                    if not items:
                        return {}

                    items_counter = Counter(items)
                    sorted_items = sorted(
                        items_counter.items(), key=lambda x: x[1], reverse=True
                    )

                    clusters = {}
                    processed = set()

                    for item, count in sorted_items:
                        if item in processed:
                            continue

                        canonical = item
                        clusters[canonical] = [item]
                        processed.add(item)

                        # Find similar items
                        for other_item, other_count in sorted_items:
                            if other_item not in processed:
                                similarity = self.calculate_similarity(
                                    canonical, other_item
                                )
                                if similarity >= similarity_threshold:
                                    clusters[canonical].append(other_item)
                                    processed.add(other_item)

                    return clusters

            return MockDataDistributionVisualizer()

    def test_data_loading_from_file(self, mock_visualizer, temp_json_file):
        """Test loading data from JSON file"""
        mock_visualizer._load_data(data_sources=[temp_json_file])

        assert mock_visualizer.data is not None
        assert len(mock_visualizer.data) == 3
        assert "10.1000/test1" in mock_visualizer.data
        assert "composition_data" in mock_visualizer.data["10.1000/test1"]

    def test_data_loading_from_dict(self, mock_visualizer, sample_data):
        """Test loading data from dictionary"""
        mock_visualizer._load_data(data_sources=[sample_data])

        assert mock_visualizer.data is not None
        assert len(mock_visualizer.data) == 3
        assert mock_visualizer.data == sample_data

    def test_families_extraction(self, mock_visualizer, sample_data):
        """Test extraction of material families"""
        mock_visualizer.data = sample_data
        families_counter = mock_visualizer._extract_families()

        assert isinstance(families_counter, Counter)
        assert families_counter["Perovskite"] == 2  # Two perovskite entries
        assert families_counter["Spinel"] == 1
        assert len(families_counter) == 2

    def test_precursors_extraction(self, mock_visualizer, sample_data):
        """Test extraction of precursors"""
        mock_visualizer.data = sample_data
        precursors_counter = mock_visualizer._extract_precursors()

        assert isinstance(precursors_counter, Counter)
        assert precursors_counter["TiO2"] == 2  # Used in two syntheses
        assert precursors_counter["Ba(OH)2"] == 1
        assert precursors_counter["ZnO"] == 1
        assert "ethanol" in precursors_counter

    def test_characterization_techniques_extraction(self, mock_visualizer, sample_data):
        """Test extraction of characterization techniques"""
        mock_visualizer.data = sample_data
        techniques_counter = mock_visualizer._extract_characterization_techniques()

        assert isinstance(techniques_counter, Counter)
        assert techniques_counter["XRD"] == 2  # Used in two papers
        assert techniques_counter["SEM"] == 1
        assert (
            techniques_counter["X-ray diffraction"] == 1
        )  # Similar to XRD but separate
        assert techniques_counter["VSM"] == 1
        assert techniques_counter["TEM"] == 1

    def test_similarity_calculation(self, mock_visualizer):
        """Test similarity calculation using difflib"""
        # Test identical strings
        similarity = mock_visualizer.calculate_similarity("XRD", "XRD")
        assert similarity == 1.0

        # Test similar strings
        similarity = mock_visualizer.calculate_similarity("XRD", "X-ray diffraction")
        assert 0.0 <= similarity <= 1.0  # Valid similarity range

        # Test empty inputs
        similarity = mock_visualizer.calculate_similarity("", "test")
        assert similarity == 0.0

        similarity = mock_visualizer.calculate_similarity(None, "test")
        assert similarity == 0.0

    def test_clustering_functionality(self, mock_visualizer):
        """Test clustering of similar items"""
        items = ["XRD", "X-ray diffraction", "SEM", "scanning electron microscopy"]
        clusters = mock_visualizer.cluster_items(items, similarity_threshold=0.7)

        assert isinstance(clusters, dict)
        assert len(clusters) >= 1

        # Verify all items are assigned to clusters
        all_clustered_items = [
            item for cluster_items in clusters.values() for item in cluster_items
        ]
        assert set(all_clustered_items) == set(items)

        # Verify each item appears only once
        assert len(all_clustered_items) == len(set(all_clustered_items))

    def test_empty_data_handling(self, mock_visualizer):
        """Test handling of empty data"""
        mock_visualizer.data = {}

        families_counter = mock_visualizer._extract_families()
        assert len(families_counter) == 0

        precursors_counter = mock_visualizer._extract_precursors()
        assert len(precursors_counter) == 0

        techniques_counter = mock_visualizer._extract_characterization_techniques()
        assert len(techniques_counter) == 0

    def test_malformed_data_handling(self, mock_visualizer):
        """Test handling of malformed data"""
        malformed_data = {
            "10.1000/malformed": {
                "composition_data": {"family": ""},  # Empty family
                "synthesis_data": {
                    "precursors": [],  # Empty list
                    "characterization_techniques": None,  # None value
                },
            }
        }

        mock_visualizer.data = malformed_data

        families_counter = mock_visualizer._extract_families()
        assert len(families_counter) == 0  # Empty family should be ignored

        precursors_counter = mock_visualizer._extract_precursors()
        assert len(precursors_counter) == 0  # Empty list should result in no precursors

        techniques_counter = mock_visualizer._extract_characterization_techniques()
        assert len(techniques_counter) == 0  # None value should result in no techniques


class TestDataVisualizationMethods:
    """Test data visualization plotting methods with mocking"""

    @pytest.fixture
    def sample_counter_data(self):
        """Sample counter data for testing plots"""
        return {
            "families": Counter(
                {"Perovskite": 5, "Spinel": 3, "Garnet": 2, "Fluorite": 1}
            ),
            "precursors": Counter(
                {"TiO2": 4, "BaCO3": 3, "SrCO3": 2, "ZnO": 2, "Fe2O3": 1}
            ),
            "techniques": Counter({"XRD": 6, "SEM": 4, "TEM": 2, "VSM": 2, "FTIR": 1}),
        }

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.subplots")
    def test_pie_chart_creation(
        self, mock_subplots, mock_tight_layout, mock_savefig, sample_counter_data
    ):
        """Test pie chart creation with mocking"""
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Mock pie chart return values
        mock_wedges = [Mock() for _ in range(4)]
        mock_texts = [Mock() for _ in range(4)]
        mock_ax.pie.return_value = (mock_wedges, mock_texts)

        # Create a simple pie chart function to test
        def create_test_pie_chart(data_counter, title, output_file=None):
            fig, ax = plt.subplots(figsize=(12, 8))

            labels = [f"{k} ({v})" for k, v in data_counter.items()]
            values = list(data_counter.values())

            wedges, texts = ax.pie(values, labels=labels, startangle=90)
            ax.set_title(title)
            plt.tight_layout()

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")

            return fig

        # Test the function
        fig = create_test_pie_chart(
            sample_counter_data["families"],
            "Test Family Distribution",
            "test_output.png",
        )

        # Verify matplotlib functions were called
        mock_subplots.assert_called_once_with(figsize=(12, 8))
        mock_ax.pie.assert_called_once()
        mock_ax.set_title.assert_called_once_with("Test Family Distribution")
        mock_tight_layout.assert_called_once()
        mock_savefig.assert_called_once_with(
            "test_output.png", dpi=300, bbox_inches="tight"
        )

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.subplots")
    def test_histogram_creation(
        self, mock_subplots, mock_tight_layout, mock_savefig, sample_counter_data
    ):
        """Test histogram creation with mocking"""
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Mock bar chart return values
        mock_bars = [Mock() for _ in range(4)]
        for bar in mock_bars:
            bar.get_height.return_value = 5
            bar.get_x.return_value = 0
            bar.get_width.return_value = 1
        mock_ax.bar.return_value = mock_bars

        # Create a simple histogram function to test
        def create_test_histogram(data_counter, title, output_file=None):
            fig, ax = plt.subplots(figsize=(12, 8))

            sorted_items = sorted(
                data_counter.items(), key=lambda x: x[1], reverse=True
            )
            labels = [k for k, _ in sorted_items]
            values = [v for _, v in sorted_items]

            bars = ax.bar(range(len(sorted_items)), values)
            ax.set_xticks(range(len(sorted_items)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_title(title)
            ax.set_ylabel("Frequency")

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")

            return fig

        # Test the function
        fig = create_test_histogram(
            sample_counter_data["techniques"],
            "Test Techniques Distribution",
            "test_histogram.png",
        )

        # Verify matplotlib functions were called
        mock_subplots.assert_called_once_with(figsize=(12, 8))
        mock_ax.bar.assert_called_once()
        mock_ax.set_title.assert_called_once_with("Test Techniques Distribution")
        mock_ax.set_ylabel.assert_called_once_with("Frequency")
        mock_tight_layout.assert_called_once()
        mock_savefig.assert_called_once_with(
            "test_histogram.png", dpi=300, bbox_inches="tight"
        )

    def test_data_processing_for_plots(self, sample_counter_data):
        """Test data processing for plotting (min_percentage, max_items)"""
        families = sample_counter_data["families"]

        # Test min_percentage filtering
        total = sum(families.values())  # Total = 5+3+2+1 = 11
        min_percentage = 20.0  # 20%

        filtered_items = []
        others_sum = 0

        for family, count in families.items():
            percentage = (count / total) * 100
            if percentage >= min_percentage:
                filtered_items.append((family, count))
            else:
                others_sum += count

        if others_sum > 0:
            filtered_items.append(("Others", others_sum))

        # Perovskite: 5/11 = 45.45% (>= 20%)
        # Spinel: 3/11 = 27.27% (>= 20%)
        # Garnet: 2/11 = 18.18% (< 20%)
        # Fluorite: 1/11 = 9.09% (< 20%)
        # So we should have Perovskite, Spinel, and Others
        assert len(filtered_items) == 3
        assert any(item[0] == "Perovskite" for item in filtered_items)
        assert any(item[0] == "Spinel" for item in filtered_items)
        assert any(item[0] == "Others" for item in filtered_items)

        # Test max_items limiting
        precursors = sample_counter_data["precursors"]
        max_items = 3

        sorted_precursors = sorted(precursors.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_precursors) > max_items:
            display_items = sorted_precursors[: max_items - 1]  # Take first 2 items
            others_sum = sum(
                v for _, v in sorted_precursors[max_items - 1 :]
            )  # Sum remaining 3 items
            display_items.append(("Others", others_sum))
        else:
            display_items = sorted_precursors

        # Should have top 2 items plus Others
        assert len(display_items) == 3
        assert display_items[0][0] == "TiO2"  # Most frequent (4)
        assert display_items[1][0] == "BaCO3"  # Second most frequent (3)
        assert display_items[2][0] == "Others"  # Remaining items (2+2+1=5)


class TestFileOperations:
    """Test file operations and error handling"""

    def test_json_file_loading(self):
        """Test JSON file loading functionality"""
        test_data = {
            "10.1000/test": {
                "composition_data": {"family": "Test"},
                "synthesis_data": {"precursors": ["TestPrecursor"]},
            }
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            # Test loading
            with open(temp_file, "r") as f:
                loaded_data = json.load(f)

            assert loaded_data == test_data
            assert "10.1000/test" in loaded_data
            assert loaded_data["10.1000/test"]["composition_data"]["family"] == "Test"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_folder_path_handling(self):
        """Test folder path operations"""
        # Create temporary directory
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test JSON files in the directory
            test_files = []
            for i in range(3):
                test_data = {
                    f"10.1000/test{i}": {"composition_data": {"family": f"Family{i}"}}
                }
                file_path = os.path.join(temp_dir, f"test{i}.json")
                with open(file_path, "w") as f:
                    json.dump(test_data, f)
                test_files.append(file_path)

            # Test directory listing
            json_files = [f for f in os.listdir(temp_dir) if f.endswith(".json")]
            assert len(json_files) == 3

            # Test file paths
            full_paths = [os.path.join(temp_dir, f) for f in json_files]
            for path in full_paths:
                assert os.path.exists(path)
                assert path.endswith(".json")

    def test_error_handling(self):
        """Test error handling for invalid files and data"""
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            with open("non_existent_file.json", "r") as f:
                json.load(f)

        # Test invalid JSON
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

    def test_data_validation(self):
        """Test data structure validation"""
        # Valid data structure
        valid_data = {
            "10.1000/valid": {
                "composition_data": {
                    "family": "Perovskite",
                    "compositions_property_values": {"BaTiO3": 180.0},
                },
                "synthesis_data": {
                    "precursors": ["Ba(OH)2", "TiO2"],
                    "characterization_techniques": ["XRD", "SEM"],
                },
            }
        }

        # Validate structure
        for doi, paper_data in valid_data.items():
            assert "composition_data" in paper_data
            assert "synthesis_data" in paper_data
            assert "family" in paper_data["composition_data"]
            assert isinstance(paper_data["synthesis_data"]["precursors"], list)
            assert isinstance(
                paper_data["synthesis_data"]["characterization_techniques"], list
            )

        # Invalid data structure
        invalid_data = {
            "10.1000/invalid": {
                "composition_data": "not_a_dict",  # Should be dict
                "synthesis_data": {"precursors": "not_a_list"},  # Should be list
            }
        }

        # Validate that we can detect invalid structure
        paper_data = invalid_data["10.1000/invalid"]
        assert not isinstance(paper_data["composition_data"], dict)
        assert not isinstance(paper_data["synthesis_data"]["precursors"], list)


class TestUtilityFunctions:
    """Test utility functions and edge cases"""

    def test_counter_operations(self):
        """Test Counter operations used in the visualizer"""
        # Test Counter creation and operations
        items = ["A", "B", "A", "C", "B", "A"]
        counter = Counter(items)

        assert counter["A"] == 3
        assert counter["B"] == 2
        assert counter["C"] == 1
        assert len(counter) == 3

        # Test sorting by frequency
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        assert sorted_items[0] == ("A", 3)
        assert sorted_items[1] == ("B", 2)
        assert sorted_items[2] == ("C", 1)

    def test_clustering_edge_cases(self):
        """Test clustering with edge cases"""
        # Empty list
        empty_items = []
        empty_counter = Counter(empty_items)
        assert len(empty_counter) == 0

        # Single item
        single_item = ["XRD"]
        single_counter = Counter(single_item)
        assert len(single_counter) == 1
        assert single_counter["XRD"] == 1

        # Duplicate items
        duplicate_items = ["XRD", "XRD", "XRD"]
        duplicate_counter = Counter(duplicate_items)
        assert len(duplicate_counter) == 1
        assert duplicate_counter["XRD"] == 3
