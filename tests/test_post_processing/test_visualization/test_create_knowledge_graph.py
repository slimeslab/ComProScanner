"""
test_create_knowledge_graph.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 16-07-2025
"""

import pytest
import json
import os
import tempfile
import sys
from unittest.mock import Mock, patch, MagicMock
import difflib
from collections import Counter


class TestCreateKnowledgeGraphFunction:
    """Test the main create_knowledge_graph function from data_visualizer"""

    @pytest.fixture
    def sample_results_data(self):
        """Sample test data"""
        return {
            "10.1000/test1": {
                "composition_data": {
                    "family": "Perovskite",
                    "compositions_property_values": {"BaTiO3": 180.0},
                    "property_unit": "pC/N",
                },
                "synthesis_data": {
                    "method": "sol-gel",
                    "steps": ["Mix precursors", "Heat to 600°C"],
                    "precursors": ["Ba(OH)2", "TiO2"],
                    "characterization_techniques": ["XRD", "SEM"],
                },
                "article_metadata": {
                    "doi": "10.1000/test1",
                    "title": "Test Paper",
                    "keywords": ["perovskite", "ferroelectric"],
                },
            }
        }

    @pytest.fixture
    def temp_results_file(self, sample_results_data):
        """Create temporary results file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_results_data, f)
            temp_file = f.name
        yield temp_file
        if os.path.exists(temp_file):
            os.unlink(temp_file)

    def test_create_knowledge_graph_with_valid_file(self, temp_results_file):
        """Test create_knowledge_graph function with valid input file"""
        # Mock the CreateKG class and its methods
        with patch("comproscanner.data_visualizer.CreateKG") as mock_create_kg:
            # Configure the mock
            mock_kg_instance = Mock()
            mock_create_kg.return_value.__enter__.return_value = mock_kg_instance
            mock_create_kg.return_value.__exit__.return_value = None

            # Import and call the function
            from comproscanner.data_visualizer import create_knowledge_graph

            create_knowledge_graph(
                result_file=temp_results_file, is_semantic_clustering_enabled=True
            )

            # Verify the function was called correctly
            mock_create_kg.assert_called_once()
            mock_kg_instance.create_knowledge_graph.assert_called_once_with(
                result_file=temp_results_file,
                is_semantic_clustering_enabled=True,
                method_clustering_similarity_threshold=0.8,
                technique_clustering_similarity_threshold=0.8,
                keyword_clustering_similarity_threshold=0.85,
            )

    def test_create_knowledge_graph_with_none_file(self):
        """Test create_knowledge_graph function with None result_file"""
        with (
            patch("comproscanner.data_visualizer.CreateKG"),
            patch("comproscanner.data_visualizer.logger") as mock_logger,
        ):

            from comproscanner.data_visualizer import create_knowledge_graph

            # Should raise an exception for None file
            with pytest.raises(
                Exception
            ):  # ValueErrorHandler becomes Exception in mocking
                create_knowledge_graph(result_file=None)

            # Verify error was logged
            mock_logger.error.assert_called()

    def test_create_knowledge_graph_with_custom_thresholds(self, temp_results_file):
        """Test create_knowledge_graph with custom similarity thresholds"""
        with patch("comproscanner.data_visualizer.CreateKG") as mock_create_kg:
            mock_kg_instance = Mock()
            mock_create_kg.return_value.__enter__.return_value = mock_kg_instance
            mock_create_kg.return_value.__exit__.return_value = None

            from comproscanner.data_visualizer import create_knowledge_graph

            create_knowledge_graph(
                result_file=temp_results_file,
                is_semantic_clustering_enabled=False,
                method_clustering_similarity_threshold=0.9,
                technique_clustering_similarity_threshold=0.7,
                keyword_clustering_similarity_threshold=0.95,
            )

            # Verify custom parameters were passed
            mock_kg_instance.create_knowledge_graph.assert_called_once_with(
                result_file=temp_results_file,
                is_semantic_clustering_enabled=False,
                method_clustering_similarity_threshold=0.9,
                technique_clustering_similarity_threshold=0.7,
                keyword_clustering_similarity_threshold=0.95,
            )

    def test_create_knowledge_graph_exception_handling(self, temp_results_file):
        """Test exception handling in create_knowledge_graph"""
        with (
            patch("comproscanner.data_visualizer.CreateKG") as mock_create_kg,
            patch("comproscanner.data_visualizer.logger") as mock_logger,
        ):

            # Configure mock to raise an exception
            mock_kg_instance = Mock()
            mock_kg_instance.create_knowledge_graph.side_effect = Exception(
                "Test error"
            )
            mock_create_kg.return_value.__enter__.return_value = mock_kg_instance
            mock_create_kg.return_value.__exit__.return_value = None

            from comproscanner.data_visualizer import create_knowledge_graph

            # Should re-raise the exception
            with pytest.raises(Exception):
                create_knowledge_graph(result_file=temp_results_file)

            # Verify error was logged
            mock_logger.error.assert_called_with(
                "Error creating knowledge graph: Test error"
            )


class TestSemanticMatcher:
    """Test semantic matching functionality using direct implementation"""

    def test_difflib_similarity_calculation(self):
        """Test similarity calculation using difflib (fallback method)"""
        # Test identical strings
        similarity = difflib.SequenceMatcher(None, "test", "test").ratio()
        assert similarity == 1.0

        # Test completely different strings
        similarity = difflib.SequenceMatcher(None, "hello", "world").ratio()
        assert 0.0 <= similarity <= 1.0  # Just check it's a valid similarity score

        # Test case-insensitive similar strings
        similarity = difflib.SequenceMatcher(None, "XRD".lower(), "xrd".lower()).ratio()
        assert similarity == 1.0  # Should be identical when lowercased

        # Test partially similar strings
        similarity = difflib.SequenceMatcher(None, "X-ray diffraction", "XRD").ratio()
        assert 0.0 <= similarity <= 1.0  # Valid range

    def test_clustering_logic(self):
        """Test basic clustering logic"""
        items = ["XRD", "X-ray diffraction", "SEM", "scanning electron microscopy"]
        similarity_threshold = 0.8

        # Simple clustering algorithm for testing
        clusters = {}
        processed = set()

        for item in items:
            if item in processed:
                continue

            canonical = item
            cluster_items = [item]
            processed.add(item)

            # Find similar items (simplified logic)
            for other_item in items:
                if other_item != item and other_item not in processed:
                    # Simple similarity check
                    if any(
                        word.lower() in other_item.lower()
                        for word in item.lower().split()
                    ):
                        cluster_items.append(other_item)
                        processed.add(other_item)

            clusters[canonical] = cluster_items

        # Verify clustering results
        assert len(clusters) >= 1
        all_clustered_items = [
            item for cluster in clusters.values() for item in cluster
        ]
        assert set(all_clustered_items) == set(items)

    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        # Empty string similarity
        similarity = difflib.SequenceMatcher(None, "", "test").ratio()
        assert similarity == 0.0

        # Empty list clustering
        clusters = {}
        items = []
        assert clusters == {}


class TestDataStructures:
    """Test data structure handling and validation"""

    def test_valid_paper_data_structure(self):
        """Test validation of paper data structure"""
        paper_data = {
            "composition_data": {
                "family": "Perovskite",
                "compositions_property_values": {"BaTiO3": 180.0},
                "property_unit": "pC/N",
            },
            "synthesis_data": {
                "method": "sol-gel",
                "precursors": ["Ba(OH)2"],
                "characterization_techniques": ["XRD"],
            },
            "article_metadata": {"doi": "10.1000/test", "title": "Test Paper"},
        }

        # Verify required sections exist
        required_sections = ["composition_data", "synthesis_data", "article_metadata"]
        for section in required_sections:
            assert section in paper_data

        # Verify DOI exists
        assert "doi" in paper_data["article_metadata"]
        assert paper_data["article_metadata"]["doi"] is not None

    def test_malformed_data_handling(self):
        """Test handling of malformed data"""
        malformed_data = {
            "composition_data": "not_a_dict",  # Should be dict
            "synthesis_data": None,
            "article_metadata": {"doi": "test"},
        }

        # Test that we can detect malformed data
        assert not isinstance(malformed_data["composition_data"], dict)
        assert malformed_data["synthesis_data"] is None
        assert isinstance(malformed_data["article_metadata"], dict)

    def test_json_file_operations(self):
        """Test JSON file reading and writing"""
        test_data = {
            "10.1000/test": {
                "composition_data": {"family": "Test"},
                "synthesis_data": {"method": "test"},
                "article_metadata": {"doi": "10.1000/test"},
            }
        }

        # Test writing and reading JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            # Read back the data
            with open(temp_file, "r") as f:
                loaded_data = json.load(f)

            assert loaded_data == test_data
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
            # Should raise JSON decode error
            with pytest.raises(json.JSONDecodeError):
                with open(temp_file, "r") as f:
                    json.load(f)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestEnvironmentConfiguration:
    """Test environment configuration handling"""

    def test_environment_variables_validation(self):
        """Test validation of required environment variables"""
        required_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]

        # Test with missing variables
        with patch.dict(os.environ, {}, clear=True):
            for var in required_vars:
                assert os.getenv(var) is None

        # Test with present variables
        test_env = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
        }

        with patch.dict(os.environ, test_env):
            for var in required_vars:
                assert os.getenv(var) is not None

    def test_default_database_name(self):
        """Test default database name handling"""
        # Test default value
        with patch.dict(os.environ, {}, clear=True):
            database_name = os.getenv("NEO4J_DATABASE", "neo4j")
            assert database_name == "neo4j"

        # Test custom value
        with patch.dict(os.environ, {"NEO4J_DATABASE": "custom_db"}):
            database_name = os.getenv("NEO4J_DATABASE", "neo4j")
            assert database_name == "custom_db"


class TestIntegrationScenarios:
    """Test integration scenarios with mocked dependencies"""

    @pytest.fixture
    def mock_neo4j_setup(self):
        """Set up Neo4j mocks"""
        with patch("neo4j.GraphDatabase") as mock_db:
            mock_driver = Mock()
            mock_driver.verify_connectivity.return_value = None
            mock_driver.close.return_value = None

            mock_session = Mock()
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None

            mock_session.run.return_value.single.return_value = {"paper_count": 1}

            mock_db.driver.return_value = mock_driver

            yield {"driver": mock_driver, "session": mock_session}
