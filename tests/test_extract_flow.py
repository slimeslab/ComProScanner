import pytest
import json
from unittest.mock import MagicMock, patch

from comproscanner.extract_flow.main_extraction_flow import DataExtractionFlow
from comproscanner.utils.error_handler import ValueErrorHandler
from comproscanner.utils.configs.rag_config import RAGConfig


@pytest.fixture
def sample_doi():
    return "10.1186/s12951-024-03087-y"


@pytest.fixture
def sample_main_extraction_keyword():
    return "piezoelectric"


@pytest.fixture
def sample_composition_property_text():
    return """
    The Ba0.99Ca0.01Ti0.68Zr0.32O3 composition exhibited a d33 value of 375 pC/N.
    Additionally, Ba0.98Ca0.02Ti0.78Zr0.22O3 showed a d33 of 350 pC/N,
    while Ba0.97Ca0.03Ti0.88Zr0.12O3 had a value of 325 pC/N.
    """


@pytest.fixture
def sample_synthesis_text():
    return """
    The samples were prepared using solid-state reaction method. 
    BaCO3, CaCO3, TiO2, and ZrO2 were used as raw materials. 
    The raw materials were mixed in a ball mill for 24 h using ethanol as media.
    The mixtures were calcined at 1200°C for 2 h, and then sintered at 1450°C for 2 h.
    The phase structures were examined by X-ray diffraction (XRD).
    """


@pytest.fixture
def mock_rag_config():
    return MagicMock(spec=RAGConfig)


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def sample_composition_extracted_data():
    return {
        "compositions": {
            "Ba0.99Ca0.01Ti0.68Zr0.32O3": 375,
            "Ba0.98Ca0.02Ti0.78Zr0.22O3": 350,
            "Ba0.97Ca0.03Ti0.88Zr0.12O3": 325,
        },
        "property_unit": "pC/N",
        "family": "BaTiO3",
    }


@pytest.fixture
def sample_composition_formatted_data():
    return {
        "compositions": {
            "Ba0.99Ca0.01Ti0.68Zr0.32O3": 375,
            "Ba0.98Ca0.02Ti0.78Zr0.22O3": 350,
            "Ba0.97Ca0.03Ti0.88Zr0.12O3": 325,
        },
        "property_unit": "pC/N",
        "family": "BaTiO3",
        "doi": "10.1186/s12951-024-03087-y",
        "property_name": "piezoelectric_coefficient",
    }


@pytest.fixture
def sample_synthesis_extracted_data():
    return {
        "synthesis_data": {
            "method": "solid-state reaction",
            "precursors": ["BaCO3", "CaCO3", "TiO2", "ZrO2"],
            "steps": [
                "The raw materials were mixed in a ball mill for 24 h using ethanol as media.",
                "The mixtures were calcined at 1200°C for 2 h.",
                "The mixtures were sintered at 1450°C for 2 h.",
            ],
            "characterization_techniques": ["XRD"],
        }
    }


@pytest.fixture
def sample_synthesis_formatted_data():
    return {
        "method": "solid_state_reaction",
        "precursors": ["BaCO3", "CaCO3", "TiO2", "ZrO2"],
        "steps": [
            "The raw materials were mixed in a ball mill for 24 h using ethanol as media.",
            "The mixtures were calcined at 1200°C for 2 h.",
            "The mixtures were sintered at 1450°C for 2 h.",
        ],
        "characterization_techniques": ["x_ray_diffraction"],
    }


class TestSimpleExtractionFlow:
    """A simplified test suite for DataExtractionFlow that avoids complex mocking"""

    def test_initialization_with_required_params(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test initialization with only required parameters"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )
        assert flow.state.doi == sample_doi
        assert flow.state.main_extraction_keyword == sample_main_extraction_keyword
        assert flow.state.extract_synthesis_data is False
        assert flow.state.verbose is True

    def test_initialization_missing_doi(
        self, sample_main_extraction_keyword, sample_composition_property_text
    ):
        """Test initialization with missing required DOI parameter"""
        with pytest.raises(ValueErrorHandler, match="DOI is required"):
            DataExtractionFlow(
                main_extraction_keyword=sample_main_extraction_keyword,
                composition_property_text_data=sample_composition_property_text,
            )

    def test_initialization_missing_main_keyword(
        self, sample_doi, sample_composition_property_text
    ):
        """Test initialization with missing required keyword parameter"""
        with pytest.raises(
            ValueErrorHandler, match="Main property keyword is required"
        ):
            DataExtractionFlow(
                doi=sample_doi,
                composition_property_text_data=sample_composition_property_text,
            )

    def test_initialization_missing_text_data(
        self, sample_doi, sample_main_extraction_keyword
    ):
        """Test initialization with missing required text data parameter"""
        with pytest.raises(
            ValueErrorHandler, match="Composition property text data is required"
        ):
            DataExtractionFlow(
                doi=sample_doi, main_extraction_keyword=sample_main_extraction_keyword
            )

    def test_json_parsing_valid(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test JSON parsing with valid input"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        json_str = '{"key": "value", "number": 42}'
        result = flow._parse_json_output(json_str)

        assert result == {"key": "value", "number": 42}

    def test_json_parsing_invalid(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test JSON parsing with invalid input"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        json_str = "invalid json"
        default_value = {"default": "value"}
        result = flow._parse_json_output(json_str, default_value=default_value)

        assert result == default_value

    def test_flow_state_routing_with_yes(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test flow routing when materials are found"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Set the state directly
        flow.state.is_materials_mentioned = "yes"

        # Test the routing
        result = flow.route_process()
        assert result == "extract_compositions"

    def test_flow_state_routing_with_no(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test flow routing when materials are not found"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Set the state directly
        flow.state.is_materials_mentioned = "no"

        # Test the routing
        result = flow.route_process()
        assert result == "end_flow"

    def test_terminate_process(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test termination of the process"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        result = flow.terminate_process()

        expected = {
            "composition_data": {},
            "synthesis_data": {},
        }
        assert result == expected

    def test_finalize_results(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        sample_composition_formatted_data,
        sample_synthesis_formatted_data,
    ):
        """Test finalizing results from the flow"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Set state directly
        flow.state.composition_formatted_data = sample_composition_formatted_data
        flow.state.synthesis_formatted_data = sample_synthesis_formatted_data

        # Run the test
        result = flow.finalize_results()

        # Verify result
        expected = {
            "composition_data": sample_composition_formatted_data,
            "synthesis_data": sample_synthesis_formatted_data,
        }
        assert result == expected
