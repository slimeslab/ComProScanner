"""
test_extract_flow.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 16-04-2025
"""

import pytest
import json
from unittest.mock import MagicMock, patch, call

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
        assert flow.state.is_extract_synthesis_data is True
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


class TestDataExtractionFlowCore:
    """Test suite for core DataExtractionFlow methods with mocking"""

    @patch(
        "comproscanner.extract_flow.main_extraction_flow.MaterialsDataIdentifierCrew"
    )
    def test_identify_materials_data_presence_with_llm(
        self,
        mock_crew_class,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        mock_llm,
    ):
        """Test materials identification with LLM"""
        # Setup mock crew and result
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.raw = "yes"
        mock_crew_instance.kickoff.return_value = mock_result

        # Create flow with LLM
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            llm=mock_llm,
        )

        # Execute method
        result = flow.identify_materials_data_presence()

        # Verify crew was called with correct parameters
        mock_crew_class.assert_called_once_with(
            doi=sample_doi,
            llm=mock_llm,
            rag_config=flow.state.rag_config,
            output_log_folder=None,
            task_output_folder=None,
            is_log_json=False,
            verbose=True,
        )

        # Verify result
        assert result == "yes"
        assert flow.state.is_materials_mentioned == "yes"

    @patch(
        "comproscanner.extract_flow.main_extraction_flow.MaterialsDataIdentifierCrew"
    )
    def test_identify_materials_data_presence_without_llm(
        self,
        mock_crew_class,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test materials identification without LLM"""
        # Setup mock crew and result
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.raw = '"no"'  # Test quoted response cleaning
        mock_crew_instance.kickoff.return_value = mock_result

        # Create flow without LLM
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Execute method
        result = flow.identify_materials_data_presence()

        # Verify crew was called without LLM
        mock_crew_class.assert_called_once_with(
            doi=sample_doi,
            rag_config=flow.state.rag_config,
            output_log_folder=None,
            task_output_folder=None,
            is_log_json=False,
            verbose=True,
        )

        # Verify result is cleaned (quotes removed)
        assert result == "no"
        assert flow.state.is_materials_mentioned == "no"

    @patch("comproscanner.extract_flow.main_extraction_flow.CompositionExtractionCrew")
    def test_extract_composition_property_data_with_llm(
        self,
        mock_crew_class,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        sample_composition_extracted_data,
        mock_llm,
    ):
        """Test composition extraction with LLM"""
        # Setup mock crew and result
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.raw = json.dumps(sample_composition_extracted_data)
        mock_crew_instance.kickoff.return_value = mock_result

        # Create flow
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            llm=mock_llm,
        )

        # Execute method
        result = flow.extract_composition_property_data()

        # Verify crew was called correctly
        mock_crew_class.assert_called_once_with(
            doi=sample_doi,
            llm=mock_llm,
            output_log_folder=None,
            task_output_folder=None,
            is_log_json=False,
            verbose=True,
        )

        # Verify kickoff was called with correct inputs
        expected_inputs = {
            "composition_property_text_data": sample_composition_property_text,
            "main_extraction_keyword": sample_main_extraction_keyword,
            "composition_property_extraction_agent_note": flow.state.composition_property_extraction_agent_note,
            "composition_property_extraction_task_note": flow.state.composition_property_extraction_task_note,
            "expected_composition_property_example": flow.state.expected_composition_property_example,
            "expected_variable_composition_property_example": flow.state.expected_variable_composition_property_example,
        }
        mock_crew_instance.kickoff.assert_called_once_with(inputs=expected_inputs)

        # Verify result
        assert result == sample_composition_extracted_data
        assert (
            flow.state.composition_extracted_data == sample_composition_extracted_data
        )

    @patch("comproscanner.extract_flow.main_extraction_flow.CompositionFormatCrew")
    def test_extract_final_composition_property_data(
        self,
        mock_crew_class,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        sample_composition_extracted_data,
        sample_composition_formatted_data,
        mock_llm,
    ):
        """Test final composition formatting"""
        # Setup mock crew and result
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.raw = json.dumps(
            {"composition_formatted_data": sample_composition_formatted_data}
        )
        mock_crew_instance.kickoff.return_value = mock_result

        # Create flow and set state
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            llm=mock_llm,
        )
        flow.state.composition_extracted_data = sample_composition_extracted_data

        # Execute method
        flow.extract_final_composition_property_data()

        # Verify crew was called correctly
        mock_crew_class.assert_called_once_with(
            doi=sample_doi,
            llm=mock_llm,
            output_log_folder=None,
            task_output_folder=None,
            is_log_json=False,
            verbose=True,
        )

        # Verify kickoff was called with correct inputs
        expected_inputs = {
            "extracted_composition_data": sample_composition_extracted_data,
            "composition_property_formatting_agent_note": flow.state.composition_property_formatting_agent_note,
            "composition_property_formatting_task_note": flow.state.composition_property_formatting_task_note,
            "main_extraction_keyword": sample_main_extraction_keyword,
            "expected_composition_property_example": flow.state.expected_composition_property_example,
        }
        mock_crew_instance.kickoff.assert_called_once_with(inputs=expected_inputs)

        # Verify state was updated
        assert (
            flow.state.composition_formatted_data == sample_composition_formatted_data
        )

    @patch("comproscanner.extract_flow.main_extraction_flow.SynthesisExtractionCrew")
    def test_extract_synthesis_data_with_llm(
        self,
        mock_crew_class,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        sample_synthesis_text,
        sample_synthesis_extracted_data,
        sample_composition_formatted_data,
        mock_llm,
    ):
        """Test synthesis extraction with LLM"""
        # Setup mock crew and result
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.raw = json.dumps(sample_synthesis_extracted_data)
        mock_crew_instance.kickoff.return_value = mock_result

        # Create flow with synthesis data
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            synthesis_text_data=sample_synthesis_text,
            llm=mock_llm,
        )
        flow.state.composition_formatted_data = sample_composition_formatted_data

        # Execute method
        result = flow.extract_synthesis_data()

        # Verify crew was called correctly
        mock_crew_class.assert_called_once_with(
            doi=sample_doi,
            llm=mock_llm,
            output_log_folder=None,
            task_output_folder=None,
            is_log_json=False,
            verbose=True,
        )

        # Verify kickoff was called with correct inputs
        expected_inputs = {
            "synthesis_text_data": sample_synthesis_text,
            "composition_formatted_data": sample_composition_formatted_data,
            "synthesis_extraction_agent_note": flow.state.synthesis_extraction_agent_note,
            "synthesis_extraction_task_note": flow.state.synthesis_extraction_task_note,
        }
        mock_crew_instance.kickoff.assert_called_once_with(inputs=expected_inputs)

        # Verify result
        assert result == sample_synthesis_extracted_data
        assert flow.state.synthesis_extracted_data == sample_synthesis_extracted_data

    def test_extract_synthesis_data_empty_text(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test synthesis extraction with empty synthesis text"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            synthesis_text_data="",  # Empty text
        )

        result = flow.extract_synthesis_data()

        # Should return default empty synthesis data
        expected = {
            "synthesis_data": {
                "method": "",
                "precursors": [],
                "steps": [],
                "characterization_techniques": [],
            }
        }
        assert result == expected
        assert flow.state.synthesis_extracted_data == expected

    def test_extract_synthesis_data_disabled(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        sample_synthesis_text,
    ):
        """Test synthesis extraction when disabled"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            synthesis_text_data=sample_synthesis_text,
            is_extract_synthesis_data=False,  # Disabled
        )

        result = flow.extract_synthesis_data()

        # Should return default empty synthesis data
        expected = {
            "synthesis_data": {
                "method": "",
                "precursors": [],
                "steps": [],
                "characterization_techniques": [],
            }
        }
        assert result == expected

    @patch("comproscanner.extract_flow.main_extraction_flow.SynthesisFormatCrew")
    def test_extract_final_synthesis_data_with_data(
        self,
        mock_crew_class,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        sample_synthesis_extracted_data,
        sample_synthesis_formatted_data,
        mock_llm,
    ):
        """Test final synthesis formatting with valid data"""
        # Setup mock crew and result
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.raw = json.dumps(
            {"synthesis_formatted_data": sample_synthesis_formatted_data}
        )
        mock_crew_instance.kickoff.return_value = mock_result

        # Create flow and set state
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            llm=mock_llm,
        )
        flow.state.synthesis_extracted_data = sample_synthesis_extracted_data

        # Execute method
        flow.extract_final_synthesis_data()

        # Verify crew was called correctly
        mock_crew_class.assert_called_once_with(
            doi=sample_doi,
            llm=mock_llm,
            output_log_folder=None,
            task_output_folder=None,
            is_log_json=False,
            verbose=True,
        )

        # Verify kickoff was called with correct inputs
        expected_inputs = {
            "extracted_synthesis_data": sample_synthesis_extracted_data,
            "synthesis_formatting_agent_note": flow.state.synthesis_formatting_agent_note,
            "synthesis_formatting_task_note": flow.state.synthesis_formatting_task_note,
            "allowed_synthesis_methods": flow.state.allowed_synthesis_methods,
            "allowed_characterization_techniques": flow.state.allowed_characterization_techniques,
        }
        mock_crew_instance.kickoff.assert_called_once_with(inputs=expected_inputs)

        # Verify state was updated
        assert flow.state.synthesis_formatted_data == sample_synthesis_formatted_data

    def test_extract_final_synthesis_data_empty_data(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test final synthesis formatting with empty synthesis data"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Set empty synthesis data
        flow.state.synthesis_extracted_data = {
            "synthesis_data": {
                "method": "",
                "precursors": [],
                "steps": [],
            }
        }

        # Execute method - should return early without calling crew
        flow.extract_final_synthesis_data()

        # Verify state was set directly from extracted data
        assert flow.state.synthesis_formatted_data == {
            "method": "",
            "precursors": [],
            "steps": [],
        }

    def test_json_parsing_with_python_literals(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test JSON parsing with Python-style literals"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Test with Python boolean/null values
        python_str = '{"active": true, "data": null, "enabled": false}'
        result = flow._parse_json_output(python_str)
        assert result == {"active": True, "data": None, "enabled": False}

    def test_initialization_with_all_optional_params(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        sample_synthesis_text,
        mock_llm,
        mock_rag_config,
    ):
        """Test initialization with all optional parameters"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            synthesis_text_data=sample_synthesis_text,
            llm=mock_llm,
            materials_data_identifier_query="Custom query?",
            is_extract_synthesis_data=False,
            rag_config=mock_rag_config,
            output_log_folder="/test/logs",
            task_output_folder="/test/tasks",
            is_log_json=True,
            verbose=False,
            expected_composition_property_example="Custom example",
            expected_variable_composition_property_example="Variable example",
            composition_property_extraction_agent_notes=["Note 1"],
            composition_property_extraction_task_notes=["Task note 1"],
            composition_property_formatting_agent_notes=["Format note 1"],
            composition_property_formatting_task_notes=["Format task note 1"],
            synthesis_extraction_agent_notes=["Synthesis note 1"],
            synthesis_extraction_task_notes=["Synthesis task note 1"],
            synthesis_formatting_agent_notes=["Synthesis format note 1"],
            synthesis_formatting_task_notes=["Synthesis format task note 1"],
            allowed_synthesis_methods=["method1", "method2"],
            allowed_characterization_techniques=["XRD", "SEM"],
        )

        # Verify all state values are set correctly
        assert flow.state.doi == sample_doi
        assert flow.state.main_extraction_keyword == sample_main_extraction_keyword
        assert (
            flow.state.composition_property_text_data
            == sample_composition_property_text
        )
        assert flow.state.synthesis_text_data == sample_synthesis_text
        assert flow.state.llm == mock_llm
        assert flow.state.materials_data_identifier_query == "Custom query?"
        assert flow.state.is_extract_synthesis_data is False
        assert flow.state.rag_config == mock_rag_config
        assert flow.state.output_log_folder == "/test/logs"
        assert flow.state.task_output_folder == "/test/tasks"
        assert flow.state.is_log_json is True
        assert flow.state.verbose is False

    def test_route_process_case_insensitive(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test flow routing with different case variations"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Test various "yes" formats
        test_cases = ["YES", "Yes", "yes", "Yes, materials found"]
        for case in test_cases:
            flow.state.is_materials_mentioned = case
            assert flow.route_process() == "extract_compositions"

        # Test various "no" formats
        test_cases = ["NO", "No", "no", "No materials found"]
        for case in test_cases:
            flow.state.is_materials_mentioned = case
            assert flow.route_process() == "end_flow"


class TestDataExtractionFlowIntegration:
    """Integration tests for DataExtractionFlow"""

    @patch(
        "comproscanner.extract_flow.main_extraction_flow.MaterialsDataIdentifierCrew"
    )
    @patch("comproscanner.extract_flow.main_extraction_flow.CompositionExtractionCrew")
    @patch("comproscanner.extract_flow.main_extraction_flow.CompositionFormatCrew")
    @patch("comproscanner.extract_flow.main_extraction_flow.SynthesisExtractionCrew")
    @patch("comproscanner.extract_flow.main_extraction_flow.SynthesisFormatCrew")
    def test_full_flow_with_synthesis_success(
        self,
        mock_synthesis_format_crew,
        mock_synthesis_extraction_crew,
        mock_composition_format_crew,
        mock_composition_extraction_crew,
        mock_materials_crew,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        sample_synthesis_text,
        sample_composition_extracted_data,
        sample_composition_formatted_data,
        sample_synthesis_extracted_data,
        sample_synthesis_formatted_data,
        mock_llm,
    ):
        """Test complete flow execution with synthesis data"""

        # Setup all mock crews and results
        def setup_mock_crew(mock_crew_class, raw_response):
            mock_crew_instance = MagicMock()
            mock_crew_class.return_value.crew.return_value = mock_crew_instance
            mock_result = MagicMock()
            mock_result.raw = raw_response
            mock_crew_instance.kickoff.return_value = mock_result
            return mock_crew_instance

        materials_crew = setup_mock_crew(mock_materials_crew, "yes")
        composition_crew = setup_mock_crew(
            mock_composition_extraction_crew,
            json.dumps(sample_composition_extracted_data),
        )
        format_crew = setup_mock_crew(
            mock_composition_format_crew,
            json.dumps(
                {"composition_formatted_data": sample_composition_formatted_data}
            ),
        )
        synthesis_crew = setup_mock_crew(
            mock_synthesis_extraction_crew, json.dumps(sample_synthesis_extracted_data)
        )
        synthesis_format_crew = setup_mock_crew(
            mock_synthesis_format_crew,
            json.dumps({"synthesis_formatted_data": sample_synthesis_formatted_data}),
        )

        # Create and execute flow
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            synthesis_text_data=sample_synthesis_text,
            llm=mock_llm,
        )

        # Mock the flow execution (since we can't actually run kickoff in unit tests)
        # We'll test each step manually to simulate the flow

        # Step 1: Identify materials
        materials_result = flow.identify_materials_data_presence()
        assert materials_result == "yes"

        # Step 2: Route process
        route_result = flow.route_process()
        assert route_result == "extract_compositions"

        # Step 3: Extract compositions
        comp_result = flow.extract_composition_property_data()
        assert comp_result == sample_composition_extracted_data

        # Step 4: Format compositions
        flow.extract_final_composition_property_data()
        assert (
            flow.state.composition_formatted_data == sample_composition_formatted_data
        )

        # Step 5: Extract synthesis
        synth_result = flow.extract_synthesis_data()
        assert synth_result == sample_synthesis_extracted_data

        # Step 6: Format synthesis
        flow.extract_final_synthesis_data()
        assert flow.state.synthesis_formatted_data == sample_synthesis_formatted_data

        # Step 7: Finalize
        final_result = flow.finalize_results()
        expected_final = {
            "composition_data": sample_composition_formatted_data,
            "synthesis_data": sample_synthesis_formatted_data,
        }
        assert final_result == expected_final

    @patch(
        "comproscanner.extract_flow.main_extraction_flow.MaterialsDataIdentifierCrew"
    )
    def test_full_flow_no_materials_found(
        self,
        mock_materials_crew,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        mock_llm,
    ):
        """Test complete flow when no materials are found"""
        # Setup mock crew to return "no"
        mock_crew_instance = MagicMock()
        mock_materials_crew.return_value.crew.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.raw = "no"
        mock_crew_instance.kickoff.return_value = mock_result

        # Create flow
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            llm=mock_llm,
        )

        # Execute flow steps
        materials_result = flow.identify_materials_data_presence()
        assert materials_result == "no"

        route_result = flow.route_process()
        assert route_result == "end_flow"

        # When routed to end_flow, should terminate
        terminate_result = flow.terminate_process()
        expected = {
            "composition_data": {},
            "synthesis_data": {},
        }
        assert terminate_result == expected


class TestDataExtractionFlowErrorHandling:
    """Test error handling scenarios"""

    @patch("comproscanner.extract_flow.main_extraction_flow.CompositionExtractionCrew")
    def test_composition_extraction_crew_failure(
        self,
        mock_crew_class,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        mock_llm,
    ):
        """Test handling of crew execution failure"""
        # Setup mock crew to raise exception
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_crew_instance.kickoff.side_effect = Exception("Crew execution failed")

        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            llm=mock_llm,
        )

        # Should handle exception gracefully
        with pytest.raises(Exception, match="Crew execution failed"):
            flow.extract_composition_property_data()

    def test_json_parsing_malformed_response(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test JSON parsing with various malformed responses"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Test with malformed JSON that falls back to ast.literal_eval
        malformed_json = "{'key': 'value', 'bool': True, 'none': None}"
        default_value = {"fallback": "data"}
        result = flow._parse_json_output(malformed_json, default_value=default_value)
        assert result == {"key": "value", "bool": True, "none": None}

        # Test with completely invalid data
        invalid_data = "completely invalid data that can't be parsed"
        result = flow._parse_json_output(invalid_data, default_value=default_value)
        assert result == default_value

    def test_json_parsing_empty_response(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test JSON parsing with empty response"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        default_value = {"empty": "response"}

        # Test with empty string
        result = flow._parse_json_output("", default_value=default_value)
        assert result == default_value

        # Test with whitespace only
        result = flow._parse_json_output("   \n\t  ", default_value=default_value)
        assert result == default_value

    @patch("comproscanner.extract_flow.main_extraction_flow.SynthesisExtractionCrew")
    def test_synthesis_extraction_with_none_text_data(
        self,
        mock_crew_class,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test synthesis extraction with None synthesis text data"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            synthesis_text_data=None,  # None instead of empty string
        )

        result = flow.extract_synthesis_data()

        # Should not call crew and return default data
        mock_crew_class.assert_not_called()
        expected = {
            "synthesis_data": {
                "method": "",
                "precursors": [],
                "steps": [],
                "characterization_techniques": [],
            }
        }
        assert result == expected


class TestDataExtractionFlowStateManagement:
    """Test state management and transitions"""

    def test_state_initialization_defaults(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test that state is initialized with correct defaults"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Check MaterialsState defaults
        assert flow.state.is_materials_mentioned == ""
        assert flow.state.composition_extracted_data == {}
        assert flow.state.composition_formatted_data == {}
        assert flow.state.synthesis_extracted_data == {}
        assert flow.state.synthesis_formatted_data == {}
        assert flow.state.is_extract_synthesis_data is True
        assert flow.state.llm is None
        assert flow.state.verbose is True

    def test_main_extraction_keyword_space_replacement(
        self,
        sample_doi,
        sample_composition_property_text,
    ):
        """Test that spaces in main extraction keyword are replaced with underscores"""
        keyword_with_spaces = "piezoelectric coefficient"
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=keyword_with_spaces,
            composition_property_text_data=sample_composition_property_text,
        )

        assert flow.state.main_extraction_keyword == "piezoelectric_coefficient"

    def test_state_persistence_across_methods(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        sample_composition_extracted_data,
    ):
        """Test that state persists correctly across method calls"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Set some state manually
        flow.state.is_materials_mentioned = "yes"
        flow.state.composition_extracted_data = sample_composition_extracted_data

        # Verify state persists
        assert flow.state.is_materials_mentioned == "yes"
        assert (
            flow.state.composition_extracted_data == sample_composition_extracted_data
        )

        # Test routing still works with persistent state
        route_result = flow.route_process()
        assert route_result == "extract_compositions"


class TestDataExtractionFlowEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_synthesis_extraction_with_malformed_extracted_data(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        sample_synthesis_text,
    ):
        """Test final synthesis extraction with malformed extracted data"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            synthesis_text_data=sample_synthesis_text,
        )

        # Test with string instead of dict
        flow.state.synthesis_extracted_data = "not a dict"
        flow.extract_final_synthesis_data()
        # Should not crash and proceed to crew execution

        # Test with dict but no synthesis_data key
        flow.state.synthesis_extracted_data = {"other_key": "value"}
        flow.extract_final_synthesis_data()
        # Should not crash and proceed to crew execution

    def test_composition_formatting_with_missing_key(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        mock_llm,
    ):
        """Test composition formatting when response is missing expected key"""
        with patch(
            "comproscanner.extract_flow.main_extraction_flow.CompositionFormatCrew"
        ) as mock_crew_class:
            # Setup mock crew with response missing the expected key
            mock_crew_instance = MagicMock()
            mock_crew_class.return_value.crew.return_value = mock_crew_instance
            mock_result = MagicMock()
            mock_result.raw = json.dumps({"unexpected_key": "value"})
            mock_crew_instance.kickoff.return_value = mock_result

            flow = DataExtractionFlow(
                doi=sample_doi,
                main_extraction_keyword=sample_main_extraction_keyword,
                composition_property_text_data=sample_composition_property_text,
                llm=mock_llm,
            )

            # Execute method
            flow.extract_final_composition_property_data()

            # Should handle missing key gracefully and use empty dict
            assert flow.state.composition_formatted_data == {}

    def test_route_process_with_unexpected_response(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test routing with unexpected materials identification response"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Test with unexpected response
        flow.state.is_materials_mentioned = "maybe"
        result = flow.route_process()
        assert result == "end_flow"  # Should default to end_flow

        # Test with empty response
        flow.state.is_materials_mentioned = ""
        result = flow.route_process()
        assert result == "end_flow"

    def test_json_parsing_with_nested_json_strings(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test JSON parsing with nested JSON strings"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Test with escaped JSON
        escaped_json = '"{"key": "value", "number": 42}"'
        result = flow._parse_json_output(escaped_json)
        # Should handle this gracefully, may return default if can't parse

        # Test with double-escaped JSON
        double_escaped = '\\"{\\\\"key\\\\": \\\\"value\\\\"}\\"'
        default_value = {"fallback": True}
        result = flow._parse_json_output(double_escaped, default_value=default_value)
        # Should either parse correctly or return default

    @patch("comproscanner.extract_flow.main_extraction_flow.SynthesisExtractionCrew")
    def test_synthesis_extraction_without_llm(
        self,
        mock_crew_class,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        sample_synthesis_text,
        sample_synthesis_extracted_data,
    ):
        """Test synthesis extraction without LLM provided"""
        # Setup mock crew and result
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.raw = json.dumps(sample_synthesis_extracted_data)
        mock_crew_instance.kickoff.return_value = mock_result

        # Create flow without LLM
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            synthesis_text_data=sample_synthesis_text,
            # No LLM provided
        )

        # Execute method
        result = flow.extract_synthesis_data()

        # Verify crew was called without LLM parameter
        mock_crew_class.assert_called_once_with(
            doi=sample_doi,
            output_log_folder=None,
            task_output_folder=None,
            is_log_json=False,
            verbose=True,
        )

        assert result == sample_synthesis_extracted_data

    def test_initialization_with_custom_notes_and_methods(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test initialization with custom notes and allowed methods"""
        custom_notes = ["Custom note 1", "Custom note 2"]
        custom_methods = ["method1", "method2"]
        custom_techniques = ["XRD", "SEM", "TEM"]

        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            composition_property_extraction_agent_notes=custom_notes,
            synthesis_extraction_task_notes=custom_notes,
            allowed_synthesis_methods=custom_methods,
            allowed_characterization_techniques=custom_techniques,
        )

        # Verify notes are set (they get processed by internal methods)
        assert isinstance(flow.state.composition_property_extraction_agent_note, str)
        assert isinstance(flow.state.synthesis_extraction_task_note, str)
        assert isinstance(flow.state.allowed_synthesis_methods, str)
        assert isinstance(flow.state.allowed_characterization_techniques, str)

    def test_flow_with_custom_materials_identifier_query(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test flow with custom materials identifier query"""
        custom_query = "Does this paper contain any material compositions and property data? Answer yes or no."

        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            materials_data_identifier_query=custom_query,
        )

        assert flow.state.materials_data_identifier_query == custom_query


class TestDataExtractionFlowParameterValidation:
    """Test parameter validation and edge cases"""

    def test_initialization_with_empty_strings(self):
        """Test initialization with empty string parameters"""
        with pytest.raises(ValueErrorHandler, match="DOI is required"):
            DataExtractionFlow(
                doi="",  # Empty string should be treated as None
                main_extraction_keyword="piezoelectric",
                composition_property_text_data="some text",
            )

        with pytest.raises(
            ValueErrorHandler, match="Main property keyword is required"
        ):
            DataExtractionFlow(
                doi="10.1186/test",
                main_extraction_keyword="",  # Empty string
                composition_property_text_data="some text",
            )

        with pytest.raises(
            ValueErrorHandler, match="Composition property text data is required"
        ):
            DataExtractionFlow(
                doi="10.1186/test",
                main_extraction_keyword="piezoelectric",
                composition_property_text_data="",  # Empty string
            )

    def test_boolean_parameter_handling(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test boolean parameter handling"""
        # Test with explicit False values
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            is_extract_synthesis_data=False,
            is_log_json=False,
            verbose=False,
        )

        assert flow.state.is_extract_synthesis_data is False
        assert flow.state.is_log_json is False
        assert flow.state.verbose is False

    def test_flow_routing_edge_cases(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test flow routing with various edge case responses"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        # Test case variations
        test_cases = [
            ("YES MATERIALS FOUND", "extract_compositions"),
            ("No materials present", "end_flow"),
            ("yes, found materials", "extract_compositions"),
            ("DEFINITELY YES", "extract_compositions"),
            ("absolutely no", "end_flow"),
            ("uncertain", "end_flow"),  # Default to end_flow for unclear responses
        ]

        for response, expected_route in test_cases:
            flow.state.is_materials_mentioned = response
            result = flow.route_process()
            assert result == expected_route, f"Failed for response: '{response}'"


# Additional fixtures for complex testing scenarios
@pytest.fixture
def mock_crew_result():
    """Factory for creating mock crew results"""

    def _create_mock_result(raw_response):
        mock_result = MagicMock()
        mock_result.raw = raw_response
        return mock_result

    return _create_mock_result


@pytest.fixture
def mock_crew_factory():
    """Factory for creating mock crews"""

    def _create_mock_crew(crew_class_mock, raw_response):
        mock_crew_instance = MagicMock()
        crew_class_mock.return_value.crew.return_value = mock_crew_instance
        mock_result = MagicMock()
        mock_result.raw = raw_response
        mock_crew_instance.kickoff.return_value = mock_result
        return mock_crew_instance

    return _create_mock_crew


class TestDataExtractionFlowComplexScenarios:
    """Test complex real-world scenarios"""

    def test_flow_with_logging_enabled(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
        sample_synthesis_text,
    ):
        """Test flow initialization with logging parameters"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            synthesis_text_data=sample_synthesis_text,
            output_log_folder="/test/logs",
            task_output_folder="/test/tasks",
            is_log_json=True,
        )

        # Verify logging parameters are set
        assert flow.state.output_log_folder == "/test/logs"
        assert flow.state.task_output_folder == "/test/tasks"
        assert flow.state.is_log_json is True

    def test_materials_identification_response_cleaning(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test various response formats from materials identification"""
        flow = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
        )

        with patch(
            "comproscanner.extract_flow.main_extraction_flow.MaterialsDataIdentifierCrew"
        ) as mock_crew:
            mock_crew_instance = MagicMock()
            mock_crew.return_value.crew.return_value = mock_crew_instance

            # Test various response formats that need cleaning
            test_cases = [
                ('"yes"', "yes"),
                ("'no'", "'no'"),  # Single quotes should remain
                ("  yes  ", "yes"),
                ('"YES"', "YES"),
                ('""no""', '"no"'),  # Only outer quotes removed
            ]

            for raw_response, expected_cleaned in test_cases:
                mock_result = MagicMock()
                mock_result.raw = raw_response
                mock_crew_instance.kickoff.return_value = mock_result

                result = flow.identify_materials_data_presence()
                assert result == expected_cleaned
                assert flow.state.is_materials_mentioned == expected_cleaned

    def test_synthesis_data_skipping_conditions(
        self,
        sample_doi,
        sample_main_extraction_keyword,
        sample_composition_property_text,
    ):
        """Test all conditions that cause synthesis data extraction to be skipped"""
        # Test with whitespace-only synthesis text
        flow1 = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            synthesis_text_data="   \n\t   ",  # Only whitespace
        )
        result1 = flow1.extract_synthesis_data()
        assert "synthesis_data" in result1
        assert result1["synthesis_data"]["method"] == ""

        # Test with synthesis extraction disabled
        flow2 = DataExtractionFlow(
            doi=sample_doi,
            main_extraction_keyword=sample_main_extraction_keyword,
            composition_property_text_data=sample_composition_property_text,
            synthesis_text_data="Valid synthesis text",
            is_extract_synthesis_data=False,
        )
        result2 = flow2.extract_synthesis_data()
        assert "synthesis_data" in result2
        assert result2["synthesis_data"]["method"] == ""
