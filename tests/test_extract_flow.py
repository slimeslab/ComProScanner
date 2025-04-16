import pytest
import json
from unittest.mock import MagicMock, patch

# Import the DataExtractionFlow class and related dependencies
from comproscanner.extract_flow.main_extraction_flow import (
    DataExtractionFlow,
    MaterialsState,
)
from comproscanner.utils.error_handler import ValueErrorHandler
from comproscanner.utils.configs.rag_config import RAGConfig


class TestResult:
    """Simple class to simulate CrewAI task results"""

    def __init__(self, raw):
        self.raw = raw


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


@pytest.fixture
def basic_flow(
    sample_doi, sample_main_extraction_keyword, sample_composition_property_text
):
    """Create a basic DataExtractionFlow instance with required parameters only"""
    return DataExtractionFlow(
        doi=sample_doi,
        main_extraction_keyword=sample_main_extraction_keyword,
        composition_property_text_data=sample_composition_property_text,
    )


@pytest.fixture
def complete_flow(
    sample_doi,
    sample_main_extraction_keyword,
    sample_composition_property_text,
    sample_synthesis_text,
    mock_rag_config,
    mock_llm,
):
    """Create a DataExtractionFlow instance with all parameters"""
    return DataExtractionFlow(
        doi=sample_doi,
        main_extraction_keyword=sample_main_extraction_keyword,
        composition_property_text_data=sample_composition_property_text,
        synthesis_text_data=sample_synthesis_text,
        llm=mock_llm,
        materials_data_identifier_query="Custom query?",
        extract_synthesis_data=True,
        rag_config=mock_rag_config,
        output_log_folder="output_logs",
        task_output_folder="task_output",
        is_log_json=True,
        verbose=True,
        allowed_synthesis_methods=["solid_state_reaction", "sol_gel", "hydrothermal"],
        allowed_characterization_techniques=["x_ray_diffraction", "sem", "tem"],
    )


class TestDataExtractionFlow:
    def test_initialization_with_required_params(
        self, basic_flow, sample_doi, sample_main_extraction_keyword
    ):
        """Test initialization with only required parameters"""
        assert basic_flow.state.doi == sample_doi
        assert (
            basic_flow.state.main_extraction_keyword == sample_main_extraction_keyword
        )
        assert basic_flow.state.extract_synthesis_data is False
        assert basic_flow.state.verbose is True

    def test_initialization_with_all_params(
        self, complete_flow, sample_doi, mock_llm, mock_rag_config
    ):
        """Test initialization with all parameters"""
        assert complete_flow.state.doi == sample_doi
        assert complete_flow.state.llm == mock_llm
        assert complete_flow.state.rag_config == mock_rag_config
        assert complete_flow.state.extract_synthesis_data is True
        assert complete_flow.state.output_log_folder == "output_logs"
        assert "solid_state_reaction" in complete_flow.state.allowed_synthesis_methods
        assert (
            "x_ray_diffraction"
            in complete_flow.state.allowed_characterization_techniques
        )

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

    def test_parse_json_output_valid_json(self, basic_flow):
        """Test parsing valid JSON output"""
        json_str = '{"key": "value", "number": 42}'
        result = basic_flow._parse_json_output(json_str)
        assert result == {"key": "value", "number": 42}

    def test_parse_json_output_markdown_json(self, basic_flow):
        """Test parsing JSON inside markdown code blocks"""
        markdown_json = '```json\n{"key": "value", "number": 42}\n```'
        result = basic_flow._parse_json_output(markdown_json)
        assert result == {"key": "value", "number": 42}

    def test_parse_json_output_python_format(self, basic_flow):
        """Test parsing Python-style JSON with booleans and None"""
        python_json = '{"key": "value", "active": true, "data": null}'
        result = basic_flow._parse_json_output(python_json)
        assert result == {"key": "value", "active": True, "data": None}

    def test_parse_json_output_invalid_json(self, basic_flow):
        """Test handling of invalid JSON"""
        invalid_json = "{this is not valid json}"
        default_value = {"default": "value"}
        result = basic_flow._parse_json_output(
            invalid_json, default_value=default_value
        )
        assert result == default_value

    @patch(
        "comproscanner.extraction_flowcrews.materials_data_identifier_crew.materials_data_identifier_crew.MaterialsDataIdentifierCrew"
    )
    def test_identify_materials_data_presence_positive(
        self, mock_crew_class, basic_flow
    ):
        """Test identification of materials data presence (positive case)"""
        # Configure mocks
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_crew_instance.kickoff.return_value = TestResult("YES")

        # Run the test
        result = basic_flow.identify_materials_data_presence()

        # Verify result
        assert result == "YES"
        assert basic_flow.state.is_materials_mentioned == "YES"
        mock_crew_instance.kickoff.assert_called_once()

    @patch(
        "comproscanner.extraction_flowcrews.materials_data_identifier_crew.materials_data_identifier_crew.MaterialsDataIdentifierCrew"
    )
    def test_router_with_materials_present(self, mock_crew_class, basic_flow):
        """Test routing when materials are present"""
        # Setup
        basic_flow.state.is_materials_mentioned = "YES"

        # Execute
        result = basic_flow.route_process()

        # Verify
        assert result == "extract_compositions"

    @patch(
        "comproscanner.extraction_flowcrews.materials_data_identifier_crew.materials_data_identifier_crew.MaterialsDataIdentifierCrew"
    )
    def test_router_with_no_materials(self, mock_crew_class, basic_flow):
        """Test routing when no materials are present"""
        # Setup
        basic_flow.state.is_materials_mentioned = "NO"

        # Execute
        result = basic_flow.route_process()

        # Verify
        assert result == "end_flow"

    @patch(
        "comproscanner.extraction_flowcrews.composition_crew.composition_extraction_crew.composition_extraction_crew.CompositionExtractionCrew"
    )
    def test_extract_composition_property_data(
        self, mock_crew_class, basic_flow, sample_composition_extracted_data
    ):
        """Test extraction of composition and property data"""
        # Configure mocks
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_crew_instance.kickoff.return_value = TestResult(
            json.dumps(sample_composition_extracted_data)
        )

        # Run the test
        result = basic_flow.extract_composition_property_data()

        # Verify result
        assert result == sample_composition_extracted_data
        assert (
            basic_flow.state.composition_extracted_data
            == sample_composition_extracted_data
        )
        mock_crew_instance.kickoff.assert_called_once()

    @patch(
        "comproscanner.extraction_flowcrews.composition_crew.composition_format_crew.composition_format_crew.CompositionFormatCrew"
    )
    def test_extract_final_composition_property_data(
        self,
        mock_crew_class,
        basic_flow,
        sample_composition_extracted_data,
        sample_composition_formatted_data,
    ):
        """Test extraction of final composition and property data"""
        # Setup state
        basic_flow.state.composition_extracted_data = sample_composition_extracted_data

        # Configure mocks
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_crew_instance.kickoff.return_value = TestResult(
            json.dumps(
                {"composition_formatted_data": sample_composition_formatted_data}
            )
        )

        # Run the test
        basic_flow.extract_final_composition_property_data()

        # Verify result
        assert (
            basic_flow.state.composition_formatted_data
            == sample_composition_formatted_data
        )
        mock_crew_instance.kickoff.assert_called_once()

    @patch(
        "comproscanner.extraction_flowcrews.synthesis_crew.synthesis_extraction_crew.synthesis_extraction_crew.SynthesisExtractionCrew"
    )
    def test_extract_synthesis_data(
        self,
        mock_crew_class,
        basic_flow,
        sample_synthesis_text,
        sample_composition_formatted_data,
        sample_synthesis_extracted_data,
    ):
        """Test extraction of synthesis data"""
        # Setup state
        basic_flow.state.synthesis_text_data = sample_synthesis_text
        basic_flow.state.extract_synthesis_data = True
        basic_flow.state.composition_formatted_data = sample_composition_formatted_data

        # Configure mocks
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_crew_instance.kickoff.return_value = TestResult(
            json.dumps(sample_synthesis_extracted_data)
        )

        # Run the test
        result = basic_flow.extract_synthesis_data()

        # Verify result
        assert result == sample_synthesis_extracted_data
        assert (
            basic_flow.state.synthesis_extracted_data == sample_synthesis_extracted_data
        )
        mock_crew_instance.kickoff.assert_called_once()

    def test_extract_synthesis_data_empty_text(self, basic_flow):
        """Test extraction of synthesis data with empty text"""
        # Setup state
        basic_flow.state.synthesis_text_data = ""
        basic_flow.state.extract_synthesis_data = True

        # Run the test
        result = basic_flow.extract_synthesis_data()

        # Verify result
        assert result == {
            "synthesis_data": {
                "method": "",
                "precursors": [],
                "steps": [],
                "characterization_techniques": [],
            }
        }

    def test_extract_synthesis_data_disabled(self, basic_flow, sample_synthesis_text):
        """Test when synthesis data extraction is disabled"""
        # Setup state
        basic_flow.state.synthesis_text_data = sample_synthesis_text
        basic_flow.state.extract_synthesis_data = False

        # Run the test
        result = basic_flow.extract_synthesis_data()

        # Verify result
        assert result == {
            "synthesis_data": {
                "method": "",
                "precursors": [],
                "steps": [],
                "characterization_techniques": [],
            }
        }

    @patch(
        "comproscanner.extraction_flowcrews.synthesis_crew.synthesis_format_crew.synthesis_format_crew.SynthesisFormatCrew"
    )
    def test_extract_final_synthesis_data(
        self,
        mock_crew_class,
        basic_flow,
        sample_synthesis_extracted_data,
        sample_synthesis_formatted_data,
    ):
        """Test extraction of final synthesis data"""
        # Setup state
        basic_flow.state.synthesis_extracted_data = sample_synthesis_extracted_data

        # Configure mocks
        mock_crew_instance = MagicMock()
        mock_crew_class.return_value.crew.return_value = mock_crew_instance
        mock_crew_instance.kickoff.return_value = TestResult(
            json.dumps({"synthesis_formatted_data": sample_synthesis_formatted_data})
        )

        # Run the test
        basic_flow.extract_final_synthesis_data()

        # Verify result
        assert (
            basic_flow.state.synthesis_formatted_data == sample_synthesis_formatted_data
        )
        mock_crew_instance.kickoff.assert_called_once()

    def test_extract_final_synthesis_data_empty(self, basic_flow):
        """Test handling of empty synthesis data"""
        # Setup state with empty synthesis data
        basic_flow.state.synthesis_extracted_data = {
            "synthesis_data": {
                "method": "",
                "precursors": [],
                "steps": [],
                "characterization_techniques": [],
            }
        }

        # Run the test
        basic_flow.extract_final_synthesis_data()

        # Verify that the empty data is used directly
        assert basic_flow.state.synthesis_formatted_data == {
            "method": "",
            "precursors": [],
            "steps": [],
            "characterization_techniques": [],
        }

    def test_finalize_results(
        self,
        basic_flow,
        sample_composition_formatted_data,
        sample_synthesis_formatted_data,
    ):
        """Test finalizing results from the flow"""
        # Setup state
        basic_flow.state.composition_formatted_data = sample_composition_formatted_data
        basic_flow.state.synthesis_formatted_data = sample_synthesis_formatted_data

        # Run the test
        result = basic_flow.finalize_results()

        # Verify result
        expected = {
            "composition_data": sample_composition_formatted_data,
            "synthesis_data": sample_synthesis_formatted_data,
        }
        assert result == expected

    def test_terminate_process(self, basic_flow):
        """Test termination of the process"""
        # Run the test
        result = basic_flow.terminate_process()

        # Verify result
        expected = {
            "composition_data": {},
            "synthesis_data": {},
        }
        assert result == expected

    @patch(
        "comproscanner.extraction_flowcrews.materials_data_identifier_crew.materials_data_identifier_crew.MaterialsDataIdentifierCrew"
    )
    @patch(
        "comproscanner.extraction_flowcrews.composition_crew.composition_extraction_crew.composition_extraction_crew.CompositionExtractionCrew"
    )
    @patch(
        "comproscanner.extraction_flowcrews.composition_crew.composition_format_crew.composition_format_crew.CompositionFormatCrew"
    )
    def test_full_flow_without_synthesis(
        self,
        mock_format_crew,
        mock_extract_crew,
        mock_identifier_crew,
        basic_flow,
        sample_composition_extracted_data,
        sample_composition_formatted_data,
    ):
        """Test full flow without synthesis extraction"""
        # Configure mocks
        mock_identifier_instance = MagicMock()
        mock_identifier_crew.return_value.crew.return_value = mock_identifier_instance
        mock_identifier_instance.kickoff.return_value = TestResult("YES")

        mock_extract_instance = MagicMock()
        mock_extract_crew.return_value.crew.return_value = mock_extract_instance
        mock_extract_instance.kickoff.return_value = TestResult(
            json.dumps(sample_composition_extracted_data)
        )

        mock_format_instance = MagicMock()
        mock_format_crew.return_value.crew.return_value = mock_format_instance
        mock_format_instance.kickoff.return_value = TestResult(
            json.dumps(
                {"composition_formatted_data": sample_composition_formatted_data}
            )
        )

        # Run identify
        result = basic_flow.identify_materials_data_presence()
        assert result == "YES"

        # Run router
        route = basic_flow.route_process()
        assert route == "extract_compositions"

        # Run extraction
        basic_flow.extract_composition_property_data()
        basic_flow.extract_final_composition_property_data()

        # Set empty synthesis data (no extraction)
        basic_flow.state.synthesis_formatted_data = {
            "method": "",
            "precursors": [],
            "steps": [],
            "characterization_techniques": [],
        }

        # Finalize
        final_result = basic_flow.finalize_results()
        expected = {
            "composition_data": sample_composition_formatted_data,
            "synthesis_data": {
                "method": "",
                "precursors": [],
                "steps": [],
                "characterization_techniques": [],
            },
        }
        assert final_result == expected
