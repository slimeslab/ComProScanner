"""
main_extraction_flow.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 19-03-2025
"""

# Standard library imports
from typing import Dict, Optional
import json
import re
from textwrap import dedent
import ast

# Third party imports
from pydantic import BaseModel, ConfigDict
from crewai import LLM
from crewai.flow.flow import Flow, listen, start, router

# Custom imports
from ..utils.error_handler import ValueErrorHandler
from ..utils.logger import setup_logger
from ..utils.configs.rag_config import RAGConfig
from .crews.materials_data_identifier_crew.materials_data_identifier_crew import (
    MaterialsDataIdentifierCrew,
)
from .crews.composition_crew.composition_extraction_crew.composition_extraction_crew import (
    CompositionExtractionCrew,
)
from .crews.composition_crew.composition_format_crew.composition_format_crew import (
    CompositionFormatCrew,
)
from .crews.synthesis_crew.synthesis_extraction_crew.synthesis_extraction_crew import (
    SynthesisExtractionCrew,
)
from .crews.synthesis_crew.synthesis_format_crew.synthesis_format_crew import (
    SynthesisFormatCrew,
)

######## logger Configuration ########
logger = setup_logger("composition_property_extractor.log")


class MaterialsState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    is_materials_mentioned: str = ""
    composition_extracted_data: Dict = {}
    composition_formatted_data: Dict = {}
    synthesis_extracted_data: Dict = {}
    synthesis_formatted_data: Dict = {}
    doi: str = ""
    materials_data_identifier_query: str = ""
    main_extraction_keyword: str = ""
    composition_property_text_data: str = ""
    synthesis_text_data: str = ""
    extract_synthesis_data: bool = False
    llm: Optional[LLM] = None
    rag_config: Optional[RAGConfig] = None
    output_log_folder: Optional[str] = None
    task_output_folder: Optional[str] = None
    is_log_json: Optional[bool] = False
    verbose: Optional[bool] = True
    expected_composition_property_example: str = ""
    expected_variable_composition_property_example: str = ""
    composition_property_extraction_agent_note: str = ""
    composition_property_extraction_task_note: str = ""
    composition_property_formatting_agent_note: str = ""
    composition_property_formatting_task_note: str = ""
    synthesis_extraction_agent_note: str = ""
    synthesis_extraction_task_note: str = ""
    synthesis_formatting_agent_note: str = ""
    synthesis_formatting_task_note: str = ""
    allowed_synthesis_methods: str = ""
    allowed_characterization_techniques: str = ""


class DataExtractionFlow(Flow[MaterialsState]):
    """CrewAI Flow for extracting composition, property value and synthesis data from text

    Args:
        doi (str: required): DOI of the research paper
        main_extraction_keyword (str: required): Main property keyword to extract from the text
        composition_property_text_data (str: required): Text data to extract composition and property data
        synthesis_text_data (str: required): Text data to extract synthesis data
        llm (LLM: optional): LLM instance for the agents. Default: None
        materials_data_identifier_query (str: optional): Query to identify if materials data is present in the text. Must be an 'Yes/No' answer. Default: "Is there any material chemical composition (not abbreviations) and corresponding {main_extraction_keyword} value mentioned in the paper? GIVE ONE WORD ANSWER. Either YES or NO."
        extract_synthesis_data (bool: optional): Flag to extract synthesis data. Default: False
        rag_config (RAGConfig: optional): RAG configuration. Default: None
        output_log_folder (str, optional): Set to True to save logs inside {provided folder}/{doi} folder as .txt files. Logs will be in JSON format if the is_log_json is True, otherwise .txt. Defaults to None.
        task_output_folder (str, optional): Folder path for storing the task outputs as .txt files inside {provided foler}/{doi} folder. Defaults to None.
        is_log_json (bool, optional): Flag to save logs in JSON format. Defaults to False.
        verbose (bool: optional): Flag to enable verbose logging. Default: True
        expected_composition_property_example (str: optional): Expected example of composition and property data.
        expected_variable_composition_property_example (str: optional): Expected example of variable composition and property data.
        composition_property_extraction_agent_notes (list: optional): Notes for composition extraction agent. Default: [].
        composition_property_extraction_task_notes (list: optional): Notes for composition extraction task. Default: [].
        composition_property_formatting_agent_notes (list: optional): Notes for composition formatting agent. Default: [].
        composition_property_formatting_task_notes (list: optional): Notes for composition formatting task. Default: [].
        synthesis_extraction_agent_notes (list: optional): Notes for synthesis extraction agent. Default: [].
        synthesis_extraction_task_notes (list: optional): Notes for synthesis extraction task. Default: [].
        synthesis_formatting_agent_notes (list: optional): Notes for synthesis formatting agent. Default: [].
        synthesis_formatting_task_notes (list: optional): Notes for synthesis formatting task. Default: [].
        allowed_synthesis_methods (list: optional): Allowed synthesis methods for knowledge-graph nodes. Default: [].
        allowed_characterization_techniques (list: optional): Allowed characterization techniques for knowledge-graph nodes. Default: [].

    Returns:
        result_json (dict): Final result in JSON format
        result_dict (dict): Final result in dictionary format
    """

    def __init__(
        self,
        doi: str = None,
        main_extraction_keyword: str = None,
        composition_property_text_data: str = None,
        synthesis_text_data: str = None,
        llm: Optional[LLM] = None,
        materials_data_identifier_query: str = None,
        extract_synthesis_data: bool = False,
        rag_config: Optional[RAGConfig] = None,
        output_log_folder: Optional[str] = None,
        task_output_folder: Optional[str] = None,
        is_log_json: Optional[bool] = False,
        verbose: bool = True,
        expected_composition_property_example: str = "",
        expected_variable_composition_property_example: str = "",
        composition_property_extraction_agent_notes: list = [],
        composition_property_extraction_task_notes: list = [],
        composition_property_formatting_agent_notes: list = [],
        composition_property_formatting_task_notes: list = [],
        synthesis_extraction_agent_notes: list = [],
        synthesis_extraction_task_notes: list = [],
        synthesis_formatting_agent_notes: list = [],
        synthesis_formatting_task_notes: list = [],
        allowed_synthesis_methods: list = [],
        allowed_characterization_techniques: list = [],
    ):
        super().__init__()
        if not doi:
            raise ValueErrorHandler("DOI is required")
        if not main_extraction_keyword:
            raise ValueErrorHandler("Main property keyword is required")
        if not composition_property_text_data:
            raise ValueErrorHandler("Composition property text data is required")

        self.state.doi = doi
        self.state.llm = llm
        self.state.extract_synthesis_data = extract_synthesis_data
        self.state.rag_config = rag_config
        self.state.output_log_folder = output_log_folder
        self.state.task_output_folder = task_output_folder
        self.state.is_log_json = is_log_json
        self.state.verbose = verbose
        self.state.expected_composition_property_example = (
            expected_composition_property_example
        )
        self.state.expected_variable_composition_property_example = (
            expected_variable_composition_property_example
        )
        main_extraction_keyword = main_extraction_keyword.replace(" ", "_")
        self.state.main_extraction_keyword = main_extraction_keyword
        self.state.materials_data_identifier_query = materials_data_identifier_query
        self.state.composition_property_text_data = composition_property_text_data
        self.state.synthesis_text_data = synthesis_text_data

        default_expected_composition_property_example = dedent(
            f"""
{{
  "compositions":
  {{
    "Ba0.99Ca0.01Ti0.68Zr0.32O3": 375, 
    "Ba0.98Ca0.02Ti0.78Zr0.22O3": 350, 
    "Ba0.97Ca0.03Ti0.88Zr0.12O3": 325, 
    "Ba0.96Ca0.04Ti0.98Zr0.02O3": 300
  }},
  "property_unit": "pC/N", 
  "family": "BaTiO3"
}}"""
        )

        default_expected_variable_composition_property_example = dedent(
            f"""
{{
"compositions": 
  {{
    "0.5NaNbO3": 375, 
    "(1-x)Na0.2K2(x)Bi0.5TiO3 - (y)NaNbO3 where x=0, y=0.5": 350, 
    "(1-x)Na0.2K2(x)Bi0.5TiO3 - (y)NaNbO3 where x=0.1, y=0.4": 325, 
    "(1-x)Na0.2K2(x)Bi0.5TiO3 - (y)NaNbO3 where x=0.2, y=0.3": 375, 
    "(1-x)Na0.2K2(x)Bi0.5TiO3 - (y)NaNbO3 where x=0.3, y=0.1": 425 
  }},
  "property_unit": "pC/N", 
  "family": "NaNbO3"
}}"""
        )
        if not expected_composition_property_example:
            self.state.expected_composition_property_example = (
                default_expected_composition_property_example
            )
        if not expected_variable_composition_property_example:
            self.state.expected_variable_composition_property_example = (
                default_expected_variable_composition_property_example
            )

        def _update_notes(note: list, default_note: list = None):
            if not note and not default_note:
                return ""

            # Create bullet points for default notes
            if default_note:
                default_points = dedent(
                    """{}""".format("\n".join(f"- {n}" for n in default_note))
                )
            else:
                default_points = ""

            # Add custom notes if they exist
            if default_points and note:
                all_notes = dedent(
                    """{}""".format(
                        default_points + "\n" + "\n".join(f"- {n}" for n in note)
                    )
                )
            elif note:
                all_notes = dedent("""{}""".format("\n".join(f"- {n}" for n in note)))
            else:
                all_notes = default_points

            return dedent(f"""**Notes**:\n{all_notes}""")

        def _update_methods_techniques(item_list: list):
            return dedent("""{}""".format("\n".join(f"- {n}" for n in item_list)))

        # optional notes to pass to the agents and tasks
        composition_property_extraction_default_notes = [
            "The unit, given here, along with other data, is just an example. It can change depending on the property."
        ]
        synthesis_extraction_default_notes = [
            "For precursors, just use the chemical name (no company or purity) and for characterization_techniques, only the instrument name or short name of the instrument for the instruments like XRD, Raman Spectroscopy, SEM, TEM etc."
        ]
        synthesis_formatting_default_notes = [
            "Only modify 'methods' and 'characterization_techniques' values to match allowed entities",
            "Keep 'steps' and 'precursors' values unmodified",
        ]

        self.state.composition_property_extraction_agent_note = _update_notes(
            composition_property_extraction_agent_notes,
            composition_property_extraction_default_notes,
        )
        self.state.composition_property_extraction_task_note = _update_notes(
            composition_property_extraction_task_notes
        )
        self.state.composition_property_formatting_agent_note = _update_notes(
            composition_property_formatting_agent_notes
        )
        self.state.composition_property_formatting_task_note = _update_notes(
            composition_property_formatting_task_notes,
            composition_property_extraction_default_notes,
        )

        self.state.synthesis_extraction_agent_note = _update_notes(
            synthesis_extraction_agent_notes
        )
        self.state.synthesis_extraction_task_note = _update_notes(
            synthesis_extraction_task_notes, synthesis_extraction_default_notes
        )
        self.state.synthesis_formatting_agent_note = _update_notes(
            synthesis_formatting_agent_notes
        )
        self.state.synthesis_formatting_task_note = _update_notes(
            synthesis_formatting_task_notes, synthesis_formatting_default_notes
        )

        # allowed synthesis methods and characterization techniques
        self.state.allowed_synthesis_methods = _update_methods_techniques(
            allowed_synthesis_methods
        )
        self.state.allowed_characterization_techniques = _update_methods_techniques(
            allowed_characterization_techniques
        )

    def _parse_json_output(self, raw_output, default_value=None, log_prefix=""):
        """
        Parses JSON output from LLM responses, handling different output formats including Python escape sequences.

        Args:
            raw_output: The raw output from the LLM
            default_value: The default value to return if parsing fails
            log_prefix: Prefix for log messages

        Returns:
            Parsed data or default_value if parsing fails
        """
        if default_value is None:
            default_value = {}

        try:
            # Handle non-string input
            if not isinstance(raw_output, str):
                return raw_output

            # Clean up markdown code blocks
            cleaned_output = raw_output.strip()
            if "```json" in cleaned_output:
                json_match = re.search(
                    r"```json\s*(.*?)\s*```", cleaned_output, re.DOTALL
                )
                if json_match:
                    cleaned_output = json_match.group(1)
                else:
                    json_match = re.search(
                        r"```\s*(.*?)\s*```", cleaned_output, re.DOTALL
                    )
                    if json_match:
                        cleaned_output = json_match.group(1)

            # Attempt 1: Standard JSON parsing
            try:
                parsed_output = json.loads(cleaned_output)
                logger.debug(f"{log_prefix} parsed successfully using standard JSON")
                return parsed_output
            except json.JSONDecodeError as e:
                logger.debug(
                    f"Standard JSON parsing failed: {e}, trying ast.literal_eval"
                )

            # Attempt 2: Use ast.literal_eval to handle Python-style string with escape sequences
            try:
                # Convert JSON booleans/null to Python format
                python_literal = (
                    cleaned_output.replace("true", "True")
                    .replace("false", "False")
                    .replace("null", "None")
                )
                evaluated_dict = ast.literal_eval(python_literal)
                logger.debug(f"{log_prefix} parsed successfully using ast.literal_eval")
                return evaluated_dict
            except (ValueError, SyntaxError) as e:
                logger.error(
                    f"Failed to parse {log_prefix} using ast.literal_eval: {e}"
                )
                return default_value

        except Exception as e:
            logger.error(f"Error processing {log_prefix} data: {e}")
            return default_value

    @start()
    def identify_materials_data_presence(self):
        """Identify if threre is any material and corresponding property present in the text"""
        logger.debug("Starting material identification process...")
        if self.state.llm:
            rag_crew = MaterialsDataIdentifierCrew(
                doi=self.state.doi,
                llm=self.state.llm,
                rag_config=self.state.rag_config,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
            ).crew()
        else:
            rag_crew = MaterialsDataIdentifierCrew(
                doi=self.state.doi,
                rag_config=self.state.rag_config,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
            ).crew()
        result = rag_crew.kickoff(
            inputs={
                "doi": self.state.doi,
                "materials_data_identifier_query": self.state.materials_data_identifier_query,
                "main_extraction_keyword": self.state.main_extraction_keyword,
            }
        )
        self.state.is_materials_mentioned = result.raw
        return self.state.is_materials_mentioned

    @router(identify_materials_data_presence)
    def route_process(self):
        """Routes based on material type in state"""
        if "yes" in self.state.is_materials_mentioned.lower():
            return "extract_compositions"
        else:
            return "end_flow"

    @listen("extract_compositions")
    def extract_composition_property_data(self):
        """Extract composition and property data"""
        logger.debug("Extracting composition and property data...")
        if self.state.llm:
            composition_property_crew = CompositionExtractionCrew(
                doi=self.state.doi,
                llm=self.state.llm,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
            ).crew()
        else:
            composition_property_crew = CompositionExtractionCrew(
                doi=self.state.doi,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
            ).crew()

        result = composition_property_crew.kickoff(
            inputs={
                "composition_property_text_data": self.state.composition_property_text_data,
                "main_extraction_keyword": self.state.main_extraction_keyword,
                "composition_property_extraction_agent_note": self.state.composition_property_extraction_agent_note,
                "composition_property_extraction_task_note": self.state.composition_property_extraction_task_note,
                "expected_composition_property_example": self.state.expected_composition_property_example,
                "expected_variable_composition_property_example": self.state.expected_variable_composition_property_example,
            }
        )

        self.state.composition_extracted_data = self._parse_json_output(
            result.raw, default_value={}, log_prefix="Composition extraction"
        )
        return self.state.composition_extracted_data

    @listen(extract_composition_property_data)
    def extract_final_composition_property_data(self):
        """Extract final composition and property data"""
        logger.debug("Extracting final composition and property data...")
        if self.state.llm:
            composition_format_crew = CompositionFormatCrew(
                doi=self.state.doi,
                llm=self.state.llm,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
            ).crew()
        else:
            composition_format_crew = CompositionFormatCrew(
                doi=self.state.doi,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
            ).crew()

        result = composition_format_crew.kickoff(
            inputs={
                "extracted_composition_data": self.state.composition_extracted_data,
                "composition_property_formatting_agent_note": self.state.composition_property_formatting_agent_note,
                "composition_property_formatting_task_note": self.state.composition_property_formatting_task_note,
                "main_extraction_keyword": self.state.main_extraction_keyword,
                "expected_composition_property_example": self.state.expected_composition_property_example,
            }
        )

        parsed_data = self._parse_json_output(
            result.raw,
            default_value={"composition_formatted_data": {}},
            log_prefix="Composition formatting",
        )
        self.state.composition_formatted_data = parsed_data.get(
            "composition_formatted_data", {}
        )

    @listen(extract_final_composition_property_data)
    def extract_synthesis_data(self):
        """Extract synthesis data"""
        if (
            not self.state.synthesis_text_data
            or self.state.synthesis_text_data.strip() == ""
            or self.state.extract_synthesis_data == False
        ):
            logger.warning(
                "Synthesis text data is empty or extraction skipped, not performing synthesis extraction"
            )
            self.state.synthesis_extracted_data = {
                "synthesis_data": {
                    "method": "",
                    "precursors": [],
                    "steps": [],
                    "characterization_techniques": [],
                }
            }
            return self.state.synthesis_extracted_data

        logger.debug("Extracting synthesis data...")
        if self.state.llm:
            synthesis_extraction_crew = SynthesisExtractionCrew(
                doi=self.state.doi,
                llm=self.state.llm,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
            ).crew()
        else:
            synthesis_extraction_crew = SynthesisExtractionCrew(
                doi=self.state.doi,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
            ).crew()

        result = synthesis_extraction_crew.kickoff(
            inputs={
                "synthesis_text_data": self.state.synthesis_text_data,
                "composition_formatted_data": self.state.composition_formatted_data,
                "synthesis_extraction_agent_note": self.state.synthesis_extraction_agent_note,
                "synthesis_extraction_task_note": self.state.synthesis_extraction_task_note,
            }
        )

        default_synthesis_data = {
            "synthesis_data": {
                "method": "",
                "precursors": [],
                "steps": [],
                "characterization_techniques": [],
            }
        }

        self.state.synthesis_extracted_data = self._parse_json_output(
            result.raw,
            default_value=default_synthesis_data,
            log_prefix="Synthesis extraction",
        )
        return self.state.synthesis_extracted_data

    @listen(extract_synthesis_data)
    def extract_final_synthesis_data(self):
        """Extract final synthesis data"""
        if (
            isinstance(self.state.synthesis_extracted_data, dict)
            and "synthesis_data" in self.state.synthesis_extracted_data
        ):
            extracted_data = self.state.synthesis_extracted_data.get(
                "synthesis_data", {}
            )
            if (
                not extracted_data.get("method")
                and not extracted_data.get("precursors")
                and not extracted_data.get("steps")
            ):
                self.state.synthesis_formatted_data = (
                    self.state.synthesis_extracted_data.get("synthesis_data", {})
                )
                return

        logger.debug("Extracting final synthesis data...")
        if self.state.llm:
            synthesis_format_crew = SynthesisFormatCrew(
                doi=self.state.doi,
                llm=self.state.llm,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
            ).crew()
        else:
            synthesis_format_crew = SynthesisFormatCrew(
                doi=self.state.doi,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
            ).crew()

        result = synthesis_format_crew.kickoff(
            inputs={
                "extracted_synthesis_data": self.state.synthesis_extracted_data,
                "synthesis_formatting_agent_note": self.state.synthesis_formatting_agent_note,
                "synthesis_formatting_task_note": self.state.synthesis_formatting_task_note,
                "allowed_synthesis_methods": self.state.allowed_synthesis_methods,
                "allowed_characterization_techniques": self.state.allowed_characterization_techniques,
            }
        )

        parsed_data = self._parse_json_output(
            result.raw,
            default_value={"synthesis_formatted_data": {}},
            log_prefix="Synthesis formatting",
        )
        self.state.synthesis_formatted_data = parsed_data.get(
            "synthesis_formatted_data", {}
        )

    @listen(extract_final_synthesis_data)
    def finalize_results(self):
        """Combine and return final results"""
        logger.debug("Finalizing results...")
        return {
            "composition_data": self.state.composition_formatted_data,
            "synthesis_data": self.state.synthesis_formatted_data,
        }

    @listen("end_flow")
    def terminate_process(self):
        logger.warning("Terminating process...")
        return {
            "composition_data": {},
            "synthesis_data": {},
        }
