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

# configure logger
logger = setup_logger("comproscanner.log", module_name="main_extraction_flow")


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
    is_extract_synthesis_data: bool = True
    vlm_model: str = "gemini/gemini-3-flash-preview"
    related_figures_base_path: str = "results/related_figures"
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
        materials_data_identifier_query (str: optional): Query to identify if materials data is present in the text. Must be an 'yes/no' answer. Default: "Is there any material chemical composition (not abbreviations) and corresponding {main_extraction_keyword} value mentioned in the paper? GIVE ONE WORD ANSWER. Either yes or no."
        is_extract_synthesis_data (bool: optional): Flag to extract synthesis data. Default: True
        rag_config (RAGConfig: optional): RAG configuration. Default: None
        output_log_folder (str, optional): Base folder path to save logs. Logs will be saved in {output_log_folder}/{doi}/ subdirectory. Logs will be in JSON format if is_log_json is True, otherwise plain text. Defaults to None (no logging).
        task_output_folder (str, optional): Base folder path to save task outputs. Task outputs will be saved as .txt files in {task_output_folder}/{doi}/ subdirectory. Defaults to None (no task output saving).
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
        is_extract_synthesis_data: bool = True,
        vlm_model: str = "gemini/gemini-3-flash-preview",
        related_figures_base_path: str = "results/related_figures",
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
        self.state.is_extract_synthesis_data = is_extract_synthesis_data
        self.state.vlm_model = vlm_model
        self.state.related_figures_base_path = related_figures_base_path
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

    def _parse_json_output(
        self, raw_output, default_value=None, log_prefix="", expected_keys=None
    ):
        """
        Parses JSON output from LLM responses, handling different output formats and nested schemas.

        Handles:
        - Already parsed dictionaries/lists (returns as-is)
        - JSON strings with markdown code blocks (```json...```)
        - Standard JSON strings
        - Python-style literals with escape sequences
        - Nested JSON schemas (crawls to find actual data)

        Args:
            raw_output: The raw output from the LLM (can be dict, list, or string)
            default_value: The default value to return if parsing fails
            log_prefix: Prefix for log messages
            expected_keys: List of keys expected in the final data (for schema extraction)

        Returns:
            Parsed data or default_value if parsing fails
        """
        if default_value is None:
            default_value = {}

        def crawl_and_extract(data, depth=0, max_depth=10):
            """Recursively search for actual data in nested structures."""
            if depth > max_depth or not isinstance(data, dict):
                return None

            # If expected_keys provided, check if current level has all expected keys
            if expected_keys:
                if all(key in data for key in expected_keys):
                    logger.debug(f"{log_prefix} - Found actual data at depth {depth}")
                    return {key: data[key] for key in expected_keys if key in data}

                # Check for partial match (at least 50% of expected keys)
                found_keys = [key for key in expected_keys if key in data]
                if found_keys and len(found_keys) >= len(expected_keys) * 0.5:
                    logger.debug(
                        f"{log_prefix} - Found partial data at depth {depth} with keys: {found_keys}"
                    )
                    result = {key: data[key] for key in found_keys}
                    # Try to find remaining keys in nested structures
                    for key, value in data.items():
                        if key not in expected_keys and isinstance(value, dict):
                            nested_result = crawl_and_extract(
                                value, depth + 1, max_depth
                            )
                            if nested_result:
                                result.update(nested_result)
                    return result if len(result) > 0 else None

            # Search in common nested locations
            search_locations = [
                "properties",
                "data",
                "result",
                "output",
                "content",
                "value",
                "composition_formatted_data",
                "synthesis_formatted_data",
            ]

            for location in search_locations:
                if location in data and isinstance(data[location], dict):
                    result = crawl_and_extract(data[location], depth + 1, max_depth)
                    if result:
                        return result

            # If not found in common locations, search all nested dicts
            for key, value in data.items():
                if isinstance(value, dict):
                    result = crawl_and_extract(value, depth + 1, max_depth)
                    if result:
                        return result

            return None

        try:
            # Case 1: Already a valid dict or list
            if isinstance(raw_output, (dict, list)):
                logger.debug(
                    f"{log_prefix} - Already parsed as {type(raw_output).__name__}"
                )

                # If it's a dict and expected_keys provided, check if we need to extract from schema
                if isinstance(raw_output, dict) and expected_keys:
                    # Check if this looks like a schema (has "type", "properties", etc.)
                    if "type" in raw_output or "properties" in raw_output:
                        logger.warning(
                            f"{log_prefix} - Detected schema structure, extracting actual data"
                        )
                        extracted = crawl_and_extract(raw_output)
                        if extracted:
                            return extracted
                        else:
                            logger.error(
                                f"{log_prefix} - Failed to extract data from schema"
                            )
                            return default_value

                    # Check if data is directly accessible with expected keys
                    if all(key in raw_output for key in expected_keys):
                        return raw_output

                    # Try crawling to find the data
                    extracted = crawl_and_extract(raw_output)
                    if extracted:
                        return extracted

                return raw_output

            # Case 2: Not a string - unexpected type
            if not isinstance(raw_output, str):
                logger.warning(
                    f"{log_prefix} - Unexpected type: {type(raw_output)}, returning default"
                )
                return default_value

            # Case 3: Empty or whitespace-only string
            if not raw_output or not raw_output.strip():
                logger.warning(f"{log_prefix} - Empty string, returning default")
                return default_value

            # Clean up the string
            cleaned_output = raw_output.strip()

            # Case 4: Remove markdown code blocks (```json ... ``` or ``` ... ```)
            if "```" in cleaned_output:
                # Try to extract content from ```json...```
                json_match = re.search(
                    r"```json\s*(.*?)\s*```", cleaned_output, re.DOTALL
                )
                if json_match:
                    cleaned_output = json_match.group(1).strip()
                    logger.debug(f"{log_prefix} - Extracted from ```json block")
                else:
                    # Try to extract from generic ```...```
                    json_match = re.search(
                        r"```\s*(.*?)\s*```", cleaned_output, re.DOTALL
                    )
                    if json_match:
                        cleaned_output = json_match.group(1).strip()
                        logger.debug(f"{log_prefix} - Extracted from ``` block")

            parsed_output = None

            # Case 5: Try standard JSON parsing
            try:
                parsed_output = json.loads(cleaned_output)
                logger.debug(f"{log_prefix} - Successfully parsed with json.loads()")
            except json.JSONDecodeError as e:
                logger.debug(f"{log_prefix} - json.loads() failed: {str(e)[:100]}")

            # Case 6: Try Python literal evaluation (handles escape sequences)
            if parsed_output is None:
                try:
                    # Convert JSON booleans/null to Python equivalents
                    python_literal = (
                        cleaned_output.replace("true", "True")
                        .replace("false", "False")
                        .replace("null", "None")
                    )
                    parsed_output = ast.literal_eval(python_literal)
                    logger.debug(
                        f"{log_prefix} - Successfully parsed with ast.literal_eval()"
                    )
                except (ValueError, SyntaxError) as e:
                    logger.debug(
                        f"{log_prefix} - ast.literal_eval() failed: {str(e)[:100]}"
                    )

            # Case 7: Parsing successful, now check if we need to extract from schema
            if parsed_output is not None:
                if isinstance(parsed_output, dict) and expected_keys:
                    # Check if this looks like a schema
                    if "type" in parsed_output or "properties" in parsed_output:
                        logger.warning(
                            f"{log_prefix} - Detected schema structure, extracting actual data"
                        )
                        extracted = crawl_and_extract(parsed_output)
                        if extracted:
                            return extracted
                        else:
                            logger.error(
                                f"{log_prefix} - Failed to extract data from schema"
                            )
                            return default_value

                    # Check if data is directly accessible
                    if all(key in parsed_output for key in expected_keys):
                        return parsed_output

                    # Try crawling
                    extracted = crawl_and_extract(parsed_output)
                    if extracted:
                        return extracted

                return parsed_output

            # Case 8: All parsing methods failed
            logger.error(f"{log_prefix} - All parsing methods failed")
            logger.error(f"{log_prefix} - First 300 chars: {cleaned_output[:300]}")
            return default_value

        except Exception as e:
            logger.error(f"{log_prefix} - Unexpected error: {e}")
            logger.exception(f"{log_prefix} - Full traceback:")
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
        # Store the raw result
        raw_result = result.raw if hasattr(result, "raw") else str(result)

        logger.debug(f"Raw materials identification result: '{raw_result}'")

        # Parse the result using the centralized parser
        parsed_result = self._parse_json_output(
            raw_output=raw_result,
            log_prefix="Materials Identifier",
            expected_keys=["answer"],
            default_value={"answer": "no"},
        )

        # Extract the answer value - handle both direct string and nested dict
        answer_value = parsed_result.get("answer", "no")

        # Handle case where answer might be a dict itself
        if isinstance(answer_value, dict):
            answer = str(answer_value.get("value", answer_value.get("answer", "no")))
        elif isinstance(answer_value, str):
            answer = answer_value
        else:
            answer = str(answer_value)

        # Normalize the answer
        answer = answer.lower().strip()

        # Validate the answer
        if answer not in ["yes", "no"]:
            logger.warning(f"Invalid answer value: '{answer}', defaulting to 'no'")
            answer = "no"

        self.state.is_materials_mentioned = answer
        logger.info(f"Materials identification result: '{answer}'")

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
                vlm_model=self.state.vlm_model,
                related_figures_base_path=self.state.related_figures_base_path,
                main_extraction_keyword=self.state.main_extraction_keyword,
            ).crew()
        else:
            composition_property_crew = CompositionExtractionCrew(
                doi=self.state.doi,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
                vlm_model=self.state.vlm_model,
                related_figures_base_path=self.state.related_figures_base_path,
                main_extraction_keyword=self.state.main_extraction_keyword,
            ).crew()

        result = composition_property_crew.kickoff(
            inputs={
                "doi": self.state.doi,
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
        composition_expected_keys = [
            "compositions_property_values",
            "property_unit",
            "family",
        ]
        parsed_composition_data = self._parse_json_output(
            result.raw,
            log_prefix="Composition formatting",
            expected_keys=composition_expected_keys,
        )
        self.state.composition_formatted_data = parsed_composition_data

    @listen(extract_final_composition_property_data)
    def extract_synthesis_data(self):
        """Extract synthesis data"""
        if (
            not self.state.synthesis_text_data
            or self.state.synthesis_text_data.strip() == ""
            or self.state.is_extract_synthesis_data == False
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

        parsed_synthesis_data = self._parse_json_output(
            result.raw,
            log_prefix="Synthesis formatting",
            expected_keys=[
                "method",
                "precursors",
                "steps",
                "characterization_techniques",
            ],
        )
        self.state.synthesis_formatted_data = parsed_synthesis_data

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
