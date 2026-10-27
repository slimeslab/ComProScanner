"""
main_extraction_flow.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24
"""

# Standard library imports
from pathlib import Path
from typing import Dict, Optional, Union
import base64
import json
import os
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
from ..utils.configs.instructions_config import InstructionsConfig, load_instructions_config
from ..utils.configs.schema_config import (
    DEFAULT_SCHEMA_CONFIG,
    ExtractionSchemaConfig,
    load_schema_config,
)
from ..utils.configs.tools_config import ToolsConfig, load_tools_config, render_prompt
from ..utils.schema_tools import render_example_json
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
    """Shared state object passed between all crews in the extraction flow.

    Attributes:
        is_materials_mentioned (str): Crew output indicating whether materials data was found.
        composition_extracted_data (Dict): Raw composition data from the extraction crew.
        composition_formatted_data (Dict): Formatted composition data from the format crew.
        synthesis_extracted_data (Dict): Raw synthesis data from the extraction crew.
        synthesis_formatted_data (Dict): Formatted synthesis data from the format crew.
        doi (str): DOI of the article being processed.
        materials_data_identifier_query (str): Query used by the identifier crew.
        main_extraction_keyword (str): Primary property keyword driving extraction.
        composition_property_text_data (str): Article text passed to the composition crew.
        synthesis_text_data (str): Article text passed to the synthesis crew.
        is_extract_synthesis_data (bool): Whether synthesis data extraction is enabled.
        vlm_model (str): Vision LLM model name for the graph extractor tool.
        related_figures_base_path (str): Base directory for saved article figures.
        tools_config (ToolsConfig, optional): Enabled tools and their prompts.
        equation_model (str, optional): Explicit model override for the equation tool.
        llm (LLM, optional): CrewAI LLM instance shared across all crews.
        rag_config (RAGConfig, optional): RAG configuration for the RAG tool.
        output_log_folder (str, optional): Directory for crew task output JSON logs.
        task_output_folder (str, optional): Directory for crew task artefacts.
        is_log_json (bool, optional): Whether to write per-task JSON logs.
        verbose (bool, optional): Whether crews run in verbose mode.
        expected_composition_property_example (str): Few-shot example for the composition extraction task.
        expected_variable_composition_property_example (str): Few-shot variable-composition example.
        composition_property_extraction_agent_note (str): Extra instructions for the extraction agent.
        composition_property_extraction_task_note (str): Extra instructions for the extraction task.
        composition_property_formatting_agent_note (str): Extra instructions for the format agent.
        composition_property_formatting_task_note (str): Extra instructions for the format task.
        synthesis_extraction_agent_note (str): Extra instructions for the synthesis extraction agent.
        synthesis_extraction_task_note (str): Extra instructions for the synthesis extraction task.
        synthesis_formatting_agent_note (str): Extra instructions for the synthesis format agent.
        synthesis_formatting_task_note (str): Extra instructions for the synthesis format task.
        formula_notation_rules (str): Formula notation rules for the composition extraction task.
    """

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
    tools_config: Optional[ToolsConfig] = None
    equation_model: Optional[str] = None
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
    formula_notation_rules: str = ""


class DataExtractionFlow(Flow[MaterialsState]):
    """CrewAI Flow for extracting composition, property value and synthesis data from text.

    Args:
        doi (str, required): DOI of the research paper.
        main_extraction_keyword (str, required): Main property keyword to extract from the text.
        composition_property_text_data (str, required): Text data to extract composition and property data.
        synthesis_text_data (str, required): Text data to extract synthesis data.
        llm (LLM, optional): LLM instance for the agents. Defaults to None.
        materials_data_identifier_query (str, optional): Query to identify if materials data is present in the text. Must be a 'yes/no' answer. Defaults to "Is there any material chemical composition (not abbreviations) and corresponding {main_extraction_keyword} value mentioned in the paper? GIVE ONE WORD ANSWER. Either yes or no."
        is_extract_synthesis_data (bool, optional): Flag to extract synthesis data. Defaults to True.
        vlm_model (str, optional): Vision LLM model for graph extraction. Defaults to "gemini/gemini-3-flash-preview".
        related_figures_base_path (str, optional): Base path for saved figures. Defaults to "results/related_figures".
        equation_model (str, optional): Explicit litellm model name for EquationTool. Defaults to None.
        rag_config (RAGConfig, optional): RAG configuration. Defaults to None.
        output_log_folder (str, optional): Base folder path to save logs. Logs will be saved in {output_log_folder}/{doi}/ subdirectory. Logs will be in JSON format if is_log_json is True, otherwise plain text. Defaults to None (no logging).
        task_output_folder (str, optional): Base folder path to save task outputs. Task outputs will be saved as .txt files in {task_output_folder}/{doi}/ subdirectory. Defaults to None (no task output saving).
        is_log_json (bool, optional): Flag to save logs in JSON format. Defaults to False.
        verbose (bool, optional): Flag to enable verbose logging. Defaults to True.
        instructions_config (InstructionsConfig, optional): Notes, formula notation rules and examples, with deprecated arguments already folded in. Defaults to None (bundled defaults).
        tools_config (ToolsConfig, optional): Enabled tools and their prompts. Defaults to None (bundled defaults).
        schema_config (ExtractionSchemaConfig, optional): Loaded extraction schema; takes precedence over schema_config_path. Defaults to None.
        schema_config_path (str | Path, optional): Path to an extraction schema YAML file. Defaults to None (bundled default schema).

    Returns:
        dict: final result of kickoff(), with keys:
            - composition_data (dict): formatted composition and property data.
            - synthesis_data (dict): formatted synthesis data.

    Raises:
        ValueErrorHandler: If doi, main_extraction_keyword, or composition_property_text_data is not provided.
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
        equation_model: Optional[str] = None,
        rag_config: Optional[RAGConfig] = None,
        output_log_folder: Optional[str] = None,
        task_output_folder: Optional[str] = None,
        is_log_json: Optional[bool] = False,
        verbose: bool = True,
        instructions_config: Optional[InstructionsConfig] = None,
        tools_config: Optional[ToolsConfig] = None,
        schema_config: Optional[ExtractionSchemaConfig] = None,
        schema_config_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.schema_config = (
            schema_config if schema_config is not None else load_schema_config(schema_config_path)
        )
        instructions = instructions_config or load_instructions_config()
        # Pass schema to crews only when custom (crews default to the bundled one)
        self._crew_schema_kwargs = (
            {}
            if self.schema_config is DEFAULT_SCHEMA_CONFIG
            else {"schema_config": self.schema_config}
        )
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
        self.state.tools_config = tools_config or load_tools_config()
        self.state.equation_model = equation_model
        self.state.rag_config = rag_config
        self.state.output_log_folder = output_log_folder
        self.state.task_output_folder = task_output_folder
        self.state.is_log_json = is_log_json
        self.state.verbose = verbose
        main_extraction_keyword = main_extraction_keyword.replace(" ", "_")
        self.state.main_extraction_keyword = main_extraction_keyword
        self.state.materials_data_identifier_query = materials_data_identifier_query
        self.state.composition_property_text_data = composition_property_text_data
        self.state.synthesis_text_data = synthesis_text_data

        # Default examples rendered from the schema
        final_composition_fields = self.schema_config.fields(
            "composition", final_only=True
        )
        default_expected_composition_property_example = render_example_json(
            final_composition_fields
        )
        default_expected_variable_composition_property_example = render_example_json(
            final_composition_fields, variant="variable"
        )
        self.state.expected_composition_property_example = (
            instructions.examples.composition or default_expected_composition_property_example
        )
        self.state.expected_variable_composition_property_example = (
            instructions.examples.variable_composition
            or default_expected_variable_composition_property_example
        )

        def _update_notes(note: list, default_note: list = None):
            """Combine default and custom notes into a single bulleted block for the prompt.

            Args:
                note (list, required): Custom notes supplied by the caller.
                default_note (list, optional): Built-in default notes to prepend. Defaults to None.

            Returns:
                str: a "**Notes**:" block with bullet points, or "" if there are no notes.
            """
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

        # optional notes to pass to the agents and tasks
        composition_property_extraction_default_notes = [
            "The unit, given here, along with other data, is just an example. It can change depending on the property."
        ]

        self.state.composition_property_extraction_agent_note = _update_notes(
            instructions.composition_extraction.agent_notes,
            composition_property_extraction_default_notes,
        )
        self.state.composition_property_extraction_task_note = _update_notes(
            instructions.composition_extraction.task_notes
        )
        self.state.composition_property_formatting_agent_note = _update_notes(
            instructions.composition_formatting.agent_notes
        )
        self.state.composition_property_formatting_task_note = _update_notes(
            instructions.composition_formatting.task_notes,
            composition_property_extraction_default_notes,
        )

        self.state.synthesis_extraction_agent_note = _update_notes(
            instructions.synthesis_extraction.agent_notes
        )
        # Field-specific guidance comes from the schema descriptions
        self.state.synthesis_extraction_task_note = _update_notes(
            instructions.synthesis_extraction.task_notes
        )
        self.state.synthesis_formatting_agent_note = _update_notes(
            instructions.synthesis_formatting.agent_notes
        )
        self.state.synthesis_formatting_task_note = _update_notes(
            instructions.synthesis_formatting.task_notes
        )
        self.state.formula_notation_rules = "\n".join(
            f"- {rule}" for rule in instructions.composition_extraction.formula_notation_rules
        )

    def _default_synthesis_data(self) -> Dict:
        """Build an empty synthesis record used as a fallback when extraction is skipped or fails.

        Returns:
            Dict: a dict with a single "synthesis_data" key holding the schema's default shape.
        """
        return {"synthesis_data": self.schema_config.default_shape("synthesis")}

    def _parse_json_output(
        self, raw_output, default_value=None, log_prefix="", expected_keys=None
    ):
        """Parse JSON output from LLM responses, handling different output formats and nested schemas.

        Handles already-parsed dicts/lists, JSON strings wrapped in markdown code blocks, standard
        JSON strings, Python-style literals with escape sequences, and nested JSON schemas (crawls to
        find the actual data).

        Args:
            raw_output (dict | list | str, required): The raw output from the LLM.
            default_value (Any, optional): The value to return if parsing fails. Defaults to None (treated as {}).
            log_prefix (str, optional): Prefix for log messages. Defaults to "".
            expected_keys (list, optional): Keys expected in the final data, used to locate it inside a nested schema. Defaults to None.

        Returns:
            The parsed data (dict or list), or default_value if parsing fails.
        """
        if default_value is None:
            default_value = {}

        def crawl_and_extract(data, depth=0, max_depth=10):
            """Recursively search nested dicts for the keys named in expected_keys.

            Args:
                data (dict, required): The structure to search.
                depth (int, optional): Current recursion depth. Defaults to 0.
                max_depth (int, optional): Maximum recursion depth before giving up. Defaults to 10.

            Returns:
                dict | None: the matched data (full or partial), or None if nothing was found.
            """
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

            # Case 6: Extract JSON from mixed text (LLM reasoning + JSON output)
            # Handles outputs like "Thought: ... reasoning ... { actual json }"
            if parsed_output is None:
                for start_char, end_char in [('{', '}'), ('[', ']')]:
                    start_idx = cleaned_output.find(start_char)
                    end_idx = cleaned_output.rfind(end_char)
                    if start_idx != -1 and end_idx > start_idx:
                        candidate = cleaned_output[start_idx:end_idx + 1]
                        try:
                            parsed_output = json.loads(candidate)
                            logger.debug(
                                f"{log_prefix} - Extracted JSON from mixed text using brace search"
                            )
                            cleaned_output = candidate
                            break
                        except json.JSONDecodeError:
                            pass

            # Case 7: Try Python literal evaluation (handles escape sequences)
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

            # Case 8: Parsing successful, now check if we need to extract from schema
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

            # Case 8 (final fallback): All parsing methods failed
            logger.error(f"{log_prefix} - All parsing methods failed")
            logger.error(f"{log_prefix} - First 300 chars: {cleaned_output[:300]}")
            return default_value

        except Exception as e:
            logger.error(f"{log_prefix} - Unexpected error: {e}")
            logger.exception(f"{log_prefix} - Full traceback:")
            return default_value

    def _check_figures_for_data(self) -> bool:
        """Check saved figures for the DOI using VLM to identify if relevant property data is present.

        The prompt is graph_extractor.relevance_instruction from the tools config.

        Returns:
            bool: True if any figure contains relevant data, False otherwise.
        """
        doi = self.state.doi
        doi_folder = doi.replace("/", "_").replace(":", "_")
        fig_dir = os.path.join(self.state.related_figures_base_path, doi_folder)

        if not os.path.isdir(fig_dir):
            logger.debug(f"No figures directory found for DOI {doi} at {fig_dir}")
            return False

        info_path = os.path.join(fig_dir, "info.json")
        captions = {}
        if os.path.isfile(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    captions = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read info.json for {doi}: {e}")

        image_files = sorted(
            f for f in os.listdir(fig_dir) if f.lower().endswith(".jpg")
        )
        if not image_files:
            logger.debug(f"No .jpg figures found in {fig_dir}")
            return False

        try:
            import litellm
        except ImportError:
            logger.error("litellm is not installed; image data checking requires litellm.")
            return False

        keyword = self.state.main_extraction_keyword.replace("_", " ")

        for img_filename in image_files:
            caption_id = os.path.splitext(img_filename)[0]
            caption_text = captions.get(caption_id, "")
            img_path = os.path.join(fig_dir, img_filename)

            try:
                with open(img_path, "rb") as f:
                    b64_image = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                logger.warning(f"Could not read image {img_path}: {e}")
                continue

            prompt = render_prompt(
                self.state.tools_config.graph_extractor.relevance_instruction,
                keyword=keyword,
                caption=caption_text,
            )

            try:
                response = litellm.completion(
                    model=self.state.vlm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )
                answer = response.choices[0].message.content.strip().lower()
                if "yes" in answer:
                    logger.info(
                        f"Image check found {keyword} data in figure '{caption_id}' for DOI {doi}"
                    )
                    return True
            except Exception as e:
                logger.warning(
                    f"VLM image check failed for figure '{caption_id}' in {doi}: {e}"
                )

        return False

    @start()
    def identify_materials_data_presence(self):
        """Identify if there is any material and corresponding property present in the text.

        Falls back to a VLM check of saved figures if the text-based RAG check returns 'no'.

        Returns:
            str: "yes" or "no", also stored on self.state.is_materials_mentioned.
        """
        logger.debug("Starting material identification process...")
        if self.state.llm:
            rag_crew = MaterialsDataIdentifierCrew(
                doi=self.state.doi,
                llm=self.state.llm,
                rag_config=self.state.rag_config,
                tools_config=self.state.tools_config,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
            ).crew()
        else:
            rag_crew = MaterialsDataIdentifierCrew(
                doi=self.state.doi,
                rag_config=self.state.rag_config,
                tools_config=self.state.tools_config,
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

        # If text RAG returned no, also check saved figures with VLM
        if answer == "no" and self.state.tools_config.graph_extractor.enabled:
            logger.info(
                f"Text RAG check returned 'no' for {self.state.doi}, checking figures with VLM..."
            )
            if self._check_figures_for_data():
                answer = "yes"
                logger.info(
                    f"Figure image check found relevant data for {self.state.doi}, overriding to 'yes'"
                )

        self.state.is_materials_mentioned = answer
        logger.info(f"Materials identification result: '{answer}'")

        return self.state.is_materials_mentioned

    @router(identify_materials_data_presence)
    def route_process(self):
        """Route to composition extraction, or end the flow, based on materials data presence.

        Returns:
            str: "extract_compositions" if materials data was found, otherwise "end_flow".
        """
        if "yes" in self.state.is_materials_mentioned.lower():
            return "extract_compositions"
        else:
            return "end_flow"

    @listen("extract_compositions")
    def extract_composition_property_data(self):
        """Run the composition extraction crew and parse its raw output.

        Returns:
            dict: the parsed composition data, also stored on self.state.composition_extracted_data.
        """
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
                tools_config=self.state.tools_config,
                equation_model=self.state.equation_model,
                **self._crew_schema_kwargs,
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
                tools_config=self.state.tools_config,
                equation_model=self.state.equation_model,
                **self._crew_schema_kwargs,
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
                "formula_notation_rules": self.state.formula_notation_rules,
            }
        )

        self.state.composition_extracted_data = self._parse_json_output(
            result.raw, default_value={}, log_prefix="Composition extraction"
        )
        return self.state.composition_extracted_data

    @listen(extract_composition_property_data)
    def extract_final_composition_property_data(self):
        """Run the composition format crew to normalize the extracted composition data.

        Stores the parsed result on self.state.composition_formatted_data; returns nothing.
        """
        logger.debug("Extracting final composition and property data...")
        if self.state.llm:
            composition_format_crew = CompositionFormatCrew(
                doi=self.state.doi,
                llm=self.state.llm,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
                **self._crew_schema_kwargs,
            ).crew()
        else:
            composition_format_crew = CompositionFormatCrew(
                doi=self.state.doi,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
                **self._crew_schema_kwargs,
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
        parsed_composition_data = self._parse_json_output(
            result.raw,
            log_prefix="Composition formatting",
            expected_keys=self.schema_config.field_names("composition"),
        )
        self.state.composition_formatted_data = parsed_composition_data

    @listen(extract_final_composition_property_data)
    def extract_synthesis_data(self):
        """Run the synthesis extraction crew, unless synthesis text is empty or disabled.

        Returns:
            dict: the parsed (or default) synthesis data, also stored on self.state.synthesis_extracted_data.
        """
        if (
            not self.state.synthesis_text_data
            or self.state.synthesis_text_data.strip() == ""
            or self.state.is_extract_synthesis_data == False
        ):
            logger.warning(
                "Synthesis text data is empty or extraction skipped, not performing synthesis extraction"
            )
            self.state.synthesis_extracted_data = self._default_synthesis_data()
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
                **self._crew_schema_kwargs,
            ).crew()
        else:
            synthesis_extraction_crew = SynthesisExtractionCrew(
                doi=self.state.doi,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
                **self._crew_schema_kwargs,
            ).crew()

        result = synthesis_extraction_crew.kickoff(
            inputs={
                "synthesis_text_data": self.state.synthesis_text_data,
                "composition_formatted_data": self.state.composition_formatted_data,
                "synthesis_extraction_agent_note": self.state.synthesis_extraction_agent_note,
                "synthesis_extraction_task_note": self.state.synthesis_extraction_task_note,
            }
        )

        self.state.synthesis_extracted_data = self._parse_json_output(
            result.raw,
            default_value=self._default_synthesis_data(),
            log_prefix="Synthesis extraction",
        )
        return self.state.synthesis_extracted_data

    @listen(extract_synthesis_data)
    def extract_final_synthesis_data(self):
        """Run the synthesis format crew to normalize the extracted synthesis data.

        Skips the format crew (reusing the extracted data as-is) if none of the schema's synthesis
        fields were populated. Stores the result on self.state.synthesis_formatted_data; returns nothing.
        """
        if (
            isinstance(self.state.synthesis_extracted_data, dict)
            and "synthesis_data" in self.state.synthesis_extracted_data
        ):
            extracted_data = self.state.synthesis_extracted_data.get(
                "synthesis_data", {}
            )
            if not any(
                extracted_data.get(name)
                for name in self.schema_config.field_names("synthesis")
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
                **self._crew_schema_kwargs,
            ).crew()
        else:
            synthesis_format_crew = SynthesisFormatCrew(
                doi=self.state.doi,
                output_log_folder=self.state.output_log_folder,
                task_output_folder=self.state.task_output_folder,
                is_log_json=self.state.is_log_json,
                verbose=self.state.verbose,
                **self._crew_schema_kwargs,
            ).crew()

        result = synthesis_format_crew.kickoff(
            inputs={
                "extracted_synthesis_data": self.state.synthesis_extracted_data,
                "synthesis_formatting_agent_note": self.state.synthesis_formatting_agent_note,
                "synthesis_formatting_task_note": self.state.synthesis_formatting_task_note,
            }
        )

        parsed_synthesis_data = self._parse_json_output(
            result.raw,
            log_prefix="Synthesis formatting",
            expected_keys=self.schema_config.field_names("synthesis"),
        )
        self.state.synthesis_formatted_data = parsed_synthesis_data

    @listen(extract_final_synthesis_data)
    def finalize_results(self):
        """Combine the formatted composition and synthesis data into the flow's final result.

        Returns:
            dict: with keys "composition_data" and "synthesis_data".
        """
        logger.debug("Finalizing results...")
        return {
            "composition_data": self.state.composition_formatted_data,
            "synthesis_data": self.state.synthesis_formatted_data,
        }

    @listen("end_flow")
    def terminate_process(self):
        """End the flow early when no materials data was found, returning empty results.

        Returns:
            dict: with keys "composition_data" and "synthesis_data", both empty.
        """
        logger.warning("Terminating process...")
        return {
            "composition_data": {},
            "synthesis_data": {},
        }
