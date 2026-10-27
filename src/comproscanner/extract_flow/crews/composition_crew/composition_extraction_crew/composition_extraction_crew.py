"""
composition_extraction_crew.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24
"""

import os
from pathlib import Path
from typing import Dict, Optional

import yaml
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from .....utils.configs.llm_config import LLMConfig
from .....utils.configs.schema_config import ExtractionSchemaConfig, load_schema_config
from .....utils.configs.tools_config import ToolsConfig, load_tools_config
from .....utils.schema_tools import (
    build_output_model,
    fill_schema_placeholders,
    schema_prompt_values,
)
from ....tools.graph_extractor_tool import GraphExtractorTool
from ....tools.equation_tool import EquationTool

with open(Path(__file__).parent / "config" / "tool_guidance.yaml", "r", encoding="utf-8") as _f:
    TOOL_GUIDANCE = yaml.safe_load(_f)


def tool_guidance_values(tools_config: ToolsConfig) -> Dict[str, str]:
    """Prompt lines for each enabled tool, keyed by their agents.yaml/tasks.yaml placeholder.

    Args:
        tools_config (ToolsConfig, required): Tools configuration with the enabled switches and the equation tool input guidance.

    Returns:
        dict: the four guidance placeholders, empty for a disabled tool.
    """
    settings = {"equation_tool": tools_config.equation_tool, "graph_extractor": tools_config.graph_extractor}
    values = {}
    for tool, tool_settings in settings.items():
        for target in ("agent", "task"):
            text = TOOL_GUIDANCE[tool][target] if tool_settings.enabled else ""
            text = text.replace("{equation_tool_input_guidance}", tools_config.equation_tool.input_guidance)
            values[f"{tool}_{target}_guidance"] = text
    return values


@CrewBase
class CompositionExtractionCrew:
    """Composition Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(
        self,
        doi: str = None,
        llm: Optional[LLM] = None,
        output_log_folder: Optional[str] = None,
        task_output_folder: Optional[str] = None,
        is_log_json: bool = False,
        verbose: Optional[bool] = True,
        vlm_model: str = "gemini/gemini-3-flash-preview",
        related_figures_base_path: str = "results/related_figures",
        main_extraction_keyword: str = "",
        equation_model: str = None,
        schema_config: Optional[ExtractionSchemaConfig] = None,
        tools_config: Optional[ToolsConfig] = None,
    ):
        """Initialize the composition extraction crew for a given article.

        Args:
            doi (str, required): DOI of the article to be processed.
            llm (LLM, optional): LLM instance to use. Defaults to None, which builds one from LLMConfig.
            output_log_folder (str, optional): Base folder for saving crew logs under {folder}/{doi}. Logs are JSON if is_log_json is True, otherwise .txt. Defaults to None.
            task_output_folder (str, optional): Base folder for saving task outputs under {folder}/{doi} as .txt files. Defaults to None.
            is_log_json (bool, optional): Whether to save logs in JSON format. Defaults to False.
            verbose (bool, optional): Verbosity flag passed to the agent and crew. Defaults to True.
            vlm_model (str, optional): Vision LLM model for graph extraction. Defaults to "gemini/gemini-3-flash-preview".
            related_figures_base_path (str, optional): Base path for saved figures. Defaults to "results/related_figures".
            main_extraction_keyword (str, optional): Property keyword used to label axes in the VLM extraction prompt. Defaults to "".
            equation_model (str, optional): Explicit litellm model name for EquationTool. Defaults to None (auto-select).
            schema_config (ExtractionSchemaConfig, optional): Extraction schema. Defaults to the bundled default schema.
            tools_config (ToolsConfig, optional): Tools configuration: enabled tools and their prompts. Defaults to None (bundled defaults).

        Raises:
            ValueError: If doi is not provided.
        """
        if doi is None:
            raise ValueError("DOI must be provided")
        self.doi = doi
        # Use provided LLM or create default one
        self.llm = llm if llm is not None else LLMConfig().get_llm()
        self.output_log_folder = output_log_folder
        self.task_output_folder = task_output_folder
        self.is_log_json = is_log_json
        self.verbose = verbose
        self.vlm_model = vlm_model
        self.related_figures_base_path = related_figures_base_path
        self.main_extraction_keyword = main_extraction_keyword
        self.equation_model = equation_model
        self.tools_config = tools_config or load_tools_config()
        self.schema_config = schema_config or load_schema_config()
        fields = self.schema_config.fields("composition")
        self.output_model = build_output_model(
            fields,
            wrapper_key="composition_extracted_data",
            model_name="ExtractedCompositionPropertyData",
            inner_model_name="DetailedCompositionPropertyData",
            model_doc="Extracted Composition Property",
            inner_model_doc="Detailed Composition Property",
        )
        self.prompt_values = {
            **schema_prompt_values(fields, stage="extraction"),
            **tool_guidance_values(self.tools_config),
        }

        # Initialize output file paths as None
        self.output_log_file = None
        self.task_output_file = None

        final_task_output_folder = f"{task_output_folder}/{self.doi.replace('/', '_').replace(':', '_')}"
        if self.output_log_folder:
            final_output_log_folder = (
                f"{output_log_folder}/{self.doi.replace('/', '_').replace(':', '_')}"
            )
            if not os.path.exists(final_output_log_folder):
                os.makedirs(final_output_log_folder)
            if self.is_log_json:
                self.output_log_file = (
                    f"{final_output_log_folder}/composition_property_extractor_log.json"
                )
            else:
                self.output_log_file = (
                    f"{final_output_log_folder}/composition_property_extractor_log.txt"
                )
        if self.task_output_folder:
            final_task_output_folder = (
                f"{task_output_folder}/{self.doi.replace('/', '_').replace(':', '_')}"
            )
            if not os.path.exists(final_task_output_folder):
                os.makedirs(final_task_output_folder)
            self.task_output_file = f"{final_task_output_folder}/composition_property_extractor_task_output.txt"

    # Agents
    @agent
    def composition_property_extractor(self) -> Agent:
        """Build the composition extraction agent with the tools enabled in the tools config.

        Returns:
            Agent: configured agent equipped with EquationTool and GraphExtractorTool unless disabled.
        """
        tools = []
        equation_settings = self.tools_config.equation_tool
        if equation_settings.enabled:
            equation_tool_kwargs = {
                "related_figures_base_path": self.related_figures_base_path,
                "formula_instruction": equation_settings.instruction,
                "figure_caption_keywords": equation_settings.figure_caption_keywords,
            }
            if self.equation_model is not None:
                equation_tool_kwargs["equation_model"] = self.equation_model
            tools.append(EquationTool(**equation_tool_kwargs))
        if self.tools_config.graph_extractor.enabled:
            tools.append(
                GraphExtractorTool(
                    vlm_model=self.vlm_model,
                    related_figures_base_path=self.related_figures_base_path,
                    vlm_property_name=self.main_extraction_keyword,
                    extraction_instruction=self.tools_config.graph_extractor.extraction_instruction,
                )
            )
        return Agent(
            config=fill_schema_placeholders(
                self.agents_config["composition_property_extractor"], self.prompt_values
            ),
            tools=tools,
            llm=self.llm,
            verbose=self.verbose,
        )

    # Tasks
    @task
    def extract_composition_property(self) -> Task:
        """Build the composition extraction task, writing to a file if configured.

        Returns:
            Task: configured task producing output_model, with output_file set when task_output_file is set.
        """
        if self.task_output_file:
            return Task(
                config=fill_schema_placeholders(
                    self.tasks_config["extract_composition_property"], self.prompt_values
                ),
                output_pydantic=self.output_model,
                output_file=self.task_output_file,
            )
        else:
            return Task(
                config=fill_schema_placeholders(
                    self.tasks_config["extract_composition_property"], self.prompt_values
                ),
                output_pydantic=self.output_model,
            )

    # Crew
    @crew
    def crew(self) -> Crew:
        """Build the sequential crew, logging to a file if configured.

        Returns:
            Crew: configured crew running the agents and tasks sequentially.
        """
        if self.output_log_file:
            return Crew(
                agents=self.agents,
                tasks=self.tasks,
                process=Process.sequential,
                verbose=self.verbose,
                output_log_file=self.output_log_file,
            )
        else:
            return Crew(
                agents=self.agents,
                tasks=self.tasks,
                process=Process.sequential,
                verbose=self.verbose,
            )
