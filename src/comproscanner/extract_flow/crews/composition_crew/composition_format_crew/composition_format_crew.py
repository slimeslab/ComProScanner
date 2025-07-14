"""
composition_format_crew.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 24-03-2025
"""

import os
from typing import Dict, Union, Optional
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel
from .....utils.configs.llm_config import LLMConfig
from ....tools.material_parser_tool import MaterialParserTool


class DetailedCompositionPropertyData(BaseModel):
    """Detailed Composition Property"""

    compositions_property_values: Dict[str, Union[int, float, None]] = {}
    property_unit: str = ""
    family: str = ""


class FormattedCompositionPropertyData(BaseModel):
    """Formatted Composition Property"""

    composition_formatted_data: DetailedCompositionPropertyData = (
        DetailedCompositionPropertyData()
    )


@CrewBase
class CompositionFormatCrew:
    """Composition Format Crew"""

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
    ):
        """
        Initialize the CompositionFormatCrew.

        Parameters:
        - doi (str): DOI of the article to be processed. This is a required parameter.
        - llm: Optional LLM instance. If not provided, will use default configuration
        - output_log_folder (str, optional): Set to True to save logs inside {provided folder}/{doi} folder as .txt files. Logs will be in JSON format if the is_log_json is True, otherwise .txt. Defaults to None.
        - task_output_folder (str, optional): Folder path for storing the task outputs as .txt files inside {provided foler}/{doi} folder. Defaults to None.
        - is_log_json (bool, optional): Flag to save logs in JSON format. Defaults to False.
        - verbose: Optional boolean for verbosity. Default is True.
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

        # Initialize output file paths as None
        self.output_log_file = None
        self.task_output_file = None

        if self.output_log_folder:
            final_output_log_folder = (
                f"{output_log_folder}/{self.doi.replace('/', '_')}"
            )
            if not os.path.exists(final_output_log_folder):
                os.makedirs(final_output_log_folder)
            if self.is_log_json:
                self.output_log_file = (
                    f"{final_output_log_folder}/composition_property_formatter_log.json"
                )
            else:
                self.output_log_file = (
                    f"{final_output_log_folder}/composition_property_formatter_log.txt"
                )
        if self.task_output_folder:
            final_task_output_folder = (
                f"{task_output_folder}/{self.doi.replace('/', '_')}"
            )
            if not os.path.exists(final_task_output_folder):
                os.makedirs(final_task_output_folder)
            self.task_output_file = f"{final_task_output_folder}/composition_property_formatter_task_output.txt"

    # Agents
    @agent
    def composition_property_formatter(self) -> Agent:
        material_parser_tool = MaterialParserTool()
        return Agent(
            config=self.agents_config["composition_property_formatter"],
            tools=[material_parser_tool],
            llm=self.llm,
            verbose=self.verbose,
        )

    # Tasks
    @task
    def format_composition_property(self) -> Task:
        if self.task_output_file:
            return Task(
                config=self.tasks_config["format_composition_property"],
                output_pydantic=FormattedCompositionPropertyData,
                output_file=self.task_output_file,
            )
        else:
            return Task(
                config=self.tasks_config["format_composition_property"],
                output_pydantic=FormattedCompositionPropertyData,
            )

    # Crew
    @crew
    def crew(self) -> Crew:
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
