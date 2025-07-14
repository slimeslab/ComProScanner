"""
synthesis_format_crew.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 27-03-2025
"""

import os
from typing import Optional, List
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel
from .....utils.configs.llm_config import LLMConfig


class DetailedSynthesisData(BaseModel):
    """Detailed Synthesis Data"""

    method: str = ""
    precursors: List[str] = []
    steps: List[str] = []
    characterization_techniques: List[str] = []


class FormattedSynthesisData(BaseModel):
    """Formatted Synthesis Data"""

    synthesis_formatted_data: DetailedSynthesisData = DetailedSynthesisData()


@CrewBase
class SynthesisFormatCrew:
    """Synthesis Format Crew"""

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
        Initialize the MaterialsDataIdentifierCrew.

        Parameters:
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
                    f"{final_output_log_folder}/synthesis_formatter_log.json"
                )
            else:
                self.output_log_file = (
                    f"{final_output_log_folder}/synthesis_formatter_log.txt"
                )
        if self.task_output_folder:
            final_task_output_folder = (
                f"{task_output_folder}/{self.doi.replace('/', '_')}"
            )
            if not os.path.exists(final_task_output_folder):
                os.makedirs(final_task_output_folder)
            self.task_output_file = (
                f"{final_task_output_folder}/synthesis_formatter_task_output.txt"
            )

    # Agents
    @agent
    def synthesis_formatter(self) -> Agent:
        return Agent(
            config=self.agents_config["synthesis_formatter"],
            llm=self.llm,
            verbose=self.verbose,
        )

    # Tasks
    @task
    def format_synthesis_data(self) -> Task:
        if self.task_output_file:
            return Task(
                config=self.tasks_config["format_synthesis_data"],
                output_pydantic=FormattedSynthesisData,
                output_file=self.task_output_file,
            )
        else:
            return Task(
                config=self.tasks_config["format_synthesis_data"],
                output_pydantic=FormattedSynthesisData,
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
