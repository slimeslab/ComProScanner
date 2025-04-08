"""
composition_extraction_crew.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 23-03-2025
"""

import os
from typing import List, Dict, Union, Optional
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel
from .....utils.configs.llm_config import LLMConfig


class DetailedCompositionPropertyData(BaseModel):
    """Detailed Composition Property"""

    compositions_property_values: Dict[str, Union[int, float, None]] = {}
    abbreviations: Dict[str, str] = []
    property_unit: str = ""
    family: str = ""


class ExtractedCompositionPropertyData(BaseModel):
    """Extracted Composition Property"""

    composition_extracted_data: DetailedCompositionPropertyData = (
        DetailedCompositionPropertyData()
    )


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
    ):
        """
        Initialize the MaterialsDataIdentifierCrew.

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

        final_task_output_folder = f"{task_output_folder}/{self.doi.replace('/', '_')}"
        if self.output_log_folder:
            final_output_log_folder = (
                f"{output_log_folder}/{self.doi.replace('/', '_')}"
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
                f"{task_output_folder}/{self.doi.replace('/', '_')}"
            )
            if not os.path.exists(final_task_output_folder):
                os.makedirs(final_task_output_folder)
            self.task_output_file = f"{final_task_output_folder}/composition_property_extractor_task_output.txt"

    # Agents
    @agent
    def composition_property_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["composition_property_extractor"],
            llm=self.llm,
            verbose=self.verbose,
        )

    # Tasks
    @task
    def extract_composition_property(self) -> Task:
        if self.task_output_file:
            return Task(
                config=self.tasks_config["extract_composition_property"],
                output_pydantic=ExtractedCompositionPropertyData,
                output_file=self.task_output_file,
            )
        else:
            return Task(
                config=self.tasks_config["extract_composition_property"],
                output_pydantic=ExtractedCompositionPropertyData,
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
