"""
materials_data_identifier_crew.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 21-03-2025
"""

import os
from typing import Optional
from crewai import Agent, Crew, Process, Task, LLM
from pydantic import BaseModel, field_validator
from crewai.project import CrewBase, agent, crew, task
from ...tools.rag_tool import RAGTool
from ....utils.configs.llm_config import LLMConfig
from ....utils.configs.rag_config import RAGConfig


class YesNoResponse(BaseModel):
    answer: str

    @field_validator("answer")
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        normalized = v.lower().strip()
        if normalized not in ("yes", "no"):
            raise ValueError('Value must be either "yes" or "no" (case insensitive)')
        # Return the normalized version in title case
        return normalized.title()


@CrewBase
class MaterialsDataIdentifierCrew:
    """Materials Data Identifier Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(
        self,
        doi: str = None,
        llm: Optional[LLM] = None,
        rag_config: Optional[RAGConfig] = None,
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
        - rag_config: Optional RAGConfig instance. If not provided, will use default configuration
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
        self.rag_config = rag_config or RAGConfig()
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
                    f"{final_output_log_folder}/materials_data_identifier_log.json"
                )
            else:
                self.output_log_file = (
                    f"{final_output_log_folder}/materials_data_identifier_log.txt"
                )
        if self.task_output_folder:
            final_task_output_folder = (
                f"{task_output_folder}/{self.doi.replace('/', '_')}"
            )
            if not os.path.exists(final_task_output_folder):
                os.makedirs(final_task_output_folder)
            self.task_output_file = (
                f"{final_task_output_folder}/materials_data_identifier_task_output.txt"
            )

    # Agents
    @agent
    def materials_data_identifier(self) -> Agent:
        rag_tool = RAGTool(rag_config=self.rag_config)
        return Agent(
            config=self.agents_config["materials_data_identifier"],
            tools=[rag_tool],
            llm=self.llm,
            verbose=self.verbose,
        )

    # Tasks
    @task
    def identify_materials_data(self) -> Task:
        if self.task_output_folder:
            return Task(
                config=self.tasks_config["identify_materials_data"],
                output_pydantic=YesNoResponse,
                output_file=self.task_output_file,
            )
        else:
            return Task(
                config=self.tasks_config["identify_materials_data"],
                output_pydantic=YesNoResponse,
            )

    # Crew
    @crew
    def crew(self) -> Crew:
        """Creates the RAG Crew"""
        if self.output_log_folder:
            return Crew(
                agents=self.agents,
                tasks=self.tasks,
                process=Process.sequential,
                verbose=self.verbose,
                output_log_folder=self.output_log_file,
            )
        else:
            return Crew(
                agents=self.agents,
                tasks=self.tasks,
                process=Process.sequential,
                verbose=self.verbose,
            )
