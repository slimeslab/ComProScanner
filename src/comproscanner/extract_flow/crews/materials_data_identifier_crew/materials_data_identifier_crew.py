"""
materials_data_identifier_crew.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24
"""

import os
from typing import Optional
from crewai import Agent, Crew, Process, Task, LLM
from pydantic import BaseModel, field_validator
from crewai.project import CrewBase, agent, crew, task
from ...tools.rag_tool import RAGTool
from ....utils.configs.llm_config import LLMConfig
from ....utils.configs.rag_config import RAGConfig
from ....utils.configs.tools_config import ToolsConfig, load_tools_config


class YesNoResponse(BaseModel):
    answer: str

    @field_validator("answer")
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        """Normalize and validate that the answer field is "yes" or "no".

        Args:
            v (str, required): Raw value provided for the answer field.

        Returns:
            str: the normalized value in title case ("Yes" or "No").

        Raises:
            ValueError: If v is not "yes" or "no" (case insensitive).
        """
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
        tools_config: Optional[ToolsConfig] = None,
    ):
        """Initialize the materials data identifier crew for a given article.

        Args:
            doi (str, required): DOI of the article to be processed.
            llm (LLM, optional): LLM instance to use. Defaults to None, which builds one from LLMConfig.
            rag_config (RAGConfig, optional): RAG configuration for the RAGTool. Defaults to None, which builds a default RAGConfig.
            output_log_folder (str, optional): Base folder for saving crew logs under {folder}/{doi}. Logs are JSON if is_log_json is True, otherwise .txt. Defaults to None.
            task_output_folder (str, optional): Base folder for saving task outputs under {folder}/{doi} as .txt files. Defaults to None.
            is_log_json (bool, optional): Whether to save logs in JSON format. Defaults to False.
            verbose (bool, optional): Verbosity flag passed to the agent and crew. Defaults to True.
            tools_config (ToolsConfig, optional): Tools configuration holding the RAG prompts. Defaults to None (bundled defaults).

        Raises:
            ValueError: If doi is not provided.
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
        self.tools_config = tools_config or load_tools_config()

        # Initialize output file paths as None
        self.output_log_file = None
        self.task_output_file = None

        if self.output_log_folder:
            final_output_log_folder = (
                f"{output_log_folder}/{self.doi.replace('/', '_').replace(':', '_')}"
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
                f"{task_output_folder}/{self.doi.replace('/', '_').replace(':', '_')}"
            )
            if not os.path.exists(final_task_output_folder):
                os.makedirs(final_task_output_folder)
            self.task_output_file = (
                f"{final_task_output_folder}/materials_data_identifier_task_output.txt"
            )

    # Agents
    @agent
    def materials_data_identifier(self) -> Agent:
        """Build the materials data identifier agent with the RAG tool.

        Returns:
            Agent: configured agent equipped with RAGTool.
        """
        rag_tool = RAGTool(
            rag_config=self.rag_config,
            system_prompt=self.tools_config.rag_tool.system_prompt,
            answer_instruction=self.tools_config.rag_tool.answer_instruction,
        )
        return Agent(
            config=self.agents_config["materials_data_identifier"],
            tools=[rag_tool],
            llm=self.llm,
            verbose=self.verbose,
        )

    # Tasks
    @task
    def identify_materials_data(self) -> Task:
        """Build the yes/no materials data identification task.

        Returns:
            Task: configured task producing a YesNoResponse, with output_file set when task_output_folder is set.
        """
        if self.task_output_folder:
            return Task(
                expected_output='A JSON object in this exact format: {{"answer": "yes"}} or {{"answer": "no"}}',
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
        """Build the sequential crew, logging to a file if configured.

        Returns:
            Crew: configured crew running the agents and tasks sequentially.
        """
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
