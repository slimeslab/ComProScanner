"""
synthesis_evaluation_crew.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 20-04-2025
"""

# Standard library imports
from typing import Dict, Optional, Any, List, Union
import json

# Third party imports
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM
from pydantic import BaseModel, Field

from .....utils.logger import setup_logger

# Logger configuration
logger = setup_logger("agent_synthesis_evaluation.log")


class SynthesisMatch(BaseModel):
    """Basic match structure with reference and test values"""

    match_value: int
    reference: Optional[Any] = None
    test: Optional[Any] = None


class ItemMatch(BaseModel):
    """Model for individual item match evaluation"""

    reference_item: str
    test_item: str
    match_value: int


class ListItemsMatch(BaseModel):
    """Model for list items match evaluation (precursors, characterization_techniques)"""

    reference: List[str] = Field(default_factory=list)
    test: List[str] = Field(default_factory=list)
    matches: List[ItemMatch] = Field(default_factory=list)
    total_ground_truth_items: int = 0
    total_match: int = 0
    missing_items: List[str] = Field(default_factory=list)
    extra_items: List[str] = Field(default_factory=list)


class StepsMatch(BaseModel):
    """Model for synthesis steps match evaluation"""

    match_value: float = 0.0  # Changed to float for value between 0 and 1
    reference_steps: List[str] = Field(default_factory=list)
    test_steps: List[str] = Field(default_factory=list)


class SynthesisDataDetails(BaseModel):
    """Synthesis data details - the only output from the evaluation task"""

    method: SynthesisMatch
    precursors: ListItemsMatch
    characterization_techniques: ListItemsMatch
    steps: StepsMatch


@CrewBase
class SynthesisEvaluationCrew:
    """
    A CrewAI crew for evaluating synthesis data using AI agent reasoning.
    This crew uses binary matching (yes/no) rather than semantic similarity or exact matching.
    """

    def __init__(self, llm: Optional[LLM] = None):
        super().__init__()
        self.llm = llm or LLM(model="o3-mini")

    @agent
    def synthesis_evaluator_agent(self) -> Agent:
        """Agent that evaluates synthesis data with binary decisions."""
        return Agent(
            config=self.agents_config["synthesis_evaluator_agent"], llm=self.llm
        )

    @task
    def evaluate_synthesis_data_task(self) -> Task:
        """Task for evaluating synthesis data with binary decisions."""
        return Task(
            config=self.tasks_config["evaluate_synthesis_data_task"],
            output_pydantic=SynthesisDataDetails,
        )

    @crew
    def crew(self) -> Crew:
        """Create and configure the synthesis evaluation crew."""
        return Crew(
            agents=[self.synthesis_evaluator_agent()],
            tasks=[self.evaluate_synthesis_data_task()],
            verbose=True,
            process=Process.sequential,
        )
