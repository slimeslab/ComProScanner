"""
composition_evaluation_crew.py

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
logger = setup_logger("agent_composition_evaluation.log")


class CompositionMatch(BaseModel):
    """Basic match structure with reference and test values"""

    match_value: int
    reference: Optional[Any] = None
    test: Optional[Any] = None


class KeyMatch(BaseModel):
    """Model for key match evaluation"""

    reference_key: str
    test_key: str
    match_value: int


class ValueMatch(BaseModel):
    """Model for value match evaluation"""

    reference_key: str
    test_key: str
    reference_value: Any
    test_value: Any
    match_value: int


class PairMatch(BaseModel):
    """Model for key-value pair match evaluation"""

    reference_pair: Dict[str, Union[int, float]] = Field(default_factory=dict)
    test_pair: Dict[str, Union[int, float]] = Field(default_factory=dict)
    match_value: int


class CompositionsPropertyValuesMatch(BaseModel):
    """Model for compositions_property_values match evaluation"""

    reference: Dict[str, Any] = Field(default_factory=dict)
    test: Dict[str, Any] = Field(default_factory=dict)
    key_matches: List[KeyMatch] = Field(default_factory=list)
    value_matches: List[ValueMatch] = Field(default_factory=list)
    pair_matches: List[PairMatch] = Field(default_factory=list)
    total_ground_truth_keys: int = 0
    total_match: int = 0
    missing_keys: List[str] = Field(default_factory=list)
    extra_keys: List[str] = Field(default_factory=list)


class CompositionDataDetails(BaseModel):
    """Composition data details - the only output from the evaluation task"""

    property_unit: CompositionMatch
    family: CompositionMatch
    compositions_property_values: CompositionsPropertyValuesMatch


@CrewBase
class CompositionEvaluationCrew:
    """
    A CrewAI crew for evaluating composition data using AI agent reasoning.
    This crew uses binary matching (yes/no) rather than semantic similarity or exact matching.
    """

    def __init__(self, llm: Optional[LLM] = None):
        super().__init__()
        self.llm = llm or LLM(model="o3-mini")

    @agent
    def composition_evaluator_agent(self) -> Agent:
        """Agent that evaluates composition data with binary decisions."""
        return Agent(
            config=self.agents_config["composition_evaluator_agent"], llm=self.llm
        )

    @task
    def evaluate_composition_data_task(self) -> Task:
        """Task for evaluating composition data with binary decisions."""
        return Task(
            config=self.tasks_config["evaluate_composition_data_task"],
            output_pydantic=CompositionDataDetails,
        )

    @crew
    def crew(self) -> Crew:
        """Create and configure the composition evaluation crew."""
        return Crew(
            agents=[self.composition_evaluator_agent()],
            tasks=[self.evaluate_composition_data_task()],
            verbose=True,
            process=Process.sequential,
        )
