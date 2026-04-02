"""
composition_evaluation_crew.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 20-04-2025
"""

# Standard library imports
from typing import Dict, Optional, Any, List, Union, Tuple, Type

# Third party imports
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ThresholdToolInput(BaseModel):
    """Input schema for GetValueErrorThresholdTool."""

    reference_value: str = Field(
        ...,
        description="The ground-truth numeric property value as a string, e.g. '150' or '-300.5'.",
    )


class GetValueErrorThresholdTool(BaseTool):
    """
    Returns the allowed absolute error tolerance for a numeric ground-truth property value.

    When an evaluator has been configured with a ``value_error_thresholds`` mapping, this
    tool lets the agent look up how much the extracted value is allowed to differ from the
    ground-truth value before the comparison is counted as a mismatch.

    The returned string is one of:
    - ``"threshold:<N>"`` — the extracted value matches if |extracted - reference| <= N.
    - ``"exact"`` — no tolerance configured; use exact matching (within 1e-6).
    """

    name: str = "get_value_error_threshold"
    description: str = (
        "Look up the allowed absolute error tolerance for a numeric ground-truth property value. "
        "Call this tool with the ground-truth (reference) value before deciding whether a "
        "numeric property value extracted from the test data is a match. "
        "If the tool returns 'threshold:<N>', the test value is accepted if "
        "|test_value - reference_value| <= N. If the tool returns 'exact', require exact equality."
    )
    args_schema: Type[BaseModel] = ThresholdToolInput

    # Each element is (lo, hi, threshold) with lo <= hi
    thresholds_list: List[Tuple[float, float, float]] = Field(default_factory=list)

    def _run(self, reference_value: str) -> str:
        try:
            ref_num = float(reference_value)
        except (ValueError, TypeError):
            return "exact: reference value is not numeric"

        for lo, hi, threshold in self.thresholds_list:
            if lo <= ref_num <= hi:
                return f"threshold:{threshold}"

        return "exact"


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
    """Composition data details"""

    property_unit: CompositionMatch
    family: CompositionMatch
    compositions_property_values: CompositionsPropertyValuesMatch


class CompositionDataWrapper(BaseModel):
    """Wrapper for composition data - the output from the evaluation task"""

    composition_data: CompositionDataDetails


@CrewBase
class CompositionEvaluationCrew:
    """
    A CrewAI crew for evaluating composition data using AI agent reasoning.
    This crew uses binary matching (yes/no) rather than semantic similarity or exact matching.
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        value_error_thresholds: Optional[Dict] = None,
    ):
        """
        Args:
            llm: LLM instance for the agent.
            value_error_thresholds: Mapping of ``(min, max)`` tuples to absolute error
                tolerances for numeric property-value comparisons.  When provided, the
                ``get_value_error_threshold`` tool is added to the evaluator agent so that
                it can look up the tolerance before deciding on value matches.  Example::

                    {
                        (-200, 200): 5,
                        (201, 500): 8,
                        (-500, -201): 8,
                        (501, float('inf')): 10,
                        (float('-inf'), -501): 10,
                    }
        """
        self.llm = llm or LLM(model="o3-mini")
        # Convert the dict to an internal list of (lo, hi, threshold) triples
        self._thresholds_list: List[Tuple[float, float, float]] = []
        if value_error_thresholds:
            for range_key, threshold in value_error_thresholds.items():
                lo = min(range_key)
                hi = max(range_key)
                self._thresholds_list.append((lo, hi, float(threshold)))

    @agent
    def composition_evaluator_agent(self) -> Agent:
        """Agent that evaluates composition data with binary decisions."""
        tools = []
        if self._thresholds_list:
            tools = [
                GetValueErrorThresholdTool(thresholds_list=self._thresholds_list)
            ]
        return Agent(
            config=self.agents_config["composition_evaluator_agent"],
            llm=self.llm,
            tools=tools,
        )

    @task
    def evaluate_composition_data_task(self) -> Task:
        """Task for evaluating composition data with binary decisions."""
        return Task(
            config=self.tasks_config["evaluate_composition_data_task"],
            output_pydantic=CompositionDataWrapper,
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
