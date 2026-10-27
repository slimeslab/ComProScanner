"""
instructions_config.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24

Instructions configuration: agent and task notes, formula notation rules and few-shot examples.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .yaml_config import check_placeholders, load_config_file

DEFAULT_INSTRUCTIONS_PATH = Path(__file__).parent / "default_instructions.yaml"

# Deprecated flow_optional_args keys and the instructions key replacing each
LEGACY_INSTRUCTION_ARGS = {
    "expected_composition_property_example": "examples.composition",
    "expected_variable_composition_property_example": "examples.variable_composition",
    "composition_property_extraction_agent_notes": "composition_extraction.agent_notes",
    "composition_property_extraction_task_notes": "composition_extraction.task_notes",
    "composition_property_formatting_agent_notes": "composition_formatting.agent_notes",
    "composition_property_formatting_task_notes": "composition_formatting.task_notes",
    "synthesis_extraction_agent_notes": "synthesis_extraction.agent_notes",
    "synthesis_extraction_task_notes": "synthesis_extraction.task_notes",
    "synthesis_formatting_agent_notes": "synthesis_formatting.agent_notes",
    "synthesis_formatting_task_notes": "synthesis_formatting.task_notes",
}


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExampleInstructions(_Section):
    """Few-shot examples for the composition agents.

    Args:
        composition (str): Example output; empty means generated from the schema.
        variable_composition (str): Example output for compositions with variables; empty means generated from the schema.
    """

    composition: str = ""
    variable_composition: str = ""

    @field_validator("composition", "variable_composition", mode="before")
    @classmethod
    def _mapping_to_json(cls, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2, ensure_ascii=False)
        return value


class CrewInstructions(_Section):
    """Notes added after a crew's built-in agent and task prompts.

    Args:
        agent_notes (list): Extra instructions for the agent.
        task_notes (list): Extra instructions for the task.
    """

    agent_notes: List[str] = Field(default_factory=list)
    task_notes: List[str] = Field(default_factory=list)

    @field_validator("agent_notes", "task_notes", mode="before")
    @classmethod
    def _none_to_empty(cls, value: Any) -> Any:
        return [] if value is None else value


class CompositionExtractionInstructions(CrewInstructions):
    """Instructions of the composition extraction crew.

    Args:
        formula_notation_rules (list): How compositions are written; replaces the built-in rules.
    """

    formula_notation_rules: List[str] = Field(default_factory=list)

    @field_validator("formula_notation_rules", mode="before")
    @classmethod
    def _rules_none_to_empty(cls, value: Any) -> Any:
        return [] if value is None else value

    @field_validator("formula_notation_rules")
    @classmethod
    def _no_placeholders(cls, value: List[str]) -> List[str]:
        # Filled into the CrewAI prompt, which fails on unknown {name} tokens
        for rule in value:
            check_placeholders(rule, set(), "formula_notation_rules")
        return value


class InstructionsConfig(_Section):
    """Instructions configuration loaded from a YAML file over the bundled defaults.

    Args:
        version (str): Format version.
        examples (ExampleInstructions): Few-shot examples.
        composition_extraction (CompositionExtractionInstructions): Composition extraction notes and rules.
        composition_formatting (CrewInstructions): Composition formatting notes.
        synthesis_extraction (CrewInstructions): Synthesis extraction notes.
        synthesis_formatting (CrewInstructions): Synthesis formatting notes.
        user_keys (set): Dotted keys set by the user file, so a deprecated argument can tell whether the file overrides it.
    """

    version: str
    examples: ExampleInstructions
    composition_extraction: CompositionExtractionInstructions
    composition_formatting: CrewInstructions
    synthesis_extraction: CrewInstructions
    synthesis_formatting: CrewInstructions
    user_keys: Set[str] = Field(default_factory=set, exclude=True)


def load_instructions_config(
    instructions_config_path: Optional[Union[str, Path]] = None,
) -> InstructionsConfig:
    """Load an instructions YAML file over the bundled defaults, or the defaults alone.

    Args:
        instructions_config_path (str | Path, optional): Path to an instructions YAML file. Defaults to None (bundled defaults).

    Returns:
        InstructionsConfig: the validated configuration.
    """
    if instructions_config_path is None:
        return DEFAULT_INSTRUCTIONS_CONFIG
    return load_config_file(InstructionsConfig, DEFAULT_INSTRUCTIONS_PATH, instructions_config_path)


def apply_legacy_instruction_args(
    config: InstructionsConfig, legacy_args: Dict[str, Any]
) -> Tuple[InstructionsConfig, List[str]]:
    """Fold deprecated flow_optional_args notes and examples into an instructions configuration.

    A key the instructions file also sets keeps the file's value.

    Args:
        config (InstructionsConfig, required): Configuration loaded from the instructions file or defaults.
        legacy_args (dict, required): Deprecated keys from LEGACY_INSTRUCTION_ARGS with their values.

    Returns:
        tuple: two items:
            - InstructionsConfig: the configuration to use.
            - list: deprecation messages to report.
    """
    if not legacy_args:
        return config, []
    raw = config.model_dump()
    ignored = []
    for arg, value in legacy_args.items():
        target = LEGACY_INSTRUCTION_ARGS[arg]
        if target in config.user_keys:
            ignored.append(arg)
            continue
        section, key = target.split(".")
        raw[section][key] = value
    moves = ", ".join(f"{arg} -> {LEGACY_INSTRUCTION_ARGS[arg]}" for arg in legacy_args)
    messages = [
        "These flow_optional_args are deprecated; move them to an instructions config file "
        f"passed as instructions_config_path: {moves}"
    ]
    if ignored:
        messages.append(
            "Ignored because the instructions config sets the same keys: " + ", ".join(ignored)
        )
    raw["user_keys"] = config.user_keys
    return InstructionsConfig.model_validate(raw), messages


DEFAULT_INSTRUCTIONS_CONFIG = load_config_file(InstructionsConfig, DEFAULT_INSTRUCTIONS_PATH)
