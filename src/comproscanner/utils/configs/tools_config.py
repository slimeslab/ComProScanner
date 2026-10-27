"""
tools_config.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24

Tools configuration: which extraction tools are enabled and the prompts they send.
"""

from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .yaml_config import PLACEHOLDER, check_placeholders, load_config_file

DEFAULT_TOOLS_PATH = Path(__file__).parent / "default_tools.yaml"

_FIGURE_PLACEHOLDERS = {"keyword", "caption"}


def render_prompt(template: str, **values: str) -> str:
    """Fill {name} placeholders in a tool prompt, leaving every other brace untouched.

    Args:
        template (str, required): Prompt text with placeholders.
        **values (str): Placeholder values by name.

    Returns:
        str: the prompt with the given placeholders replaced.
    """
    return PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), template)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EquationToolSettings(_Section):
    """Settings of the equation tool.

    Args:
        enabled (bool): Whether the composition extraction agent gets the tool.
        instruction (str): System prompt sent to the equation model.
        input_guidance (str): What the agent should include in the summary it passes to the tool.
        figure_caption_keywords (list): Caption keywords that select the figures attached to the request.
    """

    enabled: bool
    instruction: str
    input_guidance: str
    figure_caption_keywords: List[str]

    @field_validator("input_guidance")
    @classmethod
    def _no_placeholders(cls, value: str) -> str:
        # Filled into the CrewAI prompt, which fails on unknown {name} tokens
        return check_placeholders(value, set(), "input_guidance")


class GraphExtractorSettings(_Section):
    """Settings of the graph extractor and the figure relevance check.

    Args:
        enabled (bool): Whether the agent gets the tool and the figure relevance check runs.
        relevance_instruction (str): Prompt asking whether a figure holds data for the property.
        extraction_instruction (str): Prompt extracting the data points from a figure.
    """

    enabled: bool
    relevance_instruction: str
    extraction_instruction: str

    @field_validator("relevance_instruction", "extraction_instruction")
    @classmethod
    def _figure_placeholders(cls, value: str, info) -> str:
        return check_placeholders(value, _FIGURE_PLACEHOLDERS, info.field_name)


class RAGToolSettings(_Section):
    """Prompts of the RAG tool.

    Args:
        system_prompt (str): System message of the RAG answer model.
        answer_instruction (str): Closing line after the question and the retrieved context.
    """

    system_prompt: str
    answer_instruction: str


class ToolsConfig(_Section):
    """Tools configuration loaded from a YAML file over the bundled defaults.

    Args:
        version (str): Format version.
        equation_tool (EquationToolSettings): Equation tool settings.
        graph_extractor (GraphExtractorSettings): Graph extractor settings.
        rag_tool (RAGToolSettings): RAG tool prompts.
        user_keys (set): Dotted keys set by the user file, so a deprecated argument can tell whether the file overrides it.
    """

    version: str
    equation_tool: EquationToolSettings
    graph_extractor: GraphExtractorSettings
    rag_tool: RAGToolSettings
    user_keys: Set[str] = Field(default_factory=set, exclude=True)


def load_tools_config(tools_config_path: Optional[Union[str, Path]] = None) -> ToolsConfig:
    """Load a tools YAML file over the bundled defaults, or the defaults alone.

    Args:
        tools_config_path (str | Path, optional): Path to a tools YAML file. Defaults to None (bundled defaults).

    Returns:
        ToolsConfig: the validated configuration.
    """
    if tools_config_path is None:
        return DEFAULT_TOOLS_CONFIG
    return load_config_file(ToolsConfig, DEFAULT_TOOLS_PATH, tools_config_path)


def apply_legacy_tools_args(
    config: ToolsConfig, formula_instruction: Optional[str]
) -> Tuple[ToolsConfig, List[str]]:
    """Fold the deprecated formula_instruction argument into a tools configuration.

    Args:
        config (ToolsConfig, required): Configuration loaded from the tools file or defaults.
        formula_instruction (str, optional): Deprecated formula_instruction argument.

    Returns:
        tuple: two items:
            - ToolsConfig: the configuration to use.
            - list: deprecation messages to report.
    """
    if formula_instruction is None:
        return config, []
    if "equation_tool.instruction" in config.user_keys:
        return config, [
            "formula_instruction is deprecated and was ignored because the tools config sets equation_tool.instruction."
        ]
    config = config.model_copy(deep=True)
    config.equation_tool.instruction = formula_instruction
    return config, [
        "formula_instruction is deprecated; set equation_tool.instruction in a tools config file passed as tools_config_path."
    ]


DEFAULT_TOOLS_CONFIG = load_config_file(ToolsConfig, DEFAULT_TOOLS_PATH)
