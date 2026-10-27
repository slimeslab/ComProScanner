"""
equation_tool.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24
"""

# Standard library imports
import base64
import json
import os
from typing import List, Optional, Type

# Third-party imports
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Local imports
from ...utils.configs.tools_config import DEFAULT_TOOLS_CONFIG
from ...utils.logger import setup_logger

logger = setup_logger("comproscanner.log", module_name="equation_tool")

# Built-in instruction, kept as a module constant for callers that import it
FORMULA_INSTRUCTION = DEFAULT_TOOLS_CONFIG.equation_tool.instruction

# Priority-ordered (env_var, litellm_model) pairs for fallback LLM selection
_API_KEY_MODEL_PAIRS = [
    ("ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-6"),
    ("GEMINI_API_KEY", "gemini/gemini-3-flash-preview"),
    ("OPENAI_API_KEY", "openai/gpt-5.4-mini"),
    ("DEEPSEEK_API_KEY", "deepseek/deepseek-v4-flash"),
    ("OPENROUTER_API_KEY", "openrouter/google/gemini-2.0-flash"),
    ("TOGETHER_API_KEY", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("COHERE_API_KEY", "cohere/command-r-plus"),
    (
        "FIREWORKS_API_KEY",
        "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct",
    ),
]


class EquationToolInput(BaseModel):
    """Input schema for EquationTool."""

    doi: str = Field(
        ...,
        description="The DOI of the article to analyse.",
    )
    paper_text: str = Field(
        ...,
        description=(
            "The full or relevant text content of the paper used to determine "
            "phase formation and derive the doped chemical formula."
        ),
    )


class EquationTool(BaseTool):
    """
    LLM-powered tool that analyses paper text (and saved figures whose captions match figure_caption_keywords, such as XRD, SEM or TEM) to determine whether doping produces a single-phase compound and, if so, to derive the general doped chemical formula.

    The instruction and caption keywords come from the tools config. The tool selects its LLM automatically based on which API key is present in the environment: ANTHROPIC_API_KEY (claude-sonnet-4-6, default), then GEMINI_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY, TOGETHER_API_KEY, COHERE_API_KEY, FIREWORKS_API_KEY.
    """

    name: str = "Equation Tool"
    description: str = (
        "Analyses paper text and crystal structure figures (XRD, SEM, TEM) to "
        "determine whether the synthesised material is a single-phase compound "
        "and derive the general doped chemical formula. "
        "Call this tool BEFORE the Graph Data Extractor. "
        "Pass the article DOI and a concise, equation-relevant evidence summary "
        "as text input (avoid sending the full paper text)."
    )
    args_schema: Type[BaseModel] = EquationToolInput

    formula_instruction: str = Field(default=FORMULA_INSTRUCTION)
    figure_caption_keywords: List[str] = list(DEFAULT_TOOLS_CONFIG.equation_tool.figure_caption_keywords)
    equation_model: Optional[str] = None
    related_figures_base_path: str = "results/related_figures"

    def _has_model_configuration(self) -> bool:
        """Return True if any EquationTool model source is configured.

        Returns:
            bool: True if `equation_model` is set or any supported provider API key is present.
        """
        if self.equation_model:
            return True
        return any(os.getenv(env_var) for env_var, _ in _API_KEY_MODEL_PAIRS)

    def _select_model(self) -> str:
        """Return the litellm model string to use, preferring the explicit override.

        Returns:
            str: litellm model identifier (e.g. "anthropic/claude-sonnet-4-6").
        """
        if self.equation_model:
            return self.equation_model

        for env_var, model in _API_KEY_MODEL_PAIRS:
            if os.getenv(env_var):
                return model
        # Final fallback — caller must have ANTHROPIC_API_KEY set
        return "anthropic/claude-sonnet-4-6"

    def _get_crystal_structure_images(self, doi: str) -> list:
        """Return base64-encoded crystal structure images (XRD/SEM/TEM) for the DOI.

        Args:
            doi (str, required): Article DOI used to locate the figure directory.

        Returns:
            list: List of dicts with keys `"caption"` (str) and `"b64"` (str, base64 JPEG).
                  Empty list if the figure directory does not exist or no matching images are found.
        """
        doi_folder = doi.replace("/", "_").replace(":", "_")
        fig_dir = os.path.join(self.related_figures_base_path, doi_folder)

        if not os.path.isdir(fig_dir):
            return []

        info_path = os.path.join(fig_dir, "info.json")
        captions: dict = {}
        if os.path.isfile(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    captions = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read info.json for {doi}: {e}")

        images = []
        for img_filename in sorted(os.listdir(fig_dir)):
            if not img_filename.lower().endswith(".jpg"):
                continue
            caption_id = os.path.splitext(img_filename)[0]
            caption_lower = captions.get(caption_id, "").lower()
            if not any(kw.lower() in caption_lower for kw in self.figure_caption_keywords):
                continue
            try:
                with open(os.path.join(fig_dir, img_filename), "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                images.append(
                    {
                        "caption": captions.get(caption_id, ""),
                        "b64": b64,
                    }
                )
            except Exception as e:
                logger.warning(f"Could not read image {img_filename} for {doi}: {e}")

        return images

    def _run(self, doi: str, paper_text: str) -> str:
        """Analyse the paper (and any saved crystal structure figures) and derive the doped formula.

        Args:
            doi (str, required): DOI of the article to analyse.
            paper_text (str, required): Paper text used to determine phase formation and derive the formula.

        Returns:
            str: the derived chemical formula, the literal "single compound is not being synthesized",
                or an error/status message if litellm is missing, no model is configured, or the LLM call fails.
        """
        try:
            import litellm
        except ImportError:
            msg = "litellm is not installed; EquationTool requires litellm."
            logger.error(msg)
            return msg

        if not self._has_model_configuration():
            msg = (
                "No EquationTool model/provider is configured. "
                "Set one of: equation_model argument, "
                "or provider API key (ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, "
                "DEEPSEEK_API_KEY, OPENROUTER_API_KEY, TOGETHER_API_KEY, COHERE_API_KEY, "
                "FIREWORKS_API_KEY)."
            )
            logger.error(msg)
            return msg

        model = self._select_model()
        images = self._get_crystal_structure_images(doi)

        content = [
            {"type": "text", "text": self.formula_instruction},
            {"type": "text", "text": f"\n\n---\n\nPAPER TEXT:\n{paper_text}"},
        ]

        for img in images:
            content.append(
                {
                    "type": "text",
                    "text": f'\nCrystal structure figure (caption: "{img["caption"]}"):',
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img['b64']}"},
                }
            )

        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
            )
            result = response.choices[0].message.content.strip()
            logger.info(f"EquationTool result for DOI {doi}: {result[:120]}")
            return result
        except Exception as e:
            logger.error(f"EquationTool LLM call failed for DOI {doi}: {e}")
            return f"Error: {str(e)}"
