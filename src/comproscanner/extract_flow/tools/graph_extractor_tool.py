"""
graph_extractor_tool.py - VLM-based graph data extraction tool for CrewAI agents.

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24
"""

# Standard library imports
import os
import json
import base64
from typing import Type, Dict

# Third-party imports
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Local imports
from ...utils.configs.tools_config import DEFAULT_TOOLS_CONFIG, render_prompt
from ...utils.logger import setup_logger

# configure logger
logger = setup_logger("comproscanner.log", module_name="graph_extractor_tool")


class GraphExtractorToolInput(BaseModel):
    """Input schema for GraphExtractorTool."""

    doi: str = Field(
        ...,
        description=(
            "The DOI of the article whose saved figures should be analysed. "
            "Figures must have been saved during article processing."
        ),
    )


class GraphExtractorTool(BaseTool):
    """
    VLM-based tool that scans saved figures for a given article DOI and extracts
    quantitative composition-property data from graphs/charts using a vision LLM.

    Figures are expected at: {related_figures_base_path}/{doi_}/{caption_id}.jpg
    Captions are read from: {related_figures_base_path}/{doi_}/info.json
    The prompt is extraction_instruction from the tools config, with {keyword} and {caption} filled in.
    """

    name: str = "Graph Data Extractor"
    description: str = (
        "Scans the saved figures for an article DOI and uses a vision language model "
        "to extract composition-property value pairs from graphs or charts. "
        "Use this tool when the text data is insufficient or when the agent detects "
        "that data may be presented graphically. Pass the article DOI as input."
    )
    args_schema: Type[BaseModel] = GraphExtractorToolInput

    vlm_model: str = "gemini/gemini-3-flash-preview"
    related_figures_base_path: str = "results/related_figures"
    vlm_property_name: str = "the target property"
    extraction_instruction: str = DEFAULT_TOOLS_CONFIG.graph_extractor.extraction_instruction

    def _run(self, doi: str) -> str:
        """Scan saved figures for the given DOI, call the vision LLM for each image, and aggregate the results.

        Args:
            doi (str, required): Article DOI.

        Returns:
            str: JSON string with extracted data per figure, or a plain error/status message
                if no figures are found or litellm is missing.
        """
        doi_folder = doi.replace("/", "_").replace(":", "_")
        fig_dir = os.path.join(self.related_figures_base_path, doi_folder)

        if not os.path.isdir(fig_dir):
            return (
                f"No saved figures found for DOI '{doi}' "
                f"(expected directory: {fig_dir}). "
                "Ensure figure extraction ran during article processing."
            )

        # Load captions from info.json
        info_path = os.path.join(fig_dir, "info.json")
        captions: Dict[str, str] = {}
        if os.path.isfile(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    captions = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read info.json for {doi}: {e}")

        # Collect .jpg image files
        image_files = sorted(
            f for f in os.listdir(fig_dir) if f.lower().endswith(".jpg")
        )
        if not image_files:
            return (
                f"No .jpg figures found in {fig_dir}. "
                "Captions available: " + json.dumps(captions)
            )

        property_name = self.vlm_property_name or "the target property"

        results: Dict[str, Any] = {}

        for img_filename in image_files:
            caption_id = os.path.splitext(img_filename)[0]
            caption_text = captions.get(caption_id, "")
            img_path = os.path.join(fig_dir, img_filename)

            try:
                with open(img_path, "rb") as f:
                    image_bytes = f.read()
                b64_image = base64.b64encode(image_bytes).decode("utf-8")
            except Exception as e:
                logger.warning(f"Could not read image {img_path}: {e}")
                results[caption_id] = {"error": f"Could not read image: {e}"}
                continue

            prompt = render_prompt(
                self.extraction_instruction, keyword=property_name, caption=caption_text
            )

            try:
                import litellm

                response = litellm.completion(
                    model=self.vlm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )
                raw_text = response.choices[0].message.content.strip()
                # Try to parse as JSON; fall back to raw text
                try:
                    extracted = json.loads(raw_text)
                except Exception:
                    # Strip markdown code fences if present
                    cleaned = raw_text.strip("`").strip()
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()
                    try:
                        extracted = json.loads(cleaned)
                    except Exception:
                        extracted = {"raw_response": raw_text}

                results[caption_id] = {
                    "caption": caption_text,
                    "extracted_data": extracted,
                }
                logger.info(f"VLM extracted data from figure '{caption_id}' for {doi}")

            except ImportError:
                msg = "litellm is not installed; graph extraction requires litellm."
                logger.error(msg)
                return msg
            except Exception as e:
                logger.warning(
                    f"VLM call failed for figure '{caption_id}' in {doi}: {e}"
                )
                results[caption_id] = {
                    "caption": caption_text,
                    "error": str(e),
                }

        if not results:
            return "No figures could be processed."

        return json.dumps(results, ensure_ascii=False, indent=2)
