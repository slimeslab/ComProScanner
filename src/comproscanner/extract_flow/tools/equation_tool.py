"""
equation_tool.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 17-04-2025
"""

# Standard library imports
import base64
import json
import os
from typing import Optional, Type

# Third-party imports
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Local imports
from ...utils.logger import setup_logger

logger = setup_logger("comproscanner.log", module_name="equation_tool")

FORMULA_INSTRUCTION = """
You are a materials science expert specializing in ceramic synthesis and solid-state chemistry.

You will be given a research paper. Your job is to:
1. Determine if doping produces a single-phase compound.
2. If yes, derive the general doped chemical formula.
3. If no, return exactly: single compound is not being synthesized

---

STEP 1 - IDENTIFY THE HOST CRYSTAL STRUCTURE AND COMPOSITION

Extract:
- The crystal structure type (perovskite, spinel, fluorite, wurtzite, rocksalt, layered, etc.)
- All constituent sublattices and the ions occupying each sublattice in the undoped base
- The mole fractions of each ion on each sublattice
- The general stoichiometry formula of the undoped compound (e.g., ABO3, AB2O4, AO2, ABX3)

---

STEP 2 - CHECK FOR SINGLE PHASE FORMATION

Look for:
- Diffraction evidence (XRD, neutron) of a single phase with no secondary phases
- Language like "pure phase", "single phase", "complete solid solution", "no impurity peaks"
- No reported phase separation or multiphase coexistence in the doped system

If secondary phases, phase separation or multiphase coexistence is reported, return:
single compound is not being synthesized

---

STEP 3 - IDENTIFY THE DOPANT

Extract:
- Dopant identity and its ionic valence in the host lattice
- Dopant concentration variable (commonly x, y or mol%)
- Which sublattice the dopant substitutes
- The host ion being replaced and its valence
- Whether the dopant is donor type (higher valence than replaced ion) or acceptor type (lower valence)

---

STEP 4 - DETERMINE THE CHARGE COMPENSATION MECHANISM

Calculate the valence mismatch: delta_q = valence of dopant - valence of replaced ion

Then apply the appropriate mechanism based on what the paper reports or what is physically standard for the structure:

DONOR DOPING (delta_q > 0, dopant has higher valence):
- Cation vacancies on the substituted sublattice
- Anion interstitials
- Reduction of another cation valence state
- Anion deficiency (write anion stoichiometry as nominal value in final formula)

ACCEPTOR DOPING (delta_q < 0, dopant has lower valence):
- Anion vacancies (write anion stoichiometry as nominal value in final formula)
- Cation interstitials
- Oxidation of another cation to a higher valence state

ISOVALENT DOPING (delta_q = 0):
- No charge compensation needed
- Direct 1-for-1 substitution on the sublattice

SUBSTITUTION RATIO RULE:
When direct vacancy compensation applies on the substituted sublattice, the substitution ratio
is determined by charge balance. For a dopant of valence Vd replacing a host ion of valence Vh:
- The number of host ions removed per dopant added is Vd/Vh (if Vd > Vh) or Vh/Vd (if Vd < Vh)
- Scale existing sublattice coefficients by (1-x) and insert dopant coefficient derived from the ratio
- Vacancies are implicit in the reduced sublattice sum and must NOT appear in the final formula

For compensation via anion stoichiometry or valence change on another sublattice:
- Substitute 1-for-1 on the doped sublattice
- Adjust the anion coefficient or the other cation coefficient accordingly
- Still write the anion stoichiometry as the nominal integer value in the final formula

---

STEP 5 - CONSTRUCT THE FORMULA

Rules:
1. Preserve the general stoichiometry template of the host structure (e.g., ABO3, AB2O4, AO2).
2. Scale all original sublattice coefficients by (1-x) or the appropriate dilution factor from Step 4.
3. Insert the dopant term on the correct sublattice with the coefficient derived from Step 4.
4. Always write anion stoichiometry as the nominal integer value regardless of the compensation mechanism.
5. Do NOT use + or - signs anywhere in the formula.
6. Do NOT write vacancy symbols such as Va, V_A or [] in the formula. Vacancies are implicit.
7. All coefficients must be explicit algebraic expressions in the dopant variable (x or as specified).
8. If dopant concentration is given in wt%, convert to mol fraction first. Do not treat wt% as mol%.
9. If multiple dopants are present, apply Steps 3 and 4 sequentially for each dopant and combine.
10. Write sublattice groups in parentheses separated by sublattice, followed by the anion block.
11. Use explicit multiplication operators in variable expressions: write 0.5*x and 0.5*(1-x-y), not 0.5x or 0.5(1-x-y).
12. Add parentheses around variable expressions where scope may be ambiguous.

---

STEP 6 - VALIDATE CHARGE BALANCE

For ionic compounds, verify:
Sum of (ionic charge x coefficient) across all cation sublattices = Sum of (ionic charge x coefficient) across all anion sublattices

If the balance does not hold, revisit the compensation mechanism in Step 4.

---

OUTPUT FORMAT

Return one of the following and nothing else:

- If single phase forms: return only the plain text chemical formula with no LaTeX, no + or - signs,
  no vacancy notation and no explanation.
  Example: (Na0.53*(1-x)K0.404*(1-x)Li0.066*(1-x)Ca(x))(Nb0.92*(1-x)Sb0.08*(1-x)Al(2*x))O3

- If not single phase: return exactly the string:
  single compound is not being synthesized
"""

# Caption keywords that indicate crystal structure characterisation figures
_CRYSTAL_STRUCTURE_KEYWORDS = {
    "xrd",
    "x-ray diffraction",
    "x-ray",
    "diffraction",
    "sem",
    "scanning electron",
    "tem",
    "transmission electron",
    "crystal structure",
    "microstructure",
    "rietveld",
    "bragg",
    "crystal phase",
    "phase purity",
    "phase identification",
}

# Priority-ordered (env_var, litellm_model) pairs for fallback LLM selection
_API_KEY_MODEL_PAIRS = [
    ("ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-6"),
    ("GEMINI_API_KEY", "gemini/gemini-3-flash-preview"),
    ("OPENAI_API_KEY", "openai/gpt-5.4-mini"),
    ("DEEPSEEK_API_KEY", "deepseek/deepseek-chat"),
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
    LLM-powered tool that analyses paper text (and any saved crystal structure
    images — XRD, SEM, TEM) to determine whether doping produces a single-phase
    perovskite compound and, if so, to derive the general doped chemical formula.

    The tool selects its LLM automatically based on which API key is present in
    the environment: ANTHROPIC_API_KEY (claude-sonnet-4-6, default) →
    GEMINI_API_KEY → OPENAI_API_KEY → DEEPSEEK_API_KEY → OPENROUTER_API_KEY →
    TOGETHER_API_KEY → COHERE_API_KEY → FIREWORKS_API_KEY.
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
            doi (str): Article DOI used to locate the figure directory.

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
            if not any(kw in caption_lower for kw in _CRYSTAL_STRUCTURE_KEYWORDS):
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
        """Analyse the paper and return the derived formula or the not-single-phase string."""
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
