"""
piezo_agent_tools.py

Core extraction functions for piezoelectric materials using LLMs.
Based on ComProScanner piezo_test structure.
"""

import json
import json5
import ast
import re
from typing import Dict, List
from langchain_core.prompts import PromptTemplate


def robust_json_parse(text: str) -> dict:
    """Tries multiple strategies to recover valid JSON from LLM output."""
    if hasattr(text, "content"):
        text = text.content

    # Strip Markdown formatting
    text = text.strip().removeprefix("```json").removesuffix("```").strip()

    # Try to extract first complete JSON object or array
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        text = match.group(1)

    # Clean trailing commas
    text = re.sub(r",\s*([\]}])", r"\1", text)

    # Replace invalid constructs
    text = text.replace("None", "null")
    text = text.replace("'", '"')

    try:
        return json.loads(text)
    except:
        pass

    try:
        return json5.loads(text)
    except:
        pass

    try:
        return ast.literal_eval(text)
    except:
        pass

    print(f"⚠️ All JSON parsing failed")
    return {
        "composition_data": {
            "compositions_property_values": {},
            "property_unit": "",
            "family": "",
        },
        "synthesis_data": {
            "method": "",
            "precursors": [],
            "steps": [],
            "characterization_techniques": [],
        },
    }


def extract_material_candidates(
    fulltext: str, llm, max_materials: int = 20
) -> List[str]:
    """Quick scan to find piezoelectric materials mentioned"""
    prompt = f"""
You are a materials science assistant specialising in piezoelectric materials. 
Scan this text and list piezoelectric composition names that have d33 (piezoelectric charge coefficient) values mentioned.
Return ONLY a JSON array of composition names, maximum {max_materials}.

Example: ["Pb(Zr0.52Ti0.48)O3", "BaTiO3", "0.5Ba(Zr0.2Ti0.8)O3-0.5(Ba0.7Ca0.3)TiO3"]

Text:
`````{fulltext[:8000]}```

JSON array:
"""
    output = llm.invoke(prompt)
    try:
        candidates = json.loads(output.content)
        return candidates[:max_materials] if isinstance(candidates, list) else []
    except:
        return []


def extract_piezo_properties(
    fulltext: str, llm, material_names: List[str] = None
) -> Dict:
    """Extract piezoelectric composition and property data"""
    material_hint = ""
    if material_names:
        formatted = ", ".join(f'"{name}"' for name in material_names)
        material_hint = f"Focus on extracting properties for the following compositions: {formatted}.\n"

    prompt = PromptTemplate.from_template(
        """
You are an expert in piezoelectric materials extraction. Your task is to extract composition-property relationships from scientific text.

{material_hint}

Extract the following information:
1. **compositions_property_values**: A dictionary mapping each composition (chemical formula) to its d33 value (as a number)
2. **property_unit**: The unit of measurement for d33 (typically "pC/N" or "pm/V")
3. **family**: The material family or system (e.g., "PZT", "BCZT", "KNN-based", "BaTiO3-based", "PbNb2O6")

**Critical Instructions:**
- Extract ALL compositions with their d33 values from the text
- Composition formulas must be exact as written in the text (e.g., "Pb0.95K0.1[Nb0.96Ta0.04]2O6")
- d33 values must be numbers only (e.g., 141, not "141 pC/N")
- If multiple d33 values exist for one composition, use the maximum value
- Property unit should be consistent for all compositions (typically "pC/N")
- Family should describe the broader material system
- Set missing or unclear values to empty string "" or empty dict {{}}
- If more than 15 compositions are found, include only the top 15 by d33 value
- Do NOT include any explanatory text, only the JSON output

Return structured JSON in this exact format:
{{
  "composition_data": {{
    "compositions_property_values": {{
      "Composition1": d33_value_as_number,
      "Composition2": d33_value_as_number
    }},
    "property_unit": "pC/N",
    "family": "MaterialFamily"
  }}
}}

Text:
````{fulltext}```
"""
    )
    output = llm.invoke(prompt.format(fulltext=fulltext, material_hint=material_hint))
    result = robust_json_parse(output.content)

    # Ensure proper structure
    if "composition_data" not in result:
        return {
            "composition_data": {
                "compositions_property_values": {},
                "property_unit": "",
                "family": "",
            }
        }
    return result


def extract_synthesis_properties(
    fulltext: str, llm, material_names: List[str] = None
) -> Dict:
    """Extract synthesis and characterisation information"""
    material_hint = ""
    if material_names:
        formatted = ", ".join(f'"{name}"' for name in material_names)
        material_hint = f"Focus on synthesis information for the following compositions: {formatted}.\n"

    prompt = PromptTemplate.from_template(
        """
You are an expert in materials synthesis extraction. Your task is to extract synthesis methodology from scientific text.

{material_hint}

Extract the following information:
1. **method**: Primary synthesis or preparation method (e.g., "solid-state reaction", "sol-gel", "hydrothermal synthesis")
2. **precursors**: List of starting materials/precursors used (e.g., ["PbO", "K2CO3", "Nb2O5", "Ta2O5"])
3. **steps**: List of synthesis steps in sequential order, as detailed as possible
4. **characterization_techniques**: List of characterisation techniques used (e.g., ["XRD", "SEM", "EDS", "Raman spectroscopy"])

**Critical Instructions:**
- Extract synthesis information that applies to the piezoelectric compositions in the paper
- Steps should be extracted as complete sentences describing each synthesis stage
- Precursors should be chemical formulas or compound names
- Characterisation techniques should use standard abbreviations (XRD, SEM, TEM, etc.)
- If information is not found, use empty list [] or empty string ""
- Extract as much detail as possible from the experimental/methods sections
- Do NOT include any explanatory text, only the JSON output

Return structured JSON in this exact format:
{{
  "synthesis_data": {{
    "method": "synthesis method",
    "precursors": ["precursor1", "precursor2", ...],
    "steps": ["step 1 description", "step 2 description", ...],
    "characterization_techniques": ["XRD", "SEM", ...]
  }}
}}

Text:
```{fulltext}```
"""
    )
    output = llm.invoke(prompt.format(fulltext=fulltext, material_hint=material_hint))
    result = robust_json_parse(output.content)

    # Ensure proper structure
    if "synthesis_data" not in result:
        return {
            "synthesis_data": {
                "method": "",
                "precursors": [],
                "steps": [],
                "characterization_techniques": [],
            }
        }
    return result


def extract_from_tables(
    table_data: list, llm, material_names: List[str] = None
) -> dict:
    """Extract piezoelectric data from tables"""
    if not table_data:
        return {
            "composition_data": {
                "compositions_property_values": {},
                "property_unit": "",
                "family": "",
            }
        }

    material_hint = ""
    if material_names:
        formatted = ", ".join(f'"{name}"' for name in material_names)
        material_hint = (
            f"Focus on extracting data for these compositions: {formatted}.\n"
        )

    combined_block = ""
    for i, table in enumerate(table_data, 1):
        combined_block += f"### Table {i} Caption:\n{table['caption']}\n\n"
        combined_block += (
            f"### Table {i} CSV Data:\n{json.dumps(table['rows'], indent=2)}\n\n"
        )

    prompt = f"""
You are a scientific table extraction agent for piezoelectric materials.

{material_hint}

Extract composition-property relationships from these tables.

**Extract:**
1. All composition formulas and their corresponding d33 values
2. The unit of measurement (typically pC/N or pm/V)
3. The material family if identifiable from the table

**Critical Instructions:**
- Map each composition to its d33 value as a number
- Ensure composition formulas are exact as shown in tables
- If multiple d33 values exist for one composition, use the maximum
- Property unit should be consistent
- Do NOT include any explanatory text, only the JSON output

Return structured JSON in this exact format:
{{
  "composition_data": {{
    "compositions_property_values": {{
      "Composition1": d33_value_as_number,
      "Composition2": d33_value_as_number
    }},
    "property_unit": "pC/N",
    "family": "MaterialFamily"
  }}
}}

Tables:
{combined_block}
"""
    output = llm.invoke(prompt)
    result = robust_json_parse(output.content)

    # Ensure proper structure
    if "composition_data" not in result:
        return {
            "composition_data": {
                "compositions_property_values": {},
                "property_unit": "",
                "family": "",
            }
        }
    return result


def merge_extraction_results(
    piezo_data: dict, synthesis_data: dict, table_data: dict
) -> dict:
    """
    Merge results from different extraction agents into final ComProScanner format

    Args:
        piezo_data: Result from extract_piezo_properties
        synthesis_data: Result from extract_synthesis_properties
        table_data: Result from extract_from_tables

    Returns:
        dict: Merged result in ComProScanner piezo_test format
    """
    # Start with piezo data (composition_data)
    merged = {
        "composition_data": piezo_data.get(
            "composition_data",
            {"compositions_property_values": {}, "property_unit": "", "family": ""},
        ),
        "synthesis_data": synthesis_data.get(
            "synthesis_data",
            {
                "method": "",
                "precursors": [],
                "steps": [],
                "characterization_techniques": [],
            },
        ),
    }

    # Merge table data into composition_data if available
    table_comp_data = table_data.get("composition_data", {})
    if table_comp_data.get("compositions_property_values"):
        # Add table compositions that aren't already present
        existing_comps = merged["composition_data"]["compositions_property_values"]
        table_comps = table_comp_data["compositions_property_values"]

        for comp, value in table_comps.items():
            if comp not in existing_comps:
                existing_comps[comp] = value

        # Update unit and family if they were empty
        if not merged["composition_data"]["property_unit"] and table_comp_data.get(
            "property_unit"
        ):
            merged["composition_data"]["property_unit"] = table_comp_data[
                "property_unit"
            ]
        if not merged["composition_data"]["family"] and table_comp_data.get("family"):
            merged["composition_data"]["family"] = table_comp_data["family"]

    return merged
