"""
material_parser_tool.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 23-03-2025
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests
from typing import Type, Dict, Any, Union
import json


class MaterialParserInput(BaseModel):
    """Input schema for MaterialParserTool."""

    data: Union[Dict[str, Any], str] = Field(
        ...,
        description="""JSON data containing compositions and their associated values, either as a dictionary or a string representation of a dictionary.""",
    )


class MaterialParserTool(BaseTool):
    name: str = "Material Formula Parser"
    description: str = (
        "Parses and resolves chemical formulas with variables from a JSON structure containing compositions and their associated values. "
        "It can handle formulas like 'Na(1-x)Li(y)TiO3 where x=0.1, y=0.4' or '(Mo 0.96 Zr 0.04) 0.85 B x (x=0.15)' and returns the API's resolved version."
    )
    args_schema: Type[BaseModel] = MaterialParserInput

    def _run(self, data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Sends chemical formulas from the compositions dictionary to the Material Parser API and returns resolved versions.

        Args:
            data: JSON data containing compositions and their associated values, either as a dictionary or string

        Returns:
            Dictionary with the same structure as input but with resolved formulas as keys
        """
        try:
            # Handle string input by parsing it to a dictionary
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    # Double-escaped JSON string (common in LLM outputs)
                    data = data.replace('\\"', '"')
                    try:
                        data = json.loads(data)
                    except:
                        return {"error": "Failed to parse input string as JSON"}

            # Handle nested description field which sometimes appears in LLM outputs
            if (
                isinstance(data, dict)
                and "description" in data
                and isinstance(data["description"], str)
            ):
                try:
                    data = json.loads(data["description"])
                except:
                    # Try with replaced quotes
                    try:
                        fixed_str = data["description"].replace('\\"', '"')
                        data = json.loads(fixed_str)
                    except:
                        return {"error": "Failed to parse description field as JSON"}

            # Extract composition data, look for various possible field names
            compositions = {}
            if isinstance(data, dict):
                if "compositions" in data:
                    compositions = data["compositions"]
                elif "compositions_property_values" in data:
                    compositions = data["compositions_property_values"]

            # Get unit information, checking various possible field names
            unit = ""
            if isinstance(data, dict):
                if "{composition_property_text_data}_unit" in data:
                    unit = data["{composition_property_text_data}_unit"]
                elif "property_unit" in data:
                    unit = data["property_unit"]

            # Get family information
            family = ""
            if isinstance(data, dict) and "family" in data:
                family = data["family"]

            # Prepare result structure
            result_dict = {
                "compositions": {},
                "property_unit": unit,
                "family": family,
            }

            # Process each composition formula
            for formula, value in compositions.items():
                url = "https://lfoppiano-material-parsers.hf.space/process/material"
                files = {"text": (None, formula)}
                response = requests.post(url, files=files)
                if response.status_code != 200:
                    result_dict["compositions"][formula] = value
                    continue
                api_data = response.json()
                if not api_data or not isinstance(api_data, list) or not api_data[0]:
                    result_dict["compositions"][formula] = value
                    continue

                # Extract the resolved formula
                try:
                    resolved_formulas = api_data[0][0].get("resolvedFormulas", [])
                    if not resolved_formulas:
                        result_dict["compositions"][formula] = value
                        continue
                    raw_value = resolved_formulas[0].get("rawValue", "")
                    if not raw_value:
                        result_dict["compositions"][formula] = value
                        continue
                    result_dict["compositions"][raw_value] = value

                except (IndexError, KeyError):
                    result_dict["compositions"][formula] = value

            return result_dict

        except Exception as e:
            # Return error information
            return {
                "error": f"Unexpected error: {str(e)}",
                "input_type": str(type(data)),
                "input_preview": (
                    str(data)[:100] + "..."
                    if isinstance(data, str) and len(str(data)) > 100
                    else str(data)
                ),
            }
