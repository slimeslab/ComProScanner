import json
from typing import List, Optional, Dict

from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from pydantic import BaseModel, Field


def parse_to_dict(result, paper_id):
    """
    Parses the given result into a structured dictionary format.

    This function first defines the data model for piezoelectric compositions
    using the Pydantic library. It then parses the result using a parser
    tailored for the piezoelectric composition model, and finally, it constructs 
    a dictionary with proper hierarchical structure separating composition_data 
    and synthesis_data.

    Parameters:
    - result (str): The result string to be parsed.
    - paper_id (str/int): The ID associated with a particular research paper or source.

    Returns:
    - dict: A dictionary with 'composition_data' and 'synthesis_data' as top-level keys.

    Raises:
    - ValueError: If there's an issue whilst parsing the result.

    Example Usage:
    >>> result = "some_output_string_from_model"
    >>> paper_id = "10.1016/j.jeurceramsoc.2025.117193"
    >>> parsed_dict = parse_to_dict(result, paper_id)
    """

    class PiezoComposition(BaseModel):
        """Pydantic data model for a piezoelectric composition."""

        composition: str = Field(description="complete chemical composition or formula")
        d33_value: Optional[str] = Field(
            description="d33 piezoelectric charge coefficient value, 'Not provided' if unavailable"
        )
        unit: Optional[str] = Field(
            description="unit of d33 measurement (e.g., pC/N, pm/V), 'Not provided' if unavailable"
        )
        composition_family: Optional[str] = Field(
            description="composition family or material system, 'Not provided' if unavailable"
        )
        synthesis_method: Optional[str] = Field(
            description="synthesis or preparation method, 'Not provided' if unavailable"
        )
        precursors: Optional[List[str]] = Field(
            description="list of precursors used in synthesis, empty list if not provided"
        )
        steps: Optional[List[str]] = Field(
            description="list of synthesis steps in sequential order, empty list if not provided"
        )
        characterisation_techniques: Optional[List[str]] = Field(
            description="list of characterisation techniques employed, empty list if not provided"
        )
        score: float = Field(description="probability score of extraction accuracy")
        justification: str = Field(description="justification for extraction with exact sentences from document")
        DOI: str = Field(description="DOI of the paper")

    class PiezoCompositionList(BaseModel):
        """Pydantic data model for a list of piezoelectric compositions."""

        compositions: List[PiezoComposition]

    # Initialise a Pydantic output parser for the piezoelectric composition list model
    parser = PydanticOutputParser(pydantic_object=PiezoCompositionList)

    # Enhance the parser with capabilities to handle specific LLM outputs
    llm_parser = OutputFixingParser.from_llm(
        parser=parser, llm=ChatOpenAI(temperature=0, model="gpt-4")
    )

    # Check if result is a valid JSON string
    try:
        json.loads(result, strict=False)
    except json.JSONDecodeError:
        # If not, convert it to a valid JSON string
        result = json.dumps({"dummy_key": result})
    
    # Try parsing the result using the enhanced parser
    try:
        parsed_result = llm_parser.parse(result)
    except ValueError as e:
        raise ValueError(f"Failed to parse piezoelectric composition: {e}")

    # Extract compositions and property values
    compositions_property_values = {}
    
    # Use first composition's data for common synthesis information
    if not parsed_result.compositions:
        return {}
    
    # Get common data from first composition (assuming synthesis data is same across all)
    first_comp = parsed_result.compositions[0]
    
    # Build compositions_property_values dictionary
    for comp in parsed_result.compositions:
        if comp.d33_value and comp.d33_value != "Not provided":
            try:
                # Try to convert to float, if successful use as number, else use as string
                compositions_property_values[comp.composition] = float(comp.d33_value)
            except ValueError:
                compositions_property_values[comp.composition] = comp.d33_value
    
    # Construct the restructured dictionary
    restructured_dict = {
        "composition_data": {
            "compositions_property_values": compositions_property_values,
            "property_unit": first_comp.unit if first_comp.unit else "Not provided",
            "family": first_comp.composition_family if first_comp.composition_family else "Not provided"
        },
        "synthesis_data": {
            "method": first_comp.synthesis_method if first_comp.synthesis_method else "Not provided",
            "precursors": first_comp.precursors if first_comp.precursors else [],
            "steps": first_comp.steps if first_comp.steps else [],
            "characterization_techniques": first_comp.characterisation_techniques if first_comp.characterisation_techniques else []
        }
    }

    return restructured_dict