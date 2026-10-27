"""
schema_tools.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24
"""

import json
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, Field, create_model

from .configs.schema_config import ParsedType, SchemaField

_ATOMIC_TYPES = {"str": str, "int": int, "float": float, "bool": bool, "None": type(None)}


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def _annotation(parsed: ParsedType, schema_field: SchemaField, owner: str) -> Any:
    if parsed.kind in _ATOMIC_TYPES:
        return _ATOMIC_TYPES[parsed.kind]
    if parsed.kind == "union":
        members = tuple(_annotation(a, schema_field, owner) for a in parsed.args)
        return Union[members]
    if parsed.kind == "list":
        return List[_annotation(parsed.args[0], schema_field, owner)]
    if parsed.kind == "dict":
        return Dict[str, _annotation(parsed.args[0], schema_field, owner)]
    # Nested model from value_fields
    return _build_fields_model(
        schema_field.value_fields, f"{owner}{_camel(schema_field.name)}"
    )


def _build_fields_model(
    fields: List[SchemaField], model_name: str, doc: Optional[str] = None
) -> Type[BaseModel]:
    definitions = {}
    for schema_field in fields:
        annotation = _annotation(schema_field.parsed_type, schema_field, model_name)
        default = schema_field.get_default()
        if default is None and not schema_field.parsed_type.is_nullable():
            # Allow null default
            annotation = Optional[annotation]
        definitions[schema_field.name] = (annotation, default)
    return create_model(model_name, __doc__=doc, **definitions)


def build_output_model(
    fields: List[SchemaField],
    wrapper_key: str,
    model_name: str,
    inner_model_name: str,
    model_doc: Optional[str] = None,
    inner_model_doc: Optional[str] = None,
) -> Type[BaseModel]:
    """
    Build a crew's output_pydantic model with a single wrapper key around the schema fields.

    Args:
        fields (list): Schema fields, in output order.
        wrapper_key (str): Top-level key, e.g. "composition_extracted_data".
        model_name (str): Class name of the outer model.
        inner_model_name (str): Class name of the inner model.
        model_doc (str, optional): Docstring of the outer model.
        inner_model_doc (str, optional): Docstring of the inner model.
    """
    inner_model = _build_fields_model(fields, inner_model_name, inner_model_doc)
    return create_model(
        model_name, __doc__=model_doc, **{wrapper_key: (inner_model, inner_model())}
    )


Stage = Literal["extraction", "format"]


def render_fields_description(
    fields: List[SchemaField], stage: Stage = "extraction", depth: int = 0
) -> str:
    """Render fields as a bullet list for the schema_fields_description placeholder."""
    pad = "  " * depth
    lines = []
    for schema_field in fields:
        requirement = "required" if schema_field.required else "optional"
        line = f"{pad}- {schema_field.name} ({schema_field.type}, {requirement})"
        if schema_field.description:
            line += f": {schema_field.description}"
        lines.append(line)
        if stage == "format":
            if schema_field.format_note:
                lines.append(f"{pad}  Formatting rule: {schema_field.format_note}")
            if schema_field.allowed_values:
                bullets = "\n".join(f"- {value}" for value in schema_field.allowed_values)
                lines.append(f"{pad}  Allowed values:\n{bullets}")
        if schema_field.value_fields:
            if schema_field.is_object_dict:
                lines.append(f"{pad}  Each value is an object with these fields:")
            else:
                lines.append(f"{pad}  An object with these fields:")
            lines.append(
                render_fields_description(schema_field.value_fields, stage, depth + 2)
            )
    return "\n".join(lines)


def _placeholder_value(parsed: ParsedType, schema_field: SchemaField, variant: str) -> Any:
    kind = parsed.kind
    if kind == "union":
        members = [a for a in parsed.args if a.kind != "None"] or parsed.args
        return _placeholder_value(members[0], schema_field, variant)
    if kind == "str":
        return f"<{schema_field.name}>"
    if kind in ("int", "float"):
        return 0
    if kind == "bool":
        return False
    if kind == "None":
        return None
    if kind == "list":
        return [_placeholder_value(parsed.args[0], schema_field, variant)]
    if kind == "dict":
        return {"<key>": _placeholder_value(parsed.args[0], schema_field, variant)}
    return build_example(schema_field.value_fields, variant)


def build_example(fields: List[SchemaField], variant: str = "default") -> Dict[str, Any]:
    """Build an example record from the fields' example values."""
    example = {}
    for schema_field in fields:
        value = schema_field.example
        if variant == "variable" and schema_field.variable_example is not None:
            value = schema_field.variable_example
        if value is None:
            value = _placeholder_value(schema_field.parsed_type, schema_field, variant)
        example[schema_field.name] = value
    return example


def render_example_json(fields: List[SchemaField], variant: str = "default") -> str:
    return json.dumps(build_example(fields, variant), indent=2, ensure_ascii=False)


def schema_prompt_values(fields: List[SchemaField], stage: Stage) -> Dict[str, str]:
    """Values for the schema placeholders in agent/task YAML."""
    return {
        "schema_fields_description": render_fields_description(fields, stage),
        "schema_example_json": render_example_json(fields),
        "schema_field_names": ", ".join(f.name for f in fields),
    }


def fill_schema_placeholders(config: Dict[str, Any], values: Dict[str, str]) -> Dict[str, Any]:
    """Return a copy of an agent/task config with the schema placeholders filled in."""
    filled = dict(config)
    for key, text in filled.items():
        if isinstance(text, str):
            for name, value in values.items():
                text = text.replace("{" + name + "}", value)
            filled[key] = text
    return filled


# Agentic evaluation: response models and prompt text per match mode

_SCALAR_MODES = ("exact", "semantic", "numeric")
_SCALAR_MODE_INSTRUCTIONS = {
    "exact": "Decide if the values match exactly (1) or not (0)",
    "semantic": "Decide if the values match in meaning (1) or not (0)",
    "numeric": "Decide if the numeric values match (1) or not (0). A ground truth range [min, max] matches any test value inside it.",
}


def leaf_paths(fields: List[SchemaField], prefix: str = "") -> List[str]:
    """Dotted paths of every leaf below a list of value fields."""
    paths = []
    for schema_field in fields:
        path = f"{prefix}{schema_field.name}"
        if schema_field.value_fields and schema_field.is_object:
            paths.extend(leaf_paths(schema_field.value_fields, f"{path}."))
        else:
            paths.append(path)
    return paths


def _has_numeric_values(schema_field: SchemaField) -> bool:
    value_type = schema_field.parsed_type.args[0]
    members = value_type.args if value_type.kind == "union" else [value_type]
    members = [m for m in members if m.kind != "None"]
    return bool(members) and all(m.kind in ("int", "float") for m in members)


def build_evaluation_model(fields: List[SchemaField], domain: str) -> Type[BaseModel]:
    """
    Build the agentic evaluator's response model for one domain.

    Args:
        fields (list): Evaluated schema fields of the domain, in output order.
        domain (str): "composition" or "synthesis".
    """
    title = domain.capitalize()
    shared = {}

    def model(name, doc, **definitions):
        if name not in shared:
            shared[name] = create_model(name, __doc__=doc, **definitions)
        return shared[name]

    def list_field(item_type):
        return (List[item_type], Field(default_factory=list))

    item_list_names = [f.name for f in fields if f.match_mode == "item_list"]
    definitions = {}
    for schema_field in fields:
        mode = schema_field.match_mode
        camel = _camel(schema_field.name)
        if mode in _SCALAR_MODES:
            annotation = model(
                f"{title}Match",
                "Basic match structure with reference and test values",
                match_value=(int, ...),
                reference=(Optional[Any], None),
                test=(Optional[Any], None),
            )
        elif mode == "item_list":
            item_match = model(
                "ItemMatch",
                "Model for individual item match evaluation",
                reference_item=(str, ...),
                test_item=(str, ...),
                match_value=(int, ...),
            )
            annotation = model(
                "ListItemsMatch",
                f"Model for list items match evaluation ({', '.join(item_list_names)})",
                reference=list_field(str),
                test=list_field(str),
                matches=list_field(item_match),
                total_ground_truth_items=(int, 0),
                total_match=(int, 0),
                missing_items=list_field(str),
                extra_items=list_field(str),
            )
        elif mode == "joined_text":
            annotation = model(
                f"{camel}Match",
                f"Model for {domain} {schema_field.name} match evaluation",
                match_value=(float, 0.0),
                reference_steps=list_field(str),
                test_steps=list_field(str),
            )
        elif mode == "keyed_values":
            key_match = model(
                "KeyMatch",
                "Model for key match evaluation",
                reference_key=(str, ...),
                test_key=(str, ...),
                match_value=(int, ...),
            )
            if schema_field.value_fields:
                value_match = model(
                    "LeafValueMatch",
                    "Model for value match evaluation of one leaf of a nested value",
                    reference_key=(str, ...),
                    test_key=(str, ...),
                    leaf=(str, ...),
                    reference_value=(Any, ...),
                    test_value=(Any, ...),
                    match_value=(int, ...),
                )
            else:
                value_match = model(
                    "ValueMatch",
                    "Model for value match evaluation",
                    reference_key=(str, ...),
                    test_key=(str, ...),
                    reference_value=(Any, ...),
                    test_value=(Any, ...),
                    match_value=(int, ...),
                )
            numeric_values = _has_numeric_values(schema_field)
            pair_value = Union[int, float] if numeric_values else Any
            pair_match = model(
                "PairMatch" if numeric_values else "ObjectPairMatch",
                "Model for key-value pair match evaluation",
                reference_pair=(Dict[str, pair_value], Field(default_factory=dict)),
                test_pair=(Dict[str, pair_value], Field(default_factory=dict)),
                match_value=(int, ...),
            )
            annotation = model(
                f"{camel}Match",
                f"Model for {schema_field.name} match evaluation",
                reference=(Dict[str, Any], Field(default_factory=dict)),
                test=(Dict[str, Any], Field(default_factory=dict)),
                key_matches=list_field(key_match),
                value_matches=list_field(value_match),
                pair_matches=list_field(pair_match),
                total_ground_truth_keys=(int, 0),
                total_match=(int, 0),
                missing_keys=list_field(str),
                extra_keys=list_field(str),
            )
        else:
            leaf_match = model(
                "LeafMatch",
                "Model for nested sub-field match evaluation",
                leaf=(str, ...),
                reference_value=(Any, None),
                test_value=(Any, None),
                match_value=(int, ...),
            )
            annotation = model(
                f"{camel}Match",
                f"Model for {schema_field.name} match evaluation",
                leaf_matches=list_field(leaf_match),
            )
        definitions[schema_field.name] = (annotation, ...)

    details = create_model(
        f"{title}DataDetails", __doc__=f"{title} data details", **definitions
    )
    return create_model(
        f"{title}DataWrapper",
        __doc__=f"Wrapper for {domain} data - the output from the evaluation task",
        **{f"{domain}_data": (details, ...)},
    )


def _evaluation_instruction(schema_field: SchemaField, domain: str) -> str:
    mode = schema_field.match_mode
    if schema_field.evaluation_note:
        instruction = schema_field.evaluation_note
    elif mode in _SCALAR_MODES:
        instruction = _SCALAR_MODE_INSTRUCTIONS[mode]
    elif mode == "item_list":
        instruction = (
            "For each item, decide if it's correctly identified (1) or not (0)\n"
            "   Note: The order of items doesn't matter, only whether each ground truth item is correctly identified in the test data."
        )
    elif mode == "joined_text":
        instruction = (
            f"Decide if the {schema_field.name} match or not as a whole. The match value should be a float "
            "between 0.00 and 1.00 and it should be round up to two decimal points."
        )
    elif mode == "keyed_values":
        instruction = (
            "For each entry, decide:\n"
            "   a. Keys: Determine if each ground truth key is matched in test data (1) or missing (0)\n"
            "   b. Values: Do the values match (1) or not (0) corresponding to the keys?"
        )
        if domain == "composition":
            instruction += "\n      {value_error_threshold_instructions}"
    else:
        instruction = (
            "Compare each sub-field and decide if it matches (1) or not (0): "
            + ", ".join(leaf_paths(schema_field.value_fields))
        )
    if mode == "keyed_values" and schema_field.value_fields:
        instruction += (
            "\n   Each value is compared leaf by leaf: give one value_matches entry per leaf ("
            + ", ".join(leaf_paths(schema_field.value_fields))
            + ") of every matched key, with the leaf's path in 'leaf'."
        )
    return instruction


def render_evaluation_criteria(fields: List[SchemaField], domain: str) -> str:
    """Render the numbered per-field judging rules for the agentic evaluator."""
    return "\n".join(
        f"{index}. For '{schema_field.name}': {_evaluation_instruction(schema_field, domain)}"
        for index, schema_field in enumerate(fields, start=1)
    )


def _evaluation_skeleton(schema_field: SchemaField) -> List[str]:
    mode = schema_field.match_mode
    label = schema_field.name.replace("_", " ")
    if mode in _SCALAR_MODES:
        return [
            '"match_value": 1 or 0,',
            '"reference": "ground truth value",',
            '"test": "test value"',
        ]
    if mode == "item_list":
        return [
            '"reference": ["item1", "item2"],',
            '"test": ["item1", "item3"],',
            '"matches": [',
            "  {",
            '    "reference_item": "item1",',
            '    "test_item": "item1",',
            '    "match_value": 1 or 0',
            "  }",
            "],",
            f'"total_ground_truth_items": total number of {label} in ground truth data,',
            f'"total_match": total number of matched {label},',
            '"missing_items": ["item2"],',
            '"extra_items": ["item3"]',
        ]
    if mode == "joined_text":
        return [
            '"match_value": value between 0 to 1 (float upto two decimal),',
            '"reference_steps": ["item1", "item2"],',
            '"test_steps": ["item1", "item3"]',
        ]
    if mode == "keyed_values":
        leaf_line = []
        if schema_field.value_fields:
            leaf_line = [f'    "leaf": "{leaf_paths(schema_field.value_fields)[0]}",']
        return [
            '"reference": {...},',
            '"test": {...},',
            '"key_matches": [',
            "  {",
            '    "reference_key": "key1",',
            '    "test_key": "key1",',
            '    "match_value": 1 or 0',
            "  }",
            "],",
            '"value_matches": [',
            "  {",
            '    "reference_key": "key1",',
            '    "test_key": "key1",',
            *leaf_line,
            '    "reference_value": "value1",',
            '    "test_value": "value1",',
            '    "match_value": 1 or 0',
            "  }",
            "],",
            '"pair_matches": [',
            "  {",
            '    "reference_pair": {"key1": "value1"},',
            '    "test_pair": {"key1": "value1"},',
            '    "match_value": 1 or 0',
            "  }",
            "],",
            f'"total_ground_truth_keys": total number of {label} keys in ground truth data,',
            f'"total_match": total number of matched {label} keys,',
            '"missing_keys": ["key2"],',
            '"extra_keys": ["key3"]',
        ]
    return [
        '"leaf_matches": [',
        "  {",
        f'    "leaf": "{leaf_paths(schema_field.value_fields)[0]}",',
        '    "reference_value": "ground truth value",',
        '    "test_value": "test value",',
        '    "match_value": 1 or 0',
        "  }",
        "]",
    ]


def render_evaluation_output_format(fields: List[SchemaField], domain: str) -> str:
    """Render the response skeleton the agentic evaluator must return."""
    lines = ["{", f'  "{domain}_data": {{']
    for index, schema_field in enumerate(fields):
        lines.append(f'    "{schema_field.name}": {{')
        lines.extend(f"      {line}" for line in _evaluation_skeleton(schema_field))
        lines.append("    }," if index < len(fields) - 1 else "    }")
    lines.extend(["  }", "}"])
    return "\n".join(lines)


def evaluation_prompt_values(fields: List[SchemaField], domain: str) -> Dict[str, str]:
    """Values for the schema placeholders in the evaluation crews' YAML."""
    return {
        "schema_evaluation_criteria": render_evaluation_criteria(fields, domain),
        "schema_evaluation_output_format": render_evaluation_output_format(fields, domain),
    }
