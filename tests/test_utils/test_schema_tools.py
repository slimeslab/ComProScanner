"""
test_schema_tools.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 24-10-2026
"""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from comproscanner.utils.configs.schema_config import (
    DEFAULT_SCHEMA_CONFIG,
    ExtractionSchemaConfig,
)
from comproscanner.utils.schema_tools import (
    build_evaluation_model,
    build_example,
    build_output_model,
    fill_schema_placeholders,
    render_evaluation_criteria,
    render_evaluation_output_format,
    render_fields_description,
    schema_prompt_values,
)

SNAPSHOT = json.loads(
    (Path(__file__).parent / "fixtures" / "legacy_output_models_snapshot.json").read_text(
        encoding="utf-8"
    )
)

# Arguments each crew passes to build_output_model
CREW_MODELS = {
    "composition_extraction": dict(
        domain="composition",
        final_only=False,
        wrapper_key="composition_extracted_data",
        model_name="ExtractedCompositionPropertyData",
        inner_model_name="DetailedCompositionPropertyData",
        model_doc="Extracted Composition Property",
        inner_model_doc="Detailed Composition Property",
    ),
    "composition_format": dict(
        domain="composition",
        final_only=True,
        wrapper_key="composition_formatted_data",
        model_name="FormattedCompositionPropertyData",
        inner_model_name="DetailedCompositionPropertyData",
        model_doc="Formatted Composition Property",
        inner_model_doc="Detailed Composition Property",
    ),
    "synthesis_extraction": dict(
        domain="synthesis",
        final_only=False,
        wrapper_key="synthesis_extracted_data",
        model_name="ExtractedSynthesisData",
        inner_model_name="DetailedSynthesisData",
        model_doc="Extracted Synthesis Data",
        inner_model_doc="Detailed Synthesis Data",
    ),
    "synthesis_format": dict(
        domain="synthesis",
        final_only=True,
        wrapper_key="synthesis_formatted_data",
        model_name="FormattedSynthesisData",
        inner_model_name="DetailedSynthesisData",
        model_doc="Formatted Synthesis Data",
        inner_model_doc="Detailed Synthesis Data",
    ),
}


def build(schema, domain, final_only, **kwargs):
    return build_output_model(schema.fields(domain, final_only=final_only), **kwargs)


def annotations(model):
    return {
        name: annotations(f.annotation)
        if isinstance(f.annotation, type) and issubclass(f.annotation, BaseModel)
        else str(f.annotation)
        for name, f in model.model_fields.items()
    }


@pytest.mark.parametrize("crew", sorted(CREW_MODELS))
class TestDefaultSchemaMatchesLegacyModels:
    def test_json_schema(self, crew):
        model = build(DEFAULT_SCHEMA_CONFIG, **CREW_MODELS[crew])
        assert model.model_json_schema() == SNAPSHOT[crew]["model_json_schema"]

    def test_default_dump(self, crew):
        model = build(DEFAULT_SCHEMA_CONFIG, **CREW_MODELS[crew])
        assert model().model_dump() == SNAPSHOT[crew]["default_dump"]

    def test_annotations(self, crew):
        model = build(DEFAULT_SCHEMA_CONFIG, **CREW_MODELS[crew])
        assert annotations(model) == SNAPSHOT[crew]["annotations"]


class TestCustomSchemaModels:
    @pytest.fixture
    def branched_schema(self):
        return ExtractionSchemaConfig.from_dict(
            {
                "version": "1.0",
                "composition_fields": [
                    {
                        "name": "compositions_property_values",
                        "type": "dict[str, object]",
                        "value_fields": [
                            {
                                "name": "d33_data",
                                "type": "object",
                                "value_fields": [
                                    {"name": "value", "type": "float|int|None"},
                                    {"name": "unit", "type": "str"},
                                ],
                            },
                            {"name": "sintering_temp", "type": "float|None"},
                        ],
                    },
                    {"name": "property_unit", "type": "str"},
                ],
                "synthesis_fields": [
                    {"name": "method", "type": "str"},
                    {"name": "sintering_atmosphere", "type": "str", "required": False},
                ],
            }
        )

    def test_branched_composition_values_validate(self, branched_schema):
        model = build(branched_schema, **CREW_MODELS["composition_format"])
        record = {
            "composition_formatted_data": {
                "compositions_property_values": {
                    "BaTiO3": {
                        "d33_data": {"value": 190, "unit": "pC/N"},
                        "sintering_temp": 1300,
                    },
                    "Ba0.9Ca0.1TiO3": {"d33_data": {"value": 250, "unit": "pC/N"}},
                },
                "property_unit": "pC/N",
            }
        }
        dumped = model.model_validate(record).model_dump()
        values = dumped["composition_formatted_data"]["compositions_property_values"]
        assert values["BaTiO3"]["d33_data"] == {"value": 190, "unit": "pC/N"}
        assert values["BaTiO3"]["sintering_temp"] == 1300
        # Missing optional block falls back to default
        assert values["Ba0.9Ca0.1TiO3"]["sintering_temp"] is None

    def test_abbreviations_kept_for_extraction_only(self, branched_schema):
        extraction = build(branched_schema, **CREW_MODELS["composition_extraction"])
        formatted = build(branched_schema, **CREW_MODELS["composition_format"])
        assert "abbreviations" in extraction().composition_extracted_data.model_dump()
        assert "abbreviations" not in formatted().composition_formatted_data.model_dump()

    def test_extra_synthesis_field(self, branched_schema):
        model = build(branched_schema, **CREW_MODELS["synthesis_format"])
        assert model().model_dump() == {
            "synthesis_formatted_data": {"method": "", "sintering_atmosphere": ""}
        }

    def test_nested_objects_accept_null(self, branched_schema):
        model = build(branched_schema, **CREW_MODELS["composition_format"])
        record = {
            "composition_formatted_data": {
                "compositions_property_values": {"BaTiO3": {"d33_data": None}}
            }
        }
        dumped = model.model_validate(record).model_dump()
        assert (
            dumped["composition_formatted_data"]["compositions_property_values"][
                "BaTiO3"
            ]["d33_data"]
            is None
        )


class TestPromptRendering:
    def test_extraction_description_lists_every_field(self):
        text = render_fields_description(
            DEFAULT_SCHEMA_CONFIG.fields("composition"), stage="extraction"
        )
        for name in DEFAULT_SCHEMA_CONFIG.field_names("composition", final_only=False):
            assert f"- {name} (" in text
        # Formatting rules only in format stage
        assert "Formatting rule" not in text

    def test_format_description_shows_rules_without_empty_allowed_values(self):
        text = render_fields_description(
            DEFAULT_SCHEMA_CONFIG.fields("synthesis", final_only=True), stage="format"
        )
        assert "Formatting rule: Keep the steps unmodified." in text
        # The default schema lists no allowed values and no runtime placeholders
        assert "Allowed values" not in text
        assert "{allowed_" not in text

    def test_nested_fields_are_indented(self):
        schema = ExtractionSchemaConfig.from_dict(
            {
                "version": "1.0",
                "composition_fields": [
                    {
                        "name": "compositions_property_values",
                        "type": "dict[str, object]",
                        "value_fields": [
                            {
                                "name": "d33_data",
                                "type": "object",
                                "value_fields": [{"name": "value", "type": "float|None"}],
                            }
                        ],
                    }
                ],
                "synthesis_fields": [
                    {
                        "name": "sintering_atmosphere",
                        "type": "str",
                        "allowed_values": ["air", "oxygen"],
                    }
                ],
            }
        )
        text = render_fields_description(schema.fields("composition"))
        assert "\n    - d33_data (object, required)" in text
        assert "\n        - value (float|None, required)" in text
        assert "Allowed values:\n- air\n- oxygen" in render_fields_description(
            schema.fields("synthesis"), stage="format"
        )
        example = build_example(schema.fields("composition", final_only=True))
        assert example == {
            "compositions_property_values": {"<key>": {"d33_data": {"value": 0}}}
        }

    def test_default_examples_use_schema_examples(self):
        fields = DEFAULT_SCHEMA_CONFIG.fields("composition", final_only=True)
        assert build_example(fields)["family"] == "BaTiO3"
        assert build_example(fields, variant="variable")["family"] == "NaNbO3"
        assert "0.5NaNbO3" in build_example(fields, variant="variable")[
            "compositions_property_values"
        ]
        assert "abbreviations" not in build_example(fields)

    def test_fill_schema_placeholders_leaves_other_placeholders(self):
        config = {
            "description": "Fields:\n{schema_fields_description}\nData: {extracted_synthesis_data}",
            "agent": "synthesis_formatter",
        }
        filled = fill_schema_placeholders(
            config, schema_prompt_values(DEFAULT_SCHEMA_CONFIG.fields("synthesis"), "format")
        )
        assert "{schema_fields_description}" not in filled["description"]
        assert "{extracted_synthesis_data}" in filled["description"]
        # Original config not mutated
        assert "{schema_fields_description}" in config["description"]

    def test_rendered_example_has_no_crewai_placeholders(self):
        import re

        for domain in ("composition", "synthesis"):
            values = schema_prompt_values(DEFAULT_SCHEMA_CONFIG.fields(domain), "format")
            assert not re.search(r"\{[A-Za-z_][A-Za-z0-9_\-]*\}", values["schema_example_json"])


EVALUATION_SNAPSHOT = json.loads(
    (
        Path(__file__).parents[1]
        / "test_post_processing"
        / "test_evaluation"
        / "fixtures"
        / "legacy_evaluation_models_snapshot.json"
    ).read_text(encoding="utf-8")
)


def sort_required(schema):
    """Sort 'required' lists, since field order follows the schema rather than the legacy models"""
    if isinstance(schema, dict):
        return {
            k: sorted(v) if k == "required" else sort_required(v)
            for k, v in schema.items()
        }
    return schema


class TestEvaluationModels:
    @pytest.mark.parametrize("domain", ["composition", "synthesis"])
    def test_default_models_match_legacy_apart_from_field_order(self, domain):
        model = build_evaluation_model(
            DEFAULT_SCHEMA_CONFIG.fields(domain, final_only=True), domain
        )
        assert sort_required(model.model_json_schema()) == sort_required(
            EVALUATION_SNAPSHOT[f"{domain}_evaluation"]["model_json_schema"]
        )

    def test_default_criteria_use_legacy_wording(self):
        text = render_evaluation_criteria(
            DEFAULT_SCHEMA_CONFIG.fields("composition", final_only=True), "composition"
        )
        assert "For 'property_unit': Decide if the units match exactly (1) or not (0)" in text
        assert "{value_error_threshold_instructions}" in text
        assert "BE STRICT about composition matching" in text

    def test_branched_model_accepts_leaf_matches(self):
        schema = ExtractionSchemaConfig.from_dict(
            {
                "version": "1.0",
                "composition_fields": [
                    {
                        "name": "compositions_property_values",
                        "type": "dict[str, object]",
                        "value_fields": [{"name": "d33_value", "type": "float|None"}],
                    }
                ],
                "synthesis_fields": [{"name": "method", "type": "str"}],
            }
        )
        model = build_evaluation_model(
            schema.fields("composition", final_only=True), "composition"
        )
        response = {
            "composition_data": {
                "compositions_property_values": {
                    "value_matches": [
                        {
                            "reference_key": "BaTiO3",
                            "test_key": "BaTiO3",
                            "leaf": "d33_value",
                            "reference_value": 190,
                            "test_value": 190,
                            "match_value": 1,
                        }
                    ]
                }
            }
        }
        parsed = model.model_validate(response).model_dump()
        entry = parsed["composition_data"]["compositions_property_values"]["value_matches"][0]
        assert entry["leaf"] == "d33_value"
        output_format = render_evaluation_output_format(
            schema.fields("composition", final_only=True), "composition"
        )
        assert '"leaf": "d33_value"' in output_format
