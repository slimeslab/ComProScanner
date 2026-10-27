"""
test_prompt_configs.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24

Tests for the instructions and tools configs and the deprecated arguments they replace.
"""

import hashlib
import json

import pytest

from comproscanner.utils.configs.instructions_config import (
    LEGACY_INSTRUCTION_ARGS,
    apply_legacy_instruction_args,
    load_instructions_config,
)
from comproscanner.utils.configs.schema_config import (
    DEFAULT_SCHEMA_CONFIG,
    ExtractionSchemaConfig,
    SchemaValidationError,
    apply_legacy_allowed_values,
)
from comproscanner.utils.configs.tools_config import (
    apply_legacy_tools_args,
    load_tools_config,
    render_prompt,
)
from comproscanner.utils.configs.yaml_config import ConfigFileError
from comproscanner.utils.schema_tools import render_fields_description
from comproscanner.extract_flow.crews.composition_crew.composition_extraction_crew.composition_extraction_crew import (
    tool_guidance_values,
)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# Prompts as they were hard-coded before the tools config existed
def old_relevance_prompt(caption_text, keyword):
    return (
        f"You are a materials science expert.\n"
        f'Figure caption: "{caption_text}"\n\n'
        f"Examine this scientific figure. Does it contain {keyword} property data "
        f"for ceramic, composite, doped, or crystal materials (excluding polymers like PVDF)?\n"
        f"Note: In graphs, relevant data may appear as doping concentrations (e.g. x values, "
        f"mol%, at%) on one axis with {keyword} values on the other axis — this counts as "
        f"relevant data even without full chemical formulas.\n"
        f"Answer with a single word only: 'yes' or 'no'."
    )


def old_graph_prompt(caption_text, property_name):
    return (
        f"You are a materials science data extraction assistant.\n"
        f"Analyse the scientific graph shown in the image.\n"
        f'Figure caption: "{caption_text}"\n\n'
        f"Extract all (composition, {property_name}) data points visible "
        f"in the graph, including any data series labels or legend entries.\n"
        f"Accuracy requirement: estimate each point from its plotted position using axis ticks/scales.\n"
        f"Use interpolation between neighboring ticks/gridlines for points between labels.\n"
        f"Do NOT snap values to 'nice numbers' (e.g., multiples of 5 or 10) unless the point clearly lies exactly there.\n"
        f"If a point appears near 193, output 193 (not 195). If near 197, output 197 (not 195 or 200).\n"
        f"Output integers only (no decimals), but they must be the closest whole-number estimate from the visual position.\n"
        f"When uncertain between two adjacent integers, choose the one closer to the marker center.\n"
        f"Return your answer as valid JSON in this exact format:\n"
        f'{{"data_points": [{{"composition": "...", "value": <integer>, '
        f'"unit": "...", "series": "..."}}]}}\n'
        f"If no numeric data is extractable, return: "
        f'{{"data_points": []}}'
    )


OLD_FORMULA_INSTRUCTION_SHA256 = "1066701962f8fbeeb57f38a38c83bfdaf80ae1a9676ecab66d11f7a5eb33ec72"
OLD_CAPTION_KEYWORDS = {
    "xrd", "x-ray diffraction", "x-ray", "diffraction", "sem", "scanning electron", "tem",
    "transmission electron", "crystal structure", "microstructure", "rietveld", "bragg",
    "crystal phase", "phase purity", "phase identification",
}


class TestToolsConfig:
    def test_defaults_reproduce_previous_prompts(self):
        config = load_tools_config()
        caption, keyword = 'XRD patterns of "BaTiO3"', "d33"
        assert render_prompt(
            config.graph_extractor.relevance_instruction, keyword=keyword, caption=caption
        ) == old_relevance_prompt(caption, keyword)
        assert render_prompt(
            config.graph_extractor.extraction_instruction, keyword=keyword, caption=caption
        ) == old_graph_prompt(caption, keyword)
        instruction = config.equation_tool.instruction.encode()
        assert hashlib.sha256(instruction).hexdigest() == OLD_FORMULA_INSTRUCTION_SHA256
        assert set(config.equation_tool.figure_caption_keywords) == OLD_CAPTION_KEYWORDS
        assert config.rag_tool.system_prompt.startswith("You are a helpful assistant specializing")
        assert config.rag_tool.answer_instruction == "Please provide a suitable answer to the provided question."
        assert config.equation_tool.enabled and config.graph_extractor.enabled
        assert config.user_keys == set()

    def test_partial_file_overrides_only_given_keys(self, tmp_path):
        path = write(tmp_path, "tools.yaml", "equation_tool:\n  enabled: false\nrag_tool:\n  system_prompt: Be brief.\n")
        config = load_tools_config(path)
        default = load_tools_config()
        assert config.equation_tool.enabled is False
        assert config.equation_tool.instruction == default.equation_tool.instruction
        assert config.rag_tool.system_prompt == "Be brief."
        assert config.rag_tool.answer_instruction == default.rag_tool.answer_instruction
        assert config.user_keys == {"equation_tool.enabled", "rag_tool.system_prompt"}

    @pytest.mark.parametrize(
        "text, message",
        [
            ("equation_tool:\n  instrction: x\n", "equation_tool.instrction"),
            ("rag:\n  system_prompt: x\n", "unknown top-level key 'rag'"),
            ("version: '2.0'\n", "unsupported config version"),
            ("graph_extractor:\n  relevance_instruction: 'Is {property} shown?'\n", "['property']"),
            ("equation_tool:\n  input_guidance: 'Mention {dopant}.'\n", "['dopant']"),
            ("- not a mapping\n", "must be a YAML mapping"),
        ],
    )
    def test_invalid_files_are_rejected(self, tmp_path, text, message):
        path = write(tmp_path, "tools.yaml", text)
        with pytest.raises(ConfigFileError, match=message.replace("[", r"\[").replace("]", r"\]")):
            load_tools_config(path)

    def test_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(ConfigFileError, match="not found"):
            load_tools_config(tmp_path / "missing.yaml")

    def test_render_prompt_leaves_other_braces(self):
        text = 'Find {keyword} in "{caption}". Reply {"points": []} or {other}.'
        assert render_prompt(text, keyword="d33", caption="Fig. 1") == (
            'Find d33 in "Fig. 1". Reply {"points": []} or {other}.'
        )

    def test_legacy_formula_instruction(self, tmp_path):
        config, messages = apply_legacy_tools_args(load_tools_config(), None)
        assert messages == []
        config, messages = apply_legacy_tools_args(load_tools_config(), "Custom.")
        assert config.equation_tool.instruction == "Custom."
        assert "deprecated" in messages[0]
        # The shared default config is not modified
        assert load_tools_config().equation_tool.instruction != "Custom."

        path = write(tmp_path, "tools.yaml", "equation_tool:\n  instruction: From file.\n")
        config, messages = apply_legacy_tools_args(load_tools_config(path), "Custom.")
        assert config.equation_tool.instruction == "From file."
        assert "ignored" in messages[0]


class TestInstructionsConfig:
    def test_defaults(self):
        config = load_instructions_config()
        assert len(config.composition_extraction.formula_notation_rules) == 4
        assert config.composition_extraction.agent_notes == []
        assert config.examples.composition == ""

    def test_file_values(self, tmp_path):
        path = write(
            tmp_path,
            "instructions.yaml",
            "examples:\n"
            "  composition:\n"
            "    compositions_property_values: {BaTiO3: 375}\n"
            "    family: BaTiO3\n"
            "composition_extraction:\n"
            "  task_notes:\n"
            "  formula_notation_rules: []\n"
            "synthesis_formatting:\n"
            "  agent_notes: [Use short names.]\n",
        )
        config = load_instructions_config(path)
        assert json.loads(config.examples.composition) == {
            "compositions_property_values": {"BaTiO3": 375},
            "family": "BaTiO3",
        }
        assert config.composition_extraction.task_notes == []
        assert config.composition_extraction.formula_notation_rules == []
        assert config.synthesis_formatting.agent_notes == ["Use short names."]

    def test_rules_with_placeholders_are_rejected(self, tmp_path):
        path = write(tmp_path, "instructions.yaml", "composition_extraction:\n  formula_notation_rules: ['Use {x}.']\n")
        with pytest.raises(ConfigFileError, match="formula_notation_rules"):
            load_instructions_config(path)

    def test_notes_may_contain_braces(self, tmp_path):
        path = write(tmp_path, "instructions.yaml", "synthesis_extraction:\n  task_notes: ['Return {\"method\": ...}']\n")
        assert load_instructions_config(path).synthesis_extraction.task_notes == ['Return {"method": ...}']

    def test_every_legacy_argument_is_mapped(self):
        legacy = {arg: [arg] if "notes" in arg else arg for arg in LEGACY_INSTRUCTION_ARGS}
        config, messages = apply_legacy_instruction_args(load_instructions_config(), legacy)
        for arg, target in LEGACY_INSTRUCTION_ARGS.items():
            section, key = target.split(".")
            assert getattr(getattr(config, section), key) == legacy[arg]
        assert len(messages) == 1 and "deprecated" in messages[0]

    def test_file_wins_over_legacy_argument(self, tmp_path):
        path = write(tmp_path, "instructions.yaml", "synthesis_extraction:\n  task_notes: [From file.]\n")
        config, messages = apply_legacy_instruction_args(
            load_instructions_config(path),
            {
                "synthesis_extraction_task_notes": ["From argument."],
                "synthesis_formatting_task_notes": ["Kept."],
            },
        )
        assert config.synthesis_extraction.task_notes == ["From file."]
        assert config.synthesis_formatting.task_notes == ["Kept."]
        assert "synthesis_extraction_task_notes" in messages[1]


def ref_schema(**method_extra):
    return ExtractionSchemaConfig.from_dict(
        {
            "version": "1.0",
            "composition_fields": [{"name": "compositions_property_values", "type": "dict[str, float|None]"}],
            "synthesis_fields": [
                {"name": "route", "type": "str", "allowed_values_ref": "allowed_synthesis_methods"},
                {"name": "method", "type": "str", **method_extra},
            ],
        }
    )


class TestLegacyAllowedValues:
    def test_lists_fill_default_fields_with_the_previous_rendering(self):
        schema, messages = apply_legacy_allowed_values(
            DEFAULT_SCHEMA_CONFIG,
            {"allowed_synthesis_methods": ["Sol-gel", "Hydrothermal"], "allowed_characterization_techniques": ["XRD"]},
        )
        text = render_fields_description(schema.fields("synthesis", final_only=True), stage="format")
        assert "  Allowed values:\n- Sol-gel\n- Hydrothermal" in text
        assert "  Allowed values:\n- XRD" in text
        assert len(messages) == 2
        # The shared default schema is not modified
        assert all(not f.allowed_values for f in DEFAULT_SCHEMA_CONFIG.synthesis_fields)

    def test_reference_is_deprecated_and_takes_precedence(self):
        with pytest.warns(DeprecationWarning, match="allowed_values_ref is deprecated"):
            schema = ref_schema()
        schema, _ = apply_legacy_allowed_values(schema, {"allowed_synthesis_methods": ["Sol-gel"]})
        fields = {f.name: f for f in schema.synthesis_fields}
        assert fields["route"].allowed_values == ["Sol-gel"]
        assert fields["method"].allowed_values is None

    def test_inline_values_win(self):
        schema = ExtractionSchemaConfig.from_dict(
            {
                "version": "1.0",
                "composition_fields": [{"name": "compositions_property_values", "type": "dict[str, float|None]"}],
                "synthesis_fields": [{"name": "method", "type": "str", "allowed_values": ["Sol-gel"]}],
            }
        )
        schema, messages = apply_legacy_allowed_values(schema, {"allowed_synthesis_methods": ["Other"]})
        assert schema.synthesis_fields[0].allowed_values == ["Sol-gel"]
        assert "ignored" in messages[0]

    def test_missing_field_raises(self):
        with pytest.raises(SchemaValidationError, match="characterization_techniques"):
            apply_legacy_allowed_values(
                ExtractionSchemaConfig.from_dict(
                    {
                        "version": "1.0",
                        "composition_fields": [{"name": "compositions_property_values", "type": "dict[str, float|None]"}],
                        "synthesis_fields": [{"name": "method", "type": "str"}],
                    }
                ),
                {"allowed_characterization_techniques": ["XRD"]},
            )


class TestToolGuidance:
    def test_enabled_tools_add_their_lines(self):
        values = tool_guidance_values(load_tools_config())
        assert values["equation_tool_agent_guidance"].startswith("You MUST first call the Equation Tool with doi={doi}.")
        assert load_tools_config().equation_tool.input_guidance in values["equation_tool_task_guidance"]
        assert "{equation_tool_input_guidance}" not in "".join(values.values())
        assert values["graph_extractor_task_guidance"].startswith("- You MUST invoke the Graph Data Extractor")

    def test_disabled_tools_add_nothing(self, tmp_path):
        path = write(tmp_path, "tools.yaml", "equation_tool:\n  enabled: false\ngraph_extractor:\n  enabled: false\n")
        values = tool_guidance_values(load_tools_config(path))
        assert set(values.values()) == {""}
