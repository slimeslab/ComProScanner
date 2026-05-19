"""
test_material_parser_tool.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 27-02-2026
"""

import json
import sys


class _DummyBaseTool:
    pass


if "crewai.tools" in sys.modules:
    sys.modules["crewai.tools"].BaseTool = _DummyBaseTool


from comproscanner.extract_flow.tools.material_parser_tool import MaterialParserTool


def _make_tool():
    return MaterialParserTool()


def _api_response(status_code=200, raw_value="BaTiO3"):
    """Build a minimal mock requests.Response."""

    class MockResponse:
        def __init__(self):
            self.status_code = status_code

        def json(self):
            if not raw_value:
                return [[{"resolvedFormulas": []}]]
            return [[{"resolvedFormulas": [{"rawValue": raw_value}]}]]

    return MockResponse()


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def test_run_resolves_formula_from_dict_input(monkeypatch):
    tool = _make_tool()
    data = {
        "compositions": {"BaTiO3 x=0.1": 150},
        "property_unit": "pC/N",
        "family": "perovskite",
    }
    monkeypatch.setattr(
        "requests.post", lambda url, files: _api_response(200, "BaTiO3")
    )

    result = tool._run(data)

    assert "BaTiO3" in result["compositions"]
    assert result["property_unit"] == "pC/N"
    assert result["family"] == "perovskite"


def test_run_uses_compositions_property_values_field(monkeypatch):
    tool = _make_tool()
    data = {"compositions_property_values": {"PbTiO3": 200}, "property_unit": "pm/V"}
    monkeypatch.setattr(
        "requests.post", lambda url, files: _api_response(200, "PbTiO3")
    )

    result = tool._run(data)

    assert "PbTiO3" in result["compositions"]


def test_run_parses_json_string_input(monkeypatch):
    tool = _make_tool()
    data = json.dumps({"compositions": {"KNbO3": 90}, "property_unit": "pC/N"})
    monkeypatch.setattr("requests.post", lambda url, files: _api_response(200, "KNbO3"))

    result = tool._run(data)

    assert "KNbO3" in result["compositions"]


def test_run_handles_double_escaped_json_string(monkeypatch):
    tool = _make_tool()
    # Simulate the double-escaped string that LLMs sometimes produce
    data = '{"compositions": {\\"SrTiO3\\": 60}}'
    monkeypatch.setattr(
        "requests.post", lambda url, files: _api_response(200, "SrTiO3")
    )

    result = tool._run(data)

    assert "compositions" in result


def test_run_handles_description_field_wrapping(monkeypatch):
    tool = _make_tool()
    inner = json.dumps({"compositions": {"BiFeO3": 55}})
    data = {"description": inner}
    monkeypatch.setattr(
        "requests.post", lambda url, files: _api_response(200, "BiFeO3")
    )

    result = tool._run(data)

    assert "BiFeO3" in result["compositions"]


def test_run_returns_error_on_invalid_json_string():
    tool = _make_tool()

    result = tool._run("this is definitely not json!!")

    assert "error" in result


def test_run_extracts_unit_from_property_unit_field(monkeypatch):
    tool = _make_tool()
    data = {
        "compositions": {"PbZrTiO3": 300},
        "property_unit": "pC/N",
    }
    monkeypatch.setattr(
        "requests.post", lambda url, files: _api_response(200, "PbZrTiO3")
    )

    result = tool._run(data)

    assert result["property_unit"] == "pC/N"


def test_run_ignores_template_literal_unit_key(monkeypatch):
    tool = _make_tool()
    data = {
        "compositions": {"PbZrTiO3": 300},
        "{composition_property_text_data}_unit": "pC/N",
    }
    monkeypatch.setattr(
        "requests.post", lambda url, files: _api_response(200, "PbZrTiO3")
    )

    result = tool._run(data)

    assert result["property_unit"] == ""


# ---------------------------------------------------------------------------
# API failure / degraded responses
# ---------------------------------------------------------------------------


def test_run_keeps_original_formula_on_api_non_200(monkeypatch):
    tool = _make_tool()
    data = {"compositions": {"BadFormula": 99}}

    class _FailResponse:
        status_code = 500

    monkeypatch.setattr("requests.post", lambda url, files: _FailResponse())

    result = tool._run(data)

    assert "BadFormula" in result["compositions"]


def test_run_keeps_original_formula_on_empty_api_response(monkeypatch):
    tool = _make_tool()
    data = {"compositions": {"NaNbO3": 75}}

    class _EmptyResponse:
        status_code = 200

        def json(self):
            return []

    monkeypatch.setattr("requests.post", lambda url, files: _EmptyResponse())

    result = tool._run(data)

    assert "NaNbO3" in result["compositions"]


def test_run_keeps_original_formula_when_resolved_formulas_empty(monkeypatch):
    tool = _make_tool()
    data = {"compositions": {"PZT": 300}}
    # raw_value="" triggers the branch that returns {"resolvedFormulas": []}
    monkeypatch.setattr("requests.post", lambda url, files: _api_response(200, ""))

    result = tool._run(data)

    assert "PZT" in result["compositions"]


def test_run_keeps_original_formula_when_raw_value_missing(monkeypatch):
    tool = _make_tool()
    data = {"compositions": {"PVDF": 20}}

    class _NoRawValueResponse:
        status_code = 200

        def json(self):
            # resolvedFormulas present but rawValue key absent
            return [[{"resolvedFormulas": [{"otherKey": "x"}]}]]

    monkeypatch.setattr("requests.post", lambda url, files: _NoRawValueResponse())

    result = tool._run(data)

    assert "PVDF" in result["compositions"]
