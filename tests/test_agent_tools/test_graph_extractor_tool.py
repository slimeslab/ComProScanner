"""
test_graph_extractor_tool.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 27-02-2026
"""

import io
import json
import sys
from types import SimpleNamespace


class _DummyBaseTool:
    pass


if "crewai.tools" in sys.modules:
    sys.modules["crewai.tools"].BaseTool = _DummyBaseTool

from comproscanner.extract_flow.tools.graph_extractor_tool import GraphExtractorTool


def _make_tool(base_path="results/related_figures", vlm_property_name="the target property"):
    tool = GraphExtractorTool()
    tool.related_figures_base_path = base_path
    tool.vlm_property_name = vlm_property_name
    return tool


def _install_fake_litellm(response_content):
    sys.modules["litellm"] = SimpleNamespace(
        completion=lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_content))]
        )
    )


def test_run_returns_message_when_figure_directory_missing(monkeypatch):
    tool = _make_tool(base_path="fake/base")
    doi = "10.1000/missing"

    monkeypatch.setattr("os.path.isdir", lambda _: False)
    result = tool._run(doi)

    assert "No saved figures found for DOI" in result
    assert "10.1000/missing" in result


def test_run_returns_message_when_no_jpg_files(monkeypatch):
    tool = _make_tool(base_path="fake/base")
    doi = "10.1000/no-jpg"

    monkeypatch.setattr("os.path.isdir", lambda _: True)
    monkeypatch.setattr("os.path.isfile", lambda path: path.endswith("info.json"))
    monkeypatch.setattr("os.listdir", lambda _: ["info.json", "figure.png"])

    def _open(path, mode="r", encoding=None):
        if path.endswith("info.json"):
            return io.StringIO(json.dumps({"fig1": "d33 vs composition"}))
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", _open)
    result = tool._run(doi)

    assert "No .jpg figures found" in result
    assert "Captions available" in result


def test_run_returns_error_when_litellm_not_installed(monkeypatch):
    tool = _make_tool(base_path="fake/base")
    doi = "10.1000/no-litellm"

    monkeypatch.setattr("os.path.isdir", lambda _: True)
    monkeypatch.setattr("os.path.isfile", lambda path: path.endswith("info.json"))
    monkeypatch.setattr("os.listdir", lambda _: ["fig1.jpg"])

    def _open(path, mode="r", encoding=None):
        if path.endswith("info.json"):
            return io.StringIO(json.dumps({"fig1": "d33 vs composition"}))
        if path.endswith("fig1.jpg"):
            return io.BytesIO(b"fake-jpeg-content")
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", _open)

    original_import = __import__

    def _mock_import(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("litellm missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _mock_import)
    result = tool._run(doi)

    assert result == "litellm is not installed; graph extraction requires litellm."


def test_run_extracts_data_and_parses_markdown_json(monkeypatch):
    tool = _make_tool(
        base_path="fake/base",
        vlm_property_name="d33",
    )
    doi = "10.1000/ok"

    monkeypatch.setattr("os.path.isdir", lambda _: True)
    monkeypatch.setattr("os.path.isfile", lambda path: path.endswith("info.json"))
    monkeypatch.setattr("os.listdir", lambda _: ["fig1.jpg"])

    def _open(path, mode="r", encoding=None):
        if path.endswith("info.json"):
            return io.StringIO(json.dumps({"fig1": "d33 vs composition"}))
        if path.endswith("fig1.jpg"):
            return io.BytesIO(b"fake-jpeg-content")
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", _open)
    _install_fake_litellm(
        '```json\n{"data_points":[{"composition":"BaTiO3","value":193,"unit":"pC/N","series":"main"}]}\n```'
    )

    result = tool._run(doi)
    parsed = json.loads(result)
    assert parsed["fig1"]["caption"] == "d33 vs composition"
    assert parsed["fig1"]["extracted_data"]["data_points"][0]["composition"] == "BaTiO3"
    assert parsed["fig1"]["extracted_data"]["data_points"][0]["value"] == 193
