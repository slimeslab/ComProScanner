"""
test_equation_tool.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 17-04-2026
"""

import io
import json
import os
import sys
from types import SimpleNamespace


class _DummyBaseTool:
    pass


if "crewai.tools" in sys.modules:
    sys.modules["crewai.tools"].BaseTool = _DummyBaseTool

from comproscanner.extract_flow.tools.equation_tool import EquationTool


def _make_tool(base_path="results/related_figures"):
    tool = EquationTool()
    tool.related_figures_base_path = base_path
    return tool


def _install_fake_litellm(response_content):
    sys.modules["litellm"] = SimpleNamespace(
        completion=lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_content))]
        )
    )


# ---------------------------------------------------------------------------
# _select_model
# ---------------------------------------------------------------------------


def test_select_model_returns_anthropic_when_key_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
                "OPENROUTER_API_KEY", "TOGETHER_API_KEY", "COHERE_API_KEY",
                "FIREWORKS_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    tool = _make_tool()
    assert tool._select_model() == "anthropic/claude-sonnet-4-6"


def test_select_model_falls_back_to_gemini_when_anthropic_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    for var in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
                "TOGETHER_API_KEY", "COHERE_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    tool = _make_tool()
    assert tool._select_model() == "gemini/gemini-3-flash-preview"


def test_select_model_falls_back_to_openai(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    for var in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "TOGETHER_API_KEY",
                "COHERE_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    tool = _make_tool()
    assert tool._select_model() == "openai/gpt-5.4-mini"


def test_select_model_defaults_to_anthropic_when_no_key_set(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "TOGETHER_API_KEY",
                "COHERE_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    tool = _make_tool()
    assert tool._select_model() == "anthropic/claude-sonnet-4-6"


def test_select_model_prefers_explicit_equation_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    tool = _make_tool()
    tool.equation_model = "gemini/gemini-3-flash-preview"
    assert tool._select_model() == "gemini/gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# _get_crystal_structure_images
# ---------------------------------------------------------------------------


def test_get_crystal_structure_images_returns_empty_for_missing_dir(monkeypatch):
    monkeypatch.setattr("os.path.isdir", lambda _: False)
    tool = _make_tool(base_path="fake/base")
    result = tool._get_crystal_structure_images("10.1000/missing")
    assert result == []


def test_get_crystal_structure_images_skips_non_matching_captions(monkeypatch):
    monkeypatch.setattr("os.path.isdir", lambda _: True)
    monkeypatch.setattr("os.path.isfile", lambda path: path.endswith("info.json"))
    monkeypatch.setattr("os.listdir", lambda _: ["fig1.jpg"])

    def _open(path, mode="r", encoding=None):
        if path.endswith("info.json"):
            return io.StringIO(json.dumps({"fig1": "d33 vs doping concentration"}))
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", _open)

    tool = _make_tool(base_path="fake/base")
    result = tool._get_crystal_structure_images("10.1000/test")
    assert result == []


def test_get_crystal_structure_images_returns_xrd_figure(monkeypatch):
    monkeypatch.setattr("os.path.isdir", lambda _: True)
    monkeypatch.setattr("os.path.isfile", lambda path: path.endswith("info.json"))
    monkeypatch.setattr("os.listdir", lambda _: ["fig1.jpg"])

    def _open(path, mode="r", encoding=None):
        if path.endswith("info.json"):
            return io.StringIO(json.dumps({"fig1": "XRD patterns of sintered samples"}))
        if path.endswith("fig1.jpg"):
            return io.BytesIO(b"fake-jpeg")
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", _open)

    tool = _make_tool(base_path="fake/base")
    result = tool._get_crystal_structure_images("10.1000/test")
    assert len(result) == 1
    assert result[0]["caption"] == "XRD patterns of sintered samples"
    assert isinstance(result[0]["b64"], str)


def test_get_crystal_structure_images_skips_non_jpg(monkeypatch):
    monkeypatch.setattr("os.path.isdir", lambda _: True)
    monkeypatch.setattr("os.path.isfile", lambda path: path.endswith("info.json"))
    monkeypatch.setattr("os.listdir", lambda _: ["fig1.png", "fig2.tiff"])

    monkeypatch.setattr(
        "builtins.open",
        lambda path, mode="r", encoding=None: io.StringIO(json.dumps({})),
    )

    tool = _make_tool(base_path="fake/base")
    result = tool._get_crystal_structure_images("10.1000/test")
    assert result == []


# ---------------------------------------------------------------------------
# _run
# ---------------------------------------------------------------------------


def test_run_returns_error_when_litellm_not_installed(monkeypatch):
    monkeypatch.setattr("os.path.isdir", lambda _: False)

    original_import = __import__

    def _mock_import(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("litellm missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _mock_import)

    tool = _make_tool()
    result = tool._run(doi="10.1000/test", paper_text="some text")
    assert "litellm is not installed" in result


def test_run_returns_formula_for_single_phase(monkeypatch):
    monkeypatch.setattr("os.path.isdir", lambda _: False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    _install_fake_litellm("(Ba1-xCax)TiO3")

    tool = _make_tool()
    result = tool._run(doi="10.1000/ok", paper_text="BaTiO3 doped with Ca ...")
    assert result == "(Ba1-xCax)TiO3"


def test_run_returns_not_single_phase_string(monkeypatch):
    monkeypatch.setattr("os.path.isdir", lambda _: False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    _install_fake_litellm("single compound is not being synthesized")

    tool = _make_tool()
    result = tool._run(doi="10.1000/multi", paper_text="Two phase mixture ...")
    assert result == "single compound is not being synthesized"


def test_run_includes_crystal_structure_images_in_prompt(monkeypatch):
    monkeypatch.setattr("os.path.isdir", lambda _: True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setattr("os.path.isfile", lambda path: path.endswith("info.json"))
    monkeypatch.setattr("os.listdir", lambda _: ["fig1.jpg"])

    def _open(path, mode="r", encoding=None):
        if path.endswith("info.json"):
            return io.StringIO(json.dumps({"fig1": "XRD of doped BaTiO3"}))
        if path.endswith("fig1.jpg"):
            return io.BytesIO(b"fake-jpeg")
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", _open)

    captured = {}

    def _fake_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="(Ba1-xCax)TiO3"))]
        )

    sys.modules["litellm"] = SimpleNamespace(completion=_fake_completion)

    tool = _make_tool(base_path="fake/base")
    result = tool._run(doi="10.1000/img", paper_text="XRD confirms single phase ...")

    content = captured["messages"][0]["content"]
    types = [block["type"] for block in content]
    assert "image_url" in types
    assert result == "(Ba1-xCax)TiO3"


def test_run_uses_custom_formula_instruction(monkeypatch):
    monkeypatch.setattr("os.path.isdir", lambda _: False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

    captured = {}

    def _fake_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="AB2O4"))]
        )

    sys.modules["litellm"] = SimpleNamespace(completion=_fake_completion)

    tool = _make_tool()
    tool.formula_instruction = "Custom instruction for spinel."
    tool._run(doi="10.1000/custom", paper_text="spinel synthesis ...")

    first_text_block = captured["messages"][0]["content"][0]
    assert first_text_block["text"] == "Custom instruction for spinel."


def test_run_returns_config_error_when_no_model_source_set(monkeypatch):
    monkeypatch.setattr("os.path.isdir", lambda _: False)
    for var in (
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
        "TOGETHER_API_KEY",
        "COHERE_API_KEY",
        "FIREWORKS_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    _install_fake_litellm("(Ba1-xCax)TiO3")
    tool = _make_tool()
    result = tool._run(doi="10.1000/nokey", paper_text="test")
    assert "No EquationTool model/provider is configured" in result
