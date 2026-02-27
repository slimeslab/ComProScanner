"""
test_rag_tool.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 27-02-2026
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _DummyBaseTool:
    pass


if "crewai.tools" in sys.modules:
    sys.modules["crewai.tools"].BaseTool = _DummyBaseTool


from comproscanner.extract_flow.tools.rag_tool import RAGTool
from comproscanner.utils.configs.rag_config import RAGConfig


def _make_tool():
    """Create a RAGTool with VectorDatabaseManager fully mocked out."""
    with patch(
        "comproscanner.extract_flow.tools.rag_tool.VectorDatabaseManager"
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        tool = RAGTool(rag_config=None)
    # tool._vector_db_manager is the MagicMock instance returned above;
    # it outlives the patch context and can be configured per test.
    return tool


class _MockDoc:
    def __init__(self, content):
        self.page_content = content


# ---------------------------------------------------------------------------
# _format_documents
# ---------------------------------------------------------------------------


def test_format_documents_numbers_each_document():
    tool = _make_tool()
    docs = [(_MockDoc("text about BaTiO3"), 0.95), (_MockDoc("more context"), 0.82)]

    result = tool._format_documents(docs)

    assert "Document 1" in result
    assert "Document 2" in result


def test_format_documents_includes_page_content():
    tool = _make_tool()
    docs = [(_MockDoc("unique-content-xyz"), 0.9)]

    result = tool._format_documents(docs)

    assert "unique-content-xyz" in result


def test_format_documents_includes_relevance_scores():
    tool = _make_tool()
    docs = [(_MockDoc("doc"), 0.9500)]

    result = tool._format_documents(docs)

    assert "0.9500" in result


# ---------------------------------------------------------------------------
# _run — happy path
# ---------------------------------------------------------------------------


def test_run_generates_response_when_docs_found(monkeypatch):
    tool = _make_tool()
    docs = [(_MockDoc("BaTiO3 has d33 of 193 pC/N"), 0.9)]
    tool._vector_db_manager.query_database.return_value = docs

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = SimpleNamespace(content="d33 is 193 pC/N")
    monkeypatch.setattr(tool, "_get_llm", lambda: mock_llm)

    result = tool._run(doi="10.1000/ok", query="what is d33?")

    assert result == "d33 is 193 pC/N"
    mock_llm.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# _run — edge / error cases
# ---------------------------------------------------------------------------


def test_run_returns_no_docs_message_when_db_is_empty():
    tool = _make_tool()
    tool._vector_db_manager.query_database.return_value = []

    result = tool._run(doi="10.1000/empty", query="any query")

    assert result == "No relevant documents found"


def test_run_returns_error_message_on_db_exception():
    tool = _make_tool()
    tool._vector_db_manager.query_database.side_effect = RuntimeError("DB unavailable")

    result = tool._run(doi="10.1000/err", query="any query")

    assert "Error occurred" in result
    assert "DB unavailable" in result


# ---------------------------------------------------------------------------
# _get_llm — model routing
# ---------------------------------------------------------------------------


def test_get_llm_raises_for_unsupported_model_prefix():
    tool = _make_tool()
    tool.rag_config = RAGConfig(rag_chat_model="unsupported-xyz-9000")

    with pytest.raises(Exception) as exc_info:
        tool._get_llm()

    msg = str(exc_info.value).lower()
    assert "unsupported" in msg or "unrecognized" in msg
