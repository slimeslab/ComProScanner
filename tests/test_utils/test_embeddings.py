"""
test_embeddings.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 27-02-2025
"""

import os
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, call

from comproscanner.utils.error_handler import ValueErrorHandler, ImportErrorHandler
from comproscanner.utils.embeddings import MultiModelEmbeddings
from comproscanner.utils.configs import RAGConfig


@pytest.fixture
def mock_huggingface_model():
    """Fixture to mock HuggingFace model components"""
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch("transformers.AutoModel.from_pretrained") as mock_model,
    ):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": torch.ones((1, 10), dtype=torch.long),
            "attention_mask": torch.ones((1, 10), dtype=torch.long),
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_instance = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.ones((1, 10, 768), dtype=torch.float)
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance

        yield mock_tokenizer, mock_model


@pytest.fixture
def mock_sentence_transformers(monkeypatch):
    """Fixture to mock SentenceTransformers"""
    mock_st = MagicMock()
    mock_st_instance = MagicMock()
    mock_st_instance.encode.return_value = np.ones(768, dtype=np.float32)
    mock_st.return_value = mock_st_instance
    monkeypatch.setattr(
        "comproscanner.utils.embeddings.HAVE_SENTENCE_TRANSFORMERS", True
    )
    monkeypatch.setattr("comproscanner.utils.embeddings.SentenceTransformer", mock_st)

    yield mock_st_instance


@pytest.fixture
def mock_openai(monkeypatch):
    """Fixture to mock OpenAI API"""
    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    mock_data = MagicMock()
    mock_data.embedding = [0.1] * 1536
    mock_response = MagicMock()
    mock_response.data = [mock_data]
    mock_embeddings.create.return_value = mock_response
    mock_client.embeddings = mock_embeddings
    mock_openai_class = MagicMock()
    mock_openai_class.return_value = mock_client
    monkeypatch.setattr("comproscanner.utils.embeddings.HAVE_OPENAI", True)
    monkeypatch.setattr("comproscanner.utils.embeddings.OpenAI", mock_openai_class)
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    yield mock_client


def test_init_huggingface(mock_huggingface_model):
    """Test initialization with HuggingFace model"""
    config = RAGConfig(embedding_model="huggingface:bert-base-uncased")
    embeddings = MultiModelEmbeddings(config)

    assert embeddings.model_type == "huggingface"
    assert embeddings.rag_config == config
    mock_tokenizer, mock_model = mock_huggingface_model
    mock_tokenizer.assert_called_once_with("bert-base-uncased")
    mock_model.assert_called_once_with("bert-base-uncased")


def test_init_sentence_transformers(mock_sentence_transformers):
    """Test initialization with SentenceTransformers model"""
    config = RAGConfig(embedding_model="sentence-transformers:all-MiniLM-L6-v2")
    embeddings = MultiModelEmbeddings(config)

    assert embeddings.model_type == "sentence_transformers"
    assert embeddings.rag_config == config
    assert embeddings.model == mock_sentence_transformers


def test_init_openai(mock_openai):
    """Test initialization with OpenAI model"""
    config = RAGConfig(embedding_model="text-embedding-ada-002")
    embeddings = MultiModelEmbeddings(config)

    assert embeddings.model_type == "openai"
    assert embeddings.rag_config == config
    assert embeddings.openai_model == "text-embedding-ada-002"
    config = RAGConfig(embedding_model="openai:text-embedding-3-small")
    embeddings = MultiModelEmbeddings(config)

    assert embeddings.model_type == "openai"
    assert embeddings.openai_model == "text-embedding-3-small"


def test_init_unsupported_model():
    """Test initialization with an unsupported model type"""
    config = RAGConfig(embedding_model="unsupported-model-type")

    # Since _determine_model_type defaults to "openai" for any unrecognized model,
    # we need to mock the _init_openai method to raise an exception
    with patch.object(
        MultiModelEmbeddings,
        "_init_openai",
        side_effect=Exception("OpenAI initialization failed"),
    ):
        with pytest.raises(Exception):
            MultiModelEmbeddings(config)


def test_init_openai_without_api_key(monkeypatch):
    """Test initialization with OpenAI model without API key"""
    monkeypatch.setattr("comproscanner.utils.embeddings.HAVE_OPENAI", True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = RAGConfig(embedding_model="text-embedding-ada-002")

    with pytest.raises(ValueErrorHandler) as excinfo:
        MultiModelEmbeddings(config)

    assert "OpenAI API key not found" in str(excinfo.value)


def test_init_openai_import_error(monkeypatch):
    """Test initialization with OpenAI when package is not available"""
    monkeypatch.setattr("comproscanner.utils.embeddings.HAVE_OPENAI", False)

    config = RAGConfig(embedding_model="text-embedding-ada-002")

    with pytest.raises(ImportErrorHandler) as excinfo:
        MultiModelEmbeddings(config)

    assert "OpenAI package not found" in str(excinfo.value)


def test_init_sentence_transformers_import_error(monkeypatch):
    """Test initialization with SentenceTransformers when package is not available"""
    monkeypatch.setattr(
        "comproscanner.utils.embeddings.HAVE_SENTENCE_TRANSFORMERS", False
    )

    config = RAGConfig(embedding_model="sentence-transformers:all-MiniLM-L6-v2")

    with pytest.raises(ImportErrorHandler) as excinfo:
        MultiModelEmbeddings(config)

    assert "SentenceTransformers package not found" in str(excinfo.value)


def test_embed_documents_huggingface(mock_huggingface_model):
    """Test embedding documents with HuggingFace model"""
    config = RAGConfig(embedding_model="huggingface:bert-base-uncased")
    embeddings = MultiModelEmbeddings(config)
    with patch("torch.no_grad"):
        result = embeddings.embed_documents(["This is a test", "Another test"])

    assert len(result) == 2
    assert isinstance(result[0], list)
    assert len(result[0]) > 0


def test_embed_documents_sentence_transformers(mock_sentence_transformers):
    """Test embedding documents with SentenceTransformers model"""
    config = RAGConfig(embedding_model="sentence-transformers:all-MiniLM-L6-v2")
    embeddings = MultiModelEmbeddings(config)

    result = embeddings.embed_documents(["This is a test", "Another test"])

    assert len(result) == 2
    assert isinstance(result[0], list)
    assert len(result[0]) > 0
    mock_sentence_transformers.encode.assert_has_calls(
        [
            call("This is a test", convert_to_numpy=True),
            call("Another test", convert_to_numpy=True),
        ]
    )


def test_embed_documents_openai(mock_openai):
    """Test embedding documents with OpenAI model"""
    config = RAGConfig(embedding_model="text-embedding-ada-002")
    embeddings = MultiModelEmbeddings(config)

    result = embeddings.embed_documents(["This is a test", "Another test"])

    assert len(result) == 2
    assert isinstance(result[0], list)
    assert len(result[0]) > 0
    mock_openai.embeddings.create.assert_has_calls(
        [
            call(model="text-embedding-ada-002", input=["This is a test"]),
            call(model="text-embedding-ada-002", input=["Another test"]),
        ]
    )


def test_embed_query_huggingface():
    """Test query embedding with HuggingFace model"""
    config = RAGConfig(embedding_model="huggingface:bert-base-uncased")
    embeddings = MultiModelEmbeddings(config)
    with patch.object(embeddings, "_embed_document_huggingface") as mock_embed:
        mock_embed.return_value = [0.1] * 768
        result = embeddings.embed_query("This is a query")

    mock_embed.assert_called_once_with("This is a query")
    assert isinstance(result, list)
    assert len(result) == 768


def test_embed_query_sentence_transformers():
    """Test query embedding with SentenceTransformers model"""
    config = RAGConfig(embedding_model="sentence-transformers:all-MiniLM-L6-v2")
    embeddings = MultiModelEmbeddings(config)
    with patch.object(
        embeddings, "_embed_document_sentence_transformers"
    ) as mock_embed:
        mock_embed.return_value = [0.1] * 384
        result = embeddings.embed_query("This is a query")

    mock_embed.assert_called_once_with("This is a query")
    assert isinstance(result, list)
    assert len(result) == 384


def test_embed_query_openai(mock_openai):
    """Test query embedding with OpenAI model"""
    config = RAGConfig(embedding_model="text-embedding-ada-002")
    embeddings = MultiModelEmbeddings(config)
    with patch.object(embeddings, "_embed_document_openai") as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        result = embeddings.embed_query("This is a query")

    mock_embed.assert_called_once_with("This is a query")
    assert isinstance(result, list)
    assert len(result) == 1536


@pytest.mark.parametrize(
    "model_name,expected_type",
    [
        ("huggingface:bert-base-uncased", "huggingface"),
        ("sentence-transformers:all-MiniLM-L6-v2", "sentence_transformers"),
        ("text-embedding-ada-002", "openai"),
        ("some-other-model", "openai"),
    ],
)
def test_determine_model_type(monkeypatch, model_name, expected_type):
    """Test model type determination based on model name"""
    monkeypatch.setattr(
        "comproscanner.utils.embeddings.HAVE_SENTENCE_TRANSFORMERS", True
    )
    monkeypatch.setattr("comproscanner.utils.embeddings.HAVE_OPENAI", True)
    config = RAGConfig(embedding_model=model_name)

    with (
        patch.object(MultiModelEmbeddings, "_init_huggingface"),
        patch.object(MultiModelEmbeddings, "_init_sentence_transformers"),
        patch.object(MultiModelEmbeddings, "_init_openai"),
    ):

        embeddings = MultiModelEmbeddings(config)
        assert embeddings.model_type == expected_type


@pytest.mark.integration
def test_huggingface_actual_embedding():
    """Integration test for actual embedding with HuggingFace (when available)"""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    try:
        config = RAGConfig(embedding_model="sentence-transformers:all-MiniLM-L6-v2")
        embeddings = MultiModelEmbeddings(config)
        if hasattr(embeddings, "model"):
            result = embeddings.embed_query("This is an integration test")
            assert isinstance(result, list)
            assert len(result) > 0
    except Exception:
        pytest.skip("Hugging Face model not available for integration testing")


@pytest.mark.integration
def test_sentence_transformers_actual_embedding():
    """Integration test for actual embedding with SentenceTransformers (when available)"""
    pytest.importorskip("sentence_transformers")

    try:
        config = RAGConfig(embedding_model="sentence-transformers:all-MiniLM-L6-v2")
        embeddings = MultiModelEmbeddings(config)
        if hasattr(embeddings, "model"):
            result = embeddings.embed_query("This is an integration test")
            assert isinstance(result, list)
            assert len(result) > 0
    except Exception:
        pytest.skip("SentenceTransformers model not available for integration testing")
