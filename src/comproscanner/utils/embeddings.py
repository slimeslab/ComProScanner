"""
embeddings.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 23-02-2025
"""

# Standard library imports
import os
from typing import List, Any

# Third-party imports
import torch
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel

# Local imports
from comproscanner.utils.error_handler import ValueErrorHandler, ImportErrorHandler
from comproscanner.utils.logger import setup_logger

# configure logger
logger = setup_logger("comproscanner.log", module_name="embeddings")

# Import optional dependencies conditionally
try:
    from sentence_transformers import SentenceTransformer

    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError as e:
    logger.warning(f"SentenceTransformers package not found: {str(e)}")
    HAVE_SENTENCE_TRANSFORMERS = False

try:
    from openai import OpenAI

    HAVE_OPENAI = True
except ImportError as e:
    logger.warning(f"OpenAI package not found: {str(e)}")
    HAVE_OPENAI = False


class MultiModelEmbeddings(Embeddings):
    """Embeddings class supporting multiple model types optimized for processing pre-chunked articles"""

    def __init__(self, rag_config: Any):
        self.rag_config = rag_config
        self.model_type = self._determine_model_type(rag_config.embedding_model)

        if self.model_type == "huggingface":
            self._init_huggingface()
        elif self.model_type == "sentence_transformers":
            self._init_sentence_transformers()
        elif self.model_type == "openai":
            self._init_openai()
        else:
            logger.error(f"Unsupported embedding model: {rag_config.embedding_model}")
            raise ValueErrorHandler(
                f"Unsupported embedding model: {rag_config.embedding_model}"
            )

    def _determine_model_type(self, model_name: str) -> str:
        """Determine the embedding model type based on the model name"""
        if model_name.startswith("huggingface:"):
            return "huggingface"
        elif model_name.startswith("sentence-transformers:"):
            return "sentence_transformers"
        else:
            # Default to openai models
            return "openai"

    def _init_huggingface(self):
        """Initialize HuggingFace model (works with any transformer model)"""
        # Strip the prefix "huggingface:" from the model name
        model_name = self.rag_config.embedding_model
        if model_name.startswith("huggingface:"):
            model_name = model_name[len("huggingface:") :]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _init_sentence_transformers(self):
        """Initialize SentenceTransformers model"""
        if not HAVE_SENTENCE_TRANSFORMERS:
            logger.error(
                "SentenceTransformers package not found. Please install it with `pip install sentence-transformers`."
            )
            raise ImportErrorHandler(
                "SentenceTransformers package not found. Please install it with `pip install sentence-transformers`."
            )

        # Strip the prefix "sentence-transformers:" from the model name
        model_name = self.rag_config.embedding_model
        if model_name.startswith("sentence-transformers:"):
            model_name = model_name[len("sentence-transformers:") :]

        self.model = SentenceTransformer(model_name)

    def _init_openai(self):
        """Initialize OpenAI API configuration"""
        if not HAVE_OPENAI:
            logger.error(
                "OpenAI package not found. Please install it with `pip install openai`."
            )
            raise ImportErrorHandler(
                "OpenAI package not found. Please install it with `pip install openai`."
            )
        # Extract model name if format is "openai:model-name"
        if self.rag_config.embedding_model.startswith("openai:"):
            self.openai_model = self.rag_config.embedding_model.split(":", 1)[1]
        else:
            self.openai_model = self.rag_config.embedding_model

        # Check if API key is available and initialize client
        if not os.getenv("OPENAI_API_KEY"):
            logger.error(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment variable."
            )
            raise ValueErrorHandler(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents. Since documents are pre-chunked, simply process each one individually
        """
        embeddings = []

        for text in texts:
            if self.model_type == "huggingface":
                embedding = self._embed_document_huggingface(text)
            elif self.model_type == "sentence_transformers":
                embedding = self._embed_document_sentence_transformers(text)
            elif self.model_type == "openai":
                embedding = self._embed_document_openai(text)

            embeddings.append(embedding)

        return embeddings

    def _embed_document_huggingface(self, text: str) -> List[float]:
        """HuggingFace document embedding implementation for a single chunk"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.rag_config.rag_max_tokens,
        )

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Create attention mask to ignore padding tokens and calculate mean embeddings
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.sum(input_mask_expanded, 1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]

        return embedding.tolist()

    def _embed_document_sentence_transformers(self, text: str) -> List[float]:
        """SentenceTransformers document embedding implementation for a single chunk"""
        embedding = self.model.encode(text, convert_to_numpy=True)

        if len(embedding.shape) == 1:
            return embedding.tolist()
        return embedding[0].tolist()

    def _embed_document_openai(self, text: str) -> List[float]:
        """OpenAI document embedding implementation for a single chunk"""
        processed_text = text.replace("\n", " ")

        response = self.client.embeddings.create(
            model=self.openai_model, input=[processed_text]
        )

        return response.data[0].embedding

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query string (identical to embedding a single document)"""
        if self.model_type == "huggingface":
            return self._embed_document_huggingface(text)
        elif self.model_type == "sentence_transformers":
            return self._embed_document_sentence_transformers(text)
        elif self.model_type == "openai":
            return self._embed_document_openai(text)
