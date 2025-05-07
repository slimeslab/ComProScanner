from typing import Optional


class RAGConfig:
    """
    Configuration class for RAG model.

    Args:
        rag_db_path (str: optional): Path to the SQLite database (default: db).
        chunk_size (int: optional): Size of the chunks to split the input text into (default: 1000).
        chunk_overlap (int: optional): Overlap between the chunks (default: 25).
        embedding_model (str: optional): Name of the embedding model (default: huggingface:thellert/physbert_cased).
        rag_chat_model (str: optional): Name of the chat model (default: gemini-2.0-flash-thinking-exp).
        rag_max_tokens (int: optional): Maximum length of the input text (default: 512).
        rag_top_k (int: optional): Top k value for sampling (default: 3).
    """

    DEFAULT_DB_PATH = "db"
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 25
    DEFAULT_EMBEDDING_MODEL = "huggingface:thellert/physbert_cased"
    DEFAULT_CHAT_MODEL = "gemini-2.0-flash-thinking-exp"
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_TOP_K = 3

    def __init__(
        self,
        rag_db_path: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        rag_chat_model: Optional[str] = None,
        rag_max_tokens: Optional[int] = None,
        rag_top_k: Optional[int] = None,
        rag_base_url: Optional[str] = None,
    ):
        self.rag_db_path = rag_db_path or self.DEFAULT_DB_PATH
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or self.DEFAULT_CHUNK_OVERLAP
        self.embedding_model = embedding_model or self.DEFAULT_EMBEDDING_MODEL
        self.rag_chat_model = rag_chat_model or self.DEFAULT_CHAT_MODEL
        self.rag_max_tokens = rag_max_tokens or self.DEFAULT_MAX_TOKENS
        self.rag_top_k = rag_top_k or self.DEFAULT_TOP_K
        self.rag_base_url = rag_base_url
