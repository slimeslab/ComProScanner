"""
rag_tool.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24
"""

# Standard library imports
from typing import Optional, Type
import importlib.util

# Third-party imports
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# Local imports
from ...utils.configs.rag_config import RAGConfig
from ...utils.configs.tools_config import DEFAULT_TOOLS_CONFIG
from ...utils.database_manager import VectorDatabaseManager
from ...utils.logger import setup_logger
from ...utils.error_handler import ImportErrorHandler, ValueErrorHandler

# configure logger
logger = setup_logger("comproscanner.log", module_name="rag_tool")


class RAGToolInput(BaseModel):
    """Input schema for RAGTool."""

    doi: str = Field(
        ..., description="The DOI of the document to be used for answering the query."
    )
    query: str = Field(
        ..., description="The query to be answered using the RAG system."
    )


class RAGTool(BaseTool):
    """RAG Tool for retrieving and generating answers from documents"""

    name: str = "RAG Research Tool"
    description: str = (
        "A tool that uses RAG (Retrieval Augmented Generation) to answer questions based on provided documents."
    )
    args_schema: Type[BaseModel] = RAGToolInput
    rag_config: RAGConfig = Field(default_factory=RAGConfig)
    system_prompt: str = DEFAULT_TOOLS_CONFIG.rag_tool.system_prompt
    answer_instruction: str = DEFAULT_TOOLS_CONFIG.rag_tool.answer_instruction
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        system_prompt: Optional[str] = None,
        answer_instruction: Optional[str] = None,
    ):
        """Initialize the RAG tool and its vector database manager.

        Args:
            rag_config (RAGConfig, optional): RAG configuration. Defaults to None, which builds a default RAGConfig.
            system_prompt (str, optional): System message of the answer model. Defaults to None (tools config default).
            answer_instruction (str, optional): Closing line after the question and context. Defaults to None (tools config default).
        """
        load_dotenv()
        super().__init__()
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if answer_instruction is not None:
            self.answer_instruction = answer_instruction
        if rag_config is None:
            self.rag_config = RAGConfig()
        else:
            self.rag_config = rag_config
        self._vector_db_manager = VectorDatabaseManager(self.rag_config)

    def _check_package_exists(self, package_name, model):
        """Check if a required package is installed, raising ImportErrorHandler if not.

        Args:
            package_name (str, required): Name of the Python package to check (importlib spec name).
            model (str, required): Model name included in the error message for context.

        Raises:
            ImportErrorHandler: If the package is not found.
        """
        if not importlib.util.find_spec(package_name):
            logger.error(
                f"The package required to run model '{model}' is missing: '{package_name}'."
            )
            raise ImportErrorHandler(
                f"The package required to run model '{model}' is missing: '{package_name}'."
            )

    def _get_llm(self) -> BaseChatModel:
        """Build the appropriate LangChain chat model based on the model name prefix in rag_config.rag_chat_model.

        Returns:
            BaseChatModel: the instantiated chat model for the configured provider.

        Raises:
            ValueErrorHandler: If the model name does not match any supported provider prefix.
        """
        model = self.rag_config.rag_chat_model
        temp = getattr(self.rag_config, "temperature", 0.1)
        streaming = getattr(self.rag_config, "streaming", False)
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else None
        common_params = {
            "temperature": temp,
            "streaming": streaming,
            "callbacks": callbacks,
        }
        # OpenAI models
        if model.startswith(("openai/", "gpt-", "text-", "o1", "o3")):
            self._check_package_exists("langchain_openai", model)
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model, request_timeout=1000, **common_params)

        # Deepseek models
        if model.startswith("deepseek/"):
            self._check_package_exists("langchain_deepseek", model)
            from langchain_deepseek import ChatDeepSeek

            return ChatDeepSeek(model=model, request_timeout=1000, **common_params)

        # Google Gemini models
        elif model.startswith("gemini/"):
            self._check_package_exists("langchain_google_genai", model)
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model=model, **common_params)

        # Anthropic Claude models
        elif model.startswith("claude/"):
            self._check_package_exists("langchain_anthropic", model)
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=model, **common_params)

        # Ollama models
        elif model.startswith("ollama/"):
            self._check_package_exists("langchain_ollama", model)
            from langchain_ollama import ChatOllama

            model_name = model.replace("ollama/", "")
            base_url = self.rag_config.rag_base_url or "http://localhost:11434"
            return ChatOllama(model=model_name, base_url=base_url, **common_params)

        # Together AI models
        elif model.startswith("together/"):
            self._check_package_exists("langchain_together", model)
            from langchain_together import ChatTogether

            model_name = model.replace("together/", "")
            return ChatTogether(model=model_name, request_timeout=1000, **common_params)

        # OpenRouter models
        elif model.startswith("openrouter/"):
            self._check_package_exists("langchain_openrouter", model)
            from langchain_openrouter import ChatOpenRouter

            model_name = model.replace("openrouter/", "")
            return ChatOpenRouter(model=model_name, **common_params)

        # Cohere models
        elif model.startswith("cohere/"):
            self._check_package_exists("langchain_cohere", model)
            from langchain_cohere import ChatCohere

            model_name = model.replace("cohere/", "")
            return ChatCohere(model=model_name, **common_params)

        # Fireworks models
        elif model.startswith(("fireworks/")):
            self._check_package_exists("langchain_fireworks", model)
            from langchain_fireworks import ChatFireworks

            return ChatFireworks(model=model, request_timeout=1000, **common_params)

        else:
            logger.error(f"Unrecognized or unsupported model name: {model}")
            raise ValueErrorHandler(f"Unrecognized or unsupported model name: {model}")

    def _format_documents(self, docs: list) -> str:
        """Format retrieved (document, score) pairs into a numbered string for LLM context.

        Args:
            docs (list, required): List of (Document, float) tuples from a similarity search.

        Returns:
            str: Newline-separated document blocks, each prefixed with its relevance score.
        """
        formatted_docs = []
        for i, doc_with_score in enumerate(docs):
            doc, score = doc_with_score
            formatted_docs.append(
                f"Document {i+1} [Relevance: {score:.4f}]:\n{doc.page_content}"
            )
        return "\n\n".join(formatted_docs)

    def _generate_response(self, query: str, docs: list) -> str:
        """Generate an LLM answer grounded in the retrieved documents.

        Args:
            query (str, required): The user question to answer.
            docs (list, required): List of (Document, float) tuples returned by the vector search.

        Returns:
            str: The LLM-generated answer string.
        """
        combined_input = (
            f"Question: {query}\n\n"
            f"""Context:\n{self._format_documents(docs)}\n\n
            {self.answer_instruction}"""
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=combined_input),
        ]
        llm = self._get_llm()
        result = llm.invoke(messages)
        return result.content

    def _run(self, doi: str, query: str) -> str:
        """Query the vector database for the given DOI and return an LLM-generated answer.

        Args:
            doi (str, required): DOI of the article whose vector database to query.
            query (str, required): Question to answer using the retrieved document chunks.

        Returns:
            str: LLM-generated answer, or an error message string if the query fails.
        """
        logger.info(f"\nDOI: {doi}")
        logger.info(f"Query: {query}")

        db_name = doi.replace("/", "_").replace(":", "_")
        logger.info(f"Database name: {db_name}")
        logger.info(f"Top K: {self.rag_config.rag_top_k}")

        try:
            logger.info("Step 1: Querying vector database...")
            relevant_docs = self._vector_db_manager.query_database(
                db_name=db_name, query=query, top_k=self.rag_config.rag_top_k
            )
            logger.info(f"Found {len(relevant_docs) if relevant_docs else 0} documents")

            if not relevant_docs:
                logger.warning("No relevant documents found")
                return "No relevant documents found"

            logger.info("Step 2: Generating response...")
            response = self._generate_response(query, relevant_docs)

            logger.info("RAG tool execution completed successfully")
            return response

        except Exception as e:
            error_msg = f"Error in RAG tool for DOI {doi}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Error type: {type(e).__name__}")

            # More detailed error info
            import traceback

            logger.error("Full traceback:")
            logger.error(traceback.format_exc())

            return f"Error occurred: {str(e)}"
