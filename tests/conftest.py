import pytest
import sys
import os
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

# Set environment variable to indicate testing BEFORE any imports
os.environ["PYTEST_CURRENT_TEST"] = "true"

# Add the src directory to the path FIRST
tests_dir = Path(__file__).parent
src_dir = tests_dir.parent / "src"
sys.path.insert(0, str(src_dir))


def create_mock_module(name, **kwargs):
    """Create a mock module with proper attributes."""
    mock = MagicMock()
    mock.__name__ = name
    mock.__package__ = name.rsplit(".", 1)[0] if "." in name else ""
    mock.__path__ = [name]
    mock.__spec__ = MagicMock()
    mock.__spec__.name = name

    for key, value in kwargs.items():
        setattr(mock, key, value)

    return mock


def create_real_module(name, **kwargs):
    """Create a real module object (not MagicMock) with explicit attributes."""
    module = types.ModuleType(name)
    module.__package__ = name.rsplit(".", 1)[0] if "." in name else ""
    module.__path__ = [name]

    for key, value in kwargs.items():
        setattr(module, key, value)

    return module


def _identity_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def _decorator(func):
        return func

    return _decorator


def _event_decorator(*args, **kwargs):
    def _decorator(func):
        return func

    return _decorator


def _or_(*signals):
    return signals


class _DummyFlow:
    def __class_getitem__(cls, _item):
        return cls

    def __init__(self, *args, **kwargs):
        self.state = types.SimpleNamespace(
            is_materials_mentioned="",
            composition_extracted_data={},
            composition_formatted_data={},
            synthesis_extracted_data={},
            synthesis_formatted_data={},
            doi="",
            materials_data_identifier_query="",
            main_extraction_keyword="",
            composition_property_text_data="",
            synthesis_text_data="",
            is_extract_synthesis_data=True,
            vlm_model="gemini/gemini-3-flash-preview",
            related_figures_base_path="results/related_figures",
            llm=None,
            rag_config=None,
            output_log_folder=None,
            task_output_folder=None,
            is_log_json=False,
            verbose=True,
            expected_composition_property_example="",
            expected_variable_composition_property_example="",
            composition_property_extraction_agent_note="",
            composition_property_extraction_task_note="",
            composition_property_formatting_agent_note="",
            composition_property_formatting_task_note="",
            synthesis_extraction_agent_note="",
            synthesis_extraction_task_note="",
            synthesis_formatting_agent_note="",
            synthesis_formatting_task_note="",
            allowed_synthesis_methods="",
            allowed_characterization_techniques="",
        )


class _DummyChromaCollection:
    def __init__(self, *args, **kwargs):
        pass

    def query(self, *args, **kwargs):
        return {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}

    def upsert(self, *args, **kwargs):
        return None


class _DummyCrewAILLM:
    pass


# Mock crewai completely before any imports
crewai_modules = {
    "crewai": {
        "LLM": _DummyCrewAILLM,
        "Agent": MagicMock(),
        "Task": MagicMock(),
        "Crew": MagicMock(),
    },
    "crewai.flow": {},
    "crewai.flow.flow": {
        "Flow": _DummyFlow,
        "listen": _event_decorator,
        "start": _identity_decorator,
        "router": _event_decorator,
        "or_": _or_,
    },
    "crewai.project": {
        "CrewBase": MagicMock(),
        "agent": MagicMock(),
        "crew": MagicMock(),
        "task": MagicMock(),
    },
    "crewai.tools": {
        "BaseTool": MagicMock(),
    },
    "crewai.agent": {},
    "crewai.llm": {"LLM": _DummyCrewAILLM},
    "crewai.agents": {},
    "crewai.agents.crew_agent_executor": {"CrewAgentExecutor": MagicMock()},
    "crewai.agents.agent_builder": {},
    "crewai.agents.agent_builder.base_agent_executor_mixin": {
        "CrewAgentExecutorMixin": MagicMock()
    },
    "crewai.utilities": {},
    "crewai.utilities.evaluators": {},
    "crewai.utilities.evaluators.task_evaluator": {"TaskEvaluator": MagicMock()},
    "crewai.utilities.events": {"EventListener": MagicMock()},
    "crewai.utilities.events.event_listener": {"EventListener": MagicMock()},
}

for module_name, attrs in crewai_modules.items():
    sys.modules[module_name] = create_mock_module(module_name, **attrs)

# Mock other problematic dependencies
sys.modules["litellm"] = create_mock_module("litellm")
sys.modules["litellm.types"] = create_mock_module("litellm.types")
sys.modules["litellm.types.utils"] = create_mock_module("litellm.types.utils")
sys.modules["litellm.exceptions"] = create_real_module(
    "litellm.exceptions",
    ContextWindowExceededError=type("ContextWindowExceededError", (Exception,), {}),
)
sys.modules["litellm.utils"] = create_real_module(
    "litellm.utils",
    supports_response_schema=lambda *args, **kwargs: True,
    supports_function_calling=lambda *args, **kwargs: True,
)
sys.modules["litellm.litellm_core_utils"] = create_real_module(
    "litellm.litellm_core_utils"
)
sys.modules["litellm.litellm_core_utils.get_supported_openai_params"] = (
    create_real_module(
        "litellm.litellm_core_utils.get_supported_openai_params",
        get_supported_openai_params=lambda *args, **kwargs: ["stop"],
    )
)
sys.modules["litellm.integrations"] = create_real_module("litellm.integrations")
sys.modules["litellm.integrations.custom_logger"] = create_real_module(
    "litellm.integrations.custom_logger",
    CustomLogger=type("CustomLogger", (), {}),
)
sys.modules["instructor"] = create_mock_module("instructor")
sys.modules["crewai_tools"] = create_mock_module("crewai_tools")
sys.modules["langchain_chroma"] = create_real_module(
    "langchain_chroma", Chroma=type("Chroma", (), {})
)

# Use lightweight real types/functions (not MagicMock) for chromadb stubs.
# CrewAI/Pydantic dataclass parsing reads type annotations and fails on MagicMock.
class _PydanticAnyTypeMixin:
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema

        return core_schema.any_schema()


class _DummyPersistentClient(_PydanticAnyTypeMixin):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def get_or_create_collection(self, *args, **kwargs):
        return _DummyChromaCollection()

    def reset(self):
        return None

    def clear_system_cache(self):
        return None


class _DummySettings(_PydanticAnyTypeMixin):
    def __init__(self, *args, **kwargs):
        self._kwargs = kwargs


class _DummyAsyncClientAPI(_PydanticAnyTypeMixin):
    pass


class _DummyClientAPI(_PydanticAnyTypeMixin):
    pass


class _DummyCollectionConfigurationInterface(_PydanticAnyTypeMixin):
    pass


class _DummyCollectionMetadata(dict, _PydanticAnyTypeMixin):
    pass


class _DummyLoadable(_PydanticAnyTypeMixin):
    pass


class _DummyWhere(dict, _PydanticAnyTypeMixin):
    pass


class _DummyWhereDocument(dict, _PydanticAnyTypeMixin):
    pass


class _DummyDataLoader(_PydanticAnyTypeMixin):
    def __class_getitem__(cls, _item):
        return cls


class _DummyEmbeddingFunction(_PydanticAnyTypeMixin):
    def __class_getitem__(cls, _item):
        return cls


class _DummyInclude(list, _PydanticAnyTypeMixin):
    pass


class _DummyDocuments(list, _PydanticAnyTypeMixin):
    pass


class _DummyEmbeddings(list, _PydanticAnyTypeMixin):
    pass


class _DummyMetadata(dict, _PydanticAnyTypeMixin):
    pass


class _DummyCollection(_PydanticAnyTypeMixin):
    pass


class _DummyOpenAIEmbeddingFunction(_PydanticAnyTypeMixin):
    def __init__(self, *args, **kwargs):
        pass


class _DummyEmbeddingCallable(_PydanticAnyTypeMixin):
    def __init__(self, *args, **kwargs):
        pass


def _dummy_validate_embedding_function(*args, **kwargs):
    return None


sys.modules["chromadb"] = create_real_module(
    "chromadb",
    PersistentClient=_DummyPersistentClient,
    Collection=_DummyCollection,
    Documents=_DummyDocuments,
    EmbeddingFunction=_DummyEmbeddingFunction,
    Embeddings=_DummyEmbeddings,
    Metadata=_DummyMetadata,
)
sys.modules["chromadb.config"] = create_real_module(
    "chromadb.config", Settings=_DummySettings
)
sys.modules["chromadb.api"] = create_real_module(
    "chromadb.api",
    AsyncClientAPI=_DummyAsyncClientAPI,
    ClientAPI=_DummyClientAPI,
)
sys.modules["chromadb.api.types"] = create_real_module(
    "chromadb.api.types",
    CollectionMetadata=_DummyCollectionMetadata,
    DataLoader=_DummyDataLoader,
    Documents=_DummyDocuments,
    EmbeddingFunction=_DummyEmbeddingFunction,
    Embeddings=_DummyEmbeddings,
    Include=_DummyInclude,
    Loadable=_DummyLoadable,
    OneOrMany=list,
    Where=_DummyWhere,
    WhereDocument=_DummyWhereDocument,
    validate_embedding_function=_dummy_validate_embedding_function,
)
sys.modules["chromadb.errors"] = create_real_module(
    "chromadb.errors",
    InvalidDimensionException=type("InvalidDimensionException", (Exception,), {}),
)
sys.modules["chromadb.api.configuration"] = create_real_module(
    "chromadb.api.configuration",
    CollectionConfigurationInterface=_DummyCollectionConfigurationInterface,
)
sys.modules["chromadb.utils"] = create_real_module("chromadb.utils")
sys.modules["chromadb.utils.embedding_functions"] = create_real_module(
    "chromadb.utils.embedding_functions"
)


def _register_embedding_function_module(module_name, class_names):
    attrs = {class_name: _DummyEmbeddingCallable for class_name in class_names}
    sys.modules[module_name] = create_real_module(module_name, **attrs)


_register_embedding_function_module(
    "chromadb.utils.embedding_functions.openai_embedding_function",
    ["OpenAIEmbeddingFunction"],
)
_register_embedding_function_module(
    "chromadb.utils.embedding_functions.amazon_bedrock_embedding_function",
    ["AmazonBedrockEmbeddingFunction"],
)
_register_embedding_function_module(
    "chromadb.utils.embedding_functions.cohere_embedding_function",
    ["CohereEmbeddingFunction"],
)
_register_embedding_function_module(
    "chromadb.utils.embedding_functions.google_embedding_function",
    ["GoogleGenerativeAiEmbeddingFunction", "GoogleVertexEmbeddingFunction"],
)
_register_embedding_function_module(
    "chromadb.utils.embedding_functions.huggingface_embedding_function",
    ["HuggingFaceEmbeddingFunction", "HuggingFaceEmbeddingServer"],
)
_register_embedding_function_module(
    "chromadb.utils.embedding_functions.instructor_embedding_function",
    ["InstructorEmbeddingFunction"],
)
_register_embedding_function_module(
    "chromadb.utils.embedding_functions.jina_embedding_function",
    ["JinaEmbeddingFunction"],
)
_register_embedding_function_module(
    "chromadb.utils.embedding_functions.ollama_embedding_function",
    ["OllamaEmbeddingFunction"],
)
_register_embedding_function_module(
    "chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2",
    ["ONNXMiniLM_L6_V2"],
)
_register_embedding_function_module(
    "chromadb.utils.embedding_functions.open_clip_embedding_function",
    ["OpenCLIPEmbeddingFunction"],
)
_register_embedding_function_module(
    "chromadb.utils.embedding_functions.roboflow_embedding_function",
    ["RoboflowEmbeddingFunction"],
)
_register_embedding_function_module(
    "chromadb.utils.embedding_functions.sentence_transformer_embedding_function",
    ["SentenceTransformerEmbeddingFunction"],
)
_register_embedding_function_module(
    "chromadb.utils.embedding_functions.text2vec_embedding_function",
    ["Text2VecEmbeddingFunction"],
)

# Mock aiohttp to avoid the ConnectionTimeoutError
aiohttp_mock = create_mock_module("aiohttp")
aiohttp_mock.ConnectionTimeoutError = type("ConnectionTimeoutError", (Exception,), {})
sys.modules["aiohttp"] = aiohttp_mock


# Fixtures
@pytest.fixture
def share_scopus_api_key():
    """Provide a dummy Scopus API key for testing."""
    return "dummy_scopus_api_key"


@pytest.fixture(autouse=True)
def disable_exit_program(monkeypatch):
    """Disable exit_program for all tests to prevent SystemExit."""
    monkeypatch.setattr(
        "comproscanner.utils.error_handler.BaseError.exit_program", lambda self: None
    )


@pytest.fixture(autouse=True)
def mock_heavy_ml_dependencies():
    """Automatically mock heavy ML/AI dependencies for all tests to avoid downloading models."""

    # Only apply mocking if NOT in extract_flow tests
    import sys

    # Check if we're running extract_flow tests
    if "test_extract_flow" in str(sys.argv):
        yield
        return

    with (
        patch(
            "comproscanner.utils.database_manager.MultiModelEmbeddings"
        ) as mock_embeddings,
        patch("comproscanner.utils.embeddings.AutoTokenizer") as mock_tokenizer,
        patch("comproscanner.utils.embeddings.AutoModel") as mock_model,
    ):

        # Setup embeddings mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.return_value = [[0.1] * 768]
        mock_embeddings_instance.embed_query.return_value = [0.1] * 768
        mock_embeddings_instance.model_type = "huggingface"
        mock_embeddings.return_value = mock_embeddings_instance

        # Setup tokenizer and model mocks (for when embeddings is initialized)
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        yield


# Pytest hooks
def pytest_configure(config):
    """Called after command line options have been parsed."""
    # Add custom markers
    config.addinivalue_line("markers", "integration: mark test as an integration test")

    # Print configuration message
    print("\n" + "=" * 80)
    print("Configuring pytest for comproscanner tests")
    print("Testing mode enabled - heavy dependencies mocked")
    print("=" * 80 + "\n")


def pytest_collection_modifyitems(config, items):
    """Called after test collection has been performed."""
    print(f"\nCollected {len(items)} test items\n")
