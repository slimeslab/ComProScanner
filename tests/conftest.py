import pytest
import sys
import os
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


# Mock crewai completely before any imports
crewai_modules = {
    "crewai": {
        "LLM": MagicMock(),
        "Agent": MagicMock(),
        "Task": MagicMock(),
        "Crew": MagicMock(),
    },
    "crewai.flow": {},
    "crewai.flow.flow": {
        "Flow": MagicMock(),
        "listen": MagicMock(),
        "start": MagicMock(),
        "router": MagicMock(),
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
    "crewai.llm": {"LLM": MagicMock()},
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
sys.modules["instructor"] = create_mock_module("instructor")
sys.modules["crewai_tools"] = create_mock_module("crewai_tools")
sys.modules["langchain_chroma"] = create_mock_module(
    "langchain_chroma", Chroma=MagicMock()
)
sys.modules["chromadb"] = create_mock_module(
    "chromadb", PersistentClient=MagicMock()
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
