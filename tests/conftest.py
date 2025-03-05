import pytest


@pytest.fixture
def share_scopus_api_key():
    return "dummy_scopus_api_key"


@pytest.fixture(autouse=True)
def disable_exit_program(monkeypatch):
    """Disable exit_program for all tests to prevent SystemExit"""
    monkeypatch.setattr(
        "comproscanner.utils.error_handler.BaseError.exit_program", lambda self: None
    )


def pytest_configure(config):
    """Add custom markers to pytest"""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
