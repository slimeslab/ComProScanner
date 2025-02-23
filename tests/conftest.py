import pytest


@pytest.fixture
def share_scopus_api_key():
    return "dummy_scopus_api_key"


def pytest_configure(config):
    """Add custom markers to pytest"""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
