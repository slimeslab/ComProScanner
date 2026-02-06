"""
test_get_paper_data.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 01-12-2025
"""

import pytest
import os
from unittest.mock import MagicMock, patch, Mock
import requests

from comproscanner.utils.get_paper_data import PaperMetadataExtractor


class TestPaperMetadataExtractor:
    """Test suite for PaperMetadataExtractor class."""

    @pytest.fixture
    def mock_env_with_key(self, monkeypatch):
        """Mock environment with Scopus API key."""
        monkeypatch.setenv("SCOPUS_API_KEY", "test_api_key_12345")

    @pytest.fixture
    def mock_env_without_key(self, monkeypatch):
        """Mock environment without Scopus API key."""
        monkeypatch.delenv("SCOPUS_API_KEY", raising=False)

    @pytest.fixture
    def extractor_with_key(self, mock_env_with_key):
        """Create extractor with API key."""
        return PaperMetadataExtractor()

    @pytest.fixture
    def extractor_without_key(self, mock_env_without_key):
        """Create extractor without API key."""
        return PaperMetadataExtractor()

    @pytest.fixture
    def sample_scopus_response(self):
        """Sample Scopus API response."""
        return {
            "abstracts-retrieval-response": {
                "coredata": {
                    "dc:title": "Test Article Title",
                    "prism:publicationName": "Test Journal",
                    "openaccess": "1",
                },
                "item": {
                    "bibrecord": {
                        "head": {"source": {"publicationyear": {"@first": "2023"}}}
                    }
                },
                "authors": {
                    "author": [
                        {
                            "preferred-name": {
                                "ce:given-name": "John",
                                "ce:surname": "Doe",
                            },
                            "affiliation": {"@id": "12345"},
                        },
                        {
                            "preferred-name": {
                                "ce:given-name": "Jane",
                                "ce:surname": "Smith",
                            },
                            "affiliation": [
                                {"@id": "67890"},
                                {"@id": "11111"},
                            ],
                        },
                    ]
                },
                "authkeywords": {
                    "author-keyword": [
                        {"$": "materials science"},
                        {"$": "conductivity"},
                    ]
                },
            }
        }

    @pytest.fixture
    def sample_openalex_response(self):
        """Sample OpenAlex API response."""
        return {
            "title": "OpenAlex Test Article",
            "primary_location": {
                "source": {
                    "display_name": "OpenAlex Journal",
                    "host_organization_name": "OpenAlex Publisher",
                }
            },
            "publication_year": 2022,
            "authorships": [
                {"author": {"display_name": "Alice Johnson"}},
                {"author": {"display_name": "Bob Williams"}},
            ],
            "keywords": [
                {"display_name": "nanotechnology"},
                {"display_name": "synthesis"},
            ],
            "open_access": {"is_oa": True},
        }

    @pytest.fixture
    def sample_affiliation_response(self):
        """Sample affiliation API response."""
        return {
            "affiliation-retrieval-response": {
                "affiliation-name": "Test University",
                "country": "United States",
            }
        }

    def test_initialization_with_api_key(self, mock_env_with_key):
        """Test initialization with Scopus API key."""
        extractor = PaperMetadataExtractor()
        assert extractor.scopus_api_key == "test_api_key_12345"

    def test_initialization_without_api_key(self, mock_env_without_key):
        """Test initialization without Scopus API key."""
        extractor = PaperMetadataExtractor()
        assert extractor.scopus_api_key is None

    @patch("comproscanner.utils.get_paper_data.requests.request")
    @patch("comproscanner.utils.get_paper_data.requests.get")
    def test_get_article_metadata_scopus_api_error(
        self, mock_get, mock_request, extractor_with_key, sample_openalex_response
    ):
        """Test handling of Scopus API error."""
        mock_scopus_response = Mock()
        mock_scopus_response.status_code = 404

        mock_keywords_response = Mock()
        mock_keywords_response.status_code = 404
        mock_keywords_response.raise_for_status.side_effect = requests.HTTPError()

        mock_openalex_response = Mock()
        mock_openalex_response.status_code = 200
        mock_openalex_response.json.return_value = sample_openalex_response

        mock_request.side_effect = [mock_scopus_response]
        mock_get.side_effect = [mock_openalex_response, mock_keywords_response]

        result = extractor_with_key.get_article_metadata("10.1000/test")

        # Should fallback to OpenAlex data
        assert result["title"] == "OpenAlex Test Article"
        assert result["journal"] == "OpenAlex Journal"

    @patch("comproscanner.utils.get_paper_data.requests.request")
    @patch("comproscanner.utils.get_paper_data.requests.get")
    def test_get_article_metadata_openalex_api_error(
        self, mock_get, mock_request, extractor_with_key, sample_scopus_response
    ):
        """Test handling of OpenAlex API error."""
        mock_scopus_response = Mock()
        mock_scopus_response.status_code = 200
        mock_scopus_response.json.return_value = sample_scopus_response

        mock_keywords_response = Mock()
        mock_keywords_response.status_code = 200
        mock_keywords_response.json.return_value = sample_scopus_response
        mock_keywords_response.raise_for_status = Mock()

        mock_openalex_response = Mock()
        mock_openalex_response.status_code = 500

        mock_request.side_effect = [mock_scopus_response]
        mock_get.side_effect = [mock_openalex_response, mock_keywords_response]

        result = extractor_with_key.get_article_metadata("10.1000/test")

        # Should still have Scopus data
        assert result["title"] == "Test Article Title"
        assert result["journal"] == "Test Journal"

    @patch("comproscanner.utils.get_paper_data.requests.request")
    @patch("comproscanner.utils.get_paper_data.requests.get")
    def test_get_article_metadata_both_apis_fail(
        self, mock_get, mock_request, extractor_with_key
    ):
        """Test handling when both APIs fail."""
        mock_scopus_response = Mock()
        mock_scopus_response.status_code = 500

        mock_keywords_response = Mock()
        mock_keywords_response.status_code = 500
        mock_keywords_response.raise_for_status.side_effect = requests.HTTPError()

        mock_openalex_response = Mock()
        mock_openalex_response.status_code = 500

        mock_request.side_effect = [mock_scopus_response]
        mock_get.side_effect = [mock_openalex_response, mock_keywords_response]

        result = extractor_with_key.get_article_metadata("10.1000/test")

        # Should return empty metadata structure
        assert result["doi"] == "10.1000/test"
        assert result["title"] == ""
        assert result["journal"] == ""
        assert result["year"] == ""
        assert result["authors"] == []

    @patch("comproscanner.utils.get_paper_data.requests.request")
    @patch("comproscanner.utils.get_paper_data.requests.get")
    def test_get_article_metadata_with_closed_access(
        self, mock_get, mock_request, extractor_with_key, sample_openalex_response
    ):
        """Test metadata extraction for closed access article."""
        scopus_response = {
            "abstracts-retrieval-response": {
                "coredata": {
                    "dc:title": "Closed Access Article",
                    "openaccess": "0",
                },
                "authors": {"author": []},
            }
        }

        mock_scopus_response = Mock()
        mock_scopus_response.status_code = 200
        mock_scopus_response.json.return_value = scopus_response

        mock_keywords_response = Mock()
        mock_keywords_response.status_code = 200
        mock_keywords_response.json.return_value = scopus_response
        mock_keywords_response.raise_for_status = Mock()

        mock_openalex_response = Mock()
        mock_openalex_response.status_code = 200
        mock_openalex_response.json.return_value = sample_openalex_response

        mock_request.side_effect = [mock_scopus_response]
        mock_get.side_effect = [mock_openalex_response, mock_keywords_response]

        result = extractor_with_key.get_article_metadata("10.1000/test")

        assert result["isOpenAccess"] is False

    @patch("comproscanner.utils.get_paper_data.requests.request")
    @patch("comproscanner.utils.get_paper_data.requests.get")
    def test_get_article_metadata_keywords_single_dict(
        self, mock_get, mock_request, extractor_with_key, sample_openalex_response
    ):
        """Test keywords extraction when single keyword as dict."""
        scopus_response = {
            "abstracts-retrieval-response": {
                "coredata": {"dc:title": "Test Article"},
                "authkeywords": {"author-keyword": {"$": "single keyword"}},
                "authors": {"author": []},
            }
        }

        mock_scopus_response = Mock()
        mock_scopus_response.status_code = 200
        mock_scopus_response.json.return_value = scopus_response

        mock_keywords_response = Mock()
        mock_keywords_response.status_code = 200
        mock_keywords_response.json.return_value = scopus_response
        mock_keywords_response.raise_for_status = Mock()

        mock_openalex_response = Mock()
        mock_openalex_response.status_code = 200
        mock_openalex_response.json.return_value = sample_openalex_response

        mock_request.side_effect = [mock_scopus_response]
        mock_get.side_effect = [mock_openalex_response, mock_keywords_response]

        result = extractor_with_key.get_article_metadata("10.1000/test")

        assert "single keyword" in result["keywords"]

    @patch("comproscanner.utils.get_paper_data.requests.request")
    @patch("comproscanner.utils.get_paper_data.requests.get")
    def test_get_article_metadata_keywords_from_openalex(
        self, mock_get, mock_request, extractor_with_key
    ):
        """Test keywords extraction from OpenAlex."""
        scopus_response = {
            "abstracts-retrieval-response": {
                "coredata": {"dc:title": "Test Article"},
                "authors": {"author": []},
            }
        }

        openalex_response = {
            "title": "Test Article",
            "keywords": [{"display_name": "openalex keyword"}],
        }

        mock_scopus_response = Mock()
        mock_scopus_response.status_code = 200
        mock_scopus_response.json.return_value = scopus_response

        mock_keywords_response = Mock()
        mock_keywords_response.status_code = 200
        mock_keywords_response.json.return_value = scopus_response
        mock_keywords_response.raise_for_status = Mock()

        mock_openalex_response = Mock()
        mock_openalex_response.status_code = 200
        mock_openalex_response.json.return_value = openalex_response

        mock_request.side_effect = [mock_scopus_response]
        mock_get.side_effect = [mock_openalex_response, mock_keywords_response]

        result = extractor_with_key.get_article_metadata("10.1000/test")

        assert "openalex keyword" in result["keywords"]

    @patch("comproscanner.utils.get_paper_data.requests.request")
    @patch("comproscanner.utils.get_paper_data.requests.get")
    def test_get_article_metadata_keywords_deduplicated(
        self, mock_get, mock_request, extractor_with_key
    ):
        """Test keywords deduplication (case-insensitive)."""
        scopus_response = {
            "abstracts-retrieval-response": {
                "coredata": {"dc:title": "Test Article"},
                "authkeywords": {
                    "author-keyword": [
                        {"$": "Materials Science"},
                        {"$": "Conductivity"},
                    ]
                },
                "authors": {"author": []},
            }
        }

        openalex_response = {
            "title": "Test Article",
            "keywords": [
                {"display_name": "materials science"},
                {"display_name": "conductivity"},
                {"display_name": "Synthesis"},
            ],
        }

        mock_scopus_response = Mock()
        mock_scopus_response.status_code = 200
        mock_scopus_response.json.return_value = scopus_response

        mock_keywords_response = Mock()
        mock_keywords_response.status_code = 200
        mock_keywords_response.json.return_value = scopus_response
        mock_keywords_response.raise_for_status = Mock()

        mock_openalex_response = Mock()
        mock_openalex_response.status_code = 200
        mock_openalex_response.json.return_value = openalex_response

        mock_request.side_effect = [mock_scopus_response]
        mock_get.side_effect = [mock_openalex_response, mock_keywords_response]

        result = extractor_with_key.get_article_metadata("10.1000/test")

        # Should have 3 unique keywords (Materials Science, Conductivity, Synthesis)
        assert len(result["keywords"]) == 3

    @patch("comproscanner.utils.get_paper_data.requests.request")
    @patch("comproscanner.utils.get_paper_data.requests.get")
    def test_get_article_metadata_exception_handling(
        self, mock_get, mock_request, extractor_with_key
    ):
        """Test exception handling during API calls."""
        mock_request.side_effect = Exception("Network error")
        mock_get.side_effect = Exception("Network error")

        result = extractor_with_key.get_article_metadata("10.1000/test")

        # Should return empty metadata structure
        assert result["doi"] == "10.1000/test"
        assert result["title"] == ""
        assert result["journal"] == ""

    @patch("comproscanner.utils.get_paper_data.requests.request")
    @patch("comproscanner.utils.get_paper_data.requests.get")
    def test_get_article_metadata_scopus_keyerror(
        self, mock_get, mock_request, extractor_with_key, sample_openalex_response
    ):
        """Test handling of KeyError in Scopus response."""
        scopus_response = {
            "abstracts-retrieval-response": {
                "coredata": {"dc:title": "Test Article"},
                "authors": {
                    "author": [
                        {
                            "preferred-name": {
                                "ce:given-name": "John",
                            }
                            # Missing surname and affiliation
                        }
                    ]
                },
            }
        }

        mock_scopus_response = Mock()
        mock_scopus_response.status_code = 200
        mock_scopus_response.json.return_value = scopus_response

        mock_keywords_response = Mock()
        mock_keywords_response.status_code = 200
        mock_keywords_response.json.return_value = scopus_response
        mock_keywords_response.raise_for_status = Mock()

        mock_openalex_response = Mock()
        mock_openalex_response.status_code = 200
        mock_openalex_response.json.return_value = sample_openalex_response

        mock_request.side_effect = [mock_scopus_response]
        mock_get.side_effect = [mock_openalex_response, mock_keywords_response]

        result = extractor_with_key.get_article_metadata("10.1000/test")

        # Should handle missing fields gracefully
        assert result["title"] == "Test Article"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
