import pytest
import pandas as pd
from unittest.mock import patch
from comproscanner.metadata_extractor.filter_metadata import FilterMetadata
from comproscanner.utils.error_handler import (
    ValueErrorHandler,
    CustomErrorHandler,
    BaseError,
)

SAMPLE_SCOPUS_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<serial-metadata-response xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:prism="http://prismstandard.org/namespaces/basic/2.0/">
    <link ref="self" href="https://api.elsevier.com/content/serial/title/issn/27679713" type="application/xml"/>
    <entry>
        <dc:title>Advanced Devices and Instrumentation</dc:title>
        <dc:publisher>American Association for the Advancement of Science</dc:publisher>
        <coverageStartYear>2020</coverageStartYear>
        <coverageEndYear>2024</coverageEndYear>
        <prism:aggregationType>journal</prism:aggregationType>
        <source-id>21101238116</source-id>
        <prism:eIssn>2767-9713</prism:eIssn>
        <openaccess>1</openaccess>
        <openaccessArticle>true</openaccessArticle>
        <subject-area code="2201" abbrev="ENGI">Engineering (miscellaneous)</subject-area>
        <subject-area code="3107" abbrev="PHYS">Atomic and Molecular Physics, and Optics</subject-area>
        <subject-area code="2208" abbrev="ENGI">Electrical and Electronic Engineering</subject-area>
        <subject-area code="3105" abbrev="PHYS">Instrumentation</subject-area>
    </entry>
</serial-metadata-response>"""


@pytest.fixture
def sample_df():
    """Fixture to create a sample DataFrame for testing"""
    return pd.DataFrame(
        {
            "article_type": ["Article", "Letter", "Review", "Article"],
            "issn": ["1234-5678", "8765-4321", "1111-2222", "1234-5678"],
            "doi": ["10.1000/abc", "10.1000/def", "10.1000/ghi", "10.1000/abc"],
            "scopus_id": ["123", "456", "789", "123"],
            "publication_name": ["Journal A", "Journal B", "Journal C", "Journal A"],
            "metadata_publisher": [None, "Publisher B", None, None],
            "general_publisher": [None, None, None, None],
        }
    )


@pytest.fixture
def filter_metadata(monkeypatch, share_scopus_api_key):
    """Fixture to create a FilterMetadata instance with test parameters"""
    monkeypatch.setenv("SCOPUS_API_KEY", share_scopus_api_key)
    return FilterMetadata(main_property_keyword="test")


def test_init_without_api_key(monkeypatch):
    """Test initialization without API key"""
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    monkeypatch.setattr(BaseError, "exit_program", lambda self: None)

    with pytest.raises(ValueErrorHandler) as exc_info:
        FilterMetadata(main_property_keyword="test")
    assert "SCOPUS_API_KEY is not set in the environment variables" in str(
        exc_info.value
    )


def test_init_without_keyword(monkeypatch, share_scopus_api_key):
    """Test initialization without main property keyword"""
    monkeypatch.setenv("SCOPUS_API_KEY", share_scopus_api_key)
    monkeypatch.setattr(BaseError, "exit_program", lambda self: None)

    with pytest.raises(ValueErrorHandler) as exc_info:
        FilterMetadata()
    assert "Main property keyword not provided" in str(exc_info.value)


def test_headers_setup(filter_metadata):
    """Test that headers are properly set up"""
    assert filter_metadata.headers["X-ELS-APIKey"] == "dummy_scopus_api_key"
    assert filter_metadata.headers["Accept"] == "application/xml"


def test_remove_invalid_rows(filter_metadata, sample_df):
    """Test removal of invalid rows"""
    result = filter_metadata._remove_invalid_rows(sample_df)
    assert len(result) == 3  # Only Article and Letter types should remain
    assert all(result["article_type"].isin(["Article", "Letter"]))


def test_remove_duplicate_doi_rows(filter_metadata, sample_df):
    """Test removal of duplicate DOI rows"""
    result = filter_metadata._remove_duplicate_doi_rows(sample_df)
    assert len(result) == 3  # One duplicate DOI should be removed
    assert len(result["doi"].unique()) == len(result["doi"])


def test_get_missing_publisher_entries(filter_metadata, sample_df):
    """Test identification of missing publisher entries"""
    result = filter_metadata._get_missing_publisher_entries(sample_df)
    assert len(result) == 3  # Three entries have missing publisher info


@pytest.mark.integration
def test_get_publisher_from_issn_success(filter_metadata, mocker):
    """Test successful publisher retrieval from ISSN"""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.text = SAMPLE_SCOPUS_RESPONSE

    mocker.patch("requests.get", return_value=mock_response)
    mocker.patch("pandas.read_csv", return_value=pd.DataFrame())
    mocker.patch("pandas.DataFrame.to_csv")

    result = filter_metadata._get_publisher_from_issn("2767-9713", pd.DataFrame())
    assert result is True


@pytest.mark.integration
def test_get_publisher_from_issn_rate_limit(filter_metadata, monkeypatch, mocker):
    """Test rate limit handling for ISSN API"""
    mock_response = mocker.Mock()
    mock_response.status_code = 429

    mocker.patch("requests.get", return_value=mock_response)
    monkeypatch.setattr(BaseError, "exit_program", lambda self: None)

    with pytest.raises(CustomErrorHandler) as exc_info:
        filter_metadata._get_publisher_from_issn("1234-5678", pd.DataFrame())
    assert exc_info.value.status_code == 429
    assert "API rate limit exceeded" in str(exc_info.value)


def test_update_from_existing_data(filter_metadata, sample_df):
    """Test updating from existing publisher data"""
    # Create a DataFrame with some existing publisher information
    # Modify index 1 to share an ISSN with index 0 and 3
    sample_df.loc[1, "issn"] = "1234-5678"  # Same ISSN as index 0 and 3
    sample_df.loc[1, "metadata_publisher"] = "Publisher B"
    sample_df.loc[1, "general_publisher"] = "publisher_b"

    df_missing = sample_df[sample_df["metadata_publisher"].isna()]
    updated_df, remaining_missing = filter_metadata._update_from_existing_data(
        sample_df, df_missing
    )

    # Now we expect index 0 and 3 to be updated since they share ISSN with index 1
    assert len(remaining_missing) < len(df_missing)
    # Additional assertions to verify the updates
    assert updated_df.loc[0, "metadata_publisher"] == "Publisher B"
    assert updated_df.loc[3, "metadata_publisher"] == "Publisher B"
    assert updated_df.loc[0, "general_publisher"] == "publisher_b"
    assert updated_df.loc[3, "general_publisher"] == "publisher_b"
    # Verify that index 2 remains unchanged
    assert pd.isna(updated_df.loc[2, "metadata_publisher"])


@pytest.mark.integration
def test_filter_metadata_complete_flow(filter_metadata, mocker, sample_df):
    """Test complete metadata filtering flow"""
    mocker.patch("pandas.read_csv", return_value=sample_df)
    mocker.patch("pandas.DataFrame.to_csv")
    mocker.patch.object(filter_metadata, "_get_publisher_from_issn", return_value=True)
    mocker.patch.object(
        filter_metadata, "_get_publisher_from_scopus_id", return_value=True
    )

    try:
        result = filter_metadata.filter_metadata()
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"filter_metadata raised an unexpected exception: {e}")


def test_publisher_mapping(filter_metadata):
    """Test publisher mapping dictionary"""
    expected_publishers = {
        "IOP Publishing Ltd.": "iop",
        "elsevier": "elsevier",
        "nature": "nature",
        "American Institute of Physics": "aip",
        "American Chemical Society": "acs",
        "American Physical Society": "aps",
        "Royal Society of Chemistry": "rsc",
        "springer": "springer",
        "open access science": "springer",
        "wiley": "wiley",
    }
    assert filter_metadata.publisher_mapping == expected_publishers
