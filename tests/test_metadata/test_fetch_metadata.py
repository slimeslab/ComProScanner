import pytest
import time
from comproscanner.utils.error_handler import BaseError
from comproscanner.metadata_extractor.fetch_metadata import FetchMetadata
from comproscanner.utils.error_handler import ValueErrorHandler

SAMPLE_SCOPUS_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<search-results xmlns="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
    xmlns:prism="http://prismstandard.org/namespaces/basic/2.0/"
    xmlns:atom="http://www.w3.org/2005/Atom">
    <opensearch:totalResults>468164</opensearch:totalResults>
    <opensearch:startIndex>0</opensearch:startIndex>
    <opensearch:itemsPerPage>25</opensearch:itemsPerPage>
    <opensearch:Query role="request" searchTerms="'piezoelectric'" startPage="0" />
    <link ref="self"
        href="https://api.elsevier.com/content/search/scopus?start=0&amp;count=25&amp;query=%27piezoelectric%27"
        type="application/xml" />
    <link ref="first"
        href="https://api.elsevier.com/content/search/scopus?start=0&amp;count=25&amp;query=%27piezoelectric%27"
        type="application/xml" />
    <link ref="next"
        href="https://api.elsevier.com/content/search/scopus?start=25&amp;count=25&amp;query=%27piezoelectric%27"
        type="application/xml" />
    <link ref="last"
        href="https://api.elsevier.com/content/search/scopus?start=4975&amp;count=25&amp;query=%27piezoelectric%27"
        type="application/xml" />
    <entry>
        <link ref="self" href="https://api.elsevier.com/content/abstract/scopus_id/85217620288" />
        <link ref="author-affiliation"
            href="https://api.elsevier.com/content/abstract/scopus_id/85217620288?field=author,affiliation" />
        <link ref="scopus"
            href="https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&amp;scp=85217620288&amp;origin=inward" />
        <link ref="scopus-citedby"
            href="https://www.scopus.com/inward/citedby.uri?partnerID=HzOxMe3b&amp;scp=85217620288&amp;origin=inward" />
        <prism:url>https://api.elsevier.com/content/abstract/scopus_id/85217620288</prism:url>
        <dc:identifier>SCOPUS_ID:85217620288</dc:identifier>
        <eid>2-s2.0-85217620288</eid>
        <dc:title>Anti-inflammatory coupled anti-angiogenic airway stent effectively suppresses
            tracheal in-stents restenosis</dc:title>
        <dc:creator>Zhao Y.</dc:creator>
        <prism:publicationName>Journal of Nanobiotechnology</prism:publicationName>
        <prism:eIssn>14773155</prism:eIssn>
        <prism:volume>23</prism:volume>
        <prism:issueIdentifier>1</prism:issueIdentifier>
        <prism:pageRange />
        <prism:coverDate>2025-12-01</prism:coverDate>
        <prism:coverDisplayDate>December 2025</prism:coverDisplayDate>
        <prism:doi>10.1186/s12951-024-03087-y</prism:doi>
        <citedby-count>0</citedby-count>
        <affiliation>
            <affilname>First Affiliated Hospital of Zhengzhou University</affilname>
            <affiliation-city>Zhengzhou</affiliation-city>
            <affiliation-country>China</affiliation-country>
        </affiliation>
        <affiliation>
            <affilname>Zhengzhou University</affilname>
            <affiliation-city>Zhengzhou</affiliation-city>
            <affiliation-country>China</affiliation-country>
        </affiliation>
        <pubmed-id>39881307</pubmed-id>
        <prism:aggregationType>Journal</prism:aggregationType>
        <subtype>ar</subtype>
        <subtypeDescription>Article</subtypeDescription>
        <article-number>59</article-number>
        <source-id>16088</source-id>
        <openaccess>1</openaccess>
        <openaccessFlag>true</openaccessFlag>
        <freetoread>
            <value>all</value>
            <value>publisherfullgold</value>
        </freetoread>
        <freetoreadLabel>
            <value>All Open Access</value>
            <value>Gold</value>
        </freetoreadLabel>
    </entry>
    <entry>
        <link ref="self" href="https://api.elsevier.com/content/abstract/scopus_id/85217530051" />
        <link ref="author-affiliation"
            href="https://api.elsevier.com/content/abstract/scopus_id/85217530051?field=author,affiliation" />
        <link ref="scopus"
            href="https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&amp;scp=85217530051&amp;origin=inward" />
        <link ref="scopus-citedby"
            href="https://www.scopus.com/inward/citedby.uri?partnerID=HzOxMe3b&amp;scp=85217530051&amp;origin=inward" />
        <prism:url>https://api.elsevier.com/content/abstract/scopus_id/85217530051</prism:url>
        <dc:identifier>SCOPUS_ID:85217530051</dc:identifier>
        <eid>2-s2.0-85217530051</eid>
        <dc:title>Influence of moving heat sources on thermoviscoelastic behavior of rotating
            nanorods: a nonlocal Klein–Gordon perspective with fractional heat conduction</dc:title>
        <dc:creator>Abouelregal A.E.</dc:creator>
        <prism:publicationName>Boundary Value Problems</prism:publicationName>
        <prism:issn>16872762</prism:issn>
        <prism:eIssn>16872770</prism:eIssn>
        <prism:volume>2025</prism:volume>
        <prism:issueIdentifier>1</prism:issueIdentifier>
        <prism:pageRange />
        <prism:coverDate>2025-12-01</prism:coverDate>
        <prism:coverDisplayDate>December 2025</prism:coverDisplayDate>
        <prism:doi>10.1186/s13661-025-01992-1</prism:doi>
        <citedby-count>0</citedby-count>
        <affiliation>
            <affilname>Faculty of Science</affilname>
            <affiliation-city>Mansoura</affiliation-city>
            <affiliation-country>Egypt</affiliation-country>
        </affiliation>
        <prism:aggregationType>Journal</prism:aggregationType>
        <subtype>ar</subtype>
        <subtypeDescription>Article</subtypeDescription>
        <article-number>10</article-number>
        <source-id>4000149606</source-id>
        <openaccess>1</openaccess>
        <openaccessFlag>true</openaccessFlag>
        <freetoread>
            <value>all</value>
            <value>publisherfullgold</value>
        </freetoread>
        <freetoreadLabel>
            <value>All Open Access</value>
            <value>Gold</value>
        </freetoreadLabel>
    </entry>
</search-results>"""


@pytest.fixture
def metadata_fetcher(monkeypatch, share_scopus_api_key):
    """Fixture to create a FetchMetadata instance with default test parameters"""
    monkeypatch.setenv("SCOPUS_API_KEY", share_scopus_api_key)
    return FetchMetadata(
        main_property_keyword="piezoelectric",
        base_queries=["materials", "devices"],
        extra_queries=["advancements"],
    )


def test_basic_initialization(monkeypatch, share_scopus_api_key):
    """Test basic initialization with minimum required parameters"""
    monkeypatch.setenv("SCOPUS_API_KEY", share_scopus_api_key)
    fetcher = FetchMetadata(main_property_keyword="test")
    assert fetcher.keywords == ["test"]
    assert fetcher.extra_queries == []
    assert fetcher.api_key == share_scopus_api_key
    assert fetcher.start_year == int(time.strftime("%Y"))
    assert fetcher.end_year == int(time.strftime("%Y")) - 2


def test_base_queries_include_main_keyword(monkeypatch, share_scopus_api_key):
    """Test that main_property_keyword is added to base_queries if not present"""
    monkeypatch.setenv("SCOPUS_API_KEY", share_scopus_api_key)
    fetcher = FetchMetadata(
        main_property_keyword="piezoelectric", base_queries=["materials", "devices"]
    )
    assert "piezoelectric" in fetcher.keywords
    assert sorted(fetcher.keywords) == sorted(["materials", "devices", "piezoelectric"])


def test_base_queries_with_main_keyword_already_present(
    monkeypatch, share_scopus_api_key
):
    """Test when main_property_keyword is already in base_queries"""
    monkeypatch.setenv("SCOPUS_API_KEY", share_scopus_api_key)
    fetcher = FetchMetadata(
        main_property_keyword="piezoelectric",
        base_queries=["piezoelectric", "materials"],
    )
    assert sorted(fetcher.keywords) == sorted(["piezoelectric", "materials"])


def test_missing_api_key(monkeypatch):
    """Test initialization without API key"""
    # Explicitly remove the SCOPUS_API_KEY from environment
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)

    # Mock the exit_program method to prevent system exit during test
    monkeypatch.setattr(BaseError, "exit_program", lambda self: None)

    with pytest.raises(ValueErrorHandler) as exc_info:
        FetchMetadata(main_property_keyword="test")
    assert "SCOPUS_API_KEY is not set in the environment variables" in str(
        exc_info.value
    )


@pytest.mark.parametrize(
    "test_input,expected_error",
    [
        ({"main_property_keyword": None}, "main_property_keyword cannot be None"),
        (
            {
                "main_property_keyword": "test",
                "start_year": int(time.strftime("%Y")) + 1,
            },
            "Start year cannot be greater than the current year",
        ),
        (
            {"main_property_keyword": "test", "start_year": 2020, "end_year": 2022},
            "Start year should be greater than the end year",
        ),
        (
            {"main_property_keyword": "test", "start_year": 2022, "end_year": 2022},
            "Start year and End year cannot be the same",
        ),
    ],
)
def test_validation_errors(
    monkeypatch, share_scopus_api_key, test_input, expected_error
):
    """Test various validation error cases"""
    monkeypatch.setenv("SCOPUS_API_KEY", share_scopus_api_key)

    # Mock the exit_program method in BaseError
    monkeypatch.setattr(BaseError, "exit_program", lambda self: None)

    with pytest.raises(ValueErrorHandler) as exc_info:
        FetchMetadata(**test_input)
    assert expected_error in str(exc_info.value)


@pytest.mark.integration
def test_main_fetch_execution(metadata_fetcher, mocker):
    """Test main_fetch execution with mock Scopus response"""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.text = SAMPLE_SCOPUS_RESPONSE

    # Mock requests.get
    mocker.patch("requests.get", return_value=mock_response)

    try:
        metadata_fetcher.main_fetch()
    except Exception as e:
        pytest.fail(f"main_fetch raised an unexpected exception: {e}")


def test_headers_setup(metadata_fetcher):
    """Test that headers are properly set up"""
    assert metadata_fetcher.headers["X-ELS-APIKey"] == "dummy_scopus_api_key"
    assert metadata_fetcher.headers["Accept"] == "application/xml"
