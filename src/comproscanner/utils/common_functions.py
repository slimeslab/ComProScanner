"""
common_functions.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 23-02-2025
"""

# Standard library imports
import re
import requests
import os
import time


@staticmethod
def get_paper_metadata_from_openalex(doi: str):
    """
    Function to get the paper metadata from the OpenAlex API.

    Args:
        doi (str): DOI of the paper.

    Returns:
        tuple: (title, journal_name, publisher) strings; all empty strings on failure.
    """
    try:
        url = f"https://api.openalex.org/works/doi:{doi}"
        response = requests.get(url)
        if response.status_code != 200:
            return "", "", ""
        else:
            data = response.json()
            title = data.get("title", "")
            journal_name = ""
            publisher = ""

            if (
                "primary_location" in data
                and data["primary_location"]
                and "source" in data["primary_location"]
                and data["primary_location"]["source"]
            ):
                journal_name = data["primary_location"]["source"].get("display_name", "")
                publisher = data["primary_location"]["source"].get(
                    "host_organization_name", ""
                )

            return title, journal_name, publisher
    except Exception:
        return "", "", ""


@staticmethod
def return_error_message(missing_variable: str):
    """
    Function to return an error message based on the missing variable.

    Args:
        missing_variable (str): Name of the missing variable (e.g. "main_property_keyword").

    Returns:
        str: Human-readable error message describing how to fix the missing variable.
    """
    if missing_variable == None:
        raise ValueError("The variable is missing.")
    else:
        if missing_variable == "main_property_keyword":
            return "main_property_keyword cannot be None. Please provide a valid keyword. Example: 'piezoelectric'. Exiting..."
        if missing_variable == "property_keywords":
            return """property_keywords cannot be None. Please provide a valid dictionary of property keywords which will be used for filtering sentences and should look like the following:\n{\n\t"exact_keywords": ["example1", "example2"],\n\n\t"substring_keywords": [" example 1 ", " example 2 "]\n}\nExiting..."""
        if missing_variable == "scopus_api_key":
            return "SCOPUS_API_KEY is not set in the environment variables. Please set it before running the script. Exiting..."
        if missing_variable == "wiley_api_key":
            return "WILEY_API_KEY is not set in the environment variables. Please set it before running the script. Exiting..."
        if missing_variable == "springer_open_access_api_key":
            return "SPRINGER_OPENACCESS_API_KEY is not set in the environment variables. Please set it before running the script. Exiting..."


@staticmethod
def get_doi_from_crossref(text: str):
    """Try to get DOI from CrossRef API using the title extracted from markdown text.

    Extracts the first heading from the markdown and queries CrossRef's
    bibliographic search. Only returns a DOI when the relevance score is
    high enough to be trustworthy.

    Args:
        text (str): Markdown text of the article.

    Returns:
        str: DOI if found with sufficient confidence, empty string otherwise.
    """
    try:
        title_match = re.search(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)
        if not title_match:
            return ""

        title = title_match.group(1).strip()
        if not title or len(title) < 10:
            return ""

        url = "https://api.crossref.org/works"
        params = {
            "query.bibliographic": title,
            "select": "DOI,title,score",
            "rows": 1,
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return ""

        data = response.json()
        items = data.get("message", {}).get("items", [])
        if not items:
            return ""

        item = items[0]
        score = item.get("score", 0)
        doi = item.get("DOI", "")

        if score >= 50 and doi:
            return doi

        return ""
    except Exception:
        return ""


@staticmethod
def write_timeout_file(doi, timeout_file):
    """
    Write the DOI to the timeout file.

    Args:
        doi (str): The DOI of the article to record.
        timeout_file (str): Path to the file where timed-out DOIs are appended.
    """
    timeout_dir = os.path.dirname(timeout_file)
    if not os.path.exists(timeout_dir):
        os.makedirs(timeout_dir)

    with open(timeout_file, "a") as f:
        f.write(doi + "\n")
        time.sleep(1)
