"""
common_functions.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 23-02-2025
"""

# Standard library imports
import requests
import os
import time


@staticmethod
def get_paper_metadata_from_oaworks(doi: str):
    """
    Function to get the paper metadata from the Open Access Works API.

    Args:
        doi (str: required): DOI of the paper

    Returns:
        title (str): Title of the paper.
        journal_name (str): Name of the journal.
        publisher (str): Name of the publisher.
    """
    try:
        url = f"https://bg.api.oa.works/metadata?id={doi}"
        response = requests.request("GET", url)
        if response.status_code != 200:
            return "", "", ""
        else:
            data = response.json()
            title = data.get("title", "")
            journal_name = data.get("container-title", "")
            publisher = data.get("publisher", "")
            return title, journal_name, publisher
    except Exception:
        return "", "", ""


@staticmethod
def return_error_message(missing_variable: str):
    """
    Function to return an error message based on the missing variable.

    Args:
        missing_variable (str: required): Variable which is missing.

    Returns:
        error_message (str): Error message based on the missing variable.
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
def write_timeout_file(doi, timeout_file):
    """
    Write the DOI to the timeout file.

    Args:
        doi (str: Required): The DOI of the article.
    """
    timeout_dir = os.path.dirname(timeout_file)
    if not os.path.exists(timeout_dir):
        os.makedirs(timeout_dir)

    with open(timeout_file, "a") as f:
        f.write(doi + "\n")
        time.sleep(1)
