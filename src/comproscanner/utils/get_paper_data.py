"""
get_paper_data.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 29-03-2025
"""

import requests
from typing import Dict
import os
from dotenv import load_dotenv
from typing import List

from .logger import setup_logger

load_dotenv()

######## logger Configuration ########
logger = setup_logger("post-processing.log")


class PaperMetadataExtractor:
    def __init__(self):
        """
        Initialize the PaperMetadataExtractor class with the Scopus API key (optional) from the environment variables. If the Scopus API key is not found, only the OA.Works API will be used for metadata extraction.
        """
        self.scopus_api_key = os.getenv("SCOPUS_API_KEY")
        if not self.scopus_api_key:
            logger.warning(
                "Scopus API key not found in environment variables. Only OA.Works API will be used."
            )

    def get_article_metadata(self, doi: str) -> Dict:
        """
        Extract the journal article metadata by doing Scopus API and Unpaywall API call with the provided DOI and returns article metadata in a dictionary. The metadata includes DOI, title, journal, year, isOpenAccess, and authors. The authors are a list of dictionaries containing name and affiliation_id.

        Args:
            doi (str: required): The DOI of the article to get metadata for

        Returns:
            paper_data (dict): A dictionary containing the metadata of the article
        """

        def _get_scopus_data(doi: str):
            """
            Get the article metadata from Scopus API using the provided DOI

            Args:
                doi (str: required): The DOI of the article to get metadata for

            Returns:
                response.json()["abstracts-retrieval-response"] (dict): The metadata of the article
            """
            try:
                url = f"https://api.elsevier.com/content/abstract/doi/{doi}"
                headers = {
                    "X-ELS-APIKey": self.scopus_api_key,
                    "Accept": "application/json",
                }
                response = requests.request("POST", url, headers=headers)
                if response.status_code != 200:
                    return f"Sorry, the status code is not 200, it is {response.status_code}"
                return response.json()["abstracts-retrieval-response"]
            except KeyError:
                return "Sorry, I couldn't find anything about that, there could be an error with your scopus api key."
            except Exception as e:
                return f"Sorry, something went wrong: {e}"

        def _get_keywords_from_scopus_api(doi: str) -> List[str]:
            """
            Get the keywords from Scopus API using the provided DOI

            Args:
                doi (str: required): The DOI of the article to get keywords for

            Returns:
                keywords (list): List of keywords from Scopus API
            """
            try:
                url = f"https://api.elsevier.com/content/abstract/doi/{doi}"
                headers = {
                    "X-ELS-APIKey": self.scopus_api_key,
                    "Accept": "application/json",
                }
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                response_json = response.json()

                keywords = []
                if response_json.get("abstracts-retrieval-response"):
                    if response_json["abstracts-retrieval-response"].get(
                        "authkeywords"
                    ):
                        author_keywords = response_json["abstracts-retrieval-response"][
                            "authkeywords"
                        ].get("author-keyword", [])

                        if isinstance(author_keywords, dict):
                            if author_keywords.get("$"):
                                keywords.append(author_keywords.get("$"))
                        elif isinstance(author_keywords, list):
                            for keyword in author_keywords:
                                if keyword.get("$"):
                                    keywords.append(keyword.get("$"))

                return keywords
            except Exception as e:
                logger.error(f"Error in getting keywords from Scopus API: {e}")
                return []

        def _get_oaworks_data(doi: str):
            """
            Get the article metadata from OAWorks API using the provided DOI

            Args:
                doi (str: required): The DOI of the article to get metadata for

            Returns:
                response.json() (dict): The metadata of the article
            """
            try:
                url = f"https://bg.api.oa.works/metadata?id={doi}"
                response = requests.request("GET", url)
                if response.status_code != 200:
                    return f"Sorry, the status code is not 200, it is {response.status_code}"
                return response.json()
            except Exception as e:
                return f"Sorry, something went wrong: {e}"

        def _safe_get_nested(dictionary, keys, default=""):
            """
            Safely access nested dictionary values using a list of keys.
            Returns default value if any key in the chain doesn't exist.

            Args:
                dictionary (dict: required): The dictionary to access
                keys (list: required): List of keys to access the nested value
                default(str: optional): Value to return if path doesn't exist (default: "")

            Returns:
                The value at the specified path or the default value
            """
            current = dictionary
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    return default
                if current is None:
                    return default
            return current

        def _get_affiliation_details(affiliation_id: str):
            """
            Get the affiliation details from Scopus API using the provided affiliation ID

            Args:
                affiliation_id (str: required): The affiliation ID to get details for

            Returns:
                affiliation_name (str): The name of the affiliation
                affiliation_country (str): The affiliation_country of the affiliation
            """
            try:
                url = f"https://api.elsevier.com/content/affiliation/affiliation_id/{affiliation_id}"
                headers = {
                    "X-ELS-APIKey": self.scopus_api_key,
                    "Accept": "application/json",
                }
                affiliation_name = ""
                affiliation_country = ""
                response = requests.request("POST", url, headers=headers)
                results = response.json()["affiliation-retrieval-response"]
                if results:
                    if "affiliation-name" in results:
                        affiliation_name = results["affiliation-name"]
                    if "country" in results:
                        affiliation_country = results["country"]
                return affiliation_name, affiliation_country
            except KeyError:
                return "Sorry, I couldn't find anything about that, there could be an error with your scopus api key."
            except Exception as e:
                return f"Sorry, something went wrong: {e}"

        print("🔎 Getting metadata for doi: ", doi)
        scopus_data = None
        if self.scopus_api_key:
            scopus_data = _get_scopus_data(doi)
        oaworks_data = _get_oaworks_data(doi)

        # Check if either API returned an error string instead of data
        if isinstance(scopus_data, str):
            print(f"Error fetching Scopus data: {scopus_data}")
            scopus_data = {}

        if isinstance(oaworks_data, str):
            print(f"Error fetching OAWorks data: {oaworks_data}")
            oaworks_data = {}

        # Initialize empty dictionary for paper data
        paper_data = {
            "doi": doi,
            "title": "",
            "journal": "",
            "year": "",
            "isOpenAccess": False,
            "authors": [],
            "keywords": [],
        }

        # Get the article title
        if isinstance(scopus_data, dict) and _safe_get_nested(
            scopus_data, ["coredata", "dc:title"]
        ):
            paper_data["title"] = scopus_data["coredata"]["dc:title"]
        elif isinstance(oaworks_data, dict) and "title" in oaworks_data:
            paper_data["title"] = oaworks_data["title"]

        # Get the journal name
        if isinstance(scopus_data, dict) and _safe_get_nested(
            scopus_data, ["coredata", "prism:publicationName"]
        ):
            paper_data["journal"] = scopus_data["coredata"]["prism:publicationName"]
        elif isinstance(oaworks_data, dict) and "journal" in oaworks_data:
            paper_data["journal"] = oaworks_data["journal"]

        # Get the year
        if isinstance(scopus_data, dict):
            year = _safe_get_nested(
                scopus_data,
                ["item", "bibrecord", "head", "source", "publicationyear", "@first"],
            )
            if year:
                paper_data["year"] = year

        if (
            not paper_data["year"]
            and isinstance(oaworks_data, dict)
            and "year" in oaworks_data
        ):
            paper_data["year"] = oaworks_data["year"]

        # Get Open Access status
        if (
            isinstance(scopus_data, dict)
            and _safe_get_nested(scopus_data, ["coredata", "openaccess"]) == "0"
        ):
            paper_data["isOpenAccess"] = False
        elif isinstance(scopus_data, dict) and _safe_get_nested(
            scopus_data, ["coredata", "openaccess"]
        ):
            paper_data["isOpenAccess"] = True

        # Get authors
        authors = []
        if isinstance(scopus_data, dict) and "authors" in scopus_data:
            try:
                for author in scopus_data["authors"]["author"]:
                    # Get author name
                    author_name = "{} {}".format(
                        author["preferred-name"].get("ce:given-name", "") or "",
                        author["preferred-name"].get("ce:surname", "") or "",
                    )
                    # Handle affiliation - it might be a list or a single item
                    affiliation_id = ""
                    if "affiliation" in author:
                        if isinstance(author["affiliation"], list):
                            # If there are multiple affiliations, take the first one
                            if author["affiliation"]:
                                affiliation_id = author["affiliation"][0].get("@id", "")
                        else:
                            # Single affiliation
                            affiliation_id = author["affiliation"].get("@id", "")

                    if affiliation_id:
                        affiliation_result = _get_affiliation_details(affiliation_id)
                        if isinstance(affiliation_result, tuple):
                            affiliation_name, affiliation_country = affiliation_result
                            authors.append(
                                {
                                    "name": author_name,
                                    "affiliation_id": affiliation_id,
                                    "affiliation_name": affiliation_name,
                                    "affiliation_country": affiliation_country,
                                }
                            )
                        else:
                            # Handle error case in _get_affiliation_details
                            print(
                                f"Error getting affiliation details: {affiliation_result}"
                            )
                            authors.append(
                                {
                                    "name": author_name,
                                    "affiliation_id": affiliation_id,
                                    "affiliation_name": "",
                                    "affiliation_country": "",
                                }
                            )
                    else:
                        authors.append(
                            {
                                "name": author_name,
                                "affiliation_id": "",
                                "affiliation_name": "",
                                "affiliation_country": "",
                            }
                        )
            except KeyError as e:
                print(f"KeyError while processing authors from Scopus: {e}")

        elif isinstance(oaworks_data, dict) and "author" in oaworks_data:
            for author in oaworks_data["author"]:
                if isinstance(author, dict) and "name" in author:
                    authors.append(
                        {
                            "name": author["name"],
                            "affiliation_id": "",
                            "affiliation_name": "",
                            "affiliation_country": "",
                        }
                    )

        # Get the keywords - combine from both sources
        all_keywords = []

        # Get keywords from Scopus API if available
        if self.scopus_api_key:
            scopus_keywords = _get_keywords_from_scopus_api(doi)
            if scopus_keywords:
                all_keywords.extend(scopus_keywords)

        # Get keywords from OAWorks
        if isinstance(oaworks_data, dict) and "keyword" in oaworks_data:
            oaworks_keywords = oaworks_data["keyword"]
            if isinstance(oaworks_keywords, list):
                all_keywords.extend(oaworks_keywords)
            elif isinstance(oaworks_keywords, str):
                all_keywords.append(oaworks_keywords)

        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in all_keywords:
            if keyword and keyword.lower() not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword.lower())

        paper_data["keywords"] = unique_keywords

        return paper_data
