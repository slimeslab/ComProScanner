"""
fetch_metadata.py - Contains the class to fetch metadata from Scopus database using the Scopus API based on the given queries and years

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 21-02-2025
"""

# Standard library imports
import csv
import time
import xml.etree.ElementTree as ET
import sys
import os
import urllib.parse

# Third party imports
import requests
from dotenv import load_dotenv

# Local imports
from ..utils.configs import (
    BaseUrls,
    DefaultPaths,
)
from ..utils.error_handler import (
    ValueErrorHandler,
    KeyboardInterruptHandler,
)
from ..utils.logger import setup_logger

load_dotenv()

# configure logger
logger = setup_logger("comproscanner.log", module_name="fetch_metadata")


######## Class to fetch metadata from Scopus ########
class FetchMetadata:
    """
    Fetch metadata from Scopus using the Scopus API. Can be called from different scripts to fetch metadata for a given set of queries and years

    Args:
        main_property_keyword (str: required): The main keyword for the search and filenames
        base_queries (list: optional): The base queries for the search. If None, the main_property_keyword will be used as the base query. The list will be sorted alphabetically before processing.
        extra_queries (list: optional): The extra queries for the search which will be combined with the base queries. The list will be sorted alphabetically before processing.
        start_year (int: optional): The start year for the search (default: current year)
        end_year (int: optional): The end year for the search (default: current year  - 2)

    Raises:

        ValueError: If the base queries are None
        ValueError: If the Scopus API key is not set in the environment variables as SCOPUS_API_KEY
    """

    def __init__(
        self,
        main_property_keyword: str = None,
        base_queries: list = None,
        extra_queries: list = None,
        start_year: int = int(time.strftime("%Y")),
        end_year=int(time.strftime("%Y")) - 2,
    ):
        self.start_year = start_year
        self.end_year = end_year
        if self.start_year < self.end_year:
            raise ValueErrorHandler(
                message="Start year should be greater than the end year."
            )
        if self.start_year > int(time.strftime("%Y")):
            raise ValueErrorHandler(
                message="Start year cannot be greater than the current year."
            )
        if self.start_year == self.end_year:
            raise ValueErrorHandler(
                message="Start year and End year cannot be the same."
            )
        if main_property_keyword is None:
            raise ValueErrorHandler(
                message="main_property_keyword cannot be None. Please provide a valid keyword."
            )
        if base_queries is None:
            base_queries = [main_property_keyword]
        if main_property_keyword not in base_queries:
            base_queries.append(main_property_keyword)
        self.keywords = sorted(base_queries)
        self.extra_queries = sorted(extra_queries) if extra_queries is not None else []
        self.api_key = os.getenv("SCOPUS_API_KEY")
        if self.api_key is None:
            raise ValueErrorHandler(
                message="SCOPUS_API_KEY is not set in the environment variables."
            )
        all_filepaths = DefaultPaths(main_property_keyword)
        self.metadata_csv_filename = all_filepaths.METADATA_CSV_FILENAME
        self.base_url = BaseUrls.METADATA_QUERY_BASE_URL
        self.headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/xml",
        }
        self.is_exceeded = False

    def _construct_url(self, cursor, year, query, special_query):
        """
        Construct the URL for the request with cursor-based pagination

        Args:
            cursor (str): The cursor value ('*' for first request, or next cursor from previous response)
            year (int): The year for the request
            query (str): The query for the request
            special_query (str): The special query for the request

        Returns:
            str: The constructed URL
        """
        base = f"{self.base_url}PUBYEAR+%3D+{year}+{query}"
        url = base + (f"+{special_query}" if special_query else "") + "&count=200"
        url += f"&cursor={cursor}"
        return url

    def _send_request(self, url):
        """
        Send a request to the Scopus API and return the response

        Args:
            url (str): The URL for the request

        Returns:
            requests.models.Response: The response from the request
        """
        return requests.get(url, headers=self.headers)

    def parse_xml_data(self, response_text):
        """
        Parse the XML data from the response and return the root, entry elements, and namespaces

        Args:
            response_text (str): The response text from the request

        Returns:
            tuple: The root, entry elements, and namespaces
        """
        namespaces = {
            "default": "http://www.w3.org/2005/Atom",
            "prism": "http://prismstandard.org/namespaces/basic/2.0/",
            "dc": "http://purl.org/dc/elements/1.1/",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        }
        root = ET.fromstring(response_text)
        entry_elements = root.findall("default:entry", namespaces)
        return root, entry_elements, namespaces

    def _get_next_cursor(self, root, namespaces):
        """
        Extract the next cursor from the 'next' link in the XML response

        Args:
            root: The XML root element
            namespaces (dict): The XML namespaces

        Returns:
            str or None: The next cursor value (URL-encoded), or None if no more pages
        """
        # Find the 'next' link
        links = root.findall("default:link", namespaces)
        for link in links:
            if link.get("ref") == "next":
                next_url = link.get("href")
                if next_url and "cursor=" in next_url:
                    # Extract cursor parameter from URL (it's already URL-encoded)
                    cursor_param = next_url.split("cursor=")[1].split("&")[0]
                    return cursor_param

        return None

    def _get_total_results(self, root, namespaces):
        """
        Extract the total number of results from the response

        Args:
            root: The XML root element
            namespaces (dict): The XML namespaces

        Returns:
            int: Total number of results
        """
        total_elem = root.find("opensearch:totalResults", namespaces)
        if total_elem is not None:
            return int(total_elem.text)
        return 0

    def _process_entry_elements(self, entry_elements, namespaces, data):
        """
        Process all the entry elements using _process_single_entry() function

        Args:
            entry_elements (list): The list of entry elements
            namespaces (dict): The dictionary of namespaces
            data (dict): The dictionary to store the data

        Returns:
            dict: The dictionary with the processed data
        """
        for entry in entry_elements:
            data = self._process_single_entry(entry, namespaces, data)
        return data

    def get_element_text(self, entry, tag, namespaces):
        """
        Get the text of an element from the entry

        Args:
            entry (Element): The entry element
            tag (str): The tag to search for
            namespaces (dict): The dictionary of namespaces

        Returns:
            str: The text of the element
        """
        element = entry.find(tag, namespaces)
        return element.text if element is not None else None

    def _process_single_entry(self, entry, namespaces, data):
        """
        Process a single entry and add the data to the dictionary

        Args:
            entry (Element): The entry element
            namespaces (dict): The dictionary of namespaces
            data (dict): The dictionary to store the data

        Returns:
            dict: The dictionary with the processed data
        """
        doi_data = self.get_element_text(entry, "prism:doi", namespaces)
        if doi_data is not None and doi_data in data["doi"]:
            return data
        data["doi"].append(doi_data)
        data_tags = {
            "publication_name": "prism:publicationName",
            "issn": "prism:issn",
            "scopus_id": "dc:identifier",
            "article_title": "dc:title",
            "article_type": "default:subtypeDescription",
        }

        for key, tag in data_tags.items():
            element_text = self.get_element_text(entry, tag, namespaces)
            if element_text is None and key == "issn":
                element_text = self.get_element_text(entry, "prism:eIssn", namespaces)
            data[key].append(element_text)

        return data

    def _write_error_logs(
        self,
        year=None,
        query=None,
        special_query=None,
        page_number=None,
        status_code=None,
        exception=None,
    ):
        """
        Write the error logs to the log file

        Args:
            year (int): The year for the request (Optional)
            query (str): The query for the request (Optional)
            special_query (str): The special query for the request (Optional)
            page_number (int): The page number for the request (Optional)
            status_code (int): The status code for the request (Optional)
            exception (Exception): The exception for the request (Optional)
        """
        if year and query and special_query and page_number:
            logger.error(
                f"Year: {year}\nQuery: {query}\nSpecial Query: {special_query}\nPage Number: {page_number}\n"
            )
        elif status_code is not None:
            logger.error(f"Status Code: {status_code}\n")
        elif exception is not None:
            logger.error(f"Exception: {exception}\n")

    def fetch_and_process_data(
        self,
        cursor,
        year,
        query,
        data: dict,
        total_results: int,
        special_query: str,
        page_number: int,
    ):
        """
        Fetch and process the data for a given cursor

        Args:
            cursor (str): The cursor for the request
            year (int): The year for the request
            query (str): The query for the request
            data (dict): The dictionary to store the data
            total_results (int): The total number of results for the request
            special_query (str): The special query for the request
            page_number (int): The current page number

        Returns:
            tuple: The data, total_results, next_cursor, and a boolean indicating if the API limit is exceeded

        Raises:
            CustomErrorHandler: If an error occurs during the request
        """
        url = self._construct_url(cursor, year, query, special_query)
        next_cursor = None

        try:
            logger.debug(f"Sending request to URL: {url}")
            response = self._send_request(url)
            if response.status_code == 200:
                # process the response and add the data to the dictionary
                logger.info(
                    f"Received response for page {page_number}, adding data to dictionary..."
                )
                root, entry_elements, namespaces = self.parse_xml_data(response.text)
                logger.info(
                    f"Found {len(entry_elements)} entry elements for year {year}, page {page_number}\n"
                )
                data = self._process_entry_elements(entry_elements, namespaces, data)

                # Get total results on first page
                if total_results == 0:
                    total_results = self._get_total_results(root, namespaces)
                    logger.info(f"Total results available: {total_results}")

                # Get next cursor
                next_cursor = self._get_next_cursor(root, namespaces)

            elif response.status_code == 429:
                self._write_error_logs(
                    year,
                    query,
                    special_query,
                    page_number,
                    status_code=response.status_code,
                )
                self.is_exceeded = True
            else:
                self._write_error_logs(
                    year,
                    query,
                    special_query,
                    page_number,
                    status_code=response.status_code,
                )
        except ValueError as ve:
            logger.error(f"ValueError while processing the data: {ve}")
        except KeyError as ke:
            logger.error(f"KeyError while processing the data: {ke}")
        except Exception as e:
            logger.error(f"Exception while processing the data: {e}")

        return data, total_results, next_cursor

    def _fetch_paginated_data(self, year, query, special_query, data):
        """
        Handle the cursor-based pagination of the data for a given year, query, and special query

        Args:
            year (int): The year for the request
            query (str): The query for the request
            special_query (str): The special query for the request
            data (dict): The dictionary to store the data

        Returns:
            dict: The dictionary with the processed data
        """
        cursor = "*"  # Start with asterisk for first request
        total_results = 0
        page_number = 1

        while cursor:
            logger.debug(f"Processing page {page_number}")
            try:
                data, total_results, next_cursor = self.fetch_and_process_data(
                    cursor, year, query, data, total_results, special_query, page_number
                )

                # Log progress
                logger.info(
                    f"Progress: Fetched {len(data['doi'])} / {total_results} results"
                )

            except Exception as e:
                self._write_error_logs(
                    year, query, special_query, page_number, exception=e
                )
                logger.error(f"Exception while fetching the data: {e}")
                break

            if self.is_exceeded:
                self._write_error_logs(
                    year, query, special_query, page_number, status_code=429
                )
                logger.critical("Exceeded API limit. Exiting the program...")
                sys.exit(1)

            # Check if there's a next cursor
            if not next_cursor:
                logger.info(
                    f"No more pages to fetch. Total results fetched: {len(data['doi'])}"
                )
                break

            cursor = next_cursor
            page_number += 1

            # Rate limiting: respect 9 requests per second limit
            time.sleep(0.12)  # ~8 requests per second to stay safely under 9/sec limit

        return data

    def _write_to_csv(self, data):
        """
        Write the data to a CSV file

        Args:
            data (dict): The dictionary with the processed data
        """
        logger.debug(f"Writing data to CSV file...")
        directory = os.path.dirname(self.metadata_csv_filename)
        os.makedirs(directory, exist_ok=True)
        with open(
            self.metadata_csv_filename, "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))
        logger.info("Done...!!!\n")

    def main_fetch(self):
        """Main function to fetch metadata from Scopus using the Scopus API with cursor-based pagination

        Raises:
            CustomErrorHandler: If an error occurs during the request
        """
        for query in self.keywords:
            try:
                for year in range(self.start_year, self.end_year, -1):
                    # Define a dictionary to store the data
                    keys = [
                        "doi",
                        "publication_name",
                        "issn",
                        "scopus_id",
                        "article_title",
                        "article_type",
                    ]
                    data = {key: [] for key in keys}

                    # First request without special query
                    logger.verbose(f"Processing year {year} with query '{query}'...")
                    data = self._fetch_paginated_data(year, query, "", data)

                    # Fetch the data with special queries
                    for special_query in self.extra_queries:
                        logger.verbose(
                            f"Processing query {query} with special query '{special_query}'..."
                        )
                        data = self._fetch_paginated_data(
                            year, query, special_query, data
                        )

                    # Write the data to a CSV file
                    self._write_to_csv(data)

            except ValueError as ve:
                logger.error(f"ValueError while fetching the data: {ve}")
                continue
            except KeyError as ke:
                logger.error(f"KeyError while fetching the data: {ke}")
                continue
            except KeyboardInterrupt as kie:
                logger.error(f"Keyboard Interruption Detected. {kie}. Exiting...")
                raise KeyboardInterruptHandler()
            except Exception as e:
                logger.error(f"Exception while fetching the data: {e}")
                continue
