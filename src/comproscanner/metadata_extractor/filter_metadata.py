"""
fetch_metadata.py - Contains the class to filter metadata based on only articles and letters, remove duplicates, and update missing publisher information.

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 21-02-2025
"""

# Standard library imports
import time
import os
import sys

# Third-party imports
from dotenv import load_dotenv
from lxml import etree
from tqdm import tqdm
import pandas as pd
import requests

# Local imports
from ..utils.configs import (
    BaseUrls,
    DefaultPaths,
)
from ..utils.error_handler import (
    FileNotFoundErrorHandler,
    ValueErrorHandler,
    CustomErrorHandler,
)
from ..utils.logger import setup_logger

load_dotenv()

######## logger Configuration ########
logger = setup_logger("metadata_collector.log")


######## Class to filter metadata ########
class FilterMetadata:
    """Class to filter metadata based on only articles and letters, remove duplicates, and update missing publisher information

    Args:
        main_property_keyword (str): Main property keyword to filter metadata

    Raises:
        ValueError: If the main property keyword is not provided
        ValueError: If the SCOPUS_API_KEY is not set in the environment variables
    """

    def __init__(self, main_property_keyword: str = None):
        if main_property_keyword is None:
            raise ValueErrorHandler("Main property keyword not provided.")
        all_paths = DefaultPaths(main_property_keyword)
        self.filepath = all_paths.METADATA_CSV_FILENAME
        self.issn_base_url = BaseUrls.ISSN_BASE_URL
        self.scopusid_base_url = BaseUrls.SCOPUSID_BASE_URL
        self.api_key = os.getenv("SCOPUS_API_KEY")
        if self.api_key is None:
            raise ValueErrorHandler(
                message="SCOPUS_API_KEY is not set in the environment variables."
            )
        self.headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/xml",
        }
        self.publisher_mapping = {
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
        self.is_exceeded = False

    def _remove_invalid_rows(self, df):
        """Removes rows with missing article types or DOIs

        Args:
            df (pd.DataFrame): DataFrame containing metadata

        Returns:
            df (pd.DataFrame): DataFrame with invalid rows removed

        Raises:
            KeyError: If the column name is invalid
            ValueError: If the value is invalid
            Exception: If any other error occurs
        """
        try:
            valid_article_types = ["Article", "Letter"]
            df = df[df["article_type"].isin(valid_article_types)]
            logger.warning("Removing rows with missing ISSN or DOI...")
            df = df.dropna(subset=["issn", "doi"])
            return df
        except KeyError as e:
            logger.error(f"KeyError: Invalid column name. {e}")
        except ValueError as e:
            logger.error(f"ValueError: Invalid value. {e}")
        except Exception as e:
            logger.error(f"An error occurred. {e}")

    def _remove_duplicate_doi_rows(self, df):
        """Removes duplicate rows based on DOI

        Args:
            df (pd.DataFrame): DataFrame containing metadata

        Returns:
            df (pd.DataFrame): DataFrame with duplicate rows removed

        Raises:
            Exception: If any error occurs
        """
        try:
            initial_rows = len(df)
            df = df.drop_duplicates(subset=["doi"], keep="first")
            removed_rows = initial_rows - len(df)
            logger.debug(f"Removed {removed_rows} duplicate DOI rows.")
            return df
        except Exception as e:
            logger.error(f"An error occurred. {e}")

    def _get_missing_publisher_entries(self, df):
        """Gets entries where publisher information is missing

        Args:
            df (pd.DataFrame): DataFrame containing metadata

        Returns:
            df_missing (pd.DataFrame): DataFrame with missing publisher information

        Raises:
            Exception: If any error occurs
        """
        try:
            # Check for missing publisher information in the DataFrame
            df_missing = df[df["metadata_publisher"].isna()]
            logger.debug(
                f"Found {len(df_missing)} entries with missing publisher information."
            )
            return df_missing
        except Exception as e:
            logger.error(f"Error identifying missing publishers: {e}")

    def _get_unique_identifiers_for_missing(self, df_missing):
        """Gets unique ISSNs and Scopus IDs for entries with missing publishers"""
        try:
            # Group by journal and take the first occurrence of each
            unique_data = (
                df_missing.groupby("issn")
                .agg({"scopus_id": "first", "publication_name": "first"})
                .reset_index()
            )

            issn_list = unique_data["issn"].tolist()
            scopus_id_list = unique_data["scopus_id"].tolist()
            publication_names = unique_data["publication_name"].tolist()

            logger.debug(f"Need to process {len(issn_list)} unique journals.")
            return issn_list, scopus_id_list, publication_names
        except Exception as e:
            logger.error(f"Error getting unique identifiers: {e}")

    def _add_publisher_to_df_and_save(self, response, issn=None, scopus_id=None):
        """Parses XML response to add publisher information to DataFrame and saves immediately

        Args:
            response (requests.Response): Response object from the API
            df (pd.DataFrame): DataFrame containing metadata
            issn (str): ISSN of the publication
            scopus_id (str): Scopus ID of the publication

        Raises:
            AttributeError: If the publisher is not found
            Exception: If any other error occurs
        """
        try:
            utf8_parser = etree.XMLParser(encoding="utf-8")
            root = etree.fromstring(response.text.encode("utf-8"), parser=utf8_parser)
            namespaces = {"dc": "http://purl.org/dc/elements/1.1/"}
            publisher_element = root.find(".//dc:publisher", namespaces)

            if publisher_element is not None:
                # Create a temporary DataFrame to store all existing rows
                temp_df = pd.read_csv(self.filepath, low_memory=False)
                publisher = publisher_element.text
                logger.info(f"Publisher: {publisher} found")

                # Update publisher in the temporary DataFrame
                if scopus_id is not None:
                    mask = temp_df["scopus_id"] == scopus_id
                else:
                    mask = temp_df["issn"] == issn

                # Update only if rows exist
                if mask.any():
                    temp_df.loc[mask, "metadata_publisher"] = publisher

                    # Update general publisher based on the mapping
                    for full_name, short_name in self.publisher_mapping.items():
                        if full_name.lower() in publisher.lower():
                            temp_df.loc[mask, "general_publisher"] = short_name
                            break

                    # Save the updated DataFrame
                    temp_df.to_csv(self.filepath, index=False)
                    logger.info(
                        f"Updated {mask.sum()} rows in CSV for {'Scopus ID: ' + str(scopus_id) if scopus_id else 'ISSN: ' + str(issn)}"
                    )
                else:
                    logger.warning(
                        f"No matching rows found for {'Scopus ID: ' + str(scopus_id) if scopus_id else 'ISSN: ' + str(issn)}"
                    )
            else:
                logger.error(
                    f"Publisher not found for the journal with {'ISSN: ' + str(issn) if issn else 'Scopus ID: ' + str(scopus_id)}"
                )
        except Exception as e:
            logger.error(f"An error occurred. {e}")

    def _get_publisher_from_issn(self, issn, df):
        """Fetches publisher information using ISSN and updates CSV

        Args:
            issn (str): ISSN of the publication
            df (pd.DataFrame): DataFrame containing metadata

        Returns:
            bool: True if successful, False otherwise

        Raises:
            Exception: If any error occurs
        """
        try:
            response = requests.get(f"{self.issn_base_url}{issn}", headers=self.headers)
            if response.status_code == 200:
                self._add_publisher_to_df_and_save(response, issn=issn)
            elif response.status_code == 404:
                return False
            elif response.status_code == 429:
                self.is_exceeded = True
                raise CustomErrorHandler(
                    message="API rate limit exceeded. Please try again later or use a different API key.",
                    status_code=429,
                )
            else:
                logger.error(f"ISSN API Error: {response.status_code}")
                return False
        except CustomErrorHandler:
            raise
        except Exception as e:
            logger.error(f"Error processing ISSN {issn}: {e}")
            return False
        return True

    def _get_publisher_from_scopus_id(self, scopus_id, df):
        """Fetches publisher information using Scopus ID and updates CSV

        Args:
            scopus_id (str): Scopus ID of the publication
            df (pd.DataFrame): DataFrame containing metadata

        Returns:
            bool: True if successful, False otherwise

        Raises:
            Exception: If any error occurs
        """
        try:
            scopus_url = f"{self.scopusid_base_url}{scopus_id}?APIKey={self.api_key}"
            response = requests.get(scopus_url)
            if response.status_code == 200:
                self._add_publisher_to_df_and_save(response, scopus_id=scopus_id)
            elif response.status_code == 404:
                logger.error(f"URL not found for Scopus ID: {scopus_id}")
                return False
            elif response.status_code == 429:
                self.is_exceeded = True
                logger.critical(
                    f"API rate limit exceeded. Please try again later or use a different API key."
                )
                raise CustomErrorHandler(
                    message="API rate limit exceeded. Please try again later or use a different API key.",
                    status_code=429,
                )
            else:
                logger.error(f"Scopus ID API Error: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error processing Scopus ID {scopus_id}: {e}")
            return False
        return True

    def _process_journal(self, issn, scopus_id, df, publication_name):
        """Processes journal information, retrieves publisher, and handles API errors

        Args:
            issn (str): ISSN of the publication
            scopus_id (str): Scopus ID of the publication
            df (pd.DataFrame): DataFrame containing metadata
            publication_name (str): Name of the publication

        Raises:
            Exception: If any error occurs
        """
        logger.debug(f"\nISSN: {issn}, Scopus ID: {scopus_id}\n")
        logger.debug(f"Processing journal: {publication_name}. ISSN: {issn}")
        try:
            if not self._get_publisher_from_issn(issn, df):
                if self.is_exceeded:
                    sys.exit("API rate limit exceeded. Exiting the program...")
                if not self._get_publisher_from_scopus_id(scopus_id, df):
                    if self.is_exceeded:
                        logger.critical(
                            f"API rate limit exceeded. Exiting the program..."
                        )
                        sys.exit()
                    logger.warning(
                        f"Unable to retrieve publisher for {publication_name}"
                    )
            time.sleep(0.2)

        except Exception as e:
            logger.error(f"An error occurred. {e}")

    def _update_from_existing_data(self, df, df_missing):
        """
        Update missing publisher information using existing entries in the DataFrame

        Args:
            df (pd.DataFrame): Complete DataFrame
            df_missing (pd.DataFrame): DataFrame with missing publisher information

        Returns:
            tuple: (updated DataFrame, remaining missing entries DataFrame)
        """
        # Get entries with valid publisher information
        df_valid = df[df["metadata_publisher"].notna()].copy()
        updated_indices = []

        # For each missing entry, try to find a match
        for idx in df_missing.index:
            row = df_missing.loc[idx]

            # Try to find a matching record with valid publisher info
            matching_entry = df_valid[
                (df_valid["issn"] == row["issn"])
                | (df_valid["scopus_id"] == row["scopus_id"])
            ]

            if not matching_entry.empty:
                # Update both publisher fields
                df.loc[idx, "metadata_publisher"] = matching_entry.iloc[0][
                    "metadata_publisher"
                ]
                if pd.notna(matching_entry.iloc[0]["general_publisher"]):
                    df.loc[idx, "general_publisher"] = matching_entry.iloc[0][
                        "general_publisher"
                    ]
                updated_indices.append(idx)

        if updated_indices:
            logger.info(
                f"Updated {len(updated_indices)} entries using existing publisher information"
            )

        # Return remaining missing entries
        remaining_missing = df_missing.drop(index=updated_indices)
        return df, remaining_missing

    def filter_metadata(self):
        """Main function to update missing publisher information in the DataFrame"""
        try:
            # Read the existing CSV
            df = pd.read_csv(self.filepath, low_memory=False)

            # Remove invalid rows and duplicates
            df = self._remove_invalid_rows(df)
            df = self._remove_duplicate_doi_rows(df)

            # Save the cleaned DataFrame back to CSV
            df.to_csv(self.filepath, index=False)
            logger.info(f"Saved cleaned DataFrame with {len(df)} rows.")

            # Ensure required columns exist
            if "metadata_publisher" not in df.columns:
                df["metadata_publisher"] = None
            if "general_publisher" not in df.columns:
                df["general_publisher"] = None

            # Get entries with missing publisher information
            df_missing = self._get_missing_publisher_entries(df)

            if len(df_missing) > 0:
                # First update from existing data and save intermediate results
                df, remaining_missing = self._update_from_existing_data(df, df_missing)
                df.to_csv(self.filepath, index=False)
                logger.info("Saved DataFrame after updating from existing data.")

                # Process remaining missing entries
                if len(remaining_missing) > 0:
                    issn_list, scopus_id_list, publication_names = (
                        self._get_unique_identifiers_for_missing(remaining_missing)
                    )
                    for issn, scopus_id, publication_name in tqdm(
                        zip(issn_list, scopus_id_list, publication_names),
                        total=len(issn_list),
                        colour="#d6adff",
                    ):
                        self._process_journal(issn, scopus_id, df, publication_name)
                else:
                    logger.info("All missing entries updated using existing data.")
            else:
                logger.warning("No missing publisher information found.")

            return df

        except Exception as e:
            logger.error(f"Error in filter_metadata: {str(e)}")
