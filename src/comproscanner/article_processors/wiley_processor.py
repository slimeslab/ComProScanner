"""
wiley_processor.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 28-03-2025
"""

# Standard library imports
import os
import sys
import time
import tempfile

# third-party library imports
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
from dotenv import load_dotenv

# Custom imports
from ..utils.configs import (
    ArticleRelatedKeywords,
    RAGConfig,
    DefaultPaths,
    DatabaseConfig,
    BaseUrls,
)
from ..utils.database_manager import (
    MySQLDatabaseManager,
    CSVDatabaseManager,
    VectorDatabaseManager,
)
from ..utils.error_handler import ValueErrorHandler, KeyboardInterruptHandler
from ..utils.logger import setup_logger
from ..utils.pdf_to_markdown_text import PDFToMarkdownText
from ..utils.common_functions import (
    get_paper_metadata_from_oaworks,
    return_error_message,
)
from ..utils.common_functions import write_timeout_file

# Load environment variables from .env file
load_dotenv()

# configure logger
logger = setup_logger("comproscanner.log", module_name="wiley_processor")


######## Class to process Wiley articles ########
class WileyArticleProcessor:
    """
    Get the article as PDF using Wiley API and process it to extract the required sections of the article and save it to the MySQL database and CSV files and create a vector store if the relevant data is present in the article.

    Args:
        main_property_keyword (str: Required): The main keyword to process the articles for and file naming
        property_keywords (dict: Required): A dictionary of property keywords which will be used for filtering sentences and should look like the following:
        {
            "exact_keywords": ["example1", "example2"],
            "substring_keywords": [" example 1 ", " example 2 "],
        }
        sql_batch_size (int): The number of rows to write to the database at once (Applicable only if is_sql_db is True) (default: 500)
        csv_batch_size (int): The number of rows to write to the CSV file at once (default: 1)
        start_row (int): The row number to start processing from (default: None)
        end_row (int): The row number to end processing at (default: None)
        doi_list (list): A list of DOIs to process (default: None)
        is_sql_db (bool): A flag to indicate if the data should be written to the database (default: False)
        rag_config (RAGConfig): An instance of the RAGConfig class (default: RAGConfig())
    """

    def __init__(
        self,
        main_property_keyword: str = None,
        property_keywords: dict = None,
        sql_batch_size: int = 500,
        csv_batch_size: int = 1,
        start_row: int = None,
        end_row: int = None,
        doi_list: list = None,
        is_sql_db: bool = False,
        is_save_pdf: bool = False,
        rag_config: RAGConfig = RAGConfig(),
    ):
        keyword_message = return_error_message("main_property_keyword")
        property_keywords_message = return_error_message("property_keywords")
        api_key_message = return_error_message("wiley_api_key")
        # Required parameters
        self.keyword = main_property_keyword
        if self.keyword is None:
            logger.error(f"{keyword_message}")
            raise ValueErrorHandler(f"{keyword_message}")
        self.property_keywords = property_keywords
        if self.property_keywords is None:
            logger.error(f"{property_keywords_message}")
            raise ValueErrorHandler(f"{property_keywords_message}")
        self.api_key = os.getenv("WILEY_API_KEY")
        if self.api_key is None:
            logger.error(f"{api_key_message}")
            raise ValueErrorHandler(f"{api_key_message}")
        # create instances
        self.all_paths = DefaultPaths(self.keyword)
        self.db_configs = DatabaseConfig(self.keyword, is_sql_db)
        self.metadata_csv_filename = self.all_paths.METADATA_CSV_FILENAME
        self.csv_path = self.db_configs.EXTRACTED_CSV_FOLDERPATH
        self.paperdata_table_name = self.db_configs.PAPERDATA_TABLE_NAME
        self.sql_batch_size = sql_batch_size
        self.csv_batch_size = csv_batch_size
        # Optional parameters
        self.start_row = start_row
        self.end_row = end_row
        self.doi_list = doi_list
        self.is_sql_db = is_sql_db
        self.is_save_pdf = is_save_pdf
        self.rag_config = rag_config
        # Takes from config file
        self.timeout_file = self.all_paths.TIMEOUT_DOI_LOG_FILENAME
        self.article_related_keywords = ArticleRelatedKeywords()

        self.headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/xml",
        }
        self.df = None
        self.new_df = pd.DataFrame(
            columns=[
                "doi",
                "article_title",
                "publication_name",
                "publisher",
                "abstract",
                "introduction",
                "exp_methods",
                "comp_methods",
                "results_discussion",
                "conclusion",
                "is_property_mentioned",
            ]
        )
        self.valid_property_articles = 0
        self.source = "wiley"
        self.csv_filepath = (
            f"{self.csv_path}/{self.source}_{self.keyword}_paragraphs.csv"
        )

        self.sql_db_manager = MySQLDatabaseManager(self.keyword, self.is_sql_db)
        self.csv_db_manager = CSVDatabaseManager()
        self.vector_db_manager = VectorDatabaseManager(rag_config=self.rag_config)
        self.is_exceeded = False

    def _load_and_preprocess_data(self):
        """
        Load and preprocess the metadata CSV file to get the DOIs of the articles to process.
        """
        self.df = pd.read_csv(self.metadata_csv_filename)
        self.df = self.df.dropna(subset=["doi"])

        # Apply row limits if specified
        if self.start_row is not None and self.end_row is not None:
            self.df = self.df.iloc[self.start_row : self.end_row]
        elif self.start_row is not None:
            self.df = self.df.iloc[self.start_row :]
        elif self.end_row is not None:
            self.df = self.df.iloc[: self.end_row]

        # Filter DOIs with the main property keyword
        self.df = self.df[self.df["general_publisher"].str.lower() == "wiley"]
        self.df = self.df.reset_index(drop=True)

        # Get processed DOIs from all sources
        processed_dois = set()

        # Create CSV directory if it doesn't exist
        os.makedirs(self.csv_path, exist_ok=True)

        # Check CSV files
        if os.path.exists(self.csv_filepath):
            try:
                df = pd.read_csv(self.csv_filepath)
                processed_dois.update(df["doi"].tolist())
            except Exception as e:
                logger.warning(
                    f"Error reading CSV file: {e}. Processed DOIs from CSV will be ignored."
                )

        # Check SQL database if enabled
        if self.is_sql_db:
            try:
                sql_engine = create_engine(self.db_configs.DATABASE_CONNECTION_URL)
                if self.sql_db_manager.table_exists(self.paperdata_table_name):
                    db_df = pd.read_sql_query(
                        f"SELECT doi FROM {self.paperdata_table_name}", sql_engine
                    )
                    processed_dois.update(db_df["doi"].tolist())
                sql_engine.dispose()
            except Exception as e:
                logger.warning(
                    f"Error reading database: {e}. Processed DOIs from database will be ignored."
                )

        unprocessed_dois = set(self.df["doi"]) - processed_dois
        self.df = self.df[self.df["doi"].isin(unprocessed_dois)]

    def _send_request(self, doi):
        """
        Send a GET request to the Wiley API to get the article as PDF.

        Args:
            doi (str: Required): The DOI of the article.

        Returns:
            tmp_path (str): The path of the temporary PDF file, or "Not Found" if the article is not found, or None if there was an error.

        Raises:
            KeyboardInterruptHandler: If user interrupts the retry process

        Note:
            This method will retry indefinitely for connection errors until successful connection or KeyboardInterrupt
        """
        retry_delay = 60  # seconds
        retry_count = 0
        url = f"{BaseUrls.WILEY_ARTICLE_BASE_URL}{doi}"
        headers = {"Wiley-TDM-Client-Token": os.getenv("WILEY_API_KEY")}

        while True:
            try:
                # If we had retries before and now succeeded, log the success
                if retry_count > 0:
                    logger.info(
                        f"Connection restored successfully after {retry_count} attempts for DOI {doi}"
                    )

                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code == 200:
                    if self.is_save_pdf:
                        filepath = self._save_pdf(doi, response)
                        return filepath
                    else:
                        with tempfile.NamedTemporaryFile(
                            suffix=".pdf", delete=False
                        ) as tmp_file:
                            tmp_file.write(response.content)
                            tmp_file.flush()
                            tmp_path = tmp_file.name
                            return tmp_path

                elif response.status_code == 429:
                    logger.critical(
                        f"API rate limit exceeded. Please try again later. DOI: {doi}"
                    )
                    self.is_exceeded = True
                    return None

                elif response.status_code == 400:
                    logger.error(f"Bad request for DOI: {doi}")
                    return None

                elif response.status_code == 404:
                    logger.warning(f"Article not found for DOI: {doi}")
                    return "Not Found"

                else:
                    logger.warning(
                        f"Request failed with status code {response.status_code} for DOI: {doi}"
                    )
                    return None

            except (ConnectionError, Timeout) as e:
                retry_count += 1
                logger.warning(
                    f"Connection error occurred while fetching DOI {doi}: {type(e).__name__}: {str(e)}"
                )
                logger.info(
                    f"Retrying in {retry_delay} seconds... (Attempt #{retry_count})"
                )
                logger.info(
                    f"Waiting for connection to restore. Press Ctrl+C to cancel."
                )

                try:
                    time.sleep(retry_delay)
                except KeyboardInterrupt:
                    logger.warning("User interrupted the retry process.")
                    raise KeyboardInterruptHandler()

            except requests.exceptions.ReadTimeout as e:
                retry_count += 1
                logger.warning(f"Read timeout error for DOI: {doi}")
                logger.info(
                    f"Retrying in {retry_delay} seconds... (Attempt #{retry_count})"
                )
                logger.info(
                    f"Waiting for connection to restore. Press Ctrl+C to cancel."
                )

                # Write to timeout file for tracking
                write_timeout_file(doi, self.timeout_file)

                try:
                    time.sleep(retry_delay)
                except KeyboardInterrupt:
                    logger.warning("User interrupted the retry process.")
                    raise KeyboardInterruptHandler()

            except RequestException as e:
                retry_count += 1

                # Check if we have a response object with status code
                if hasattr(e, "response") and e.response is not None:
                    response = e.response
                    if response.status_code == 429:
                        logger.critical(f"API rate limit exceeded for DOI: {doi}")
                        self.is_exceeded = True
                        return None
                    elif response.status_code == 400:
                        logger.error(f"Bad request for DOI: {doi}")
                        return None
                    elif response.status_code == 404:
                        logger.warning(f"Article not found for DOI: {doi}")
                        return None
                    else:
                        logger.warning(
                            f"Request failed with status code {response.status_code} for DOI: {doi}"
                        )
                        # For other HTTP errors, retry
                        logger.info(
                            f"Retrying in {retry_delay} seconds... (Attempt #{retry_count})"
                        )
                        try:
                            time.sleep(retry_delay)
                        except KeyboardInterrupt:
                            logger.warning("User interrupted the retry process.")
                            raise KeyboardInterruptHandler()
                else:
                    # No response object, it's a connection issue
                    logger.error(
                        f"Request exception occurred while fetching DOI {doi}: {type(e).__name__}: {str(e)}"
                    )
                    logger.info(
                        f"Retrying in {retry_delay} seconds... (Attempt #{retry_count})"
                    )

                    try:
                        time.sleep(retry_delay)
                    except KeyboardInterrupt:
                        logger.warning("User interrupted the retry process.")
                        raise KeyboardInterruptHandler()

            except KeyboardInterrupt:
                logger.warning("User interrupted the connection attempt.")
                raise KeyboardInterruptHandler()

            except Exception as e:
                # For other unexpected errors, log and return None (don't retry)
                logger.error(
                    f"Unexpected error for DOI {doi}: {type(e).__name__}: {str(e)}"
                )
                return None

    def _save_pdf(self, doi, response):
        """
        Save the PDF file to the local disk.

        Args:
            doi (str: Required): The DOI of the article.
            response (requests.Response: Required): The response object from the GET request.
        """
        pdf_folderpath = f"downloaded_files/pdfs/wiley"
        if not os.path.exists(pdf_folderpath):
            os.makedirs(pdf_folderpath)
        modified_doi = doi.replace("/", "_")
        filepath = f"{pdf_folderpath}/{modified_doi}.pdf"
        with open(f"{filepath}", "wb") as f:
            f.write(response.content)
        return filepath

    def _process_articles(self):
        """
        Main function to process the Wiley articles and save the required sections to the MySQL database and CSV files and create a vector store if the relevant data is present in the article.
        """
        logger.debug(f"\nProcessing articles for the first time...")
        self._load_and_preprocess_data()
        sql_dataframes = []
        csv_dataframes = []

        if self.doi_list is None:
            iterable = self.df.iterrows()
            total = self.df.shape[0]
        else:
            iterable = enumerate(self.doi_list)
            total = len(self.doi_list)

        for _, item in tqdm(iterable, total=total, colour="#d6adff"):
            if self.is_exceeded:
                logger.critical(
                    "API rate limit exceeded. Exiting the Wiley processing..."
                )
                break
            try:
                doi = None
                if self.doi_list is None:
                    row = item
                    doi = row["doi"]
                else:
                    doi = item
                    self.df = pd.read_csv(self.metadata_csv_filename)
                    matching_rows = self.df[self.df["doi"] == doi]
                    if matching_rows.empty:
                        logger.warning(
                            f"DOI {doi} for Wiley article was not found in the metadata CSV."
                        )
                        continue
                    row = matching_rows.iloc[0]

                logger.debug(f"\n\nProcessing Wiley article DOI: {row['doi']}")

                # Download PDF
                file_path = self._send_request(row["doi"])
                if file_path is None:
                    logger.warning(f"Failed to download PDF for DOI {row['doi']}")
                    continue

                # Handle "Not Found" articles
                if file_path == "Not Found":
                    empty_data = {
                        "doi": row["doi"],
                        "article_title": row["article_title"],
                        "publication_name": row["publication_name"],
                        "publisher": row["metadata_publisher"],
                        "abstract": "",
                        "introduction": "",
                        "exp_methods": "",
                        "comp_methods": "",
                        "results_discussion": "",
                        "conclusion": "",
                        "is_property_mentioned": "0",
                    }
                    empty_row = pd.DataFrame([empty_data])
                    sql_dataframes.append(empty_row)
                    csv_dataframes.append(empty_row)
                    if len(sql_dataframes) == self.sql_batch_size:
                        final_sql_df = pd.concat(sql_dataframes, ignore_index=True)
                        if self.is_sql_db:
                            self.sql_db_manager.write_to_sql_db(
                                self.paperdata_table_name, final_sql_df
                            )
                        sql_dataframes = []
                        time.sleep(5)
                    if len(csv_dataframes) == self.csv_batch_size:
                        final_csv_df = pd.concat(csv_dataframes, ignore_index=True)
                        self.csv_db_manager.write_to_csv(
                            final_csv_df,
                            self.csv_path,
                            self.keyword,
                            self.source,
                            self.csv_batch_size,
                        )
                        csv_dataframes = []
                        time.sleep(5)
                    continue

                # Get metadata
                title, journal_name, publisher = get_paper_metadata_from_oaworks(
                    row["doi"]
                )

                # Convert PDF to Markdown
                pdf_to_md = PDFToMarkdownText(file_path)
                md_text = pdf_to_md.convert_to_markdown()

                # Check if conversion was successful
                if md_text is None:
                    logger.error(
                        f"Failed to convert PDF to markdown for DOI {row['doi']}"
                    )
                    # Clean up temporary file if it exists
                    if not self.is_save_pdf and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logger.warning(
                                f"Failed to remove temp file {file_path}: {e}"
                            )
                    continue

                # Process the markdown text
                all_sections = pdf_to_md.clean_text(md_text)
                row = pdf_to_md.append_section_to_df(
                    all_sections,
                    row["doi"],
                    title,
                    journal_name,
                    publisher,
                    self.property_keywords,
                    self.vector_db_manager,
                    logger,
                )
                sql_dataframes.append(row)
                csv_dataframes.append(row)

                if row["is_property_mentioned"].iloc[0] == "1":
                    self.valid_property_articles += 1

                # Clean up temporary file if not saving PDFs
                if not self.is_save_pdf and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {file_path}: {e}")

                # Write to SQL database when batch is full
                if len(sql_dataframes) == self.sql_batch_size:
                    final_df = pd.concat(sql_dataframes, ignore_index=True)
                    if self.is_sql_db:
                        self.sql_db_manager.write_to_sql_db(
                            self.paperdata_table_name, final_df
                        )
                    sql_dataframes = []
                    time.sleep(5)

                # Write to CSV when batch is full
                if len(csv_dataframes) == self.csv_batch_size:
                    final_df = pd.concat(csv_dataframes, ignore_index=True)
                    self.csv_db_manager.write_to_csv(
                        final_df,
                        self.csv_path,
                        self.keyword,
                        self.source,
                        self.csv_batch_size,
                    )
                    csv_dataframes = []
                    time.sleep(5)

                time.sleep(0.2)

            except KeyboardInterrupt as kie:
                logger.error(f"Keyboard Interruption Detected. {kie}")
                raise KeyboardInterruptHandler()
            except Exception as e:
                logger.error(f"Error processing article with DOI {row['doi']}: {e}")
                continue

        # Append any remaining dataframes at the end
        try:
            if sql_dataframes:
                remaining_sql_df = pd.concat(sql_dataframes, ignore_index=True)
                if self.is_sql_db:
                    self.sql_db_manager.write_to_sql_db(
                        self.paperdata_table_name, remaining_sql_df
                    )
            if csv_dataframes:
                remaining_csv_df = pd.concat(csv_dataframes, ignore_index=True)
                self.csv_db_manager.write_to_csv(
                    remaining_csv_df, self.csv_path, self.keyword, self.source
                )
        except Exception as e:
            logger.error(f"Error writing remaining dataframes: {e}")

    def _process_with_timeout_handling(self):
        """Process articles and handle any timeouts"""
        while os.path.isfile(self.timeout_file):
            logger.debug(f"\nProcessing articles with timeout handling...")
            with open(self.timeout_file, "r") as file:
                timeout_dois = [line.strip() for line in file]

            if not timeout_dois:
                break

            self.doi_list = timeout_dois
            if self.doi_list:
                self._process_articles()
                # delete timeout file if exists
                if os.path.exists(self.timeout_file):
                    os.remove(self.timeout_file)

    def process_wiley_articles(self):
        """Run Wiley article processing workflow"""
        logger.verbose(f"\n\nWiley articles processing started...")
        self._process_articles()
        self._process_with_timeout_handling()
        logger.verbose(f"\n\nWiley articles processing completed...")
        logger.info(f"\nTotal valid property articles: {self.valid_property_articles}")
