"""
pdfs_processor.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 21-03-2025
"""

# Importing required libraries
import logging
import time
import json
import pandas as pd
from tqdm import tqdm
import glob
import re
import os

# Custom imports
from ..utils.configs import (
    ArticleRelatedKeywords,
    RAGConfig,
    DefaultPaths,
    DatabaseConfig,
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


# configure logger
logger = setup_logger("comproscanner.log", module_name="pdfs_processor")


######## Class to process PDFs in a folder ########
class PDFsProcessor:
    def __init__(
        self,
        folder_path: str = None,
        main_property_keyword: str = None,
        property_keywords: list = None,
        sql_batch_size: int = 500,
        csv_batch_size: int = 2000,
        is_sql_db: bool = False,
        rag_config: RAGConfig = RAGConfig(),
    ):
        """Class to process PDFs in a folder and process them to extract the required sections of the articles and save them to the MySQL database and CSV files and create a vector store if the relevant data is present in the article.

        Args:
            folder_path (str, required): Path to the folder containing PDFs.
            main_property_keyword (str: Required): The main keyword to process the articles for and file naming.
            property_keywords (dict: Required): A dictionary of property keywords which will be used for filtering sentences and should look like the following:
            {
                "exact_keywords": ["example1", "example2"],
                "substring_keywords": [" example 1 ", " example 2 "],
            }
            sql_batch_size (int): The number of rows to write to the database at once (Applicable only if is_sql_db is True) (default: 500)
            csv_batch_size (int): The number of rows to write to the CSV file at once (default: 2000)
            is_sql_db (bool): A flag to indicate if the data should be written to the database (default: False)
            rag_config (RAGConfig): An instance of the RAGConfig class (default: RAGConfig())

        Raises:
            ValueErrorHandler: If the folder_path, main_property_keyword, or property_keywords is not provided.
        """
        self.folder_path = folder_path
        if self.folder_path == None:
            logger.error(f"PDF folder path cannot be empty. Exiting...")
            raise ValueErrorHandler(f"PDF folder path cannot be empty. Exiting...")
        keyword_message = return_error_message("main_property_keyword")
        property_keywords_message = return_error_message("property_keywords")
        self.keyword = main_property_keyword
        if self.keyword is None:
            logger.error(f"{keyword_message}")
            raise ValueErrorHandler(f"{keyword_message}")
        self.property_keywords = property_keywords
        if self.property_keywords is None:
            logger.error(f"{property_keywords_message}")
            raise ValueErrorHandler(f"{property_keywords_message}")
        self.is_sql_db = is_sql_db

        self.identifier = ""
        self.doi = ""
        self.all_paths = DefaultPaths(self.keyword)
        self.db_configs = DatabaseConfig(self.keyword, self.is_sql_db)
        self.csv_path = self.db_configs.EXTRACTED_CSV_FOLDERPATH
        self.paperdata_table_name = self.db_configs.PAPERDATA_TABLE_NAME
        if self.is_sql_db:
            self.sql_batch_size = sql_batch_size
        else:
            self.sql_batch_size = csv_batch_size
        self.csv_batch_size = csv_batch_size
        self.rag_config = rag_config
        self.timeout_file = self.all_paths.TIMEOUT_DOI_LOG_FILENAME
        self.article_keywords = ArticleRelatedKeywords()

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
        self.source = "pdf"

        if self.is_sql_db:
            self.sql_db_manager = MySQLDatabaseManager(self.keyword, self.is_sql_db)
        self.csv_db_manager = CSVDatabaseManager()
        self.vector_db_manager = VectorDatabaseManager(rag_config=self.rag_config)

    def _extract_doi_from_text(self, text: str):
        """Extract DOI from text using regex pattern matching.

        Args:
            text (str): The text to extract DOI from.

        Returns:
            str: The extracted DOI or empty string if not found.
        """
        try:
            # Standard DOI pattern: 10.xxxx/xxxxx
            doi_pattern = r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+'
            matches = re.findall(doi_pattern, text)

            if matches:
                # Return the first match, clean up common trailing characters
                doi = matches[0].rstrip('.,;)]')
                logger.debug(f"DOI extracted: {doi}")
                return doi
            else:
                logger.debug("No DOI found in text")
                return ""
        except Exception as e:
            logger.error(f"Error extracting DOI from text: {e}")
            return ""

    def process_pdfs(self):
        """
        Main function to process the PDFs in the folder. It reads the PDFs, extracts the text, and writes the data to CSV file, to the SQL database (if set), and creates a vector database if the keyword is found in the text.
        """
        dataframes = []
        pdf_files = glob.glob(f"{self.folder_path}/*.pdf")
        total_files = len(pdf_files)
        logger.verbose(f"\n\nParsing of PDFs started...")
        logger.debug(f"\nTotal PDF files found: {total_files}")
        for pdf_file in tqdm(
            pdf_files, desc="Processing PDFs", total=total_files, colour="#d6adff"
        ):
            try:
                # Convert PDF to Markdown text
                pdf_to_md = PDFToMarkdownText(source=pdf_file)
                md_text = pdf_to_md.convert_to_markdown()

                # Extract DOI from the converted markdown text
                self.doi = self._extract_doi_from_text(md_text)

                if self.doi:
                    self.identifier = self.doi
                    logger.debug(f"DOI found: {self.doi}")
                else:
                    # Use filename as identifier if DOI not found
                    logger.warning(
                        f"DOI not found for {pdf_file}. Using filename as identifier."
                    )
                    filename = os.path.basename(pdf_file)
                    self.identifier = filename.replace(".pdf", "")

                # Get metadata from external API using DOI
                title, journal_name, publisher = "", "", ""
                if self.doi:
                    title, journal_name, publisher = get_paper_metadata_from_oaworks(
                        self.doi
                    )
                    if not title:
                        logger.warning(f"Metadata not found for DOI: {self.doi}")

                # Process sections
                all_sections = pdf_to_md.clean_text(md_text)
                row = pdf_to_md.append_section_to_df(
                    all_sections,
                    self.doi,
                    title,
                    journal_name,
                    publisher,
                    self.property_keywords,
                    self.vector_db_manager,
                    logger,
                )
                dataframes.append(row)

                if row["is_property_mentioned"].iloc[0] == "1":
                    self.valid_property_articles += 1
                if len(dataframes) == self.sql_batch_size:
                    final_df = pd.concat(dataframes, ignore_index=True)
                    if self.is_sql_db:
                        self.sql_db_manager.write_to_sql_db(
                            self.paperdata_table_name, final_df
                        )
                if len(dataframes) == self.csv_batch_size:
                    self.csv_db_manager.write_to_csv(
                        final_df, self.csv_path, self.keyword, self.source
                    )
                    dataframes = []
                    time.sleep(5)
                time.sleep(0.2)

            except KeyboardInterrupt as kie:
                logger.error(f"Keyboard Interruption Detected. {kie}")
                raise KeyboardInterruptHandler()
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue

            # Append any remaining dataframes at the end
            try:
                if dataframes:
                    remaining_df = pd.concat(dataframes, ignore_index=True)
                    if self.is_sql_db:
                        self.sql_db_manager.write_to_sql_db(
                            self.paperdata_table_name, remaining_df
                        )
                    self.csv_db_manager.write_to_csv(
                        remaining_df, self.csv_path, self.keyword, self.source
                    )
            except Exception as e:
                logger.error(f"Error writing remaining dataframes: {e}")
        logger.verbose(f"\n\nParsing of PDFs completed...")
        logger.info(f"\nTotal valid property articles: {self.valid_property_articles}")
