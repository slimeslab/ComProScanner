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
    get_paper_metadata_from_openalex,
    get_doi_from_crossref,
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
        csv_batch_size: int = 1,
        is_sql_db: bool = False,
        rag_config: RAGConfig = RAGConfig(),
        main_figure_keywords: dict = None,
        additional_figure_keywords: dict = None,
        save_failed_pdf_report: bool = True,
        failed_pdf_report_path: str = None,
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
            csv_batch_size (int): The number of rows to write to the CSV file at once (default: 1)
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
        self.main_figure_keywords = (
            main_figure_keywords
            if main_figure_keywords is not None
            else property_keywords
        )
        self.additional_figure_keywords = additional_figure_keywords
        self.save_failed_pdf_report = save_failed_pdf_report
        self.failed_pdf_report_path = failed_pdf_report_path or os.path.join(
            self.folder_path, "failed_pdf_filenames.txt"
        )
        self.failed_pdf_records = []

        self.identifier = ""
        self.doi = ""
        self.all_paths = DefaultPaths(self.keyword)
        self.db_configs = DatabaseConfig(self.keyword, self.is_sql_db)
        self.csv_path = self.db_configs.EXTRACTED_CSV_FOLDERPATH
        self.paperdata_table_name = self.db_configs.PAPERDATA_TABLE_NAME
        self.sql_batch_size = sql_batch_size
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

    @staticmethod
    def _is_valid_doi(doi: str) -> bool:
        """Check whether a string is a valid DOI format."""
        if not doi:
            return False
        doi_pattern = r"^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$"
        return bool(re.match(doi_pattern, doi.strip()))

    def _filename_to_valid_doi(self, pdf_file: str) -> str:
        """Convert filename to DOI candidate and validate it."""
        filename = os.path.basename(pdf_file)
        candidate = filename.replace(".pdf", "").replace("_", "/").strip()
        return candidate if self._is_valid_doi(candidate) else ""

    def _record_failed_pdf(self, pdf_file: str, reason: str) -> None:
        """Record failed PDF filename cases and optionally write to report file."""
        filename = os.path.basename(pdf_file)
        entry = f"{filename}\t{reason}"
        self.failed_pdf_records.append(entry)
        logger.warning(f"Skipping {filename}: {reason}")
        if self.save_failed_pdf_report:
            try:
                with open(self.failed_pdf_report_path, "a", encoding="utf-8") as f:
                    f.write(entry + "\n")
            except Exception as e:
                logger.error(f"Error writing failed PDF report: {e}")

    def _extract_doi_from_text(self, text: str):
        """Extract DOI from text using regex pattern matching.

        Args:
            text (str): The text to extract DOI from.

        Returns:
            str: The extracted DOI or empty string if not found.
        """
        try:
            # Standard DOI pattern: 10.xxxx/xxxxx
            doi_pattern = r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+"
            matches = re.findall(doi_pattern, text)

            if matches:
                # Return the first match, clean up common trailing characters
                doi = matches[0].rstrip(".,;)]")
                logger.debug(f"DOI extracted: {doi}")
                return doi
            else:
                logger.debug("No DOI found in text")
                return ""
        except Exception as e:
            logger.error(f"Error extracting DOI from text: {e}")
            return ""

    def _create_empty_row(
        self, doi: str, title: str = "", journal_name: str = "", publisher: str = ""
    ):
        """Create a row with empty values for PDFs with no text detection.

        Args:
            doi (str): The DOI of the article (may be empty string).
            title (str): The title of the article.
            journal_name (str): The name of the publication.
            publisher (str): The name of the publisher.

        Returns:
            pd.DataFrame: DataFrame with metadata and empty section values and is_property_mentioned=0.
        """
        return pd.DataFrame(
            [
                {
                    "doi": doi,
                    "article_title": title,
                    "publication_name": journal_name,
                    "publisher": publisher,
                    "abstract": "",
                    "introduction": "",
                    "exp_methods": "",
                    "comp_methods": "",
                    "results_discussion": "",
                    "conclusion": "",
                    "is_property_mentioned": "0",
                }
            ]
        )

    def _is_corrupted_text(self, text: str) -> bool:
        """Check if the text contains corrupted GLYPH patterns from failed OCR.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if text is corrupted (high ratio of GLYPH patterns), False otherwise.
        """
        if not text:
            return True

        # Count GLYPH pattern occurrences (both raw and HTML-escaped)
        glyph_pattern = r"GLYPH(?:<|&lt;)\d+(?:>|&gt;)"
        glyph_matches = re.findall(glyph_pattern, text)
        glyph_count = len(glyph_matches)

        # If there are many GLYPH patterns, the text is corrupted
        # Threshold: if GLYPH patterns make up more than 10% of words, consider it corrupted
        words = text.split()
        word_count = len(words)

        if word_count == 0:
            return True

        glyph_ratio = glyph_count / word_count
        return glyph_ratio > 0.1  # More than 10% GLYPH patterns indicates corruption

    def _get_metadata_from_csv(self, doi: str):
        """Try to get metadata from the local metadata CSV file.

        Args:
            doi (str): The DOI to search for.

        Returns:
            tuple: (title, journal_name, publisher) or ("", "", "") if not found.
        """
        try:
            if not os.path.exists(self.metadata_csv_filename):
                return "", "", ""

            # Load metadata CSV if not already loaded
            if self.df is None:
                self.df = pd.read_csv(self.metadata_csv_filename)

            matching_rows = self.df[self.df["doi"] == doi]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                title = row.get("article_title", "")
                journal_name = row.get("publication_name", "")
                publisher = row.get("metadata_publisher", "")
                return title, journal_name, publisher
            return "", "", ""
        except Exception as e:
            logger.warning(f"Error reading metadata from CSV: {e}")
            return "", "", ""

    def process_pdfs(self):
        """
        Main function to process the PDFs in the folder. It reads the PDFs, extracts the text, and writes the data to CSV file, to the SQL database (if set), and creates a vector database if the keyword is found in the text.
        """
        sql_dataframes = []
        csv_dataframes = []
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
                print(
                    f"\n{'='*30} MARKDOWN DEBUG START: {os.path.basename(pdf_file)} {'='*30}\n"
                )
                print(md_text if md_text is not None else "None")
                print(
                    f"\n{'='*30} MARKDOWN DEBUG END: {os.path.basename(pdf_file)} {'='*30}\n"
                )

                # Handle empty or corrupted text detection result
                if (
                    md_text is None
                    or not md_text.strip()
                    or self._is_corrupted_text(md_text)
                ):
                    logger.warning(
                        f"Text detection result is empty or corrupted for {pdf_file}. "
                        "Storing with is_property_mentioned=0 and skipping vector database creation."
                    )
                    # Try to extract DOI from filename (only if valid DOI format)
                    filename = os.path.basename(pdf_file)
                    self.doi = self._filename_to_valid_doi(pdf_file)
                    self.identifier = filename.replace(".pdf", "")
                    if not self.doi:
                        self._record_failed_pdf(
                            pdf_file,
                            "empty_or_corrupted_text_and_filename_not_valid_doi",
                        )
                        continue

                    # Try to get metadata (API first, then CSV)
                    title, journal_name, publisher = "", "", ""
                    if self.doi.startswith("10."):
                        title, journal_name, publisher = (
                            get_paper_metadata_from_openalex(self.doi)
                        )

                        if not title or not journal_name or not publisher:
                            csv_title, csv_journal, csv_publisher = (
                                self._get_metadata_from_csv(self.doi)
                            )
                            title = title or csv_title
                            journal_name = journal_name or csv_journal
                            publisher = publisher or csv_publisher

                    row = self._create_empty_row(
                        self.doi, title, journal_name, publisher
                    )
                    sql_dataframes.append(row)
                    csv_dataframes.append(row)

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

                # Extract DOI from the converted markdown text
                self.doi = self._extract_doi_from_text(md_text)

                if self.doi:
                    self.identifier = self.doi
                    logger.debug(f"DOI found: {self.doi}")
                else:
                    # Try CrossRef API as fallback before using filename
                    crossref_doi = get_doi_from_crossref(md_text)
                    if crossref_doi:
                        self.doi = crossref_doi
                        self.identifier = crossref_doi
                        logger.info(
                            f"DOI resolved via CrossRef for {pdf_file}: {self.doi}"
                        )
                    else:
                        # Final fallback: derive DOI from filename only if valid DOI format.
                        filename = os.path.basename(pdf_file)
                        self.identifier = filename.replace(".pdf", "")
                        self.doi = self._filename_to_valid_doi(pdf_file)
                        if not self.doi:
                            self._record_failed_pdf(
                                pdf_file, "doi_not_found_and_filename_not_valid_doi"
                            )
                            continue
                        logger.warning(
                            f"DOI not found in text/CrossRef for {pdf_file}. "
                            f"Using filename-derived DOI: {self.doi}"
                        )

                # Get metadata from external API (with CSV fallback) using DOI
                title, journal_name, publisher = "", "", ""
                if self.doi:
                    title, journal_name, publisher = get_paper_metadata_from_openalex(
                        self.doi
                    )

                    if not title or not journal_name or not publisher:
                        csv_title, csv_journal, csv_publisher = (
                            self._get_metadata_from_csv(self.doi)
                        )
                        title = title or csv_title
                        journal_name = journal_name or csv_journal
                        publisher = publisher or csv_publisher

                    if not title:
                        logger.warning(f"Metadata not found for DOI: {self.doi}")

                has_caption_keyword_match = pdf_to_md.extract_and_save_figures(
                    self.doi,
                    self.main_figure_keywords,
                    base_path=f"results/extracted_data/{self.keyword}/related_figures",
                )
                if self.additional_figure_keywords:
                    pdf_to_md.extract_and_save_figures(
                        self.doi,
                        self.additional_figure_keywords,
                        base_path=f"results/extracted_data/{self.keyword}/related_figures",
                    )

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
                    has_caption_keyword_match=has_caption_keyword_match,
                )
                sql_dataframes.append(row)
                csv_dataframes.append(row)

                if row["is_property_mentioned"].iloc[0] == "1":
                    self.valid_property_articles += 1

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

                time.sleep(0.2)

            except KeyboardInterrupt as kie:
                logger.error(f"Keyboard Interruption Detected. {kie}")
                raise KeyboardInterruptHandler()
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
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
        logger.verbose(f"\n\nParsing of PDFs completed...")
        logger.info(f"\nTotal valid property articles: {self.valid_property_articles}")
        if self.failed_pdf_records:
            logger.warning(
                f"Total skipped PDFs due to invalid filename DOI fallback: {len(self.failed_pdf_records)}"
            )
            if self.save_failed_pdf_report:
                logger.info(
                    f"Failed PDF report saved to: {self.failed_pdf_report_path}"
                )
