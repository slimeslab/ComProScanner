"""
data_preparator.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 17-03-2025
"""

# Standard library imports
import regex as re
import json
import os
import glob
from typing import Optional

# Third party imports
import pandas as pd
from tqdm import tqdm
from typing import Dict

# Custom imports
from .configs import DatabaseConfig, DefaultPaths
from .error_handler import (
    ValueErrorHandler,
    FileNotFoundErrorHandler,
)
from .logger import setup_logger

######## logger Configuration ########
logger = setup_logger("composition_property_extractor.log")


class SectionProcessor:
    def __init__(self):
        self.section_names = {
            "article_title": "TITLE",
            "abstract": "ABSTRACT",
            "introduction": "INTRODUCTION",
            "exp_methods": "EXPERIMENTAL METHODS",
            "results_discussion": "RESULTS AND DISCUSSION",
            "conclusion": "CONCLUSION",
        }

        # Map CSV column names to section keys
        self.column_to_section_map = {
            "article_title": "article_title",
            "abstract": "abstract",
            "introduction": "introduction",
            "conclusion": "conclusion",
            "exp_methods": "exp_methods",
            "results_discussion": "results_discussion",
        }

    def _separate_tables_and_text(self, text: str) -> tuple:
        """
        Separate tables from main text and return both parts.

        Args:
            text (str): Text to be processed.

        Returns:
            tuple: Tuple containing main text and tables text.
        """
        if pd.isna(text):
            return "", ""
        segments = text.split("\nTable 1.")
        main_text = segments[0]
        tables = "\nTable 1." + segments[1] if len(segments) > 1 else ""
        return main_text.strip(), tables.strip()

    def _split_into_sentences(self, text: str) -> list:
        """
        Split text into sentences based on period + space + capital letter or newline.

        Args:
            text (str): Text to be processed.

        Returns:
            list: List of sentences.
        """
        if pd.isna(text):
            return []
        pattern = r"(?<=\.)\s+(?=[A-Z])|(?<=\.)\n"
        sentences = re.split(pattern, str(text))
        return [sent.strip() for sent in sentences if sent.strip()]

    def _has_digits_or_consecutive_caps(self, sentence: str) -> bool:
        """
        Check if sentence contains digits or consecutive capital letters.

        Args:
            sentence (str): Sentence to be processed.

        Returns:
            bool: True if sentence contains digits or consecutive capital letters, False otherwise.
        """
        has_digits = bool(re.search(r"\d", sentence))
        has_consecutive_caps = bool(re.search(r"[A-Z]{2,}", sentence))
        return has_digits or has_consecutive_caps

    def _get_relevant_sentences(self, text: str) -> list:
        """
        Get sentences containing digits or consecutive capital letters.

        Args:
            text (str): Text to be processed.

        Returns:
            list: List of relevant sentences.
        """
        if pd.isna(text):
            return []
        sentences = self._split_into_sentences(text)
        return [s for s in sentences if self._has_digits_or_consecutive_caps(s)]

    def _process_section(self, text: str, section_name: str) -> tuple:
        """
        Process a single section and format it with header.

        Args:
            text (str): Text to be processed.
            section_name (str): Name of the section.

        Returns:
            tuple: Tuple containing tables text and main section text.
        """
        if pd.isna(text):
            return "", ""

        if section_name == "results_discussion":
            main_text, all_tables = self._separate_tables_and_text(str(text))
            relevant_sentences = self._get_relevant_sentences(main_text)

            if not relevant_sentences and not all_tables:
                return "", ""

            tables_text = f"# TABLES:\n{all_tables}\n\n"
            main_section_text = (
                f"# {self.section_names[section_name]}\n{' '.join(relevant_sentences)}\n\n"
                if relevant_sentences
                else ""
            )
            return tables_text, main_section_text
        else:
            relevant_sentences = self._get_relevant_sentences(str(text))
            if not relevant_sentences:
                return "", ""
            return (
                "",
                f"# {self.section_names[section_name]}\n{' '.join(relevant_sentences)}\n\n",
            )

    def create_formatted_texts(self, row):
        """
        Create formatted texts for composition/property and synthesis.

        Args:
            row (pd.Series): Row of the dataframe.

        Returns:
            tuple: Tuple containing composition/property text and synthesis text.
        """
        tables_text = ""
        composition_property_text = ""

        if "results_discussion" in row and not pd.isna(row["results_discussion"]):
            tables_part, results_text = self._process_section(
                row["results_discussion"], "results_discussion"
            )
            tables_text = tables_part if tables_part else ""
            composition_property_text = results_text if results_text else ""

        csv_columns_to_process = [
            "article_title",
            "abstract",
            "introduction",
            "conclusion",
        ]
        for csv_column in csv_columns_to_process:
            if csv_column in row and not pd.isna(row[csv_column]):
                section_key = self.column_to_section_map[csv_column]
                _, section_text = self._process_section(row[csv_column], section_key)
                composition_property_text += section_text

        final_composition_property_text = "\n" + tables_text + composition_property_text

        synthesis_text = ""
        synthesis_columns = ["exp_methods", "results_discussion"]
        for csv_column in synthesis_columns:
            if csv_column in row and not pd.isna(row[csv_column]):
                section_key = self.column_to_section_map[csv_column]
                _, section_text = self._process_section(row[csv_column], section_key)
                synthesis_text += section_text

        return final_composition_property_text.strip(), synthesis_text.strip()


class MatPropDataPreparator:

    def __init__(
        self,
        main_property_keyword: str = None,
        main_extraction_keyword: str = None,
        json_results_file: str = None,
        start_row: int = 0,
        num_rows: int = None,
        is_test_data_preparation: bool = False,
        test_doi_list_file=None,
        total_test_data: int = None,
        test_random_seed: Optional[int] = 42,
        checked_doi_list_file: Optional[str] = "checked_dois.txt",
    ):
        """
        Initialize the MatPropDataPreparator class.

        Args:
            main_property_keyword (str: required): Main property keyword to process the articles for and file naming.
            main_extraction_keyword (str: required): Main property keyword to search for in the article.
            json_results_file (str: required): JSON results file name.
            start_row (int: optional): Start row for processing (default: 0)
            num_rows (int: optional): Number of rows to process (default: None)
            is_test_data_preparation (bool: optional): Flag to indicate if the data preparation process is a test data preparation step (default: False)
            test_doi_list_file (str: optional): Test data file name (default: None)
            total_test_data (int: optional): Total number of test data to prepare (default: 50)
            test_random_seed (int: optional): Random seed for test data preparation (default: 42)
            checked_doi_list_file (str: optional): File to store checked DOIs (default: "checked_dois.txt")

        Returns:
            None
        """
        self.main_property_keyword = main_property_keyword
        if self.main_property_keyword is None:
            logger.error("Main property keyword is required.")
            raise ValueErrorHandler(message="Main property keyword is required.")
        self.main_extraction_keyword = main_extraction_keyword
        if self.main_extraction_keyword is None:
            logger.error("Main extraction property keyword is required.")
            raise ValueErrorHandler(
                message="Main extraction property keyword is required."
            )
        self.json_results_file = json_results_file
        if self.json_results_file is None:
            logger.error("JSON results file is required.")
            raise ValueErrorHandler(
                message="JSON results file is required for checking processed data."
            )

        if is_test_data_preparation and test_doi_list_file is None:
            logger.error("Test data file name is required for test data preparation.")
            raise ValueErrorHandler(
                message="Test data file name is required for test data preparation."
            )
        self.is_test_data_preparation = is_test_data_preparation
        if is_test_data_preparation:
            self.test_doi_list_file = test_doi_list_file
            self.test_random_seed = test_random_seed
            if total_test_data is None:
                self.total_test_data = 50
            else:
                self.total_test_data = total_test_data

        self.db_configs = DatabaseConfig(self.main_property_keyword)
        self.extracted_folderpath = self.db_configs.EXTRACTED_CSV_FOLDERPATH
        self.start_row = start_row
        self.num_rows = num_rows
        self.checked_doi_list_file = checked_doi_list_file

        # Initialize results storage
        self._load_existing_results()

        # Load checked DOIs from the specified file
        self.checked_dois = self._load_checked_dois()

    def _load_existing_results(self) -> None:
        """Load existing results from JSON file."""
        self.results = {}
        if os.path.exists(self.json_results_file):
            with open(self.json_results_file, "r") as f:
                self.results = json.load(f)

    def _load_checked_dois(self) -> set:
        """Load checked DOIs from the specified file."""
        checked_dois = set()
        if os.path.exists(self.checked_doi_list_file):
            with open(self.checked_doi_list_file, "r") as f:
                checked_dois = {line.strip() for line in f if line.strip()}
        return checked_dois

    def get_unprocessed_data(self) -> list[Dict]:
        """
        Process materials data extracted from the CSV database and run CrewAI Workflow.
        """
        all_files = glob.glob(self.extracted_folderpath + "/*.csv")
        dfs = [pd.read_csv(f) for f in all_files]
        if not dfs:
            logger.error(f"No files found in the folder: {self.extracted_folderpath}")
            raise FileNotFoundErrorHandler(
                f"No files found in the folder: {self.extracted_folderpath}"
            )
        extracted_df = pd.concat(dfs, axis=0, ignore_index=True)
        extracted_df = extracted_df.iloc[self.start_row :]
        property_mentioned_df = extracted_df[
            extracted_df["is_property_mentioned"].isin([True, 1, "1"])
        ]
        if self.num_rows is not None:
            property_mentioned_df = property_mentioned_df.head(self.num_rows)
        print(f"Length of property_mentioned_df: {len(property_mentioned_df)}")

        # get the DOIs that have already been processed and remove them from the list
        processed_dois = list(self.results.keys())
        final_unprocessed_dois = property_mentioned_df[
            ~property_mentioned_df["doi"].isin(processed_dois)
        ]

        # Remove checked DOIs from the final unprocessed DOIs - applies to both test and regular processing
        final_unprocessed_dois = final_unprocessed_dois[
            ~final_unprocessed_dois["doi"].isin(self.checked_dois)
        ]

        # prepare test data if is_test_data_preparation
        if self.is_test_data_preparation:
            test_dois = []
            if os.path.exists(self.test_doi_list_file):
                try:
                    with open(self.test_doi_list_file, "r") as f:
                        test_dois = [
                            line.strip() for line in f.readlines() if line.strip()
                        ]
                        if test_dois and test_dois[-1] == "":
                            test_dois.pop()
                except Exception as e:
                    logger.warning(
                        f"Error reading test DOI file {self.test_doi_list_file}: {str(e)}"
                    )
                    test_dois = []

            unprocessed_test_dois = list(
                set(test_dois).intersection(set(final_unprocessed_dois["doi"]))
            )

            if len(test_dois) == self.total_test_data:
                final_unprocessed_dois = final_unprocessed_dois[
                    final_unprocessed_dois["doi"].isin(test_dois)
                ]
                logger.info(
                    f"Total unprocessed DOIs: {len(final_unprocessed_dois)}. Test DOIs are already selected."
                )
            else:
                remaining_dois_df = final_unprocessed_dois[
                    ~final_unprocessed_dois["doi"].isin(test_dois)
                ]
                shuffled_remaining_dois = remaining_dois_df.sample(
                    n=len(remaining_dois_df), random_state=self.test_random_seed
                )["doi"].tolist()
                final_selection = unprocessed_test_dois + shuffled_remaining_dois

                # Filter final_unprocessed_dois to include only the selected DOIs
                final_unprocessed_dois = final_unprocessed_dois[
                    final_unprocessed_dois["doi"].isin(final_selection)
                ]
                logger.info(
                    f"Total unprocessed DOIs: {len(final_unprocessed_dois)}. Test DOIs will be chosen based on the availability of composition data in the paper."
                )
        else:
            logger.info(f"Total DOIs to process: {len(final_unprocessed_dois)}")

        print(f"Total DOIs to process: {len(final_unprocessed_dois)}")

        # Continue with the regular processing for all selected DOIs
        prepared_data = []
        processor = SectionProcessor()
        for _, row in tqdm(
            final_unprocessed_dois.iterrows(),
            total=len(final_unprocessed_dois),
            desc="Processing DOIs",
            colour="#d6adff",
        ):
            try:
                comp_prop_text, synthesis_text = processor.create_formatted_texts(row)

                prepared_data.append(
                    {
                        "doi": row["doi"],
                        "main_extraction_keyword": self.main_extraction_keyword,
                        "comp_prop_text": comp_prop_text,
                        "synthesis_text": synthesis_text,
                    }
                )
            except Exception as e:
                logger.error(f"Error preparing data for DOI: {row['doi']}. Error: {e}")
                continue
        return prepared_data
