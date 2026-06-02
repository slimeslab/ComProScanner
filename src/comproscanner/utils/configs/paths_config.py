"""
paths_config.py - Contains the default paths for the project.

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 21-02-2025
"""

import os
from dotenv import load_dotenv

load_dotenv()


class DefaultPaths:
    # Keyword-independent paths — accessible as DefaultPaths.XXX without instantiation
    FAILED_AUTOMATED_ARTICLES_FILENAME = "results/failed_automated_articles.txt"
    AGENTIC_EVALUATION_RESULT_FILENAME = "agentic_evaluation_result.json"
    DETAILED_EVALUATION_FILENAME = "detailed_evaluation.json"

    def __init__(self, main_property_keyword):
        # Keyword-dependent paths — require a keyword, accessed as self.all_paths.XXX
        self.METADATA_CSV_FILENAME = f"results/{main_property_keyword}_metadata.csv"
        self.TIMEOUT_DOI_LOG_FILENAME = f"logs/{main_property_keyword}_timeout_dois.txt"
        self.PDF_PROCESSED_DOIS_FILENAME = f"logs/{main_property_keyword}_pdf_processed_dois.txt"
        self.IOP_FOLDERPATH = os.getenv("IOP_papers_path")
