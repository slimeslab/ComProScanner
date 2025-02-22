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
    def __init__(self, main_property_keyword):
        self.METADATA_CSV_FILENAME = f"results/{main_property_keyword}_metadata.csv"
        self.TIMEOUT_DOI_LOG_FILENAME = f"logs/{main_property_keyword}_timeout_dois.txt"
        self.IOP_FOLDERPATH = os.getenv("IOP_papers_path")
