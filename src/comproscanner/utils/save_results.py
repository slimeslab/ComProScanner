"""
save_results.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 03-04-2025
"""

from typing import Dict
import pandas as pd
import os
import json


class SaveResults:

    def __init__(
        self,
        json_results_file: str = "results.json",
        csv_results_file: str = "results.csv",
    ) -> None:
        self.json_results_file = json_results_file
        self.csv_results_file = csv_results_file
        self._load_existing_results()

    def update_in_csv(self, result_dict: Dict) -> None:
        """
        Update or create a CSV file with nested dictionary data using pandas.
        Headers will be just the nested key names without parent key prefixes.

        Args:
            result_dict: Dictionary containing nested dictionaries
        """
        # Extract all unique nested keys
        headers = set()
        for main_dict in result_dict.values():
            if isinstance(main_dict, dict):
                headers.update(main_dict.keys())
        headers = sorted(headers)  # Sort for consistent order

        # Create a row dictionary with all headers
        row_data = {}
        for header in headers:
            # Search for this header in all nested dictionaries
            value = None
            for main_dict in result_dict.values():
                if isinstance(main_dict, dict) and header in main_dict:
                    value = main_dict[header]
                    break
            row_data[header] = value

        # Convert the row to a pandas DataFrame
        df_new = pd.DataFrame([row_data])

        # If file exists, read and append to it; otherwise create new file
        if os.path.exists(self.csv_results_file):
            try:
                df_existing = pd.read_csv(self.csv_results_file)
                df_updated = pd.concat([df_existing, df_new], ignore_index=True)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                # Handle case where CSV exists but is empty or corrupted
                df_updated = df_new
        else:
            df_updated = df_new

        # Save to CSV
        df_updated.to_csv(self.csv_results_file, index=False)

    def _load_existing_results(self) -> None:
        """Load existing results from JSON file."""
        self.results = {}
        if os.path.exists(self.json_results_file):
            try:
                with open(self.json_results_file, "r", encoding="utf-8") as f:
                    self.results = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError):
                # Handle case where JSON file exists but is empty or corrupted
                self.results = {}

    def update_in_json(self, doi: str, result: Dict) -> None:
        """
        Save a single result to the results dictionary and JSON file.

        Args:
            doi: The DOI of the paper
            result: The result data to save
        """
        self.results[doi] = result

        # Create directory if it doesn't exist
        json_dir = os.path.dirname(self.json_results_file)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)

        try:
            with open(self.json_results_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, default=str)
        except (TypeError, ValueError) as e:
            print(f"Error serializing data to JSON: {e}")
            # Try to save with string conversion for problematic objects
            try:
                with open(self.json_results_file, "w", encoding="utf-8") as f:
                    json.dump(
                        self.results,
                        f,
                        indent=2,
                        default=lambda x: str(x),
                    )
            except Exception as fallback_error:
                print(f"Failed to save JSON file: {fallback_error}")
                raise
