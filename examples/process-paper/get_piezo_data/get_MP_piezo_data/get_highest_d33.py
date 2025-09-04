"""
get_highest_d33.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 11-08-2025
"""

import pandas as pd
import os


def print_highest_d33_material():
    """Print the material with the highest d33 value from the results file"""
    if not os.path.exists(d33_results_file):
        print("Results file not found.")
        return

    try:
        df = pd.read_csv(d33_results_file)

        if df.empty:
            print("No data found in results file.")
            return

        # Find the row with the maximum d33 value
        max_row = df.loc[df["d33_value"].idxmax()]

        print(f"Highest d33 value: {max_row['d33_value']} pC/N")

        # Print additional details if available in your CSV
        if "composition" in df.columns:
            print(f"Composition: {max_row['composition']}")
        if "family" in df.columns:
            print(f"Material family: {max_row['family']}")
        if "doi" in df.columns:
            print(f"DOI: {max_row['doi']}")

    except Exception as e:
        print(f"Error reading results file: {e}")


if __name__ == "__main__":
    d33_results_file = "d33_results.csv"

    # Print the material with the highest d33 value
    print_highest_d33_material()
