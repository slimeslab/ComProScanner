"""
get_common_piezo_materials.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 04-09-2025
"""

import os
from dotenv import load_dotenv
from mp_api.client import MPRester
import json

# Load environment variables
load_dotenv()
MP_API = os.getenv("MP_API") or os.getenv("MY_MP_API")


def get_all_piezoelectric_materials_from_mp(
    output_file="piezoelectric_materials_from_mp.txt",
):
    """
    Get all piezoelectric materials from Materials Project.
    Saves to file and reads from file if available and has data.

    Args:
        output_file (str): Path to the output file. Defaults to "piezoelectric_materials_from_mp.txt"

    Returns:
        list: List of piezoelectric material formulae
    """

    # Check if file exists and has data
    if output_file and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Reading piezoelectric materials from existing file: {output_file}")
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                piezo_material_formulae = [line.strip() for line in f if line.strip()]
            print(
                f"Total piezoelectric materials from Materials Project: {len(piezo_material_formulae)}"
            )
            return piezo_material_formulae

        except Exception as e:
            print(f"Error reading from file {output_file}: {e}")
            print("Falling back to fetching from Materials Project...")

    # Fetch from Materials Project if file doesn't exist or is empty
    print("Fetching all piezoelectric materials from Materials Project...")
    try:
        with MPRester(MP_API) as mpr:
            piezo_materials = mpr.materials.piezoelectric.search()
            piezo_material_formulae = [item.formula_pretty for item in piezo_materials]

        print(f"Found {len(piezo_material_formulae)} piezoelectric materials")

        # Save to file if output_file is specified
        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    for formula in piezo_material_formulae:
                        f.write(f"{formula}\n")
            except Exception as e:
                print(f"Warning: Could not save to file {output_file}: {e}")

        print(
            f"Total piezoelectric materials found from Materials Project: {len(piezo_material_formulae)}"
        )
        return piezo_material_formulae

    except Exception as e:
        print(f"Error fetching from Materials Project: {e}")
        return []


def get_all_extracted_compositions(input_file):
    """
    Extract all compositions from JSON file.

    Args:
        input_file (str): Path to the input JSON file
    """

    # Read the JSON file
    print(f"\nReading extracted compositions from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all compositions
    all_compositions = []

    for _, paper_data in data.items():
        if "composition_data" in paper_data:
            comp_data = paper_data["composition_data"]
            if "compositions_property_values" in comp_data:
                # Add each composition
                for composition, _ in comp_data["compositions_property_values"].items():
                    all_compositions.append(composition)

    print(f"Total compositions extracted: {len(all_compositions)}")
    return all_compositions


if __name__ == "__main__":
    mp_piezo_materials = get_all_piezoelectric_materials_from_mp()
    extracted_compositions = get_all_extracted_compositions(
        "../../../piezo_test/model-outputs/deepseek/deepseek-v3-0324-piezo-ceramic-test-results.json"
    )
    common_materials = list(
        set(mp_piezo_materials).intersection(set(extracted_compositions))
    )
    if common_materials:
        print("\nCommon piezoelectric materials found:")
        for index, material in enumerate(common_materials):
            print(f"{index + 1}. {material}")

        print(
            f"\nPercentage of new piezoelectric materials found: {(1-(len(common_materials) / len(extracted_compositions))) * 100:.2f}%"
        )
