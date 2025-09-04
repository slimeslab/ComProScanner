"""
get_total_higher_d33.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 11-08-2025
"""

import requests
import os
from dotenv import load_dotenv
import numpy as np
from mp_api.client import MPRester
import pandas as pd
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()
MP_API = os.getenv("MP_API") or os.getenv("MY_MP_API")

# File paths
piezo_ids_filepath = "piezoelectric_materials_ids.txt"
common_ids_filepath = "common_materials_ids.txt"
d33_results_file = "d33_results.csv"


def get_all_piezoelectric_materials():
    """Get all piezoelectric materials from Materials Project"""
    print("Fetching all piezoelectric materials...")
    with MPRester(MP_API) as mpr:
        piezo_materials = mpr.materials.piezoelectric.search()
        piezo_material_ids = [item.material_id for item in piezo_materials]

    # Save to file
    with open(piezo_ids_filepath, "w") as file:
        for material_id in piezo_material_ids:
            file.write(f"{material_id}\n")

    print(f"Found {len(piezo_material_ids)} piezoelectric materials")
    return piezo_material_ids


def filter_materials_with_elasticity(piezo_material_ids):
    """Filter materials that have elasticity data"""
    print("Filtering materials with elasticity data...")
    common_materials = []

    # Check if we need to rebuild the common materials list
    if os.path.exists(common_ids_filepath) and os.path.exists(piezo_ids_filepath):
        # Check if piezo_ids file has same number of lines as current materials
        with open(piezo_ids_filepath, "r") as file:
            existing_piezo_count = len(file.readlines())

        if existing_piezo_count == len(piezo_material_ids):
            # Use existing common materials list
            with open(common_ids_filepath, "r") as file:
                existing_common = [line.strip() for line in file.readlines()]
            if len(existing_common) > 0:
                print(
                    f"Using existing filtered list of {len(existing_common)} materials"
                )
                return existing_common

    # Need to rebuild - clear the common materials file
    with open(common_ids_filepath, "w") as file:
        file.write("")

    # Filter materials with elasticity data using MPRester for efficiency
    print("Checking elasticity data availability...")

    # First try using MPRester to get materials with elasticity data
    try:
        with MPRester(MP_API) as mpr:
            # Get all materials with elasticity data
            elasticity_materials = mpr.materials.elasticity.search()
            elasticity_ids = {mat.material_id for mat in elasticity_materials}

            # Find intersection with piezoelectric materials
            for material_id in tqdm(
                piezo_material_ids, desc="Finding common materials"
            ):
                if material_id in elasticity_ids:
                    common_materials.append(material_id)
                    with open(common_ids_filepath, "a") as file:
                        file.write(f"{material_id}\n")

    except Exception as e:
        print(f"MPRester method failed, falling back to API calls: {e}")
        # Fallback to original API method
        for item in tqdm(piezo_material_ids, desc="Checking elasticity data via API"):
            try:
                res = requests.get(
                    f"https://api.materialsproject.org/materials/elasticity/{item}",
                    headers={"X-API-KEY": MP_API, "accept": "application/json"},
                )
                if res.status_code == 200:
                    common_materials.append(item)
                    with open(common_ids_filepath, "a") as file:
                        file.write(f"{item}\n")
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            except Exception as e:
                continue

    print(f"Found {len(common_materials)} materials with elasticity data")
    return common_materials


def get_field_data(url, field_type):
    """Get specific field data from the Materials Project API"""
    try:
        response = requests.get(
            url, headers={"X-API-KEY": MP_API, "accept": "application/json"}
        )
        if response.status_code == 200:
            response_json = response.json()
            if field_type == "compliance_tensor":
                response_data = response_json["data"][0]["compliance_tensor"]["raw"]
            else:
                response_data = response_json["data"][0]["total"]
            return response_data
        else:
            return None
    except Exception as e:
        return None


def calculate_d33_values(common_materials):
    """Calculate d33 values for all materials and count those > 10"""
    print("Calculating d33 values...")

    successful_calculations = 0
    materials_with_high_d33 = 0
    d33_results = []

    # Check if we already have results
    if os.path.exists(d33_results_file):
        try:
            existing_results = pd.read_csv(d33_results_file)
            processed_materials = set(existing_results["material_id"].values)
            d33_results = existing_results.to_dict("records")

            # Count existing high d33 materials
            materials_with_high_d33 = len(
                existing_results[existing_results["d33_value"] > 10]
            )
            successful_calculations = len(existing_results)

            print(f"Found {len(existing_results)} existing results")
        except:
            processed_materials = set()
    else:
        processed_materials = set()

    # Process remaining materials
    remaining_materials = [m for m in common_materials if m not in processed_materials]

    if remaining_materials:
        print(f"Processing {len(remaining_materials)} new materials...")

        for item in tqdm(remaining_materials, desc="Calculating d33 values"):
            compliance_url = f"https://api.materialsproject.org/materials/elasticity/?material_ids={item}&_fields=compliance_tensor&_all_fields=false"
            piezo_url = f"https://api.materialsproject.org/materials/piezoelectric/?material_ids={item}&_fields=total&_all_fields=false"

            compliance_tensor_data = get_field_data(compliance_url, "compliance_tensor")
            piezo_tensor_data = get_field_data(piezo_url, "piezo_tensor")

            if compliance_tensor_data and piezo_tensor_data:
                try:
                    compliance_tensor_data = np.array(compliance_tensor_data)
                    piezo_tensor_data = np.array(piezo_tensor_data)

                    # Calculate d33 value: multiply 3rd column of compliance tensor with 3rd row of piezo tensor
                    d33_value = np.dot(
                        compliance_tensor_data[:, 2], piezo_tensor_data[2, :]
                    )

                    # Store result
                    result = {"material_id": item, "d33_value": float(d33_value)}
                    d33_results.append(result)

                    successful_calculations += 1
                    if float(d33_value) > 10:
                        materials_with_high_d33 += 1

                except Exception as e:
                    continue

            # Add small delay to avoid rate limiting
            time.sleep(0.1)

        # Save results to CSV
        df = pd.DataFrame(d33_results)
        df.to_csv(d33_results_file, index=False)
        print(f"Results saved to {d33_results_file}")

    return successful_calculations, materials_with_high_d33


def count_high_d33_materials():
    """Count materials with d33 > 10 from the results file"""
    if not os.path.exists(d33_results_file):
        return 0

    try:
        df = pd.read_csv(d33_results_file)
        high_d33_count = len(df[df["d33_value"] > 10])
        return high_d33_count
    except Exception as e:
        print(f"Error reading results file: {e}")
        return 0


def main():
    """Main function to run the complete analysis"""
    print("=== Simplified Piezoelectric d33 Analysis ===\n")

    if not MP_API:
        print("Error: MP_API key not found in environment variables")
        return

    # Step 1: Get all piezoelectric materials
    piezo_material_ids = get_all_piezoelectric_materials()

    # Step 2: Filter materials with elasticity data
    common_materials = filter_materials_with_elasticity(piezo_material_ids)

    # Step 3: Calculate d33 values and count those > 10
    successful_calculations, materials_with_high_d33 = calculate_d33_values(
        common_materials
    )

    # Step 4: Final count
    final_count = count_high_d33_materials()

    # Results
    print("\n=== ANALYSIS RESULTS ===")
    print(f"Total piezoelectric materials found: {len(piezo_material_ids)}")
    print(f"Materials with elasticity data: {len(common_materials)}")
    print(f"Successfully processed materials: {successful_calculations}")
    print(f"\n🎯 FINAL ANSWER: {final_count} materials have d33 > 10")
    print(f"Results saved to: {d33_results_file}")

    return final_count


if __name__ == "__main__":
    result = main()
