import json


def extract_and_sort_compositions(input_file, output_file):
    """
    Extract all compositions_property_values from JSON file and sort by value (high to low).

    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output text file
    """

    # Read the JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all composition-property pairs
    all_compositions = []

    for doi, paper_data in data.items():
        if "composition_data" in paper_data:
            comp_data = paper_data["composition_data"]
            if "compositions_property_values" in comp_data:
                property_unit = comp_data.get("property_unit", "Unknown unit")
                family = comp_data.get("family", "Unknown family")

                # Add each composition-value pair with metadata (skip None values)
                for composition, value in comp_data[
                    "compositions_property_values"
                ].items():
                    # Only include entries with valid numeric values
                    if value is not None and isinstance(value, (int, float)):
                        all_compositions.append(
                            {
                                "composition": composition,
                                "value": value,
                                "unit": property_unit,
                                "family": family,
                                "doi": doi,
                            }
                        )

    # Sort by value in descending order (high to low)
    all_compositions.sort(key=lambda x: x["value"], reverse=True)

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            "Piezoelectric Ceramic Compositions Sorted by Property Value (High to Low)\n"
        )
        f.write("=" * 75 + "\n\n")

        for i, comp in enumerate(all_compositions, 1):
            f.write(f"{i:2d}. {comp['composition']}: {comp['value']} {comp['unit']}\n")
            f.write(f"    Family: {comp['family']}\n")
            f.write(f"    DOI: {comp['doi']}\n\n")

    print(
        f"Successfully processed {len(all_compositions)} compositions with valid values"
    )
    print(f"Results written to: {output_file}")

    # Display top 5 results
    print("\nTop 5 compositions:")
    for i, comp in enumerate(all_compositions[:5], 1):
        print(f"{i}. {comp['composition']}: {comp['value']} {comp['unit']}")


if __name__ == "__main__":
    # File paths
    input_file = "../../../piezo_test/model-outputs/deepseek/deepseek-v3-0324-piezo-ceramic-test-results.json"
    output_file = "sorted_piezoelectric_compositions.txt"

    try:
        extract_and_sort_compositions(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Could not find the file '{input_file}'")
        print("Please make sure the file exists in the current directory.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{input_file}'")
        print(f"JSON Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
