import json
import os


def create_new_json(doi_list: list, json_path: str, new_filename: str) -> None:
    """Create a new JSON file with empty entries for each DOI."""
    data = {}

    # get all the items from existing json if exists with the same doi and create a new json with those items
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        for doi in doi_list:
            if doi in existing_data:
                data[doi] = existing_data[doi]
            else:
                print(f"DOI {doi} not found in existing JSON. Adding empty entry.")

    # write the new json to a file
    with open(new_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    with open("selected_dois.txt", "r", encoding="utf-8") as f:
        dois = [line.strip() for line in f if line.strip()]
    create_new_json(dois, "../ground_truth.json", "comparison_ground_truth.json")
    create_new_json(
        dois,
        "../model-outputs/deepseek/deepseek-v3-0324-piezo-ceramic-test-results.json",
        "ComProScanner/comparison_deepseek_results.json",
    )
