import random


def choose_dois(all_dois, n=5, seed=42):
    """Randomly choose n DOIs from the provided list."""
    random.seed(seed)
    return random.sample(all_dois, n)


def get_all_dois(filepath):
    """Return a list of all available DOIs from the result JSON file."""
    import json
    from pathlib import Path

    result_file = Path(filepath)
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys())


if __name__ == "__main__":
    # Randomly select 10 DOIs from the test dataset
    all_dois = get_all_dois("../ground_truth.json")
    selected_dois = choose_dois(all_dois, n=10, seed=42)
    with open("selected_dois.txt", "w", encoding="utf-8") as f:
        for doi in selected_dois:
            f.write(doi + "\n")
