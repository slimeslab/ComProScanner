import requests
import json


def parse_materials(text):
    """
    Send text to material parser API and extract all raw values

    Args:
        text (str): Text containing material descriptions

    Returns:
        list: List of all raw values found in the response
    """
    url = "https://lfoppiano-material-parsers.hf.space/process/material"

    # Prepare the form data
    files = {"text": (None, text)}

    try:
        # Make the API request
        response = requests.post(url, files=files)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the JSON response
        data = response.json()

        # Extract raw values from resolvedFormulas only
        raw_values = []

        # Navigate through the nested structure
        for outer_list in data:
            if isinstance(outer_list, list):
                for item in outer_list:
                    if isinstance(item, dict):
                        # Check for resolved formulas raw values only
                        if "resolvedFormulas" in item:
                            for resolved in item["resolvedFormulas"]:
                                if (
                                    isinstance(resolved, dict)
                                    and "rawValue" in resolved
                                ):
                                    raw_values.append(resolved["rawValue"])

        return raw_values

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def parse_multiple_materials(text_dict):
    """
    Send multiple texts to material parser API and extract all raw values

    Args:
        text_dict (dict): Dictionary with DOI/ID as key and text as value

    Returns:
        dict: Dictionary with DOI/ID as key and list of raw values as value
    """
    results = {}

    for i, (doi, text) in enumerate(text_dict.items(), 1):
        print(f"Processing text {i}/{len(text_dict)} - DOI: {doi}...")
        raw_values = parse_materials(text)
        results[doi] = raw_values

    return results


def main():
    # Test texts with DOIs
    texts = {
        "10.1016/j.jallcom.2024.176609": "The 0.12Pb(Ni1/3Ta2/3)O3-xPbZrO3-(0.88-x)PbTiO3 piezoelectric ceramics with 2 mol% MnO2 (abbreviated as PNT-xPZ-PT-Mn, x = 0.41, 0.42, 0.43, 0.44) were fabricated by the conventional solid-state reaction method",
        "10.1016/j.jeurceramsoc.2025.117193": "In this study, dense Pb(1-x)K2x[Nb0.96Ta0.04]2O6 (PKxNT, x = 0.05, 0.10, 0.15, 0.20) ceramics were prepared via the solid-state reaction method.",
        "10.1016/j.ceramint.2024.09.282": "BaCO3 (99.8 %, Aladdin), TiO2 (99.0 %, McLean, Shanghai, China), SnO2 (99.9 %, Aladdin), CaCO3 (99.0 %, Sinopharm), Bi2O3 (99.9 %, McLean), Fe2O3 (99.0 %, Sinopharm) are used as raw materials, which were accurately weighed according to a composition of (1-x) (Ba0.95Ca0.05) (Ti0.89Sn0.11)O3-xBiFeO3 (BCTSO-xBFO, x = 0, 0.1, 0.5, 0.9 mol%) and milled with ethanol for 16 h.",
        "10.1016/j.ceramint.2024.10.314": "Lead-free piezoelectric ceramics with the formula Ba1-xSrxTi0.92Zr0.08O3 [x = 0, 0.04, 0.08, 0.12, 0.16, 0.2 (mol)] were prepared using the solid-state reaction technique.",
        "10.1016/j.jeurceramsoc.2024.117065": "Pure CaBi2Nb2O9 and rare-earth thulium-substituted CaBi2Nb2O9 powders with nominal compositions of Ca1-xTmxBi2Nb2O9 (CBN-100xTm) were prepared through a solid-phase reaction method. To characterize the phase transition in detail, a composition range of x = 0.01–0.05 was selected.",
    }

    print("Sending texts to material parser API...")
    print("=" * 80)

    # Process all texts
    all_results = parse_multiple_materials(texts)

    # Print results for each DOI
    for doi, raw_values in all_results.items():
        print(f"\nDOI: {doi}")
        print(f"Input: {texts[doi]}")
        print("-" * 40)

        if raw_values:
            print(f"Found {len(raw_values)} resolved formulas:")
            for j, value in enumerate(raw_values, 1):
                print(f"  {j}. {value}")
        else:
            print("No resolved formulas found or error occurred.")


if __name__ == "__main__":
    main()
