from fuzzywuzzy import fuzz


def match_compositions(prediction_dict, ground_truth_dict, threshold=80):
    """
    Match composition names between prediction and ground truth dictionaries based on similarity.

    This function takes two dictionaries, prediction_dict and ground_truth_dict,
    where the keys represent composition names and the values are their d33 property values.
    It uses fuzzy matching to find matching pairs of composition names between the two 
    dictionaries based on similarity.

    Parameters:
        prediction_dict (dict): A dictionary containing predicted composition data.
            Expected structure: {"composition_name": d33_value, ...}
        ground_truth_dict (dict): A dictionary containing ground truth composition data.
            Expected structure: {"composition_name": d33_value, ...}
        threshold (int, optional): The minimum similarity score required to consider
            two keys as a match. The default threshold is 80.

    Returns:
        tuple: A tuple containing three elements:
            - A dictionary (combined_dict) that contains the combined information of
              the matched compositions between prediction_dict and ground_truth_dict.
              Structure: {"composition_name": {"predicted_d33": value, "ground_truth_d33": value}}
            - A list (matched_pairs) that contains tuples of matched composition pairs from
              prediction_dict and ground_truth_dict.
            - A list (unmatched_predictions) that contains composition names from prediction_dict 
              that could not be matched with ground_truth_dict.
            - A list (unmatched_ground_truth) that contains composition names from ground_truth_dict
              that could not be matched with prediction_dict.

    Example:
        >>> pred = {"Pb(Zr0.52Ti0.48)O3": 620, "BaTiO3": 190}
        >>> truth = {"Pb(Zr0.52Ti0.48)O3": 600, "Ba1TiO3": 191}
        >>> combined, matched, unmatched_pred, unmatched_truth = match_compositions(pred, truth, 80)
    """

    combined_dict = {}
    matched_pairs = []
    unmatched_predictions = []
    matched_ground_truth_keys = set()  # Set to store matched ground_truth_dict keys

    for pred_comp, pred_value in prediction_dict.items():
        matched_comp = None
        highest_similarity = 0

        for truth_comp in ground_truth_dict.keys():
            # Check if the truth_comp is already matched, if so, skip it
            if truth_comp in matched_ground_truth_keys:
                continue

            similarity = fuzz.token_sort_ratio(pred_comp, truth_comp)
            if similarity > threshold and similarity > highest_similarity:
                matched_comp = truth_comp
                highest_similarity = similarity

        if matched_comp is not None:
            combined_dict[pred_comp] = {
                "predicted_d33": pred_value,
                "ground_truth_d33": ground_truth_dict[matched_comp],
                "matched_with": matched_comp,
                "similarity_score": highest_similarity
            }
            matched_pairs.append((pred_comp, matched_comp))
            # Add the matched ground_truth_dict key to the set
            matched_ground_truth_keys.add(matched_comp)
        else:
            unmatched_predictions.append(pred_comp)

    # Find unmatched ground truth compositions
    unmatched_ground_truth = [
        comp for comp in ground_truth_dict.keys() 
        if comp not in matched_ground_truth_keys
    ]

    return combined_dict, matched_pairs, unmatched_predictions, unmatched_ground_truth


def match_paper_data(prediction_data, ground_truth_data, threshold=80):
    """
    Match composition data for a single paper between prediction and ground truth.

    This function handles the full paper structure with composition_data and synthesis_data,
    specifically matching the compositions_property_values dictionaries.

    Parameters:
        prediction_data (dict): Predicted data for a single paper with structure:
            {
                "composition_data": {
                    "compositions_property_values": {...},
                    "property_unit": "...",
                    "family": "..."
                },
                "synthesis_data": {...}
            }
        ground_truth_data (dict): Ground truth data with same structure as prediction_data.
        threshold (int, optional): The minimum similarity score for matching. Default is 80.

    Returns:
        dict: A dictionary containing:
            - "matched_compositions": combined_dict from match_compositions
            - "matched_pairs": list of matched composition pairs
            - "unmatched_predictions": list of unmatched predicted compositions
            - "unmatched_ground_truth": list of unmatched ground truth compositions
            - "metadata": comparison of property_unit and family

    Example:
        >>> pred_paper = {"composition_data": {"compositions_property_values": {...}, ...}, ...}
        >>> truth_paper = {"composition_data": {"compositions_property_values": {...}, ...}, ...}
        >>> results = match_paper_data(pred_paper, truth_paper)
    """
    
    # Extract compositions_property_values from both dictionaries
    pred_compositions = prediction_data.get("composition_data", {}).get("compositions_property_values", {})
    truth_compositions = ground_truth_data.get("composition_data", {}).get("compositions_property_values", {})
    
    # Match compositions
    combined_dict, matched_pairs, unmatched_pred, unmatched_truth = match_compositions(
        pred_compositions, truth_compositions, threshold
    )
    
    # Compare metadata
    pred_unit = prediction_data.get("composition_data", {}).get("property_unit", "Not provided")
    truth_unit = ground_truth_data.get("composition_data", {}).get("property_unit", "Not provided")
    pred_family = prediction_data.get("composition_data", {}).get("family", "Not provided")
    truth_family = ground_truth_data.get("composition_data", {}).get("family", "Not provided")
    
    return {
        "matched_compositions": combined_dict,
        "matched_pairs": matched_pairs,
        "unmatched_predictions": unmatched_pred,
        "unmatched_ground_truth": unmatched_truth,
        "metadata": {
            "property_unit_match": pred_unit == truth_unit,
            "predicted_unit": pred_unit,
            "ground_truth_unit": truth_unit,
            "family_match": fuzz.token_sort_ratio(pred_family, truth_family) > threshold,
            "predicted_family": pred_family,
            "ground_truth_family": truth_family
        },
        "statistics": {
            "total_predicted": len(pred_compositions),
            "total_ground_truth": len(truth_compositions),
            "matched": len(matched_pairs),
            "unmatched_predictions": len(unmatched_pred),
            "unmatched_ground_truth": len(unmatched_truth),
            "match_rate": len(matched_pairs) / len(pred_compositions) * 100 if pred_compositions else 0
        }
    }