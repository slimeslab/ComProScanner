import json
import pandas as pd


def load_and_combine_data(leaderboard_file, scatterplot_file, output_file):
    """
    Combine data from leaderboard and scatterplot JSON files to create price_data.json

    Args:
        leaderboard_file (str): Path to leaderboard-text.json file
        scatterplot_file (str): Path to scatterplot-data.json file
        output_file (str): Path to output price_data.json file
    """

    try:
        # Load leaderboard data (contains ratings)
        with open(leaderboard_file, "r") as f:
            leaderboard_data = json.load(f)

        # Load scatterplot data (contains pricing details)
        with open(scatterplot_file, "r") as f:
            scatterplot_data = json.load(f)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return

    # Convert to DataFrames for easier manipulation
    leaderboard_df = pd.DataFrame(leaderboard_data)
    scatterplot_df = pd.DataFrame(scatterplot_data)

    # The leaderboard data has model names as index, so reset index to make it a column
    leaderboard_df.reset_index(inplace=True)
    leaderboard_df.rename(columns={"index": "model_name"}, inplace=True)

    # Print column names to understand the structure
    print("Leaderboard columns:", leaderboard_df.columns.tolist())
    print("Scatterplot columns:", scatterplot_df.columns.tolist())
    print(f"Leaderboard shape: {leaderboard_df.shape}")
    print(f"Scatterplot shape: {scatterplot_df.shape}")

    # Sample the first few rows to understand data structure
    print("\nLeaderboard sample (first 3 models):")
    for i in range(min(3, len(leaderboard_df))):
        print(f"Model {i+1}: {leaderboard_df.iloc[i]['model_name']}")

    print("\nScatterplot sample (first 3 models with API names):")
    for i in range(min(3, len(scatterplot_df))):
        print(
            f"Model {i+1}: {scatterplot_df.iloc[i]['name']} -> API: {scatterplot_df.iloc[i]['model_api_name']}"
        )

    # Extract rating from the 'full' column (which contains overall rating)
    def extract_rating(rating_dict):
        """Extract rating value from the rating dictionary"""
        if isinstance(rating_dict, dict) and "rating" in rating_dict:
            return rating_dict["rating"]
        return None

    # Extract overall rating from the 'full' column
    leaderboard_df["rating"] = leaderboard_df["full"].apply(extract_rating)

    # Filter out models without ratings
    leaderboard_df = leaderboard_df.dropna(subset=["rating"])
    print(
        f"\nLeaderboard after filtering for valid ratings: {leaderboard_df.shape[0]} models"
    )

    # Create a simple mapping function for model names/API names
    def normalize_name(name):
        """Normalize model names for better matching"""
        if pd.isna(name):
            return ""
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

    # Normalize names for matching - use model_name from leaderboard and model_api_name from scatterplot
    leaderboard_df["normalized_name"] = leaderboard_df["model_name"].apply(
        normalize_name
    )
    scatterplot_df["normalized_api_name"] = scatterplot_df["model_api_name"].apply(
        normalize_name
    )

    # Create a set of scatterplot API names for faster lookup
    scatterplot_api_names = dict(
        zip(scatterplot_df["normalized_api_name"], scatterplot_df.index)
    )

    # Find matching models using API names
    matches = []

    print(f"\nAttempting to match leaderboard models with scatterplot API names...")

    for idx, lb_row in leaderboard_df.iterrows():
        lb_name = lb_row["normalized_name"]

        # Direct match with API name
        if lb_name in scatterplot_api_names:
            scatter_idx = scatterplot_api_names[lb_name]
            scatter_row = scatterplot_df.iloc[scatter_idx]
            matches.append(
                {
                    "name": scatter_row["name"],  # Display name from scatterplot
                    "leaderboard_name": lb_row[
                        "model_name"
                    ],  # Original name from leaderboard
                    "api_name": scatter_row["model_api_name"],  # API name
                    "rating": lb_row["rating"],
                    "output_token_price": scatter_row["output_token_price"],
                    "organization": scatter_row["organization"],
                    "input_token_price": scatter_row["input_token_price"],
                }
            )
            continue

        # If no direct match, try partial matching
        found_match = False
        for api_name, scatter_idx in scatterplot_api_names.items():
            # Check if names are similar (basic fuzzy matching)
            if (lb_name in api_name or api_name in lb_name) and len(lb_name) > 3:
                scatter_row = scatterplot_df.iloc[scatter_idx]
                matches.append(
                    {
                        "name": scatter_row["name"],
                        "leaderboard_name": lb_row["model_name"],
                        "api_name": scatter_row["model_api_name"],
                        "rating": lb_row["rating"],
                        "output_token_price": scatter_row["output_token_price"],
                        "organization": scatter_row["organization"],
                        "input_token_price": scatter_row["input_token_price"],
                    }
                )
                found_match = True
                break

        if not found_match:
            # Print unmatched models for debugging
            print(f"  No match found for: '{lb_row['model_name']}'")

    # Create combined dataframe from matches
    if matches:
        combined_df = pd.DataFrame(matches)
        print(f"\nSuccessfully matched {len(matches)} models")

        # Convert data types to ensure proper formatting
        combined_df["rating"] = pd.to_numeric(combined_df["rating"], errors="coerce")
        combined_df["output_token_price"] = pd.to_numeric(
            combined_df["output_token_price"], errors="coerce"
        )
        combined_df["input_token_price"] = pd.to_numeric(
            combined_df["input_token_price"], errors="coerce"
        )

        # Remove any rows where conversion failed
        combined_df = combined_df.dropna(subset=["rating", "output_token_price"])

        # Show some matched examples
        print("\nSample matches (Leaderboard -> Scatterplot):")
        for i, match in enumerate(matches[:8]):
            print(
                f"  {i+1}. '{match['leaderboard_name']}' -> '{match['name']}' (API: {match['api_name']})"
            )
    else:
        print("\nNo matches found between leaderboard and scatterplot data")
        return

    # The combined_df is already in the right format from matches
    # Just need to ensure we have the right column names

    # Rename to match expected column names for the plotting script
    if "model_name" in combined_df.columns:
        combined_df = combined_df.rename(columns={"model_name": "leaderboard_name"})

    print(f"\nFinal columns: {combined_df.columns.tolist()}")

    # Convert to JSON format and save
    try:
        # Convert DataFrame to list of dictionaries
        output_data = combined_df.to_dict("records")

        # Save to JSON file
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSuccessfully created {output_file}")
        print(f"Total models in combined dataset: {len(output_data)}")

        # Show sample of final data
        print(f"\nSample of final data:")
        for i, item in enumerate(output_data[:3]):
            print(f"Model {i+1}: {item}")

        # Show some statistics
        print(f"\nDataset statistics:")
        try:
            print(
                f"  Rating range: {combined_df['rating'].min():.1f} - {combined_df['rating'].max():.1f}"
            )
            print(
                f"  Price range: ${combined_df['output_token_price'].min():.3f} - ${combined_df['output_token_price'].max():.3f}"
            )
            print(f"  Organizations: {sorted(combined_df['organization'].unique())}")
        except (ValueError, TypeError) as e:
            print(f"  Could not calculate statistics: {e}")
            print(f"  Rating column type: {combined_df['rating'].dtype}")
            print(f"  Price column type: {combined_df['output_token_price'].dtype}")

    except Exception as e:
        print(f"Error saving output file: {e}")


def main():
    """Main function to run the data combination process"""

    # File paths
    leaderboard_file = "leaderboard-text_from_lmarena.json"
    scatterplot_file = "scatterplot-data_from_lmarena.json"
    output_file = "price_data.json"

    print("Combining leaderboard and scatterplot data...")
    print(f"Input files: {leaderboard_file}, {scatterplot_file}")
    print(f"Output file: {output_file}")
    print("-" * 50)

    load_and_combine_data(leaderboard_file, scatterplot_file, output_file)


if __name__ == "__main__":
    main()
