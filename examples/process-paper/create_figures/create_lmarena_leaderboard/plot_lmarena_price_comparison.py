import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


def load_data_from_json(file_path):
    """
    Load data from JSON file and categorize organizations based on model names.

    Args:
        file_path (str): Path to the JSON file containing pricing data

    Returns:
        dict: Dictionary with model data ready for plotting
    """
    with open(file_path, "r") as f:
        json_data = json.load(f)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(json_data)

    # Filter out models without ratings
    df = df.dropna(subset=["rating"])

    # Convert output_token_price to float
    df["output_token_price"] = df["output_token_price"].astype(float)

    def categorize_organization(model_name):
        """Categorize organization based on model name patterns."""
        model_name_lower = model_name.lower()

        if any(prefix in model_name_lower for prefix in ["gpt", "o1", "o3"]):
            return "OpenAI"
        elif model_name_lower.startswith("claude"):
            return "Anthropic"
        elif model_name_lower.startswith("gemini"):
            return "Google"
        elif model_name_lower.startswith("llama"):
            return "Meta"
        elif model_name_lower.startswith("deepseek"):
            return "DeepSeek"
        elif model_name_lower.startswith("qwen"):
            return "Alibaba"
        elif model_name_lower.startswith("yi"):
            return "O1 AI"
        elif model_name_lower.startswith("nova"):
            return "Amazon"
        else:
            return "Other"

    # Use existing organization or categorize based on model names if needed
    if "organization" not in df.columns:
        df["organization"] = df["name"].apply(categorize_organization)
    else:
        # Map existing organizations to our categories or keep as "Other"
        org_mapping = {
            "OpenAI": "OpenAI",
            "Google": "Google",
            "Anthropic": "Anthropic",
            "Meta": "Meta",
            "DeepSeek": "DeepSeek",
            "Alibaba": "Alibaba",
            "Amazon": "Amazon",
        }
        df["organization"] = df["organization"].map(org_mapping).fillna("Other")

    # Convert to dictionary format
    data = {
        "model_name": df["name"].tolist(),
        "rating": df["rating"].tolist(),
        "output_token_price": df["output_token_price"].tolist(),
        "organization": df["organization"].tolist(),
    }

    return data


def create_llm_pricing_plot(data, figsize=(20, 10)):
    """
    Create a scatter plot of LLM model ratings vs output token prices.

    Args:
        data (dict): Dictionary containing model data with keys:
                    - model_name: list of model names
                    - rating: list of ratings
                    - output_token_price: list of output token prices
                    - organization: list of organizations
        figsize (tuple): Figure size as (width, height)
    """
    # Create DataFrame
    df = pd.DataFrame(data)

    # Filter out models without ratings (if any)
    df = df.dropna(subset=["rating"])

    # Define color scheme for organizations
    org_colors = {
        "Google": "#efb118",
        "OpenAI": "#4169E1",
        "DeepSeek": "#9c6b4e",
        "Anthropic": "#40E0D0",
        "Meta": "#DA70D6",
        "Alibaba": "#7043a5",
        "Amazon": "#ff725c",
        "Other": "#32CD32",
    }

    # Create the plot
    plt.figure(figsize=figsize)

    # Add shaded region for high-performance, low-cost models
    # Arena score > 1250 and cost < $1.5/1M tokens
    ax = plt.gca()

    # Create the shaded region
    # Define the boundaries of the shaded area
    x_shade = [0.03, 1.5, 1.5, 0.03]  # Cost boundaries
    y_shade = [1250, 1250, 1475, 1475]  # Rating boundaries

    ax.fill(
        x_shade,
        y_shade,
        color="#f5427e",
        alpha=0.15,
        zorder=0,
        label="High Performance,\nLow Cost",
    )

    # Define the order for legend
    legend_order = [
        "Google",
        "OpenAI",
        "DeepSeek",
        "Anthropic",
        "Meta",
        "Alibaba",
        "O1 AI",
        "Amazon",
        "Other",
    ]

    # Define which organizations should show model names
    show_names_for = []

    # Plot each organization in the specified order
    for org in legend_order:
        if org in df["organization"].unique():
            org_data = df[df["organization"] == org]
            color = org_colors.get(
                org, "#32CD32"
            )  # Default to lime green for unlisted orgs

            plt.scatter(
                org_data["output_token_price"],
                org_data["rating"],
                c=color,
                label=org,
                s=150,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.5,
                zorder=2,  # Ensure scatter points are above the shaded region
            )

            # Add model names as annotations only for specified organizations
            if org in show_names_for:
                for idx, row in org_data.iterrows():
                    plt.annotate(
                        row["model_name"],
                        (row["output_token_price"], row["rating"]),
                        xytext=(10, -3),
                        textcoords="offset points",
                        fontsize=12,
                        alpha=1.0,
                        fontweight="bold",
                        bbox=None,
                    )

    # Customize the plot
    plt.xlabel("Cost ($/1M Tokens)", fontsize=16, labelpad=15)
    plt.ylabel("Arena Score", fontsize=16, labelpad=15)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set log scale for x-axis to match the original
    plt.xscale("log")

    # Set axis limits (will be adjusted based on actual data)
    plt.xlim(0.03, 500)
    plt.ylim(1150, 1475)

    # Customize x-axis ticks to match the image
    x_ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    plt.xticks(x_ticks, [str(x) for x in x_ticks], fontsize=12)
    plt.yticks(fontsize=12)

    # Customize grid
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, zorder=1)

    # Get all legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Find the shaded region handle and label
    shade_handle = None
    shade_label = None
    org_handles = []
    org_labels = []

    # Separate shaded region from organizations
    for i, label in enumerate(labels):
        if "High Performance" in label:
            shade_handle = handles[i]
            shade_label = label
        else:
            org_handles.append(handles[i])
            org_labels.append(label)

    # Create ordered handles and labels
    ordered_handles = []
    ordered_labels = []

    # First add the shaded region
    if shade_handle is not None:
        ordered_handles.append(shade_handle)
        ordered_labels.append(shade_label)

    # Add a dummy handle for the "\nOrganisations" heading
    from matplotlib.patches import Rectangle

    dummy_handle = Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", visible=False)
    ordered_handles.append(dummy_handle)
    ordered_labels.append("\nOrganisations")

    # Then add organizations in the specified order
    org_order = [
        "Google",
        "OpenAI",
        "DeepSeek",
        "Anthropic",
        "Meta",
        "Alibaba",
        "Amazon",
        "Other",
    ]

    for org in org_order:
        # Find the handle for this organization
        for i, label in enumerate(org_labels):
            if label == org:
                ordered_handles.append(org_handles[i])
                ordered_labels.append(org_labels[i])
                break

    # Customize legend - place at upper left corner with white background
    legend = plt.legend(
        ordered_handles,
        ordered_labels,
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=14,
        facecolor="white",  # White background for legend
        edgecolor="black",  # Black border for better visibility
        framealpha=1.0,  # Full opacity for white background
    )

    # Remove the automatic title
    legend.set_title("")

    # Make only the "\nOrganisations" heading bold (which should be the second item)
    legend_texts = legend.get_texts()
    for i, text in enumerate(legend_texts):
        if text.get_text() == "\nOrganisations":
            text.set_fontweight("bold")
        else:
            text.set_fontweight("normal")  # Ensure all others are normal weight

    # Adjust layout
    plt.tight_layout()


# Load data from JSON file
try:
    data = load_data_from_json("price_data.json")

    # Create the plot with custom figsize
    create_llm_pricing_plot(data, figsize=(16, 10))

    # Save the plot with proper settings
    plt.savefig(
        "../plots-raw/lmarena_leaderboard.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    # plt.show()

except FileNotFoundError:
    print(
        "Error: price_data.json file not found. Please make sure the file exists in the current directory."
    )
except json.JSONDecodeError:
    print("Error: Invalid JSON format in price_data.json file.")
except KeyError as e:
    print(f"Error: Missing required column {e} in the JSON data.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
