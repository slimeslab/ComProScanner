"""
Comparison plot for framework evaluation metrics using seaborn heatmap.

This script creates a heatmap visualization comparing performance metrics
across three different frameworks: Eunomia, CMEG-IITR, and ComProScanner.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_evaluation_results(file_path: str) -> dict:
    """Load evaluation results from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(data: dict) -> dict:
    """Extract relevant metrics from evaluation results."""
    return {
        "Overall\nAccuracy": data["overall_accuracy"],
        "Composition\nAccuracy": data["overall_composition_accuracy"],
        "Synthesis\nAccuracy": data["overall_synthesis_accuracy"],
        "Precision": data["absolute_classification_metrics"]["precision"],
        "Recall": data["absolute_classification_metrics"]["recall"],
        "F1 Score": data["absolute_classification_metrics"]["f1_score"],
        "Normalized\nPrecision": data["normalized_classification_metrics"]["precision"],
        "Normalized\nRecall": data["normalized_classification_metrics"]["recall"],
        "Normalized\nF1 Score": data["normalized_classification_metrics"]["f1_score"],
    }


def create_comparison_heatmap(
    metrics_df: pd.DataFrame,
    output_path: str = "framework_comparison_heatmap.png",
    figsize: tuple = (16, 4),
) -> None:
    """Create and save a heatmap comparing framework metrics."""

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap with seaborn
    heatmap = sns.heatmap(
        metrics_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Score", "pad": 0.04},
        annot_kws={"size": 10, "weight": "bold"},
        ax=ax,
    )

    ax.set_xlabel("Metrics", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Framework", fontsize=12, fontweight="bold", labelpad=10)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Heatmap saved to: {output_path}")

    plt.close(fig)


def main():
    """Main function to create the comparison plot."""

    # Define file paths
    base_dir = Path(__file__).parent

    framework_files = {
        "Eunomia\n(DeepSeek-V3.2)": base_dir
        / "Eunomia"
        / "piezo_evaluation_results.json",
        "CMEG-IITR\nExtraction Agent\n(DeepSeek-V3.2)": base_dir
        / "CMEG-IITR_Agentic_data_extraction"
        / "piezo_evaluation_results.json",
        "ComProScanner\n(DeepSeek-V3-0324)": base_dir
        / "ComProScanner"
        / "comparison_deepseek_evaluation_results.json",
    }

    # Load and extract metrics for each framework
    framework_metrics = {}
    for framework_name, file_path in framework_files.items():
        print(f"Loading {framework_name} from {file_path}...")
        data = load_evaluation_results(file_path)
        framework_metrics[framework_name] = extract_metrics(data)

        # Print model name if available
        if "extraction_agent_model_name" in data:
            print(f"  Model: {data['extraction_agent_model_name']}")

    # Create DataFrame
    metrics_df = pd.DataFrame(framework_metrics).T

    # Print the metrics table
    print("\n" + "=" * 80)
    print("Performance Metrics Summary")
    print("=" * 80)
    print(metrics_df.to_string())
    print("=" * 80 + "\n")

    # Create the heatmap
    output_path = str(base_dir / "framework_comparison_heatmap.png")
    create_comparison_heatmap(metrics_df, output_path)


if __name__ == "__main__":
    main()
