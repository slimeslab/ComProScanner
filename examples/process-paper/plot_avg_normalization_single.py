"""
plot_avg_normalization_single.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 11-08-2025
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Optional, Tuple, Union
from matplotlib.figure import Figure
import os
import json
import numpy as np

mpl.rcParams["font.sans-serif"] = "Calibri"
mpl.rcParams["font.family"] = "sans-serif"


def _load_model_results(semantic_file_path, agentic_file_path):
    """
    Load model results from semantic and agentic evaluation files.

    Args:
        semantic_file_path (str): Path to semantic evaluation results file
        agentic_file_path (str): Path to agentic evaluation results file

    Returns:
        tuple: (semantic_data, agentic_data, model_name) containing the loaded results and model name
    """
    semantic_data = None
    agentic_data = None
    model_name = None

    # Load semantic results
    if not os.path.exists(semantic_file_path):
        raise ValueError(f"Semantic evaluation file not found: {semantic_file_path}")

    with open(semantic_file_path, "r") as f:
        semantic_data = json.load(f)

    # Load agentic results
    if not os.path.exists(agentic_file_path):
        raise ValueError(f"Agentic evaluation file not found: {agentic_file_path}")

    with open(agentic_file_path, "r") as f:
        agentic_data = json.load(f)

    # Extract model name from semantic data (prefer) or agentic data
    if "agent_model_name" in semantic_data:
        model_name = semantic_data["agent_model_name"]
    elif "agent_model_name" in agentic_data:
        model_name = agentic_data["agent_model_name"]
    else:
        # Fallback to filename without extension
        model_name = os.path.splitext(os.path.basename(semantic_file_path))[0]

    return semantic_data, agentic_data, model_name


def _extract_normalized_metrics(data):
    """
    Extract normalized precision, recall, and F1 score from result data.

    Args:
        data (dict): Result dictionary

    Returns:
        tuple: (precision, recall, f1_score)
    """
    if "normalized_classification_metrics" not in data:
        raise ValueError("Normalized classification metrics not found in data")

    metrics = data["normalized_classification_metrics"]
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    f1_score = metrics.get("f1_score", 0)

    return precision, recall, f1_score


def plot_evaluation_comparison(
    semantic_file_path: str,
    agentic_file_path: str,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    display_values: bool = True,
    title: Optional[str] = None,
    show_grid: bool = True,
    y_label: str = "Accuracy",
    x_label: Optional[str] = None,
    group_width: float = 0.6,
    legend_loc: str = "upper right",
    legend_fontsize: int = 24,
    y_axis_range: Tuple[float, float] = (0, 1.1),
    dpi: int = 300,
):
    """
    Plot comparison between semantic and agentic evaluation for any model.

    Args:
        semantic_file_path (str): Path to semantic evaluation results file
        agentic_file_path (str): Path to agentic evaluation results file
        output_file (str, optional): Path to save the output plot image
        figsize (Tuple[int, int]): Figure size (width, height) in inches
        display_values (bool): Whether to display metric values on bars
        title (Optional[str]): Custom title for the plot
        show_grid (bool): Whether to display horizontal grid lines
        y_label (str): Label for the y-axis
        x_label (Optional[str]): Label for the x-axis
        group_width (float): Width allocated for each group of bars (0-1)
        legend_loc (str): Location of the legend
        legend_fontsize (int): Font size for the legend
        y_axis_range (Tuple[float, float]): Range for the y-axis
        dpi (int): DPI for the output image

    Returns:
        matplotlib.figure.Figure: The generated figure object
    """

    # Load model results from both files
    semantic_data, agentic_data, model_name = _load_model_results(
        semantic_file_path, agentic_file_path
    )

    # Extract normalized metrics
    semantic_precision, semantic_recall, semantic_f1 = _extract_normalized_metrics(
        semantic_data
    )
    agentic_precision, agentic_recall, agentic_f1 = _extract_normalized_metrics(
        agentic_data
    )

    # Prepare data for plotting
    metrics_names = [
        "Normalised\nPrecision",
        "Normalised\nRecall",
        "Normalised\nF1 Score",
    ]
    semantic_values = [semantic_precision, semantic_recall, semantic_f1]
    agentic_values = [agentic_precision, agentic_recall, agentic_f1]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Set up bar positions
    x = np.arange(len(metrics_names))
    bar_width = group_width / 2

    # Create bars
    semantic_bars = ax.bar(
        x - bar_width / 2,
        semantic_values,
        bar_width,
        label="Semantic Evaluation for Llama-3.3-70B-Instruct",
        color="#d62728",  # Red color
        edgecolor="gray",
        alpha=0.8,
    )

    agentic_bars = ax.bar(
        x + bar_width / 2,
        agentic_values,
        bar_width,
        label="Agentic Evaluation for Llama-3.3-70B-Instruct",
        color="#1f77b4",  # Blue color
        edgecolor="gray",
        alpha=0.8,
    )

    # Add value labels if requested
    if display_values:
        for bar, value in zip(semantic_bars, semantic_values):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=20,
                    color="black",
                )

        for bar, value in zip(agentic_bars, agentic_values):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=20,
                    color="black",
                )

    # Customize the plot
    ax.set_ylabel(y_label, fontsize=36, color="black", labelpad=35)
    if x_label:
        ax.set_xlabel(x_label, fontsize=36, color="black")

    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=32, color="black")
    ax.tick_params(axis="x", pad=20)
    ax.tick_params(axis="y", labelsize=26)

    # Set y-axis range
    ax.set_ylim(y_axis_range)

    # Add horizontal grid lines if requested
    if show_grid:
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Add legend
    ax.legend(
        loc=legend_loc,
        fontsize=legend_fontsize,
        frameon=True,
        handlelength=1.5,
        handletextpad=0.8,
        labelcolor="black",
    )

    # Set edge color for axes
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    # Finalize layout
    plt.tight_layout()

    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        print(f"Model comparison plot saved to {output_file}")

    return fig


if __name__ == "__main__":
    # Example usage
    semantic_file_path = "../piezo_test/eval-results/semantic-evaluation/llama-3.3-70b-instruct-semantic-evaluation-results.json"
    agentic_file_path = "../piezo_test/eval-results/agentic-evaluation/llama-3.3-70b-instruct-agentic-evaluation-results.json"
    output_file = "plots-raw/single_semantic_vs_agentic_comparison.png"

    # Create the comparison plot
    fig = plot_evaluation_comparison(
        semantic_file_path=semantic_file_path,
        agentic_file_path=agentic_file_path,
        output_file=output_file,
        figsize=(18, 12),
        display_values=True,
        y_label="Accuracy",
        group_width=0.6,
        legend_loc="upper center",
        legend_fontsize=28,
        y_axis_range=(0, 1.1),
        dpi=300,
    )
