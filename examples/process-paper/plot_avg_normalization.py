# Customised script to plot average normalization results from evaluation data for the ComProScanner paper.

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Optional, Tuple, Union
from matplotlib.figure import Figure
import os
import json

mpl.rcParams["font.sans-serif"] = "Calibri"
mpl.rcParams["font.family"] = "sans-serif"


def _load_results_data(result_sources=None, folder_path=None, model_names=None):
    """
    Load evaluation results data from files or dictionaries for visualization.

    Args:
        result_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
        folder_path (str, optional): Path to folder containing JSON result files. Either result_sources or folder_path must be provided.
        model_names (list, optional): Names to use for each model (defaults to names in files or "Model 1", "Model 2", etc.)

    Returns:
        tuple: (results_data, names) where:
            - results_data (list): List of loaded result dictionaries
            - names (list): List of model names corresponding to each result

    Raises:
        ValueError: If neither result_sources nor folder_path is provided, or if the specified path does not exist
    """
    results_data = []
    names = []

    # Validate input parameters
    if result_sources is None and folder_path is None:
        raise ValueError("Either result_sources or folder_path must be provided")

    # Process folder_path if provided
    if folder_path is not None:
        if not os.path.isdir(folder_path):
            raise ValueError(f"The provided folder path does not exist: {folder_path}")

        # Find all JSON files in the folder
        json_files = []
        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                json_files.append(os.path.join(folder_path, file))

        # If no result_sources were provided, use all JSON files from the folder
        if result_sources is None:
            result_sources = json_files
        # If result_sources is a list, append the JSON files from the folder
        elif isinstance(result_sources, list):
            result_sources.extend(json_files)

    # Handle the case when result_sources is a single string (filepath)
    if isinstance(result_sources, str):
        if os.path.isfile(result_sources):
            result_sources = [result_sources]
        else:
            raise ValueError(f"The provided file path does not exist: {result_sources}")

    for i, source in enumerate(result_sources):
        # Load from dict or file
        if isinstance(source, dict):
            results_data.append(source)
            if model_names and i < len(model_names):
                names.append(model_names[i])
            elif "agent_model_name" in source:
                names.append(source["agent_model_name"])
            else:
                names.append(f"Model {i+1}")
        elif isinstance(source, str):
            try:
                with open(source, "r") as f:
                    result = json.load(f)
                    results_data.append(result)

                    if model_names and i < len(model_names):
                        names.append(model_names[i])
                    elif "agent_model_name" in result:
                        names.append(result["agent_model_name"])
                    else:
                        # Extract filename without extension
                        base_name = os.path.basename(source)
                        name = os.path.splitext(base_name)[0]
                        names.append(name)
            except Exception as e:
                print(f"Error loading {source}: {e}")
                continue

    if not results_data:
        raise ValueError("No valid results data found")

    return results_data, names


def _get_available_metrics(results_data, metrics_to_include):
    """
    Collect all available metrics across multiple result dictionaries.

    Args:
        results_data (list): List of result dictionaries
        metrics_to_include (list): List of metrics to consider including

    Returns:
        list: List of metrics available in the data that are in metrics_to_include
    """
    available_metrics = set()

    for result in results_data:
        # Check overall metrics
        for key in [
            "overall_accuracy",
            "overall_composition_accuracy",
            "overall_synthesis_accuracy",
        ]:
            if key in result and key in metrics_to_include:
                available_metrics.add(key)

        # Check absolute classification metrics
        if "absolute_classification_metrics" in result:
            for key in ["precision", "recall", "f1_score"]:
                if (
                    key in result["absolute_classification_metrics"]
                    and key in metrics_to_include
                ):
                    available_metrics.add(key)

        # Check normalized classification metrics
        if "normalized_classification_metrics" in result:
            for key in ["precision", "recall", "f1_score"]:
                normalized_key = f"normalized_{key}"
                if (
                    key in result["normalized_classification_metrics"]
                    and normalized_key in metrics_to_include
                ):
                    available_metrics.add(normalized_key)

    # Return metrics in the same order as metrics_to_include
    return [m for m in metrics_to_include if m in available_metrics]


def _get_metric_display_names():
    """
    Helper method to get display names for metrics.

    Returns:
        dict: Dictionary mapping metric names to display names
    """
    return {
        "overall_accuracy": "Overall\nAccuracy",
        "overall_composition_accuracy": "Composition\nAccuracy",
        "overall_synthesis_accuracy": "Synthesis\nAccuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "normalized_precision": "Normalised\nPrecision",
        "normalized_recall": "Normalised\nRecall",
        "normalized_f1_score": "Normalised\nF1 Score",
    }


def _extract_group_metrics(result, metrics):
    """
    Extract values for given metrics from a result dictionary.

    Args:
        result (dict): Result dictionary
        metrics (list): List of metrics to extract

    Returns:
        list: List of values for the metrics
    """
    values = []

    for metric in metrics:
        if metric in ["precision", "recall", "f1_score"]:
            # Absolute classification metrics
            if (
                "absolute_classification_metrics" in result
                and metric in result["absolute_classification_metrics"]
            ):
                values.append(result["absolute_classification_metrics"][metric])
            else:
                values.append(0)
        elif metric.startswith("normalized_"):
            # Normalized classification metrics
            base_metric = metric.replace("normalized_", "")
            if (
                "normalized_classification_metrics" in result
                and base_metric in result["normalized_classification_metrics"]
            ):
                values.append(result["normalized_classification_metrics"][base_metric])
            else:
                values.append(0)
        else:
            # Overall metrics
            values.append(result.get(metric, 0))

    return values


def _plot_model_bars(
    ax,
    group_positions,
    values,
    bar_positions,
    bar_width,
    color,
    name,
    display_values=True,
):
    """
    Plot bars for a single model in a grouped bar chart.

    Args:
        ax: Matplotlib axes object
        group_positions (array): Positions for metric groups
        values (list): Metric values for this model
        bar_positions (array): Positions for individual bars
        bar_width (float): Width of bars
        color: Color for bars
        name (str): Name of the model for legend
        display_values (bool): Whether to display value labels

    Returns:
        tuple: (bars, all_positions, all_values) where:
            - bars: Bar objects
            - all_positions (list): Positions of all bars
            - all_values (list): Values of all bars
    """
    # Create bars for this model
    bars = ax.bar(
        bar_positions,
        values,
        bar_width * 0.9,  # Slightly narrower to create gaps
        color=color,
        edgecolor="gray",
        alpha=0.8,
        label=name,
    )

    # Add value labels if requested
    if display_values:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height > 0:  # Only add text for non-zero values
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=22,
                    color="black",
                )

    # Return bar positions and values for threshold line
    return bars, list(bar_positions), values


def _calculate_bar_positions(num_metrics, num_models=1, group_width=0.8):
    """
    Calculate positions for bars in grouped bar chart.

    Args:
        num_metrics (int): Number of metrics (groups)
        num_models (int): Number of models (bars per group)
        group_width (float): Width allocated for each group

    Returns:
        tuple: (group_positions, bar_width) where:
            - group_positions (array): Positions for groups
            - bar_width (float): Width for individual bars
    """
    import numpy as np

    group_positions = np.arange(num_metrics)
    bar_width = group_width / num_models

    return group_positions, bar_width


def _draw_threshold_line_with_breaks(
    ax,
    typical_threshold,
    all_bar_positions,
    all_values,
    bar_width,
    threshold_color="red",
    threashold_line_style="--",
    threashold_tolerance_range=0.03,
):
    """
    Draw a horizontal threshold line with breaks at specified positions.

    Args:
    ax : matplotlib.axes.Axes
    typical_threshold (float): The y-value where the horizontal line should be drawn
    new_bar_positions_before (list): List of x-positions where the line should break (left side of break)
    new_bar_positions_after (list): List of x-positions where the line should resume (right side of break)
    threshold_color (str, optional): Color of the threshold line, defaults to "red"
    threashold_line_style (str, optional): Style of the threshold line, defaults to "--"
    threashold_tolerance_range (float, optional): Tolerance range for the threshold line, defaults to 0.03
    """
    new_bar_positions_before = []
    new_bar_positions_after = []
    for bar_position, value in zip(all_bar_positions, all_values):
        if (
            value - threashold_tolerance_range
            < typical_threshold
            < value + threashold_tolerance_range
        ):
            new_bar_positions_before.append(bar_position - 0.5 * bar_width)
            new_bar_positions_after.append(bar_position + 0.5 * bar_width)

    if len(new_bar_positions_before) == 0:
        # If no breaks, just draw a simple line
        ax.axhline(y=typical_threshold, color=threshold_color, linestyle="--")
        return

    # Create a twin axis that shares the y-axis
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(right=False, labelright=False)
    ax2.set_ylim(ax.get_ylim())

    # Get the x limits for relative position calculation
    x_min, x_max = ax.get_xlim()
    x_width = x_max - x_min

    # Calculate relative positions for start and end sections
    rel_position_before = (new_bar_positions_before[0] - x_min) / x_width
    rel_position_after = (new_bar_positions_after[-1] - x_min) / x_width

    # Draw the beginning part of the line
    ax2.axhline(
        y=typical_threshold,
        xmin=0,
        xmax=rel_position_before,
        color=threshold_color,
        linestyle=threashold_line_style,
    )

    # Draw the end part of the line
    ax2.axhline(
        y=typical_threshold,
        xmin=rel_position_after,
        xmax=1,
        color=threshold_color,
        linestyle="--",
    )

    # Draw fragments in the middle of the line
    for start, end in zip(new_bar_positions_after[:-1], new_bar_positions_before[1:]):
        ax.plot(
            [start, end],
            [typical_threshold, typical_threshold],
            color=threshold_color,
            linestyle="--",
        )


def plot_multiple_bar_charts(
    result_sources: Union[List[str], List[Dict], str] = None,
    folder_path: Optional[str] = None,
    output_file: Optional[str] = None,
    model_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    colormap: str = "Blues",
    display_values: bool = True,
    title: Optional[str] = None,
    typical_threshold: Optional[float] = None,
    threshold_line_style: str = "--",
    threashold_tolerance_range: float = 0.03,
    threshold_color: str = "red",
    show_grid: bool = True,
    y_label: str = "Accuracy",
    x_label: Optional[str] = None,
    group_width: float = 0.8,
    bar_width: Optional[float] = None,
    legend_loc: str = "best",
    legend_fontsize: int = 10,
    y_axis_range: Tuple[float, float] = (0, 1),
    dpi: int = 300,
    metrics_to_include: Optional[List[str]] = [
        "overall_accuracy",
        "overall_composition_accuracy",
        "overall_synthesis_accuracy",
        "precision",
        "recall",
        "f1_score",
        "normalized_precision",
        "normalized_recall",
        "normalized_f1_score",
    ],
):
    """
    Plot evaluation metrics from multiple result files or dictionaries as grouped bar charts.

    Args:
        result_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
        folder_path (Optional[str], optional): Path to folder containing JSON result files
        output_file (str, optional): Path to save the output plot image
        model_names (Optional[List[str]]): Names of models to display in the legend, defaults to filename or agent_model_name from results
        figsize (Tuple[int, int]): Figure size (width, height) in inches
        colormap (str): Matplotlib colormap name for the bars
        display_values (bool): Whether to display metric values on bars
        title (Optional[str]): Custom title for the plot
        typical_threshold (Optional[float]): Typical threshold value to display as a horizontal line [If not provided, no line is drawn]
        threshold_line_style (str): Style of the threshold line (default: '--')
        threashold_tolerance_range (float): Tolerance range for the threshold line (default: 0.03)
        threshold_color (str): Color for the threshold line (default: 'red')
        show_grid (bool): Whether to display horizontal grid lines in the plot (default: True)
        y_label (str): Label for the y-axis (default: 'Score')
        x_label (Optional[str]): Label for the x-axis (default: None)
        group_width (float): Width allocated for each group of bars (0-1)
        bar_width (Optional[float]): Width of individual bars, calculated automatically if None (default: None)
        legend_loc (str): Location of the legend (default: 'upper right')
        legend_fontsize (int): Font size for the legend (default: 10)
        y_axis_range (Tuple[float, float]): Range for the y-axis (default: (0, 1))
        dpi (int): dpi for the output image (default: 300)
        metrics_to_include (Optional[List[str]]): List of metrics to include from the plot (default: ['overall_accuracy', 'overall_composition_accuracy', 'overall_synthesis_accuracy', 'precision', 'recall', 'f1_score', 'normalized_precision', 'normalized_recall', 'normalized_f1_score'])

    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    # Load results data
    results_data, names = _load_results_data(result_sources, folder_path, model_names)

    # Get available metrics and display names
    metrics = _get_available_metrics(results_data, metrics_to_include)
    metric_display_names = _get_metric_display_names()

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Determine bar positions and width
    num_models = len(results_data)
    num_metrics = len(metrics)

    group_positions, calc_bar_width = _calculate_bar_positions(
        num_metrics, num_models, group_width
    )
    if bar_width is None:
        bar_width = calc_bar_width

    # Create colormap
    cmap = plt.colormaps[colormap].resampled(num_models)

    # Plot bars for each model
    all_bar_positions = []
    all_values = []

    for i, (result, name) in enumerate(zip(results_data, names)):
        # Extract metric values for this model
        values = _extract_group_metrics(result, metrics)

        # Calculate position for each bar
        bar_positions = group_positions - group_width / 2 + (i + 0.5) * bar_width

        # Plot bars for this model
        _, model_positions, model_values = _plot_model_bars(
            ax,
            group_positions,
            values,
            bar_positions,
            bar_width,
            cmap(i),
            name,
            display_values,
        )

        # Collect all positions and values for threshold line
        all_bar_positions.extend(model_positions)
        all_values.extend(model_values)

    # Set up axes with labels and title
    ax.set_ylabel(y_label, fontsize=40, color="black", labelpad=35)
    if x_label:
        ax.set_xlabel(x_label, fontsize=40, color="black")

    if title:
        ax.set_title(title, fontsize=75, fontweight="bold", color="black")
    else:
        ax.set_title(
            "Comparison of Evaluation Metrics Across Models",
            fontsize=14,
            fontweight="bold",
        )
    ax.tick_params(axis="y", labelsize=30)
    ax.set_xticks(group_positions)
    ax.set_xticklabels(
        [metric_display_names.get(m, m) for m in metrics],
        # keep distance between labels and the axes
        fontsize=40,
        # fontweight="bold",
        color="black",
    )
    ax.tick_params(axis="x", pad=30)
    ax.set_ylim(y_axis_range)

    # Add horizontal grid lines if requested
    if show_grid:
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Add a horizontal line for typical threshold if provided
    if typical_threshold is not None:
        # Sort positions and values for proper line drawing
        position_value_pairs = list(zip(all_bar_positions, all_values))
        sorted_pairs = sorted(position_value_pairs, key=lambda x: x[0])
        sorted_positions = [pair[0] for pair in sorted_pairs]
        sorted_values = [pair[1] for pair in sorted_pairs]

        # Draw the threshold line with breaks
        _draw_threshold_line_with_breaks(
            ax,
            typical_threshold,
            sorted_positions,
            sorted_values,
            bar_width,
            threshold_color,
            threshold_line_style,
            threashold_tolerance_range,
        )

    # Add legend
    ax.legend(
        loc=legend_loc,
        # bbox_to_anchor=(0.5, 1.15),  # Move it higher to leave more space
        ncol=num_models / 2,  # Single row of items
        fontsize=legend_fontsize,  # Smaller font to fit all models
        frameon=True,
        handlelength=1.0,  # Shorter handles
        handletextpad=0.5,  # Less padding
        columnspacing=1.0,  # Less space between columns
        labelcolor="black",  # Text color for legend labels
    )

    # Set edge color for axes
    for spine in ax.spines.values():
        spine.set_edgecolor("black")

    # Finalize and save
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        print(f"Comparison plot saved to {output_file}")
    return fig


if __name__ == "__main__":
    # Example usage
    semantic_evaluation_folder_path = "../piezo_test/eval-results/semantic-evaluation"
    agentic_evaluation_folder_path = "../piezo_test/eval-results/agentic-evaluation"
    semantic_output_file = "plots-raw/overall_comparison_semantic.png"
    agentic_output_file = "plots-raw/overall_comparison_agentic.png"
    plot_multiple_bar_charts(
        folder_path=semantic_evaluation_folder_path,
        metrics_to_include=[
            "normalized_precision",
            "normalized_recall",
            "normalized_f1_score",
        ],
        output_file=semantic_output_file,
        figsize=(28, 14),
        colormap="Reds",
        display_values=True,
        title=" ",
        y_label="Accuracy",
        x_label=None,
        group_width=0.9,
        legend_loc="upper center",
        legend_fontsize=30,
        y_axis_range=(0, 1.1),
    )
    plot_multiple_bar_charts(
        folder_path=agentic_evaluation_folder_path,
        metrics_to_include=[
            "normalized_precision",
            "normalized_recall",
            "normalized_f1_score",
        ],
        output_file=agentic_output_file,
        figsize=(28, 14),
        colormap="Blues",
        display_values=True,
        title=" ",
        y_label="Accuracy",
        x_label=None,
        group_width=0.9,
        legend_loc="upper center",
        legend_fontsize=30,
        y_axis_range=(0, 1.1),
    )
