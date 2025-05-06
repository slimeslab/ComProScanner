"""
visualiser.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 22-04-2025
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from typing import List, Dict, Union, Optional, Tuple

from ...utils.error_handler import ValueErrorHandler


class EvalVisualiser:
    def __init__(self):
        pass

    def _load_results_data(
        self, result_sources=None, folder_path=None, model_names=None
    ):
        """
        Load evaluation results data from files or dictionaries for visualisation.

        Args:
            result_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
            folder_path (str, optional): Path to folder containing JSON result files. Either result_sources or folder_path must be provided.
            model_names (list, optional): Names to use for each model (defaults to names in files or "Model 1", "Model 2", etc.)

        Returns:
            tuple: (results_data, names) where:
                - results_data (list): List of loaded result dictionaries
                - names (list): List of model names corresponding to each result

        Raises:
            ValueErrorHandler: If neither result_sources nor folder_path is provided, or if the specified path does not exist
        """
        results_data = []
        names = []

        # Validate input parameters
        if result_sources is None and folder_path is None:
            raise ValueErrorHandler(
                "Either result_sources or folder_path must be provided"
            )

        # Process folder_path if provided
        if folder_path is not None:
            if not os.path.isdir(folder_path):
                raise ValueErrorHandler(
                    f"The provided folder path does not exist: {folder_path}"
                )

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
                raise ValueErrorHandler(
                    f"The provided file path does not exist: {result_sources}"
                )

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
            raise ValueErrorHandler("No valid results data found")

        return results_data, names

    def _extract_metrics_from_result(self, result, metrics_to_include):
        """
        Extract metric values from a result dictionary based on provided metrics list.

        Args:
            result (dict): Dictionary containing evaluation results
            metrics_to_include (list): List of metrics to extract

        Returns:
            tuple: (metrics, values) where:
                - metrics (list): List of formatted metric names
                - values (list): List of corresponding metric values
        """
        metrics = []
        values = []

        # Overall accuracies - only include metrics that exist in the results
        if "overall_accuracy" in metrics_to_include and "overall_accuracy" in result:
            metrics.append("Overall\nAccuracy")
            values.append(result.get("overall_accuracy", 0))
        if (
            "overall_composition_accuracy" in metrics_to_include
            and "overall_composition_accuracy" in result
        ):
            metrics.append("Composition\nAccuracy")
            values.append(result.get("overall_composition_accuracy", 0))
        if (
            "overall_synthesis_accuracy" in metrics_to_include
            and "overall_synthesis_accuracy" in result
        ):
            metrics.append("Synthesis\nAccuracy")
            values.append(result.get("overall_synthesis_accuracy", 0))

        # Absolute classification metrics
        classification = result.get("absolute_classification_metrics", {})
        if (
            "absolute_precision" in metrics_to_include
            or "precision" in metrics_to_include
        ) and "precision" in classification:
            metrics.append("Precision")
            values.append(classification.get("precision", 0))
        if (
            "absolute_recall" in metrics_to_include or "recall" in metrics_to_include
        ) and "recall" in classification:
            metrics.append("Recall")
            values.append(classification.get("recall", 0))
        if (
            "absolute_f1_score" in metrics_to_include
            or "f1_score" in metrics_to_include
        ) and "f1_score" in classification:
            metrics.append("F1 Score")
            values.append(classification.get("f1_score", 0))

        # Normalized classification metrics
        classification = result.get("normalized_classification_metrics", {})
        if (
            "normalized_precision" in metrics_to_include
            and "precision" in classification
        ):
            metrics.append("Normalized Precision")
            values.append(classification.get("precision", 0))
        if "normalized_recall" in metrics_to_include and "recall" in classification:
            metrics.append("Normalized Recall")
            values.append(classification.get("recall", 0))
        if "normalized_f1_score" in metrics_to_include and "f1_score" in classification:
            metrics.append("Normalized F1 Score")
            values.append(classification.get("f1_score", 0))

        return metrics, values

    def _get_chart_colors(self, colormap, num_items, index=None):
        """
        Create colors using a specified colormap.

        Args:
            colormap (str): Matplotlib colormap name
            num_items (int): Number of items to create colors for
            index (int, optional): If provided, return just the color at this index

        Returns:
            If index is None: list of colors
            If index is provided: single color
        """
        cmap = cm.get_cmap(colormap)
        if index is not None:
            position = index / max(1, num_items - 1) if num_items > 1 else 0.5
            return cmap(position)
        colors = []
        for i in range(num_items):
            position = i / max(1, num_items - 1) if num_items > 1 else 0.5
            colors.append(cmap(position))

        return colors

    def _plot_bars_with_values(
        self, ax, bar_positions, values, bar_width, bar_colors, display_values=True
    ):
        """
        Plot bars with optional value labels.

        Args:
            ax: Matplotlib axes object
            bar_positions (array): Positions for bars
            values (list): Values for bars
            bar_width (float): Width of bars
            bar_colors (list): Colors for bars
            display_values (bool): Whether to display value labels on bars

        Returns:
            list: List of bar objects
        """
        # Create bars
        bars = ax.bar(
            bar_positions,
            values,
            bar_width,
            color=bar_colors,
            edgecolor="gray",
            alpha=0.8,
        )

        # Display metric values on bars if requested
        if display_values:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        return bars

    def _setup_bar_chart_axes(
        self,
        ax,
        metrics,
        title,
        y_axis_label,
        x_axis_label,
        y_axis_range,
        show_grid=True,
        rotation=45,
        ha="right",
    ):
        """
        Set up bar chart axes with labels, title, and grid.

        Args:
            ax: Matplotlib axes object
            metrics (list): List of metric names for x-axis labels
            title (str): Title for the plot
            y_axis_label (str): Label for y-axis
            x_axis_label (str): Label for x-axis (optional)
            y_axis_range (tuple): Range for y-axis
            show_grid (bool): Whether to display horizontal grid lines
            rotation (int): Rotation angle for x-axis labels
            ha (str): Horizontal alignment for x-axis labels
        """
        # Set labels and title
        ax.set_ylabel(y_axis_label, fontsize=12)
        if x_axis_label:
            ax.set_xlabel(x_axis_label, fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")

        # Set x-axis labels and tick positions
        tick_positions = range(len(metrics))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(metrics, rotation=rotation, ha=ha)
        ax.set_ylim(y_axis_range)

        # Add horizontal grid lines if requested
        if show_grid:
            ax.grid(axis="y", linestyle="--", alpha=0.7)

    def _get_model_title(self, title, model_name, results):
        """
        Generate an appropriate title for the plot.

        Args:
            title (str): Custom title if provided
            model_name (str): Name of the model
            results (dict): Results dictionary that might contain model name

        Returns:
            str: Title for the plot
        """
        if title:
            return title
        elif model_name:
            return f"Evaluation Metrics for {model_name}"
        elif "agent_model_name" in results:
            return f"Evaluation Metrics for {results['agent_model_name']}"
        else:
            return "Materials Data Evaluation Metrics"

    def _calculate_bar_positions(self, num_metrics, num_models=1, group_width=0.8):
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
        self,
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
        for start, end in zip(
            new_bar_positions_after[:-1], new_bar_positions_before[1:]
        ):
            ax.plot(
                [start, end],
                [typical_threshold, typical_threshold],
                color=threshold_color,
                linestyle="--",
            )

    def _get_available_metrics(self, results_data, metrics_to_include):
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

    def _extract_group_metrics(self, result, metrics):
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
                    values.append(
                        result["normalized_classification_metrics"][base_metric]
                    )
                else:
                    values.append(0)
            else:
                # Overall metrics
                values.append(result.get(metric, 0))

        return values

    def _plot_model_bars(
        self,
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
                        height + 0.01,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )

        # Return bar positions and values for threshold line
        return bars, list(bar_positions), values

    def _setup_radar_plot(
        self,
        ax,
        metrics,
        metric_display_names,
        theta,
        radar_range,
        label_fontsize,
        label_padding,
        show_grid,
        grid_line_width,
        grid_line_style,
        grid_line_color,
        grid_line_alpha,
        show_grid_labels,
    ):
        """
        Helper method to set up the radar plot axes.

        Args:
            ax: Matplotlib axes object
            metrics (list): List of metric names
            metric_display_names (dict): Dictionary mapping metric names to display names
            theta (array): Array of angles for radar chart
            radar_range (tuple): Range for the radar axes
            label_fontsize (int): Font size for axis labels
            label_padding (float): Padding for axis labels
            show_grid (bool): Whether to display grid lines
            grid_line_width (float): Width of grid lines
            grid_line_style (str): Style of grid lines
            grid_line_color (str): Color of grid lines
            grid_line_alpha (float): Alpha value for grid lines
            show_grid_labels (bool): Whether to show grid labels
        """
        # Set the ticks at the angles for the metrics
        ax.set_xticks(theta)

        # Set labels with padding
        ax.set_xticklabels(
            [metric_display_names.get(m, m) for m in metrics],
            fontsize=label_fontsize,
        )
        ax.tick_params(axis="x", pad=label_padding * 30)  # Adjust label distance

        # Set y limits
        ax.set_ylim(radar_range)

        # Set grid properties
        if show_grid:
            ax.grid(
                True,
                linewidth=grid_line_width,
                linestyle=grid_line_style,
                color=grid_line_color,
                alpha=grid_line_alpha,
            )

            # Hide grid labels if requested
            if not show_grid_labels:
                ax.set_yticklabels([])
        else:
            ax.grid(False)

    def _add_threshold_circle(
        self, ax, typical_threshold, threshold_color, threshold_line_style
    ):
        """
        Helper method to add a threshold circle to a radar plot.

        Args:
            ax: Matplotlib axes object
            typical_threshold (float): Threshold value
            threshold_color (str): Color for the threshold line
            threshold_line_style (str): Style for the threshold line

        Returns:
            Line2D: Legend patch for the threshold
        """
        if typical_threshold is None:
            return None

        threshold_circle = plt.Circle(
            (0, 0),
            typical_threshold,
            transform=ax.transData._b,
            edgecolor=threshold_color,
            facecolor="none",
            linestyle=threshold_line_style,
            linewidth=1.5,
        )
        ax.add_artist(threshold_circle)

        return Line2D(
            [],
            [],
            color=threshold_color,
            linestyle=threshold_line_style,
            linewidth=1.5,
            label=f"Threshold: {typical_threshold:.2f}",
        )

    def _get_available_metrics(self, results_data, metrics_to_include):
        """
        Helper method to get available metrics from results data.

        Args:
            results_data (list or dict): List of result dictionaries or a single result dictionary
            metrics_to_include (list): List of metrics to include

        Returns:
            list: List of available metrics
        """
        available_metrics = set()

        # Handle both single result and list of results
        if not isinstance(results_data, list):
            results_data = [results_data]

        for result in results_data:
            for key in [
                "overall_accuracy",
                "overall_composition_accuracy",
                "overall_synthesis_accuracy",
            ]:
                if key in result and key in metrics_to_include:
                    available_metrics.add(key)
            if "absolute_classification_metrics" in result:
                for key in ["precision", "recall", "f1_score"]:
                    if (
                        key in result["absolute_classification_metrics"]
                        and key in metrics_to_include
                    ):
                        available_metrics.add(key)
            if "normalized_classification_metrics" in result:
                for key in ["precision", "recall", "f1_score"]:
                    if (
                        key in result["normalized_classification_metrics"]
                        and f"normalized_{key}" in metrics_to_include
                    ):
                        available_metrics.add(f"normalized_{key}")

        # Use specified metrics or default ordering
        if metrics_to_include:
            metrics = [m for m in metrics_to_include if m in available_metrics]
            if not metrics:
                raise ValueErrorHandler(
                    "None of the specified metrics are available in the results"
                )
        else:
            # Default ordering of metrics
            default_order = [
                "overall_accuracy",
                "overall_composition_accuracy",
                "overall_synthesis_accuracy",
                "precision",
                "recall",
                "f1_score",
                "normalized_precision",
                "normalized_recall",
                "normalized_f1_score",
            ]
            metrics = [m for m in default_order if m in available_metrics]

        return metrics

    def _get_metric_display_names(self):
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
            "normalized_precision": "Normalized\nPrecision",
            "normalized_recall": "Normalized\nRecall",
            "normalized_f1_score": "Normalized\nF1 Score",
        }

    def _setup_heatmap_data(
        self,
        result,
        metrics_to_use,
        metric_extractors,
        min_score,
        max_score,
        bin_count,
        use_percentage,
    ):
        """
        Prepare heatmap data for a single model result.

        Args:
            result (dict): Dictionary containing evaluation results
            metrics_to_use (list): List of metrics to include in the heatmap
            metric_extractors (dict): Dictionary of functions to extract metric values
            min_score (float): Minimum score for binning
            max_score (float): Maximum score for binning
            bin_count (int): Number of bins to use
            use_percentage (bool): Whether to convert counts to percentages

        Returns:
            tuple: (distribution, total_items) where:
                - distribution: 2D array with metric values distributed into bins
                - total_items: Total number of items evaluated
        """
        # Initialize distribution matrix
        distribution = np.zeros((len(metrics_to_use), bin_count))
        total_items = len(result.get("item_results", {}))

        # Count DOIs in each bin
        for doi, item in result.get("item_results", {}).items():
            for i, metric_id in enumerate(metrics_to_use):
                try:
                    if metric_id in [
                        "overall_accuracy",
                        "overall_composition_accuracy",
                        "overall_synthesis_accuracy",
                    ]:
                        value = metric_extractors[metric_id](result)
                    elif metric_id in ["precision", "recall", "f1_score"]:
                        value = item.get("absolute_classification_metrics", {}).get(
                            metric_id.split("_")[-1], 0
                        )
                    elif metric_id.startswith("normalized_"):
                        base_metric = metric_id.split("_")[-1]
                        value = item.get("normalized_classification_metrics", {}).get(
                            base_metric, 0
                        )
                    elif metric_id == "property_match":
                        value = (
                            item.get("details", {})
                            .get("composition_data", {})
                            .get("compositions_property_values", {})
                            .get("similarity_score", 0)
                        )
                    elif metric_id == "method_match":
                        value = (
                            1
                            if item.get("details", {})
                            .get("synthesis_data", {})
                            .get("method", {})
                            .get("match", False)
                            else 0
                        )
                    elif metric_id == "precursors_match":
                        value = (
                            item.get("details", {})
                            .get("synthesis_data", {})
                            .get("precursors", {})
                            .get("similarity", 0)
                        )
                    elif metric_id == "steps_match":
                        value = (
                            item.get("details", {})
                            .get("synthesis_data", {})
                            .get("steps", {})
                            .get("paragraph_similarity", 0)
                        )
                    else:
                        continue

                    # Find the appropriate bin
                    bin_idx = min(
                        max(
                            0,
                            int(
                                (value - min_score)
                                / (max_score - min_score)
                                * bin_count
                            ),
                        ),
                        bin_count - 1,
                    )
                    distribution[i, bin_idx] += 1
                except Exception as e:
                    continue

        # Convert to percentages if requested
        if use_percentage and total_items > 0:
            distribution = (distribution / total_items) * 100

        return distribution, total_items

    def _setup_combined_heatmap_data(
        self,
        results_data,
        metrics_to_use,
        metric_extractors,
        min_score,
        max_score,
        bin_count,
        use_percentage,
    ):
        """
        Prepare combined heatmap data for multiple model results.

        Args:
            results_data (list): List of dictionaries containing evaluation results
            metrics_to_use (list): List of metrics to include in the heatmap
            metric_extractors (dict): Dictionary of functions to extract metric values
            min_score (float): Minimum score for binning
            max_score (float): Maximum score for binning
            bin_count (int): Number of bins to use
            use_percentage (bool): Whether to convert counts to percentages

        Returns:
            tuple: (distribution, total_items) where:
                - distribution: 2D array with metric values distributed into bins
                - total_items: Total number of items evaluated
        """
        # Combine all models into a single distribution
        all_doi_metrics = {}

        # Collect metrics for all DOIs from all models
        for result in results_data:
            for doi, item in result.get("item_results", {}).items():
                if doi not in all_doi_metrics:
                    all_doi_metrics[doi] = {}

                # Extract metrics for this DOI
                for metric_id in metrics_to_use:
                    if metric_id in [
                        "overall_accuracy",
                        "overall_composition_accuracy",
                        "overall_synthesis_accuracy",
                    ]:
                        # These are model-level metrics, use them directly
                        all_doi_metrics[doi][metric_id] = metric_extractors[metric_id](
                            result
                        )
                    elif metric_id in ["precision", "recall", "f1_score"]:
                        all_doi_metrics[doi][metric_id] = item.get(
                            "absolute_classification_metrics", {}
                        ).get(metric_id.split("_")[-1], 0)
                    elif metric_id.startswith("normalized_"):
                        base_metric = metric_id.split("_")[-1]
                        all_doi_metrics[doi][metric_id] = item.get(
                            "normalized_classification_metrics", {}
                        ).get(base_metric, 0)
                    elif metric_id == "property_match":
                        all_doi_metrics[doi][metric_id] = (
                            item.get("details", {})
                            .get("composition_data", {})
                            .get("compositions_property_values", {})
                            .get("similarity_score", 0)
                        )
                    elif metric_id == "method_match":
                        all_doi_metrics[doi][metric_id] = (
                            1
                            if item.get("details", {})
                            .get("synthesis_data", {})
                            .get("method", {})
                            .get("match", False)
                            else 0
                        )
                    elif metric_id == "precursors_match":
                        all_doi_metrics[doi][metric_id] = (
                            item.get("details", {})
                            .get("synthesis_data", {})
                            .get("precursors", {})
                            .get("similarity", 0)
                        )
                    elif metric_id == "steps_match":
                        all_doi_metrics[doi][metric_id] = (
                            item.get("details", {})
                            .get("synthesis_data", {})
                            .get("steps", {})
                            .get("paragraph_similarity", 0)
                        )

        # Initialize distribution matrix
        distribution = np.zeros((len(metrics_to_use), bin_count))

        # Count DOIs in each bin
        total_items = len(all_doi_metrics)
        for doi, metrics in all_doi_metrics.items():
            for i, metric_id in enumerate(metrics_to_use):
                if metric_id in metrics:
                    value = metrics[metric_id]
                    bin_idx = min(
                        max(
                            0,
                            int(
                                (value - min_score)
                                / (max_score - min_score)
                                * bin_count
                            ),
                        ),
                        bin_count - 1,
                    )
                    distribution[i, bin_idx] += 1

        # Convert to percentages if requested
        if use_percentage and total_items > 0:
            distribution = (distribution / total_items) * 100

        return distribution, total_items

    def _add_group_backgrounds(
        self,
        ax,
        metric_groups,
        group_colors,
        metrics_to_use,
        bin_count,
        show_group_labels=False,
        group_label_right_margin=1,
        is_leftmost_axis=True,
    ):
        """
        Add background colors and labels for metric groups in the heatmap.

        Args:
            ax: Matplotlib axes object
            metric_groups (list): List of metric group definitions
            group_colors (list): List of colors for metric groups
            metrics_to_use (list): List of metrics included in the heatmap
            bin_count (int): Number of bins used in the heatmap
            show_group_labels (bool): Whether to show group labels
            group_label_right_margin (int): Right margin for group labels
            is_leftmost_axis (bool): Whether this is a leftmost axis in a grid

        Returns:
            list: List of group indices information
        """
        group_indices = []
        metric_indices = {m: i for i, m in enumerate(metrics_to_use)}

        for i, group in enumerate(metric_groups):
            group_metrics = [m for m in group["metrics"] if m in metrics_to_use]
            if group_metrics:
                start_idx = min(
                    metric_indices[m] for m in group_metrics if m in metric_indices
                )
                end_idx = (
                    max(metric_indices[m] for m in group_metrics if m in metric_indices)
                    + 1
                )
                group_indices.append(
                    {
                        "name": group["name"],
                        "start": start_idx,
                        "end": end_idx,
                        "color": group_colors[i % len(group_colors)],
                    }
                )

        # Draw group backgrounds
        for group in group_indices:
            ax.add_patch(
                plt.Rectangle(
                    (0, group["start"]),
                    bin_count,
                    group["end"] - group["start"],
                    fill=True,
                    color=group["color"],
                    alpha=0.3,
                    zorder=0,
                )
            )

            # Add group labels if this is a leftmost axis
            if show_group_labels and is_leftmost_axis:
                ax.text(
                    -group_label_right_margin,
                    (group["start"] + group["end"]) / 2,
                    group["name"],
                    ha="right",
                    va="center",
                    fontweight="bold",
                    rotation=90,
                )

        return group_indices

    def _add_average_values(
        self, ax, distribution, metrics_to_use, bins, bin_count, right_margin=1
    ):
        """
        Add average value annotations to the heatmap.

        Args:
            ax: Matplotlib axes object
            distribution (array): 2D array with metric values distributed into bins
            metrics_to_use (list): List of metrics included in the heatmap
            bins (array): Array of bin edges
            bin_count (int): Number of bins
            right_margin (int): Right margin for average value annotations
        """
        bin_centers = [(bins[j] + bins[j + 1]) / 2 for j in range(len(bins) - 1)]
        for i, metric_id in enumerate(metrics_to_use):
            # Calculate weighted average score
            weighted_sum = sum(
                bin_centers[j] * distribution[i, j] for j in range(bin_count)
            )
            total = sum(distribution[i, :])
            avg = weighted_sum / total if total > 0 else 0

            # Add text with average
            ax.text(
                bin_count + right_margin,
                i + 0.5,
                f"Avg: {avg:.2f}",
                ha="center",
                va="center",
                fontweight="bold",
            )

    def _get_metric_extractors(self):
        """
        Get standard metric extractor functions.

        Returns:
            dict: Dictionary of functions to extract metric values
        """
        return {
            # Overall scores
            "overall_accuracy": lambda r: r.get("overall_accuracy", 0),
            "overall_composition_accuracy": lambda r: r.get(
                "overall_composition_accuracy", 0
            ),
            "overall_synthesis_accuracy": lambda r: r.get(
                "overall_synthesis_accuracy", 0
            ),
            # Component scores
            "property_match": lambda r: np.mean(
                [
                    item.get("details", {})
                    .get("composition_data", {})
                    .get("compositions_property_values", {})
                    .get("similarity_score", 0)
                    for item in r.get("item_results", {}).values()
                ]
            ),
            "method_match": lambda r: np.mean(
                [
                    (
                        1
                        if item.get("details", {})
                        .get("synthesis_data", {})
                        .get("method", {})
                        .get("match", False)
                        else 0
                    )
                    for item in r.get("item_results", {}).values()
                ]
            ),
            "precursors_match": lambda r: np.mean(
                [
                    item.get("details", {})
                    .get("synthesis_data", {})
                    .get("precursors", {})
                    .get("similarity", 0)
                    for item in r.get("item_results", {}).values()
                ]
            ),
            "steps_match": lambda r: np.mean(
                [
                    item.get("details", {})
                    .get("synthesis_data", {})
                    .get("steps", {})
                    .get("paragraph_similarity", 0)
                    for item in r.get("item_results", {}).values()
                ]
            ),
            # Absolute metrics
            "precision": lambda r: r.get("absolute_classification_metrics", {}).get(
                "precision", 0
            ),
            "recall": lambda r: r.get("absolute_classification_metrics", {}).get(
                "recall", 0
            ),
            "f1_score": lambda r: r.get("absolute_classification_metrics", {}).get(
                "f1_score", 0
            ),
            # Normalized metrics
            "normalized_precision": lambda r: r.get(
                "normalized_classification_metrics", {}
            ).get("precision", 0),
            "normalized_recall": lambda r: r.get(
                "normalized_classification_metrics", {}
            ).get("recall", 0),
            "normalized_f1_score": lambda r: r.get(
                "normalized_classification_metrics", {}
            ).get("f1_score", 0),
        }

    def _get_detailed_metric_display_names(self):
        """
        Get display names for metrics suitable for bar charts.

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
            "normalized_precision": "Normalized\nPrecision",
            "normalized_recall": "Normalized\nRecall",
            "normalized_f1_score": "Normalized\nF1 Score",
            "property_match": "Property\nMatch",
            "method_match": "Method\nMatch",
            "precursors_match": "Precursors\nMatch",
            "steps_match": "Steps\nMatch",
        }

    def _get_default_metric_groups(self):
        """
        Get default metric groups for heatmap visualisation.

        Returns:
            list: List of default metric group definitions
        """
        return [
            {
                "name": "Overall\nScores",
                "metrics": [
                    "overall_accuracy",
                    "overall_composition_accuracy",
                    "overall_synthesis_accuracy",
                ],
            },
            {
                "name": "Component\nScores",
                "metrics": [
                    "property_match",
                    "method_match",
                    "precursors_match",
                    "steps_match",
                ],
            },
            {
                "name": "Absolute\nMetrics",
                "metrics": ["precision", "recall", "f1_score"],
            },
            {
                "name": "Normalized\nMetrics",
                "metrics": [
                    "normalized_precision",
                    "normalized_recall",
                    "normalized_f1_score",
                ],
            },
        ]

    def _extract_per_doi_metric_values(self, result, metric_name):
        """
        Extract values for a specific metric at the DOI level from a result dictionary.

        Args:
            result (dict): Result dictionary containing evaluation data
            metric_name (str): Name of the metric to extract

        Returns:
            list: List of values for the specified metric across all DOIs
        """
        values = []

        # Extract values based on metric type
        if metric_name in [
            "overall_accuracy",
            "overall_composition_accuracy",
            "overall_synthesis_accuracy",
        ]:
            # For overall metrics, extract per-DOI values
            item_results = result.get("item_results", {})
            for _, item_data in item_results.items():
                if metric_name == "overall_accuracy" and "overall_score" in item_data:
                    values.append(item_data["overall_score"])
                elif "field_scores" in item_data:
                    field_name = None
                    if metric_name == "overall_composition_accuracy":
                        field_name = "composition_data"
                    elif metric_name == "overall_synthesis_accuracy":
                        field_name = "synthesis_data"

                    if field_name and field_name in item_data["field_scores"]:
                        values.append(item_data["field_scores"][field_name])

        elif metric_name in ["precision", "recall", "f1_score"]:
            # Extract absolute metrics from each DOI
            for _, item_data in result.get("item_results", {}).items():
                if "absolute_classification_metrics" in item_data:
                    metric_value = item_data["absolute_classification_metrics"].get(
                        metric_name, 0
                    )
                    values.append(metric_value)

        elif metric_name.startswith("normalized_"):
            # Extract normalized metrics from each DOI
            base_metric = metric_name.replace("normalized_", "")
            for _, item_data in result.get("item_results", {}).items():
                if "normalized_classification_metrics" in item_data:
                    metric_value = item_data["normalized_classification_metrics"].get(
                        base_metric, 0
                    )
                    values.append(metric_value)

        elif metric_name == "property_match":
            # Extract property match scores from each DOI
            for _, item_data in result.get("item_results", {}).items():
                similarity_score = (
                    item_data.get("details", {})
                    .get("composition_data", {})
                    .get("compositions_property_values", {})
                    .get("similarity_score", 0)
                )
                values.append(similarity_score)

        elif metric_name == "method_match":
            # Extract method match scores (0 or 1) from each DOI
            for _, item_data in result.get("item_results", {}).items():
                is_match = (
                    1
                    if item_data.get("details", {})
                    .get("synthesis_data", {})
                    .get("method", {})
                    .get("match", False)
                    else 0
                )
                values.append(is_match)

        elif metric_name == "precursors_match":
            # Extract precursors similarity scores from each DOI
            for _, item_data in result.get("item_results", {}).items():
                similarity = (
                    item_data.get("details", {})
                    .get("synthesis_data", {})
                    .get("precursors", {})
                    .get("similarity", 0)
                )
                values.append(similarity)

        elif metric_name == "steps_match":
            # Extract steps similarity scores from each DOI
            for _, item_data in result.get("item_results", {}).items():
                similarity = (
                    item_data.get("details", {})
                    .get("synthesis_data", {})
                    .get("steps", {})
                    .get("paragraph_similarity", 0)
                )
                values.append(similarity)

        return values

    def plot_single_bar_chart(
        self,
        result_file: Optional[str] = None,
        result_dict: Optional[dict] = None,
        output_file: Optional[str] = None,
        model_name: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        colormap: Optional[str] = "Blues",
        display_values: bool = True,
        title: Optional[str] = None,
        typical_threshold: Optional[float] = None,
        threashold_line_style: Optional[str] = "--",
        threashold_tolerance_range: Optional[float] = 0.03,
        threshold_color: Optional[str] = "red",
        show_grid: bool = True,
        bar_width: float = 0.6,
        y_axis_label: str = "Score",
        x_axis_label: Optional[str] = None,
        y_axis_range: Tuple[float, float] = (0, 1),
        dpi: int = 300,
        metrics_to_include: Optional[List[str]] = [
            "overall_accuracy",
            "overall_composition_accuracy",
            "overall_synthesis_accuracy",
            "absolute_precision",
            "absolute_recall",
            "absolute_f1_score",
            "normalized_precision",
            "normalized_recall",
            "normalized_f1_score",
        ],
    ):
        """
        Plot evaluation metrics from results file or dictionary.

        Args:
            result_file (str, optional): Path to the JSON file containing evaluation results
            result_dict (dict, optional): Dictionary containing evaluation results
            output_file (str, optional): Path to save the output plot image
            model_name (str, optional): Name of the model used for evaluation
            figsize (tuple, optional): Figure size (width, height) in inches (default: (12, 8))
            colormap (str, optional): Matplotlib colormap name (e.g., 'Blues', 'Greens', 'Oranges', etc.) (default: 'Blues')
            display_values (bool, optional): Whether to display metric values on bars
            title (str, optional): Custom title for the plot (default: True)
            typical_threshold (float, optional): Typical threshold value to display as a horizontal line on the plot (default: None)
            threashold_line_style (str, optional): Style of the threshold line (default: '--')
            threashold_tolerance_range (float, optional): Tolerance range for the threshold line (default: 0.03)
            threshold_color (str, optional): Color for the threshold line (default: 'red')
            show_grid (bool, optional): Whether to display horizontal grid lines in the plot (default: True)
            bar_width (float, optional): Width of the bars in the plot (default: 0.6)
            y_axis_label (str, optional): Label for the y-axis (default: "Score")
            x_axis_label (str, optional): Label for the x-axis (default: None)
            y_axis_range (tuple, optional): Range for the y-axis (default: (0, 1))
            dpi (int, optional): dpi for the output image (default: 300)
            metrics_to_include (list, optional): List of metrics to include in the plot (default: ['overall_accuracy', 'overall_composition_accuracy', 'overall_synthesis_accuracy', 'absolute_precision', 'absolute_recall', 'absolute_f1_score', 'normalized_precision', 'normalized_recall', 'normalized_f1_score'])

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        if result_dict is not None:
            results = result_dict
        elif result_file is not None:
            with open(result_file, "r") as f:
                results = json.load(f)
        else:
            raise ValueErrorHandler(
                "Either result_file or result_dict must be provided"
            )

        # Extract metrics and values
        metrics, values = self._extract_metrics_from_result(results, metrics_to_include)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        bar_positions = np.arange(len(metrics))

        # Get colors for bars
        bar_colors = self._get_chart_colors(colormap, len(metrics))

        # Create bars with optional value labels
        bars = self._plot_bars_with_values(
            ax, bar_positions, values, bar_width, bar_colors, display_values
        )

        # Generate appropriate title
        plot_title = self._get_model_title(title, model_name, results)

        # Set up axes with labels, title, and grid
        self._setup_bar_chart_axes(
            ax,
            metrics,
            plot_title,
            y_axis_label,
            x_axis_label,
            y_axis_range,
            show_grid,
        )

        # Add typical threshold line if provided
        if typical_threshold:
            self._draw_threshold_line_with_breaks(
                ax,
                typical_threshold,
                bar_positions,
                values,
                bar_width,
                threshold_color,
                threashold_line_style,
                threashold_tolerance_range,
            )

        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Single model's evaluation plot saved to {output_file}")

        return fig

    def plot_multiple_bar_charts(
        self,
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
        y_label: str = "Score",
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
        results_data, names = self._load_results_data(
            result_sources, folder_path, model_names
        )

        # Get available metrics and display names
        metrics = self._get_available_metrics(results_data, metrics_to_include)
        metric_display_names = self._get_metric_display_names()

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Determine bar positions and width
        num_models = len(results_data)
        num_metrics = len(metrics)

        group_positions, calc_bar_width = self._calculate_bar_positions(
            num_metrics, num_models, group_width
        )
        if bar_width is None:
            bar_width = calc_bar_width

        # Create colormap
        cmap = cm.get_cmap(colormap, num_models)

        # Plot bars for each model
        all_bar_positions = []
        all_values = []

        for i, (result, name) in enumerate(zip(results_data, names)):
            # Extract metric values for this model
            values = self._extract_group_metrics(result, metrics)

            # Calculate position for each bar
            bar_positions = group_positions - group_width / 2 + (i + 0.5) * bar_width

            # Plot bars for this model
            _, model_positions, model_values = self._plot_model_bars(
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
        ax.set_ylabel(y_label, fontsize=12)
        if x_label:
            ax.set_xlabel(x_label, fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        else:
            ax.set_title(
                "Comparison of Evaluation Metrics Across Models",
                fontsize=14,
                fontweight="bold",
            )

        ax.set_xticks(group_positions)
        ax.set_xticklabels(
            [metric_display_names.get(m, m) for m in metrics], fontsize=11
        )
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
            self._draw_threshold_line_with_breaks(
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
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)

        # Finalize and save
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Comparison plot saved to {output_file}")
        return fig

    def plot_single_radar_chart(
        self,
        result_file: Optional[str] = None,
        result_dict: Optional[dict] = None,
        output_file: Optional[str] = None,
        model_name: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        colormap: str = "Blues",
        display_values: bool = False,
        title: Optional[str] = None,
        title_fontsize: int = 14,
        title_pad: Optional[float] = 50.0,
        typical_threshold: Optional[float] = None,
        threshold_color: str = "red",
        threshold_line_style: str = "--",
        label_fontsize: int = 12,
        value_fontsize: int = 10,
        legend_loc: str = "best",
        legend_fontsize: int = 10,
        bbox_to_anchor: Optional[Tuple[float, float]] = None,
        show_grid: bool = True,
        show_grid_labels: bool = False,
        grid_line_width: float = 1.0,
        grid_line_style: str = "-",
        grid_line_color: str = "gray",
        grid_line_alpha: float = 0.2,
        fill_alpha: float = 0.4,
        marker_size: int = 7,
        line_width: float = 2,
        label_padding: float = 0.25,
        clockwise: bool = True,
        start_angle: float = np.pi / 2,
        radar_range: Tuple[float, float] = (0, 1),
        dpi: int = 300,
        metrics_to_include: Optional[List[str]] = None,
    ):
        """
        Plot radar chart for a single evaluation result.

        Args:
            result_file (str, optional): Path to JSON file containing evaluation results
            result_dict (dict, optional): Dictionary containing evaluation results
            output_file (str, optional): Path to save the output plot image
            model_name (str, optional): Name of the model for display
            figsize (tuple, optional): Figure size (width, height) in inches
            colormap (str, optional): Matplotlib colormap name
            display_values (bool, optional): Whether to display metric values on chart
            title (str, optional): Custom title for the plot
            title_fontsize (int): Font size for the title
            title_pad (float): Padding for the title from the top of the plot
            typical_threshold (float, optional): Threshold value to display as a circular line
            threshold_color (str): Color for the threshold line
            threshold_line_style (str): Style of the threshold line
            label_fontsize (int): Font size for axis labels
            value_fontsize (int): Font size for displayed values
            legend_loc (str): Location for the legend box (default: 'best')
            legend_fontsize (int): Font size for the legend
            bbox_to_anchor (tuple, optional): Bounding box for the legend box (default: None)
            show_grid (bool): Whether to display the grid lines
            show_grid_labels (bool): Whether to display grid line values/labels
            grid_line_width (float): Width of the grid lines
            grid_line_style (str): Style of the grid lines
            grid_line_color (str): Color of the grid lines
            grid_line_alpha (float): Alpha (transparency) of the grid lines
            fill_alpha (float): Alpha (transparency) of the filled area
            marker_size (int): Size of the data point markers
            line_width (float): Width of the plot lines
            label_padding (float): Distance padding for axis labels from plot
            clockwise (bool): Direction of the radar chart
            start_angle (float): Start angle in radians
            radar_range (tuple): Range for the radar axes
            dpi (int): DPI for the output image
            metrics_to_include (list, optional): List of metrics to include

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        if result_dict is not None:
            results = result_dict
        elif result_file is not None:
            with open(result_file, "r") as f:
                results = json.load(f)
        else:
            raise ValueErrorHandler(
                "Either result_file or result_dict must be provided"
            )

        # Default metrics if not specified
        if metrics_to_include is None:
            metrics_to_include = [
                "overall_accuracy",
                "overall_composition_accuracy",
                "overall_synthesis_accuracy",
                "precision",
                "recall",
                "f1_score",
                "normalized_precision",
                "normalized_recall",
                "normalized_f1_score",
            ]

        # Get available metrics and display names
        metrics = self._get_available_metrics(results, metrics_to_include)
        metric_display_names = self._get_metric_display_names()

        # Setup plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)

        num_metrics = len(metrics)
        theta = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)

        # Adjust for starting angle and direction
        if clockwise:
            theta = 2 * np.pi - theta
        theta = (theta + start_angle) % (2 * np.pi)

        # Set up the radar plot
        self._setup_radar_plot(
            ax,
            metrics,
            metric_display_names,
            theta,
            radar_range,
            label_fontsize,
            label_padding,
            show_grid,
            grid_line_width,
            grid_line_style,
            grid_line_color,
            grid_line_alpha,
            show_grid_labels,
        )

        # Create colormap
        color = self._get_chart_colors(colormap, 1, index=0)

        # Add threshold if provided
        threshold_patch = self._add_threshold_circle(
            ax, typical_threshold, threshold_color, threshold_line_style
        )

        # Extract values
        values = self._extract_group_metrics(results, metrics)
        values = np.array(values)

        # Close the loop for radar chart
        theta_closed = np.append(theta, theta[0])
        values_closed = np.append(values, values[0])

        # Plot the radar chart
        display_name = model_name or results.get("agent_model_name", "Model")
        ax.plot(
            theta_closed,
            values_closed,
            "o-",
            linewidth=line_width,
            color=color,
            label=display_name,
            markersize=marker_size,
        )
        ax.fill(theta_closed, values_closed, color=color, alpha=fill_alpha)

        # Add value labels if requested
        if display_values:
            for t, v in zip(theta, values):
                if v > 0:
                    text_distance = v + 0.05
                    ax.text(
                        t,
                        text_distance,
                        f"{v:.2f}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=value_fontsize,
                        fontweight="bold",
                        color=color,
                    )

        # Add title using shared function
        plot_title = self._get_model_title(title, model_name, results)
        ax.set_title(
            plot_title, fontsize=title_fontsize, fontweight="bold", pad=title_pad
        )

        # Add legend
        if threshold_patch:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(threshold_patch)
            ax.legend(
                handles=handles,
                fontsize=legend_fontsize,
                loc=legend_loc,
                bbox_to_anchor=bbox_to_anchor,
            )
        else:
            ax.legend(
                fontsize=legend_fontsize, loc=legend_loc, bbox_to_anchor=bbox_to_anchor
            )

        # Save and return
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Radar chart saved to {output_file}")
        return fig

    def plot_multiple_radar_charts(
        self,
        result_sources: Union[List[str], List[Dict], str] = None,
        folder_path: Optional[str] = None,
        output_file: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 10),
        colormap: str = "viridis",
        display_values: bool = False,
        title: Optional[str] = None,
        title_fontsize: int = 14,
        title_pad: Optional[float] = 50.0,
        typical_threshold: Optional[float] = None,
        threshold_color: str = "red",
        threshold_line_style: str = "--",
        label_fontsize: int = 12,
        value_fontsize: int = 10,
        legend_loc: str = "best",
        bbox_to_anchor: Optional[Tuple[float, float]] = None,
        legend_fontsize: int = 10,
        show_grid: bool = True,
        show_grid_labels: bool = False,
        grid_line_width: float = 1.0,
        grid_line_style: str = "-",
        grid_line_color: str = "gray",
        grid_line_alpha: float = 0.2,
        fill_alpha: float = 0.25,
        marker_size: int = 7,
        line_width: float = 2,
        label_padding: float = 0.25,
        clockwise: bool = True,
        start_angle: float = np.pi / 2,
        radar_range: Tuple[float, float] = (0, 1),
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
        Plot evaluation metrics from multiple result files or dictionaries as a radar chart.

        Args:
            result_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
            folder_path (Optional[str], optional): Path to folder containing JSON result files
            output_file (str, optional): Path to save the output plot image
            model_names (Optional[List[str]]): Names of models to display in the legend
            figsize (Tuple[int, int]): Figure size (width, height) in inches
            colormap (str): Matplotlib colormap name for the plot lines and markers
            display_values (bool): Whether to display metric values on the chart
            title (Optional[str]): Custom title for the plot
            title_fontsize (int): Font size for the title
            title_pad (Optional[float]): Padding for the title from the top of the plot
            typical_threshold (Optional[float]): Typical threshold value to display as a circular line
            threshold_color (str): Color for the threshold line
            threshold_line_style (str): Style of the threshold line
            label_fontsize (int): Font size for axis labels
            value_fontsize (int): Font size for displayed values
            legend_loc (str): Location of the legend
            bbox_to_anchor (Optional[Tuple[float, float]]): Bounding box for the legend
            legend_fontsize (int): Font size for the legend
            show_grid (bool): Whether to display the grid lines
            show_grid_labels (bool): Whether to display grid line values/labels
            grid_line_width (float): Width of the grid lines
            grid_line_style (str): Style of the grid lines
            grid_line_color (str): Color of the grid lines
            grid_line_alpha (float): Alpha (transparency) of the grid lines
            fill_alpha (float): Alpha (transparency) of the filled area
            marker_size (int): Size of the data point markers
            line_width (float): Width of the plot lines
            label_padding (float): Distance padding for axis labels from plot
            clockwise (bool): Direction of the radar chart
            start_angle (float): Start angle in radians
            radar_range (Tuple[float, float]): Range for the radar axes
            dpi (int): dpi for the output image
            metrics_to_include (Optional[List[str]]): List of metrics to include in the plot

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load results data using the helper function
        results_data, names = self._load_results_data(
            result_sources, folder_path, model_names
        )

        # Get available metrics and display names
        metrics = self._get_available_metrics(results_data, metrics_to_include)
        metric_display_names = self._get_metric_display_names()

        num_metrics = len(metrics)
        num_models = len(results_data)

        # Create figure, polar axes and angles
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)
        theta = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)

        # Adjust for starting angle and direction
        if clockwise:
            theta = 2 * np.pi - theta
        theta = (theta + start_angle) % (2 * np.pi)

        # Set up the radar plot
        self._setup_radar_plot(
            ax,
            metrics,
            metric_display_names,
            theta,
            radar_range,
            label_fontsize,
            label_padding,
            show_grid,
            grid_line_width,
            grid_line_style,
            grid_line_color,
            grid_line_alpha,
            show_grid_labels,
        )

        # Get colors using shared function
        colors = self._get_chart_colors(colormap, num_models)

        # Add threshold if provided
        threshold_patch = self._add_threshold_circle(
            ax, typical_threshold, threshold_color, threshold_line_style
        )

        # Plot each model's data
        for i, (result, name) in enumerate(zip(results_data, names)):
            # Extract values
            values = self._extract_group_metrics(result, metrics)
            values = np.array(values)

            # Close the loop for radar chart
            theta_closed = np.append(theta, theta[0])
            values_closed = np.append(values, values[0])

            # Plot the radar chart
            ax.plot(
                theta_closed,
                values_closed,
                "o-",
                linewidth=line_width,
                color=colors[i],
                label=name,
                markersize=marker_size,
            )
            ax.fill(theta_closed, values_closed, color=colors[i], alpha=fill_alpha)

            # Display values if requested
            if display_values:
                for t, v in zip(theta, values):
                    if v > 0:
                        text_distance = v + 0.05
                        ax.text(
                            t,  # Angle
                            text_distance,  # Radius
                            f"{v:.2f}",  # Text
                            horizontalalignment="center",
                            verticalalignment="center",
                            fontsize=value_fontsize,
                            fontweight="bold",
                            color=colors[i],
                        )

        # Add title
        if title:
            ax.set_title(
                title, fontsize=title_fontsize, fontweight="bold", pad=title_pad
            )

        # Create legend (include threshold patch if specified)
        if threshold_patch:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(threshold_patch)
            ax.legend(
                handles=handles,
                loc=legend_loc,
                bbox_to_anchor=bbox_to_anchor,
                fontsize=legend_fontsize,
            )
        else:
            ax.legend(
                loc=legend_loc,
                bbox_to_anchor=bbox_to_anchor,
                fontsize=legend_fontsize,
            )

        # Adjust layout and save
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Radar chart comparison saved to {output_file}")

        return fig

    def plot_single_performance_heatmap(
        self,
        result_file: Optional[str] = None,
        result_dict: Optional[dict] = None,
        output_file: Optional[str] = None,
        model_name: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 12),
        colormap: str = "YlGnBu",
        bin_count: int = 10,
        score_range: Tuple[float, float] = (0, 1),
        use_percentage: bool = True,
        show_averages: bool = False,
        show_group_labels: bool = False,
        show_annotations: bool = False,
        annotation_format: Optional[str] = None,
        title: Optional[str] = None,
        title_fontsize: int = 14,
        title_pad: Optional[float] = None,
        labels: List[str] = ["Metrics", "Scores"],
        label_fontsize: int = 12,
        dpi: int = 300,
        group_metrics: bool = False,
        metric_groups: Optional[List[Dict]] = None,
        group_colors: Optional[List[str]] = None,
        include_metrics: Optional[List[str]] = None,
        exclude_metrics: Optional[List[str]] = None,
        group_label_right_margin: int = 1,
        average_value_left_margin: int = 1,
        plot_padding: float = 0.1,
    ):
        """
        Create a heatmap showing the distribution of scores across metrics for a single model.

        Args:
            result_file (str, optional): Path to JSON file containing evaluation results for the model
            result_dict (dict, optional): Dictionary containing evaluation results for the model. Either result_file or result_dict must be provided.
            output_file (str, optional): Path to save the output visualisation
            model_name (str, optional): Name to display for the model in the plot
            figsize (tuple, optional): Figure size as (width, height) in inches (default: (12, 12))
            colormap (str, optional): Matplotlib colormap name for the heatmap (default: 'YlGnBu')
            bin_count (int, optional): Number of bins to divide the score range into (default: 10)
            score_range (tuple, optional): Min and max values for score bins (default: (0, 1))
            use_percentage (bool, optional): Whether to show percentages (True) or counts (False)
            show_averages (bool, optional): Whether to show average scores per metric
            show_group_labels (bool, optional): Whether to show metric group labels
            show_annotations (bool, optional): Whether to show value annotations in cells
            annotation_format (str, optional): Format string for annotations (e.g., '.1f' or 'd')
            title (str, optional): Custom title for the plot
            title_fontsize (int, optional): Font size for the title text (default: 14)
            title_pad (float, optional): Padding for the title from the top of the plot
            labels (list, optional): Labels for the x and y axes (default: ['Metrics', 'Scores'])
            label_fontsize (int, optional): Font size for the axis labels (default: 12)
            dpi (int, optional): Resolution for saved image (default: 300)
            group_metrics (bool, optional): Whether to visually group related metrics together (default: False)
            metric_groups (list, optional): Custom metric groups definition for grouping metrics
            group_colors (list, optional): Colors for metric groups (default: ['#f8f9fa', '#e9ecef', '#f8f9fa', '#e9ecef'])
            include_metrics (list, optional): Specific metrics to include in the heatmap (default: all available)
            exclude_metrics (list, optional): Specific metrics to exclude from the heatmap (default: none)
            group_label_right_margin (int, optional): Right margin for group labels (default: 1)
            average_value_left_margin (int, optional): Left margin for average values (default: 1)
            plot_padding (float, optional): Padding between heatmap and axes (default: 0.1)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load the result data
        if result_dict is not None:
            result = result_dict
        elif result_file is not None:
            with open(result_file, "r") as f:
                result = json.load(f)
        else:
            raise ValueErrorHandler(
                "Either result_file or result_dict must be provided"
            )

        # Set default metric groups if not provided
        if metric_groups is None:
            metric_groups = self._get_default_metric_groups()

        # Set default group colors if not provided
        if group_colors is None:
            group_colors = ["#f8f9fa", "#e9ecef", "#f8f9fa", "#e9ecef"]

        # Get metric extractors
        metric_extractors = self._get_metric_extractors()

        # Get available metrics
        available_metrics = list(metric_extractors.keys())

        # Filter metrics based on include/exclude lists
        if include_metrics:
            metrics_to_use = [m for m in include_metrics if m in available_metrics]
        else:
            metrics_to_use = available_metrics.copy()

        if exclude_metrics:
            metrics_to_use = [m for m in metrics_to_use if m not in exclude_metrics]

        # Get human-readable metric names
        metric_display_names = self._get_detailed_metric_display_names()

        # Create bins for scores
        min_score, max_score = score_range
        bins = np.linspace(min_score, max_score, bin_count + 1)
        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Set up the data
        distribution, total_items = self._setup_heatmap_data(
            result,
            metrics_to_use,
            metric_extractors,
            min_score,
            max_score,
            bin_count,
            use_percentage,
        )

        # Create DataFrame for plotting
        metric_names = [metric_display_names.get(m, m) for m in metrics_to_use]
        df_dist = pd.DataFrame(distribution, index=metric_names, columns=bin_labels)

        # Draw group backgrounds if requested
        if group_metrics:
            self._add_group_backgrounds(
                ax,
                metric_groups,
                group_colors,
                metrics_to_use,
                bin_count,
                show_group_labels,
                group_label_right_margin,
            )

        # Format for annotations
        if annotation_format is None:
            fmt = ".1f" if use_percentage else ".0f"
        else:
            fmt = annotation_format

        # Create heatmap
        sns.heatmap(
            df_dist,
            annot=show_annotations,
            fmt=fmt,
            cmap=colormap,
            linewidths=0.5,
            ax=ax,
            vmin=0,
        )

        # Add side summary - average score per metric
        if show_averages:
            self._add_average_values(
                ax,
                distribution,
                metrics_to_use,
                bins,
                bin_count,
                average_value_left_margin,
            )

        # Add labels
        ax.set_xlabel(labels[1], fontsize=label_fontsize)
        ax.set_ylabel(labels[0], fontsize=label_fontsize)

        # Set title
        title_kwargs = (
            {"fontsize": title_fontsize, "pad": title_pad}
            if title_pad
            else {"fontsize": title_fontsize}
        )
        if title:
            ax.set_title(title, fontweight="bold", **title_kwargs)
        else:
            value_type = "Percentage (%)" if use_percentage else "Count"
            model_display_name = model_name or result.get("agent_model_name", "Model")
            ax.set_title(
                f"{model_display_name}: Distribution of {value_type} of Articles by Performance Score (Total Articles: {total_items})",
                fontweight="bold",
                **title_kwargs,
            )

        # Adjust layout
        plt.tight_layout(pad=plot_padding)

        # Save figure
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Single model's distribution heatmap saved to {output_file}")

        return fig

    def plot_multiple_performance_heatmaps(
        self,
        result_sources: Union[List[str], List[Dict], str] = None,
        folder_path: Optional[str] = None,
        output_file: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 12),
        colormap: str = "YlGnBu",
        bin_count: int = 10,
        score_range: Tuple[float, float] = (0, 1),
        use_percentage: bool = True,
        show_averages: bool = False,
        show_group_labels: bool = False,
        show_annotations: bool = False,
        annotation_format: Optional[str] = None,
        title: Optional[str] = None,
        title_fontsize: int = 14,
        labels: List[str] = ["Metrics", "Scores"],
        label_fontsize: int = 12,
        dpi: int = 300,
        group_metrics: bool = True,
        metric_groups: Optional[List[Dict]] = None,
        group_colors: Optional[List[str]] = None,
        include_metrics: Optional[List[str]] = None,
        exclude_metrics: Optional[List[str]] = None,
        sort_models_by: str = "overall_accuracy",
        combine_models: bool = False,
        group_label_right_margin: int = 1,
        average_value_left_margin: int = 1,
        plot_padding: float = 0.1,
    ):
        """
        Create a heatmap showing the distribution of scores across metrics for multiple models.

        Args:
            result_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
            folder_path (Optional[str], optional): Path to folder containing JSON result files. Either result_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output visualisation
            model_names (Optional[List[str]]): Names to display for models in the plots
            figsize (Tuple[int, int]): Figure size as (width, height) in inches
            colormap (str): Matplotlib colormap name for the heatmap
            bin_count (int): Number of bins to divide the score range into
            score_range (Tuple[float, float]): Min and max values for score bins (default: (0, 1))
            use_percentage (bool): Whether to show percentages (True) or counts (False)
            show_averages (bool): Whether to show average scores per metric group and model
            show_group_labels (bool): Whether to show metric group labels
            show_annotations (bool): Whether to show value annotations in cells
            annotation_format (Optional[str]): Format string for annotations (e.g., '.1f' or 'd')
            title (Optional[str]): Custom title for the plot
            title_fontsize (int): Font size for the title
            labels (List[str]): Labels for the x and y axes (default: ['Metrics', 'Scores'])
            label_fontsize (int): Font size for the axis labels
            dpi (int): Resolution for saved image
            group_metrics (bool): Whether to visually group related metrics
            metric_groups (Optional[List[Dict]]): Custom metric groups definition
            group_colors (Optional[List[str]]): Colors for metric groups
            include_metrics (Optional[List[str]]): Specific metrics to include (if None, includes all available)
            exclude_metrics (Optional[List[str]]): Specific metrics to exclude (if None, excludes none)
            sort_models_by (str): Metric to sort models by when displaying multiple models (default: 'overall_accuracy')
            combine_models (bool): Whether to combine all models into a single distribution plot (default: False)
            group_label_right_margin (int): Right margin for group labels
            average_value_left_margin (int): Left margin for average values
            plot_padding (float): Padding between heatmap and axes labels and title

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load results data
        results_data, names = self._load_results_data(
            result_sources, folder_path, model_names
        )

        # Set default metric groups if not provided
        if metric_groups is None:
            metric_groups = self._get_default_metric_groups()

        # Set default group colors if not provided
        if group_colors is None:
            group_colors = ["#f8f9fa", "#e9ecef", "#f8f9fa", "#e9ecef"]

        # Get metric extractors
        metric_extractors = self._get_metric_extractors()

        # Get available metrics and filter based on include/exclude lists
        available_metrics = list(metric_extractors.keys())

        if include_metrics:
            metrics_to_use = [m for m in include_metrics if m in available_metrics]
        else:
            metrics_to_use = available_metrics.copy()

        if exclude_metrics:
            metrics_to_use = [m for m in metrics_to_use if m not in exclude_metrics]

        # Get human-readable metric names
        metric_display_names = self._get_detailed_metric_display_names()

        # Create bins for scores
        min_score, max_score = score_range
        bins = np.linspace(min_score, max_score, bin_count + 1)
        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

        # Sort models if requested and not combining
        if not combine_models and sort_models_by and len(results_data) > 1:
            sorted_indices = []
            sort_values = []

            for result in results_data:
                if sort_models_by in result:
                    sort_values.append(result[sort_models_by])
                elif (
                    "absolute_classification_metrics" in result
                    and sort_models_by in result["absolute_classification_metrics"]
                ):
                    sort_values.append(
                        result["absolute_classification_metrics"][sort_models_by]
                    )
                elif (
                    "normalized_classification_metrics" in result
                    and sort_models_by in result["normalized_classification_metrics"]
                ):
                    sort_values.append(
                        result["normalized_classification_metrics"][sort_models_by]
                    )
                else:
                    sort_values.append(0)  # Default

            # Sort results_data and names by sort_values (descending)
            sorted_indices = np.argsort(sort_values)[::-1]
            results_data = [results_data[i] for i in sorted_indices]
            names = [names[i] for i in sorted_indices]

        # Start visualising data
        if combine_models:
            # Combined visualisation for all models
            distribution, total_items = self._setup_combined_heatmap_data(
                results_data,
                metrics_to_use,
                metric_extractors,
                min_score,
                max_score,
                bin_count,
                use_percentage,
            )

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Create DataFrame for plotting
            df_metric_names = [metric_display_names.get(m, m) for m in metrics_to_use]
            df_dist = pd.DataFrame(
                distribution, index=df_metric_names, columns=bin_labels
            )

            # Draw group backgrounds if requested
            if group_metrics:
                self._add_group_backgrounds(
                    ax,
                    metric_groups,
                    group_colors,
                    metrics_to_use,
                    bin_count,
                    show_group_labels,
                    group_label_right_margin,
                )

            # Format for annotations
            if annotation_format is None:
                fmt = ".1f" if use_percentage else ".0f"
            else:
                fmt = annotation_format

            # Create heatmap
            sns.heatmap(
                df_dist,
                annot=show_annotations,
                fmt=fmt,
                cmap=colormap,
                linewidths=0.5,
                ax=ax,
                vmin=0,
            )

            # Add side summary - average score per metric
            if show_averages:
                self._add_average_values(
                    ax,
                    distribution,
                    metrics_to_use,
                    bins,
                    bin_count,
                    average_value_left_margin,
                )

            # Add labels
            ax.set_xlabel(labels[1], fontsize=label_fontsize)
            ax.set_ylabel(labels[0], fontsize=label_fontsize)

            # Set title
            if title:
                ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
            else:
                value_type = "Percentage (%)" if use_percentage else "Count"
                combined_title = f"Combined Distribution of {value_type} of Articles by Performance Score\n"
                combined_title += f"Models: {', '.join(names[:3])}"
                if len(names) > 3:
                    combined_title += f" and {len(names) - 3} others"
                combined_title += f" (Total Articles: {total_items})"
                ax.set_title(combined_title, fontsize=title_fontsize, fontweight="bold")

        else:
            # Multiple visualisations for individual models
            num_models = len(results_data)
            ncols = min(2, num_models)
            nrows = (num_models + ncols - 1) // ncols  # Ceiling division

            # Create subplots
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, sharex=True, sharey=True
            )

            # Handle the case where there's only one model or one row
            if num_models == 1:
                axes = np.array([[axes]])
            elif num_models <= 2:
                axes = np.array([axes]).reshape(-1, ncols)

            axes_flat = axes.flatten()

            # Process each model
            for model_idx, (result, name, ax) in enumerate(
                zip(results_data, names, axes_flat)
            ):
                # Set up the data
                distribution, total_items = self._setup_heatmap_data(
                    result,
                    metrics_to_use,
                    metric_extractors,
                    min_score,
                    max_score,
                    bin_count,
                    use_percentage,
                )

                # Create DataFrame for plotting
                df_metric_names = [
                    metric_display_names.get(m, m) for m in metrics_to_use
                ]
                df_dist = pd.DataFrame(
                    distribution, index=df_metric_names, columns=bin_labels
                )

                # Draw group backgrounds if requested
                is_leftmost = model_idx % ncols == 0
                if group_metrics:
                    self._add_group_backgrounds(
                        ax,
                        metric_groups,
                        group_colors,
                        metrics_to_use,
                        bin_count,
                        show_group_labels
                        and is_leftmost,  # Only show labels on leftmost plots
                        group_label_right_margin,
                        is_leftmost,
                    )

                # Format for annotations
                if annotation_format is None:
                    fmt = ".1f" if use_percentage else ".0f"
                else:
                    fmt = annotation_format

                # Create heatmap
                sns.heatmap(
                    df_dist,
                    annot=show_annotations,
                    fmt=fmt,
                    cmap=colormap,
                    linewidths=0.5,
                    ax=ax,
                    vmin=0,
                )

                # Add side summary - average score per metric
                if show_averages:
                    self._add_average_values(
                        ax,
                        distribution,
                        metrics_to_use,
                        bins,
                        bin_count,
                        average_value_left_margin,
                    )

                # Add title for this subplot
                ax.set_title(
                    f"Model: {name} (Articles: {total_items})", fontweight="bold"
                )

                # Only add axis labels for bottom/left plots
                if model_idx >= num_models - ncols:  # Bottom row
                    ax.set_xlabel(labels[1], fontsize=label_fontsize)
                else:
                    ax.set_xlabel("")

                if model_idx % ncols == 0:  # First column
                    ax.set_ylabel(labels[0], fontsize=label_fontsize)
                else:
                    ax.set_ylabel("")

            # Hide empty subplots if any
            for i in range(num_models, len(axes_flat)):
                axes_flat[i].axis("off")

            # Add overall title if provided
            if title:
                fig.suptitle(title, fontsize=title_fontsize, y=0.98)
            else:
                value_type = "Percentage (%)" if use_percentage else "Count"
                fig.suptitle(
                    f"Distribution of {value_type} of Articles by Performance Score Range",
                    fontsize=title_fontsize,
                    fontweight="bold",
                    y=0.98,
                )

        # Adjust layout
        plt.tight_layout(pad=plot_padding)

        # Save figure
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Multiple models' distribution heatmap saved to {output_file}")

        return fig

    def plot_single_histogram_chart(
        self,
        result_file: Optional[str] = None,
        result_dict: Optional[dict] = None,
        metric_name: str = "overall_accuracy",
        output_file: Optional[dict] = None,
        model_name: Optional[dict] = None,
        figsize: Tuple[int, int] = (8, 6),
        bins: int = 10,
        color: str = "skyblue",
        color_gradient: bool = False,
        gradient_colors: Optional[List[str]] = None,
        show_kde: bool = False,
        show_mean: bool = False,
        mean_color: str = "green",
        mean_line_style: str = "-",
        show_median: bool = False,
        median_color: str = "black",
        median_line_style: str = "-",
        show_threshold: bool = False,
        threshold_value: float = 0.8,
        threshold_color: str = "red",
        threshold_line_style: str = "--",
        title: Optional[str] = None,
        title_fontsize=14,
        xlabel: Optional[str] = None,
        ylabel: str = "Count",
        xlabel_fontsize: int = 12,
        ylabel_fontsize: int = 12,
        legend_loc: Optional[str] = "best",
        bbox_to_anchor: Optional[str] = None,
        dpi: int = 300,
    ):
        """
        Create a histogram for a single metric from evaluation results.

        Args:
            result_file (str, optional): Path to JSON file containing evaluation results
            result_dict (dict, optional): Dictionary containing evaluation results. Either result_file or result_dict must be provided.
            metric_name (str): Name of the metric to plot (default: "overall_accuracy")
            output_file (str, optional): Path to save the output plot image
            model_name (str, optional): Name of the model for display in the plot title
            figsize (tuple): Figure size as (width, height) in inches (default: (8, 6))
            bins (int or sequence): Number of bins or bin edges for histogram (default: 10)
            color (str): Color for the histogram bars (default: 'skyblue')
            color_gradient (bool): Whether to use color gradient for histogram bars (default: False)
            gradient_colors (list, optional): List of colors for gradient (default: ['#D4E6F1', 'color'])
            show_kde (bool): Whether to show a KDE curve over the histogram (default: False)
            show_mean (bool): Whether to show a vertical line at the mean value (default: False)
            show_median (bool): Whether to show a vertical line at the median value (default: False)
            show_threshold (bool): Whether to show a threshold line (default: False)
            threshold_value (float): Value for the threshold line (default: 0.8)
            threshold_color (str): Color for the threshold line (default: 'red')
            threshold_line_style (str): Line style for the threshold line (default: '--')
            title (str, optional): Custom title for the plot (default: None)
            title_fontsize (int): Font size for the title (default: 14)
            xlabel (str, optional): Custom label for x-axis
            ylabel (str, optional): Label for y-axis
            xlabel_fontsize (int, optional): Font size for x-axis label
            ylabel_fontsize (int, optional): Font size for y-axis label
            legend_loc (str, optional): Location for the legend
            bbox_to_anchor (tuple, optional): Bounding box for the legend
            dpi (int, optional): DPI for the output image

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load result data
        if result_dict is not None:
            results = result_dict
        elif result_file is not None:
            with open(result_file, "r") as f:
                results = json.load(f)
        else:
            raise ValueErrorHandler(
                "Either result_file or result_dict must be provided"
            )

        # Initialize figure
        fig, ax = plt.subplots(figsize=figsize)

        # Extract data for the specified metric
        values = self._extract_per_doi_metric_values(results, metric_name)

        # Handle case with no values
        if not values:
            print(f"No values found for metric: {metric_name}")
            plt.close(fig)
            return None

        # Create color gradient if requested
        if color_gradient:
            if gradient_colors is None:
                gradient_colors = ["#D4E6F1", color]
                cmap = LinearSegmentedColormap.from_list("custom_cmap", gradient_colors)
            else:
                cmap = LinearSegmentedColormap.from_list("custom_cmap", gradient_colors)
            hist, edges = np.histogram(values, bins=bins)
            norm = plt.Normalize(0, max(hist))
            colors = [cmap(norm(h)) for h in hist]
        else:
            colors = color

        # Plot histogram with optional KDE
        if show_kde:
            sns.histplot(
                values,
                bins=bins,
                kde=True,
                color=colors if not color_gradient else color,
                ax=ax,
                element="step" if show_kde else "bars",
            )
        else:
            if color_gradient:
                # Plot each bar with its own color
                counts, bin_edges = np.histogram(values, bins=bins)
                for i in range(len(counts)):
                    ax.bar(
                        (bin_edges[i] + bin_edges[i + 1]) / 2,
                        counts[i],
                        width=bin_edges[i + 1] - bin_edges[i],
                        color=colors[i],
                        edgecolor="gray",
                        alpha=0.7,
                    )
            else:
                # Simple histogram
                ax.hist(values, bins=bins, color=color, edgecolor="black", alpha=0.7)

        # Calculate statistics
        mean_value = np.mean(values)
        median_value = np.median(values)

        # Add lines for mean, median, threshold
        lines = []
        if show_mean:
            mean_line = ax.axvline(
                mean_value, color=mean_color, linestyle=mean_line_style, linewidth=2
            )
            lines.append((mean_line, f"Mean: {mean_value:.2f}"))

        if show_median:
            median_line = ax.axvline(
                median_value,
                color=median_color,
                linestyle=median_line_style,
                linewidth=2,
            )
            lines.append((median_line, f"Median: {median_value:.2f}"))

        if show_threshold:
            threshold_line = ax.axvline(
                threshold_value,
                color=threshold_color,
                linestyle=threshold_line_style,
                linewidth=2,
            )
            lines.append((threshold_line, f"Threshold: {threshold_value:.2f}"))

        # Add legend if we have any lines
        if lines:
            ax.legend(
                [line for line, _ in lines],
                [label for _, label in lines],
                loc=legend_loc,
                bbox_to_anchor=bbox_to_anchor,
            )

        # Set labels
        display_names = self._get_metric_display_names()

        ax.set_xlabel(
            xlabel if xlabel else display_names.get(metric_name, metric_name),
            fontsize=xlabel_fontsize,
        )
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

        # Set title
        if title:
            ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
        else:
            metric_display_name = display_names.get(metric_name, metric_name)
            model_display_name = model_name or results.get("agent_model_name", "Model")
            ax.set_title(
                f"Distribution of {metric_display_name} for {model_display_name}",
                fontsize=title_fontsize,
                fontweight="bold",
            )

        # Adjust layout
        plt.tight_layout()

        # Save figure if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Histogram saved to {output_file}")

        return fig

    def plot_multiple_histogram_charts(
        self,
        result_sources: Union[List[str], List[Dict], str] = None,
        folder_path: Optional[str] = None,
        output_file: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        metric_name: str = "overall_accuracy",
        figsize: Tuple[int, int] = (14, 12),
        bins: int = 10,
        colormap: str = "tab10",
        show_kde: bool = False,
        kde_alpha: float = 0.7,
        show_mean: bool = False,
        mean_color: str = "green",
        mean_line_style: str = "-",
        show_median: bool = False,
        median_color: str = "black",
        median_line_style: str = "-",
        show_threshold: bool = False,
        threshold_value: float = 0.8,
        threshold_color: str = "red",
        threshold_line_style: str = "--",
        show_grid: bool = True,
        title: Optional[str] = None,
        title_fontsize: int = 14,
        xlabel: Optional[str] = None,
        ylabel: str = "Count",
        xlabel_fontsize: int = 12,
        ylabel_fontsize: int = 12,
        legend_loc: Optional[str] = "best",
        legend_fontsize: int = 10,
        bbox_to_anchor: Optional[str] = None,
        is_normalized: bool = True,
        shared_bins: bool = True,
        dpi: int = 300,
    ):
        """
        Create histograms for a single metric from evaluation results for multiple models.

        Args:
            =result_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
            folder_path (Optional[str], optional): Path to folder containing JSON result files. Either result_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image
            model_names (Optional[List[str]]): Names of the models for display in the plot titles
            metric_name (str, optional): Name of the metric to plot (default: "overall_accuracy")
            figsize (tuple, optional): Figure size as (width, height) in inches (default: (14, 12))
            bins (int, optional): Number of bins or bin edges for histogram (default: 10)
            colormap (str, optional): Matplotlib colormap name for the histogram colors (default: 'tab10')
            show_kde (bool, optional): Whether to show a KDE curve over the histogram (default: False)
            kde_alpha (float, optional): Alpha value for the KDE curve (default: 0.7)
            show_mean (bool, optional): Whether to show a vertical line at the mean value (default: False)
            mean_color (str, optional): Color for the mean line (default: 'green')
            mean_line_style (str, optional): Line style for the mean line (default: '-')
            show_median (bool, optional): Whether to show a vertical line at the median value (default: False)
            median_color (str, optional): Color for the median line (default: 'black')
            median_line_style (str, optional): Line style for the median line (default: '-')
            show_threshold (bool, optional): Whether to show a threshold line (default: False)
            threshold_value (float, optional): Value for the threshold line (default: 0.8)
            threshold_color (str, optional): Color for the threshold line (default: 'red')
            threshold_line_style (str, optional): Line style for the threshold line (default: '--')
            show_grid (bool, optional): Whether to show grid lines on the plot (default: True)
            title (str, optional): Custom title for the plot (default: None)
            title_fontsize (int, optional): Font size for the title (default: 14)
            xlabel (str, optional): Custom label for x-axis
            ylabel (str, optional): Label for y-axis (default: 'Count')
            xlabel_fontsize (int, optional): Font size for x-axis label (default: 12)
            ylabel_fontsize (int, optional): Font size for y-axis label (default: 12)
            legend_loc (str, optional): Location for the legend (default: 'best')
            legend_fontsize (int, optional): Font size for the legend (default: 10)
            bbox_to_anchor (tuple, optional): Bounding box for the legend
            is_normalized (bool, optional): Whether to normalize histograms to show percentages (default: True)
            shared_bins (bool, optional): Whether to use shared bins across all histograms (default: True)
            dpi (int, optional): DPI for the output image (default: 300)
        """
        # Load results data
        results_data, names = self._load_results_data(
            result_sources, folder_path, model_names
        )

        display_names = self._get_metric_display_names()
        metric_display_name = display_names.get(metric_name, metric_name)

        # Initialize figure
        fig, ax = plt.subplots(figsize=figsize)

        # Extract data for the specified metric from all models
        all_values = []
        for result in results_data:
            values = self._extract_per_doi_metric_values(result, metric_name)
            if values:
                all_values.append(values)
            else:
                all_values.append([0])
                print(
                    f"No values found for metric: {metric_name} in model: {result.get('agent_model_name')}"
                )

        # Handle case with no values
        if all(not values for values in all_values):
            print(f"No values found for metric: {metric_name}")
            plt.close(fig)
            return None

        # Get color from colormap
        cmap = plt.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, len(all_values)))

        # Determine common bin edges if shared_bins is True
        if shared_bins and isinstance(bins, int):
            flat_values = [val for model_values in all_values for val in model_values]
            if flat_values:
                min_val, max_val = min(flat_values), max(flat_values)
                paddings = (max_val - min_val) * 0.05
                bin_edges = np.linspace(
                    min_val - paddings, max_val + paddings, bins + 1
                )
            else:
                bin_edges = bins
        else:
            bin_edges = bins

        # Plot histograms for all models
        for i, (values, name, color) in enumerate(zip(all_values, names, colors)):
            if not values:
                continue

            if is_normalized:
                weights = np.ones_like(values) / len(values)
            else:
                weights = None

            # Plot histogram with optional KDE
            if show_kde:
                sns.histplot(
                    values,
                    bins=bin_edges,
                    kde=True,
                    color=color,
                    alpha=kde_alpha,
                    label=name,
                    weights=weights,
                    ax=ax,
                    kde_kws={"alpha": kde_alpha, "linewidth": 2, "color": color},
                )
            else:
                ax.hist(
                    values,
                    bins=bin_edges,
                    color=color,
                    edgecolor="black",
                    alpha=0.7,
                    label=name,
                    weights=weights,
                )

            # Add lines for mean, median, threshold for each model
            mean_value = np.mean(values)
            median_value = np.median(values)
            if show_mean:
                ax.axvline(
                    mean_value,
                    color=mean_color,
                    linestyle=mean_line_style,
                    linewidth=2,
                    alpha=0.7,
                    label=f"{name} Mean: {mean_value:.2f}",
                )

            if show_median:
                ax.axvline(
                    median_value,
                    color=median_color,
                    linestyle=median_line_style,
                    linewidth=2,
                    alpha=0.7,
                    label=f"{name} Median: {median_value:.2f}",
                )

        # Add threshold line if requested
        if show_threshold:
            ax.axvline(
                threshold_value,
                color=threshold_color,
                linestyle=threshold_line_style,
                linewidth=2,
                alpha=0.7,
                label=f"Threshold: {threshold_value:.2f}",
            )

        # Set labels
        ax.set_xlabel(
            xlabel if xlabel else metric_display_name,
            fontsize=xlabel_fontsize,
        )
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

        # Add grid if requested
        if show_grid:
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Set title
        if title:
            ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
        else:
            ax.set_title(
                f"Distribution of {metric_display_name} for Multiple Models",
                fontsize=title_fontsize,
                fontweight="bold",
            )

        # Handle legend
        if show_mean or show_median:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                loc=legend_loc,
                bbox_to_anchor=bbox_to_anchor,
                fontsize=legend_fontsize,
            )
        else:
            ax.legend(
                loc=legend_loc, bbox_to_anchor=bbox_to_anchor, fontsize=legend_fontsize
            )

        # Adjust layout
        plt.tight_layout()

        # Save figure if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Multiple models' histogram comparison saved to {output_file}")

        return fig

    def plot_single_violin_chart(
        self,
        result_file: Optional[str] = None,
        result_dict: Optional[dict] = None,
        output_file: Optional[str] = None,
        model_name: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        colormap: str = "Blues",
        title: Optional[str] = None,
        title_fontsize: int = 14,
        title_pad: Optional[float] = 10.0,
        show_box: bool = False,
        show_mean: bool = True,
        mean_marker: str = "o",
        mean_color: str = "red",
        show_median: bool = False,
        median_color: str = "green",
        median_line_style: str = "-",
        show_grid: bool = True,
        show_threshold: bool = False,
        threshold_value: float = 0.8,
        threshold_color: str = "red",
        threshold_line_style: str = "--",
        violin_alpha: float = 0.7,
        violin_width: float = 0.8,
        x_label: str = "Metrics",
        y_label: str = "Score",
        x_label_fontsize: int = 12,
        y_label_fontsize: int = 12,
        y_axis_range: Tuple[float, float] = (0, 1),
        label_rotation: int = 45,
        inner: str = "box",
        dpi: int = 300,
        include_metrics: Optional[List[str]] = None,
        exclude_metrics: Optional[List[str]] = None,
    ):
        """
        Create a violin plot for all metrics from a single model's evaluation results.

        Args:
            result_file (str, optional): Path to JSON file containing evaluation results
            result_dict (dict, optional): Dictionary containing evaluation results. Either result_file or result_dict must be provided.
            output_file (str, optional): Path to save the output visualisation
            model_name (str, optional): Name to display for the model in the plot
            figsize (tuple, optional): Figure size as (width, height) in inches (default: (14, 10))
            colormap (str, optional): Matplotlib colormap name for the violins (default: 'Blues')
            title (str, optional): Custom title for the plot (default: None)
            title_fontsize (int, optional): Font size for the title text (default: 14)
            title_pad (float, optional): Padding for the title from the top of the plot (default: 10.0)
            show_box (bool, optional): Whether to show a box plot inside the violin (default: False)
            show_mean (bool, optional): Whether to show the mean marker (default: True)
            mean_marker (str, optional): Marker style for the mean (default: 'o')
            mean_color (str, optional): Color for the mean marker (default: 'red')
            show_median (bool, optional): Whether to show the median line (default: False)
            median_color (str, optional): Color for the median line (default: 'green')
            median_line_style (str, optional): Line style for the median (default: '-')
            show_grid (bool, optional): Whether to display the grid lines (default: True)
            show_threshold (bool, optional): Whether to show a threshold line (default: False)
            threshold_value (float, optional): Value for the threshold line (default: 0.8)
            threshold_color (str, optional): Color for the threshold line (default: 'red')
            threshold_line_style (str, optional): Line style for the threshold line (default: '--')
            violin_alpha (float, optional): Alpha (transparency) of the violin plots (default: 0.7)
            violin_width (float, optional): Width of the violin plots (default: 0.8)
            x_label (str, optional): Label for the x-axis (default: 'Metrics')
            y_label (str, optional): Label for the y-axis (default: 'Score')
            x_label_fontsize (int, optional): Font size for x-axis label (default: 12)
            y_label_fontsize (int, optional): Font size for y-axis label (default: 12)
            y_axis_range (tuple, optional): Range for the y-axis (default: (0, 1))
            label_rotation (int, optional): Rotation angle for x-axis labels (default: 45)
            inner (str, optional): The representation of the data points inside the violin ('box', 'stick', 'point', or None) (default: 'box')
            dpi (int, optional): Resolution for saved image (default: 300)
            include_metrics (list, optional): Specific metrics to include in the plot (default: None - all available)
            exclude_metrics (list, optional): Specific metrics to exclude from the plot (default: None)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load result data
        if result_dict is not None:
            result = result_dict
        elif result_file is not None:
            with open(result_file, "r") as f:
                result = json.load(f)
        else:
            raise ValueErrorHandler(
                "Either result_file or result_dict must be provided"
            )

        metric_extractors = self._get_metric_extractors()
        metric_display_names = self._get_detailed_metric_display_names()
        available_metrics = list(metric_extractors.keys())

        # Filter metrics based on include/exclude lists
        if include_metrics:
            metrics_to_use = [m for m in include_metrics if m in available_metrics]
        else:
            metrics_to_use = available_metrics.copy()
        if exclude_metrics:
            metrics_to_use = [m for m in metrics_to_use if m not in exclude_metrics]

        data_dict = {}
        for metric_id in metrics_to_use:
            data_dict[metric_id] = self._extract_per_doi_metric_values(
                result, metric_id
            )

        # Filter out metrics with no data
        metrics_to_use = [m for m in metrics_to_use if data_dict[m]]

        if not metrics_to_use:
            raise ValueErrorHandler("No data available for the specified metrics")

        # Prepare data for plotting
        plot_data = []
        labels = []

        for metric_id in metrics_to_use:
            if data_dict[metric_id]:
                plot_data.append(data_dict[metric_id])
                labels.append(metric_display_names.get(metric_id, metric_id))

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create color map
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i / len(metrics_to_use)) for i in range(len(metrics_to_use))]

        # Plot violins
        parts = ax.violinplot(
            plot_data,
            showmeans=show_mean,
            showmedians=show_median,
            showextrema=True,
            widths=violin_width,
            vert=True,
        )

        # Customize violins
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor("gray")
            pc.set_alpha(violin_alpha)

        # Customize mean markers
        if show_mean:
            means = [np.mean(data) for data in plot_data]
            positions = np.arange(1, len(plot_data) + 1)
            ax.scatter(
                positions, means, color=mean_color, marker=mean_marker, s=80, zorder=3
            )
            parts["cmeans"].set_visible(False)

        # Customize median lines
        if show_median:
            parts["cmedians"].set_edgecolor(median_color)
            parts["cmedians"].set_linestyle(median_line_style)
            parts["cmedians"].set_linewidth(2)

        # Customize whiskers and caps
        for partname in ["cbars", "cmins", "cmaxes"]:
            parts[partname].set_edgecolor("black")
            parts[partname].set_linewidth(1)

        # Add boxplots inside violins if requested
        if show_box and inner == "box":
            box_positions = np.arange(1, len(plot_data) + 1)
            box_parts = ax.boxplot(
                plot_data,
                positions=box_positions,
                widths=violin_width * 0.3,
                patch_artist=True,
                showfliers=False,
                showcaps=True,
                vert=True,
            )

            for i, box in enumerate(box_parts["boxes"]):
                box.set_facecolor(colors[i])
                box.set_alpha(0.5)
                box.set_edgecolor("black")

            for whisker in box_parts["whiskers"]:
                whisker.set_color("black")

            for cap in box_parts["caps"]:
                cap.set_color("black")

            for median in box_parts["medians"]:
                median.set_color("black")

        # Set x-axis labels
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=label_rotation, ha="right")

        # Set axis labels
        ax.set_xlabel(x_label, fontsize=x_label_fontsize)
        ax.set_ylabel(y_label, fontsize=y_label_fontsize)

        # Set y-axis range
        ax.set_ylim(y_axis_range)

        # Add grid
        if show_grid:
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Set title
        if title:
            ax.set_title(
                title, fontsize=title_fontsize, fontweight="bold", pad=title_pad
            )
        else:
            model_display_name = model_name or result.get("agent_model_name", "Model")
            ax.set_title(
                f"Distribution of Evaluation Metrics for {model_display_name}",
                fontsize=title_fontsize,
                fontweight="bold",
                pad=title_pad,
            )

        # Add threshold line if requested
        if show_threshold:
            ax.axhline(
                y=threshold_value,
                color=threshold_color,
                linestyle=threshold_line_style,
                alpha=0.5,
                label=f"Threshold: {threshold_value:.2f}",
            )
            ax.legend(loc="lower right")

        # Adjust layout
        plt.tight_layout()

        # Save figure if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Single model's violin plot saved to {output_file}")

        return fig

    def plot_multiple_violin_charts(
        self,
        result_sources: Union[List[str], List[Dict], str] = None,
        folder_path: Optional[str] = None,
        output_file: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        metric_name: str = "overall_accuracy",
        figsize: Tuple[int, int] = (12, 8),
        colormap: str = "viridis",
        title: Optional[str] = None,
        title_fontsize: int = 14,
        title_pad: Optional[float] = 50.0,
        show_box: bool = False,
        show_mean: bool = True,
        mean_marker: str = "o",
        mean_color: str = "red",
        show_median: bool = False,
        median_color: str = "green",
        median_line_style: str = "-",
        show_grid: bool = True,
        show_threshold: bool = False,
        threshold_value: float = 0.8,
        threshold_color: str = "red",
        threshold_line_style: str = "--",
        violin_alpha: float = 0.7,
        violin_width: float = 0.8,
        x_label: str = "Models",
        y_label: str = "Score",
        x_label_fontsize: int = 12,
        y_label_fontsize: int = 12,
        y_axis_range: Tuple[float, float] = (0, 1),
        label_rotation: int = 45,
        inner: str = "box",
        dpi: int = 300,
    ):
        """
        Create violin plots comparing multiple models on a single metric.

        Args:
            result_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
            folder_path (Optional[str], optional): Path to folder containing JSON result files. Either result_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output visualisation
            model_names (Optional[List[str]]): Names to display for models in the plot
            metric_name (str, optional): Name of the metric to compare across models (default: "overall_accuracy")
            figsize (tuple, optional): Figure size as (width, height) in inches (default: (12, 8))
            colormap (str, optional): Matplotlib colormap name for the violins (default: 'viridis')
            title (str, optional): Custom title for the plot (default: None)
            title_fontsize (int, optional): Font size for the title text (default: 14)
            title_pad (float, optional): Padding for the title from the top of the plot
            show_box (bool, optional): Whether to show a box plot inside the violin (default: False)
            show_mean (bool, optional): Whether to show the mean marker (default: True)
            mean_marker (str, optional): Marker style for the mean (default: 'o')
            mean_color (str, optional): Color for the mean marker (default: 'red')
            show_median (bool, optional): Whether to show the median line (default: False)
            median_color (str, optional): Color for the median line (default: 'green')
            median_line_style (str, optional): Line style for the median (default: '-')
            show_grid (bool, optional): Whether to display the grid lines (default: True)
            show_threshold (bool, optional): Whether to show a threshold line (default: False)
            threshold_value (float, optional): Value for the threshold line (default: 0.8)
            threshold_color (str, optional): Color for the threshold line (default: 'red')
            threshold_line_style (str, optional): Line style for the threshold line (default: '--')
            violin_alpha (float, optional): Alpha (transparency) of the violin plots (default: 0.7)
            violin_width (float, optional): Width of the violin plots (default: 0.8)
            x_label (str, optional): Label for the x-axis (default: 'Models')
            y_label (str, optional): Label for the y-axis (default: 'Score')
            x_label_fontsize (int, optional): Font size for x-axis label (default: 12)
            y_label_fontsize (int, optional): Font size for y-axis label (default: 12)
            y_axis_range (tuple, optional): Range for the y-axis (default: (0, 1))
            label_rotation (int, optional): Rotation angle for x-axis labels (default: 45)
            inner (str, optional): The representation of the data points inside the violin ('box', 'stick', 'point', or None) (default: 'box')
            dpi (int, optional): Resolution for saved image (default: 300)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load results data
        results_data, names = self._load_results_data(
            result_sources, folder_path, model_names
        )

        # Get metric display names
        metric_display_names = self._get_metric_display_names()
        metric_display_name = metric_display_names.get(metric_name, metric_name)

        # Extract data for the specified metric from all models
        plot_data = []

        for result in results_data:
            values = self._extract_per_doi_metric_values(result, metric_name)

            # Add non-empty data for this model
            if values:
                plot_data.append(values)
            else:
                # If no data available, add a placeholder
                plot_data.append([0])
                print(
                    f"Warning: No data found for metric '{metric_name}' in model '{names[len(plot_data)-1]}'"
                )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create color map
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i / len(results_data)) for i in range(len(results_data))]

        # Plot violins
        positions = np.arange(1, len(plot_data) + 1)
        parts = ax.violinplot(
            plot_data,
            positions=positions,
            showmeans=show_mean,
            showmedians=show_median,
            showextrema=True,
            widths=violin_width,
            vert=True,
        )

        # Customize violins
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor("gray")
            pc.set_alpha(violin_alpha)

        # Customize mean markers
        if show_mean:
            # Calculate mean values
            means = [np.mean(data) for data in plot_data]
            # Plot custom markers for means
            ax.scatter(
                positions, means, color=mean_color, marker=mean_marker, s=80, zorder=3
            )
            # Hide original mean markers
            parts["cmeans"].set_visible(False)

        # Customize median lines
        if show_median:
            parts["cmedians"].set_edgecolor(median_color)
            parts["cmedians"].set_linestyle(median_line_style)
            parts["cmedians"].set_linewidth(2)

        # Customize whiskers and caps
        for partname in ["cbars", "cmins", "cmaxes"]:
            parts[partname].set_edgecolor("black")
            parts[partname].set_linewidth(1)

        # Add boxplots inside violins if requested
        if show_box and inner == "box":
            box_parts = ax.boxplot(
                plot_data,
                positions=positions,
                widths=violin_width * 0.3,
                patch_artist=True,
                showfliers=False,
                showcaps=True,
                vert=True,
            )

            for i, box in enumerate(box_parts["boxes"]):
                box.set_facecolor(colors[i])
                box.set_alpha(0.5)
                box.set_edgecolor("black")

            for whisker in box_parts["whiskers"]:
                whisker.set_color("black")

            for cap in box_parts["caps"]:
                cap.set_color("black")

            for median in box_parts["medians"]:
                median.set_color("black")

        # Set x-axis labels
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=label_rotation, ha="right")

        # Set axis labels
        ax.set_xlabel(x_label, fontsize=x_label_fontsize)
        ax.set_ylabel(y_label, fontsize=y_label_fontsize)

        # Set y-axis range
        ax.set_ylim(y_axis_range)

        # Add grid
        if show_grid:
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Add threshold line
        if show_threshold:
            ax.axhline(
                y=threshold_value,
                color=threshold_color,
                linestyle=threshold_line_style,
                alpha=0.7,
                label=f"Threshold: {threshold_value:.2f}",
            )
            ax.legend(loc="lower right")

        # Set title
        if title:
            ax.set_title(
                title, fontsize=title_fontsize, fontweight="bold", pad=title_pad
            )
        else:
            ax.set_title(
                f"Distribution of {metric_display_name} Across Models",
                fontsize=title_fontsize,
                fontweight="bold",
                pad=title_pad,
            )

        # Adjust layout
        plt.tight_layout()

        # Save figure if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Multiple models' violin plot saved to {output_file}")

        return fig
