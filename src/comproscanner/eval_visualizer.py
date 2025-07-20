"""
Visualization module for ComProScanner evaluation results.

This module provides functions for creating various visualizations of evaluation results, including bar charts, radar charts, heatmaps, histograms, and violin plots.
"""

from typing import Optional, List, Dict, Tuple, Union
import numpy as np
from .post_processing.visualization.eval_plot_visualizers import EvalVisualizer

# Import for type annotations, but use lazy loading for actual imports
if False:
    import matplotlib.figure
    from matplotlib import pyplot as plt
    import pandas as pd
    import seaborn as sns


def plot_single_bar_chart(
    result_file: str | None = None,
    result_dict: dict | None = None,
    output_file: str | None = None,
    model_name: str | None = None,
    figsize: Tuple[int, int] = (12, 8),
    colormap: str | None = "Blues",
    display_values: bool = True,
    title: str | None = None,
    typical_threshold: float | None = None,
    threashold_line_style: str | None = "--",
    threashold_tolerance_range: float | None = 0.03,
    threshold_color: str | None = "red",
    show_grid: bool = True,
    bar_width: float = 0.6,
    y_axis_label: str = "Score",
    x_axis_label: str | None = None,
    y_axis_range: Tuple[float, float] = (0, 1),
    dpi: int = 300,
    metrics_to_include: List[str] | None = [
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

    Raises:
        ValueError: If neither result_file nor result_dict is provided
    """

    visualizer = EvalVisualizer()
    fig = visualizer.plot_single_bar_chart(
        result_file=result_file,
        result_dict=result_dict,
        output_file=output_file,
        model_name=model_name,
        figsize=figsize,
        colormap=colormap,
        display_values=display_values,
        title=title,
        typical_threshold=typical_threshold,
        threashold_line_style=threashold_line_style,
        threashold_tolerance_range=threashold_tolerance_range,
        threshold_color=threshold_color,
        show_grid=show_grid,
        bar_width=bar_width,
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label,
        y_axis_range=y_axis_range,
        dpi=dpi,
        metrics_to_include=metrics_to_include,
    )
    return fig


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

    Raises:
        ValueErrorHandler: If neither result_sources nor folder_path is provided, or if the specified path does not exist
    """
    visualizer = EvalVisualizer()
    fig = visualizer.plot_multiple_bar_charts(
        result_sources=result_sources,
        folder_path=folder_path,
        output_file=output_file,
        model_names=model_names,
        figsize=figsize,
        colormap=colormap,
        display_values=display_values,
        title=title,
        typical_threshold=typical_threshold,
        threshold_line_style=threshold_line_style,
        threashold_tolerance_range=threashold_tolerance_range,
        threshold_color=threshold_color,
        show_grid=show_grid,
        y_label=y_label,
        x_label=x_label,
        group_width=group_width,
        bar_width=bar_width,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        y_axis_range=y_axis_range,
        dpi=dpi,
        metrics_to_include=metrics_to_include,
    )
    return fig


def plot_single_radar_chart(
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

    Raises:
        ValueErrorHandler: If neither result_file nor result_dict is provided, or if the specified path does not exist
    """
    visualizer = EvalVisualizer()
    fig = visualizer.plot_single_radar_chart(
        result_file=result_file,
        result_dict=result_dict,
        output_file=output_file,
        model_name=model_name,
        figsize=figsize,
        colormap=colormap,
        display_values=display_values,
        title=title,
        title_fontsize=title_fontsize,
        title_pad=title_pad,
        typical_threshold=typical_threshold,
        threshold_color=threshold_color,
        threshold_line_style=threshold_line_style,
        label_fontsize=label_fontsize,
        value_fontsize=value_fontsize,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        bbox_to_anchor=bbox_to_anchor,
        show_grid=show_grid,
        show_grid_labels=show_grid_labels,
        grid_line_width=grid_line_width,
        grid_line_style=grid_line_style,
        grid_line_color=grid_line_color,
        grid_line_alpha=grid_line_alpha,
        fill_alpha=fill_alpha,
        marker_size=marker_size,
        line_width=line_width,
        label_padding=label_padding,
        clockwise=clockwise,
        start_angle=start_angle,
        radar_range=radar_range,
        dpi=dpi,
        metrics_to_include=metrics_to_include,
    )
    return fig


def plot_multiple_radar_charts(
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

    Raises:
        ValueErrorHandler: If neither result_sources nor folder_path is provided, or if the specified path does not exist
    """
    visualizer = EvalVisualizer()
    fig = visualizer.plot_multiple_radar_charts(
        result_sources=result_sources,
        folder_path=folder_path,
        output_file=output_file,
        model_names=model_names,
        figsize=figsize,
        colormap=colormap,
        display_values=display_values,
        title=title,
        title_fontsize=title_fontsize,
        title_pad=title_pad,
        typical_threshold=typical_threshold,
        threshold_color=threshold_color,
        threshold_line_style=threshold_line_style,
        label_fontsize=label_fontsize,
        value_fontsize=value_fontsize,
        legend_loc=legend_loc,
        bbox_to_anchor=bbox_to_anchor,
        legend_fontsize=legend_fontsize,
        show_grid=show_grid,
        show_grid_labels=show_grid_labels,
        grid_line_width=grid_line_width,
        grid_line_style=grid_line_style,
        grid_line_color=grid_line_color,
        grid_line_alpha=grid_line_alpha,
        fill_alpha=fill_alpha,
        marker_size=marker_size,
        line_width=line_width,
        label_padding=label_padding,
        clockwise=clockwise,
        start_angle=start_angle,
        radar_range=radar_range,
        dpi=dpi,
        metrics_to_include=metrics_to_include,
    )
    return fig


def plot_single_performance_heatmap(
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
        output_file (str, optional): Path to save the output visualization
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

    Raises:
        ValueErrorHandler: If neither result_file nor result_dict is provided, or if the specified path does not exist
    """
    visualizer = EvalVisualizer()
    fig = visualizer.plot_single_performance_heatmap(
        result_file=result_file,
        result_dict=result_dict,
        output_file=output_file,
        model_name=model_name,
        figsize=figsize,
        colormap=colormap,
        bin_count=bin_count,
        score_range=score_range,
        use_percentage=use_percentage,
        show_averages=show_averages,
        show_group_labels=show_group_labels,
        show_annotations=show_annotations,
        annotation_format=annotation_format,
        title=title,
        title_fontsize=title_fontsize,
        title_pad=title_pad,
        labels=labels,
        label_fontsize=label_fontsize,
        dpi=dpi,
        group_metrics=group_metrics,
        metric_groups=metric_groups,
        group_colors=group_colors,
        include_metrics=include_metrics,
        exclude_metrics=exclude_metrics,
        group_label_right_margin=group_label_right_margin,
        average_value_left_margin=average_value_left_margin,
        plot_padding=plot_padding,
    )
    return fig


def plot_multiple_performance_heatmaps(
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
        output_file (str, optional): Path to save the output visualization
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

    Raises:
        ValueErrorHandler: If neither result_sources nor folder_path is provided, or if the specified path does not exist
    """
    visualizer = EvalVisualizer()
    fig = visualizer.plot_multiple_performance_heatmaps(
        result_sources=result_sources,
        folder_path=folder_path,
        output_file=output_file,
        model_names=model_names,
        figsize=figsize,
        colormap=colormap,
        bin_count=bin_count,
        score_range=score_range,
        use_percentage=use_percentage,
        show_averages=show_averages,
        show_group_labels=show_group_labels,
        show_annotations=show_annotations,
        annotation_format=annotation_format,
        title=title,
        title_fontsize=title_fontsize,
        labels=labels,
        label_fontsize=label_fontsize,
        dpi=dpi,
        group_metrics=group_metrics,
        metric_groups=metric_groups,
        group_colors=group_colors,
        include_metrics=include_metrics,
        exclude_metrics=exclude_metrics,
        sort_models_by=sort_models_by,
        combine_models=combine_models,
        group_label_right_margin=group_label_right_margin,
        average_value_left_margin=average_value_left_margin,
        plot_padding=plot_padding,
    )
    return fig


def plot_multiple_confusion_matrices_combined(
    self,
    result_sources: Union[List[str], List[Dict], str] = None,
    folder_path: Optional[str] = None,
    output_file: Optional[str] = None,
    model_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    colormap: str = "YlOrRd",
    show_annotations: bool = True,
    annotation_format: Optional[str] = None,
    annotation_fontsize: int = 10,
    title: Optional[str] = None,
    title_fontsize: int = 14,
    title_pad: Optional[float] = 20.0,
    labels: List[str] = ["Models", "Metrics"],
    label_fontsize: int = 12,
    tick_label_fontsize: int = 10,
    dpi: int = 300,
    include_metrics: Optional[List[str]] = [
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
    exclude_metrics: Optional[List[str]] = None,
    sort_models_by: str = "average",  # CHANGED: Default to "average" instead of "overall_accuracy"
    value_range: Tuple[float, float] = (0, 1),
    show_colorbar: bool = True,
    colorbar_label: str = "Score",
    colorbar_fontsize: int = 10,
    plot_padding: float = 0.1,
):
    """
    Create a confusion matrix-style heatmap showing all models vs all performance metrics in a single visualization.

    Args:
        result_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
        folder_path (Optional[str], optional): Path to folder containing JSON result files. Either result_sources or folder_path must be provided.
        output_file (str, optional): Path to save the output visualization
        model_names (Optional[List[str]]): Names to display for models in the plot
        figsize (Tuple[int, int]): Figure size as (width, height) in inches
        colormap (str): Matplotlib colormap name for the heatmap
        show_annotations (bool): Whether to show value annotations in cells
        annotation_format (Optional[str]): Format string for annotations (e.g., '.2f' or '.1f')
        annotation_fontsize (int): Font size for the annotation values inside cells
        title (Optional[str]): Custom title for the plot
        title_fontsize (int): Font size for the title
        title_pad (Optional[float]): Padding for the title from the top of the plot
        labels (List[str]): Labels for the x and y axes (default: ['Models', 'Metrics'])
        label_fontsize (int): Font size for the axis labels
        tick_label_fontsize (int): Font size for x and y tick labels
        dpi (int): Resolution for saved image
        include_metrics (Optional[List[str]]): Specific metrics to include (default: all 9 standard metrics)
        exclude_metrics (Optional[List[str]]): Specific metrics to exclude from the heatmap
        sort_models_by (str): Metric to sort models by, or "average" for average of all metrics (default: 'average')
        value_range (Tuple[float, float]): Min and max values for color mapping (default: (0, 1))
        show_colorbar (bool): Whether to show the colorbar legend
        colorbar_label (str): Label for the colorbar
        colorbar_fontsize (int): Font size for colorbar labels
        plot_padding (float): Padding between heatmap and axes labels and title

    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    visualizer = EvalVisualizer()
    fig = visualizer.plot_multiple_confusion_matrices_combined(
        result_sources=result_sources,
        folder_path=folder_path,
        output_file=output_file,
        model_names=model_names,
        figsize=figsize,
        colormap=colormap,
        show_annotations=show_annotations,
        annotation_format=annotation_format,
        annotation_fontsize=annotation_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        title_pad=title_pad,
        labels=labels,
        label_fontsize=label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        dpi=dpi,
        include_metrics=include_metrics,
        exclude_metrics=exclude_metrics,
        sort_models_by=sort_models_by,
        value_range=value_range,
        show_colorbar=show_colorbar,
        colorbar_label=colorbar_label,
        colorbar_fontsize=colorbar_fontsize,
        plot_padding=plot_padding,
    )
    return fig


def plot_single_histogram_chart(
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

    Raises:
        ValueErrorHandler: If neither result_file nor result_dict is provided, or if the specified path does not exist
    """
    visualizer = EvalVisualizer()
    fig = visualizer.plot_single_histogram_chart(
        result_file=result_file,
        result_dict=result_dict,
        metric_name=metric_name,
        output_file=output_file,
        model_name=model_name,
        figsize=figsize,
        bins=bins,
        color=color,
        color_gradient=color_gradient,
        gradient_colors=gradient_colors,
        show_kde=show_kde,
        show_mean=show_mean,
        mean_color=mean_color,
        mean_line_style=mean_line_style,
        show_median=show_median,
        median_color=median_color,
        median_line_style=median_line_style,
        show_threshold=show_threshold,
        threshold_value=threshold_value,
        threshold_color=threshold_color,
        threshold_line_style=threshold_line_style,
        title=title,
        title_fontsize=title_fontsize,
        xlabel=xlabel,
        ylabel=ylabel,
        xlabel_fontsize=xlabel_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        legend_loc=legend_loc,
        bbox_to_anchor=bbox_to_anchor,
        dpi=dpi,
    )
    return fig


def plot_multiple_histogram_charts(
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

    Returns:
        matplotlib.figure.Figure: The generated figure object

    Raises:
        ValueErrorHandler: If neither result_sources nor folder_path is provided, or if the specified path does not exist
    """
    visualizer = EvalVisualizer()
    fig = visualizer.plot_multiple_histogram_charts(
        result_sources=result_sources,
        folder_path=folder_path,
        output_file=output_file,
        model_names=model_names,
        metric_name=metric_name,
        figsize=figsize,
        bins=bins,
        colormap=colormap,
        show_kde=show_kde,
        kde_alpha=kde_alpha,
        show_mean=show_mean,
        mean_color=mean_color,
        mean_line_style=mean_line_style,
        show_median=show_median,
        median_color=median_color,
        median_line_style=median_line_style,
        show_threshold=show_threshold,
        threshold_value=threshold_value,
        threshold_color=threshold_color,
        threshold_line_style=threshold_line_style,
        show_grid=show_grid,
        title=title,
        title_fontsize=title_fontsize,
        xlabel=xlabel,
        ylabel=ylabel,
        xlabel_fontsize=xlabel_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        bbox_to_anchor=bbox_to_anchor,
        is_normalized=is_normalized,
        shared_bins=shared_bins,
        dpi=dpi,
    )
    return fig


def plot_single_violin_chart(
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
        output_file (str, optional): Path to save the output visualization
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

    Raises:
        ValueErrorHandler: If neither result_file nor result_dict is provided, or if the specified path does not exist
    """
    visualizer = EvalVisualizer()
    fig = visualizer.plot_single_violin_chart(
        result_file=result_file,
        result_dict=result_dict,
        output_file=output_file,
        model_name=model_name,
        figsize=figsize,
        colormap=colormap,
        title=title,
        title_fontsize=title_fontsize,
        title_pad=title_pad,
        show_box=show_box,
        show_mean=show_mean,
        mean_marker=mean_marker,
        mean_color=mean_color,
        show_median=show_median,
        median_color=median_color,
        median_line_style=median_line_style,
        show_grid=show_grid,
        show_threshold=show_threshold,
        threshold_value=threshold_value,
        threshold_color=threshold_color,
        threshold_line_style=threshold_line_style,
        violin_alpha=violin_alpha,
        violin_width=violin_width,
        x_label=x_label,
        y_label=y_label,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        y_axis_range=y_axis_range,
        label_rotation=label_rotation,
        inner=inner,
        dpi=dpi,
        include_metrics=include_metrics,
        exclude_metrics=exclude_metrics,
    )
    return fig


def plot_multiple_violin_charts(
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
        output_file (str, optional): Path to save the output visualization
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

    Raises:
        ValueErrorHandler: If neither result_sources nor folder_path is provided, or if the specified path does not exist
    """
    visualizer = EvalVisualizer()
    fig = visualizer.plot_multiple_violin_charts(
        result_sources=result_sources,
        folder_path=folder_path,
        output_file=output_file,
        model_names=model_names,
        metric_name=metric_name,
        figsize=figsize,
        colormap=colormap,
        title=title,
        title_fontsize=title_fontsize,
        title_pad=title_pad,
        show_box=show_box,
        show_mean=show_mean,
        mean_marker=mean_marker,
        mean_color=mean_color,
        show_median=show_median,
        median_color=median_color,
        median_line_style=median_line_style,
        show_grid=show_grid,
        show_threshold=show_threshold,
        threshold_value=threshold_value,
        threshold_color=threshold_color,
        threshold_line_style=threshold_line_style,
        violin_alpha=violin_alpha,
        violin_width=violin_width,
        x_label=x_label,
        y_label=y_label,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        y_axis_range=y_axis_range,
        label_rotation=label_rotation,
        inner=inner,
        dpi=dpi,
    )
    return fig
