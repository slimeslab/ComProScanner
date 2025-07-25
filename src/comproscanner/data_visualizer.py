"""
data_visualizer.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 07-05-2025
"""

from typing import Optional, List, Dict, Union, Tuple
import numpy as np
from comproscanner.post_processing.visualization.data_distribution_visualizers import (
    DataDistributionVisualizer,
)
from .post_processing.visualization.create_knowledge_graph import CreateKG
from .utils.logger import setup_logger

# Import for type annotations, but use lazy loading for actual imports
if False:
    import matplotlib.figure
    from matplotlib import pyplot as plt
    import pandas as pd
    import seaborn as sns


# configure logger
logger = setup_logger("comproscanner.log", module_name="data_visualizer")


def plot_family_pie_chart(
    data_sources: Union[List[str], List[Dict], str] = None,
    folder_path: Optional[str] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    min_percentage: float = 1.0,
    title: str = "Distribution of Material Families",
    color_palette: str = "Blues",
    title_fontsize: int = 14,
    label_fontsize: int = 10,
    legend_fontsize: int = 10,
    is_semantic_clustering_enabled: bool = True,
    similarity_threshold: float = 0.8,
):
    """
    Create a pie chart visualization of material families distribution.

    Args:
        data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
        folder_path (str, optional): Path to folder containing JSON data files
        output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
        figsize (tuple, optional): Figure size as (width, height) in inches (Default: (10, 8))
        dpi (int, optional): DPI for output image (Default: 300)
        min_percentage (float, optional): Minimum percentage for a category to be shown separately (Default: 1.0)
        title (str, optional): Title for the plot (Default: "Distribution of Material Families")
        color_palette (str, optional): Matplotlib colormap name for the pie sections (Default: "Blues")
        title_fontsize (int, optional): Font size for the title (Default: 14)
        label_fontsize (int, optional): Font size for the percentage labels (Default: 10)
        legend_fontsize (int, optional): Font size for the legend (Default: 10)
        is_semantic_clustering_enabled (bool, optional): Whether to use semantic similarity for clustering similar families (Default: True)
        similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)

    Returns:
        matplotlib.figure.Figure: The generated figure object

    Raises:
        ValueError: If neither data_sources nor folder_path is provided, or if no family data found
    """
    visualizer = DataDistributionVisualizer()
    fig = visualizer.plot_family_pie_chart(
        data_sources=data_sources,
        folder_path=folder_path,
        output_file=output_file,
        figsize=figsize,
        dpi=dpi,
        min_percentage=min_percentage,
        title=title,
        color_palette=color_palette,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        legend_fontsize=legend_fontsize,
        is_semantic_clustering_enabled=is_semantic_clustering_enabled,
        similarity_threshold=similarity_threshold,
    )
    return fig


def plot_family_histogram(
    data_sources: Union[List[str], List[Dict], str] = None,
    folder_path: Optional[str] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    max_items: int = 15,
    title: str = "Frequency Distribution of Material Families",
    color_palette: str = "Blues",
    x_label: str = "Material Family",
    y_label: str = "Frequency",
    rotation: int = 45,
    title_fontsize: int = 14,
    xlabel_fontsize: int = 12,
    ylabel_fontsize: int = 12,
    xtick_fontsize: int = 10,
    value_label_fontsize: int = 9,
    grid_axis: str = "y",
    grid_linestyle: str = "--",
    grid_alpha: float = 0.3,
    is_semantic_clustering_enabled: bool = True,
    similarity_threshold: float = 0.8,
):
    """
    Create a histogram visualization of material families distribution.

    Args:
        data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
        folder_path (str, optional): Path to folder containing JSON data files
        output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
        figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
        dpi (int, optional): DPI for output image (Default: 300)
        max_items (int, optional): Maximum number of items to display (Default: 15)
        title (str, optional): Title for the plot (Default: "Frequency Distribution of Material Families")
        color_palette (str, optional): Matplotlib colormap name for the bars (Default: "Blues")
        x_label (str, optional): Label for the x-axis (Default: "Material Family")
        y_label (str, optional): Label for the y-axis (Default: "Frequency")
        rotation (int, optional): Rotation angle for x-axis labels (Default: 45)
        title_fontsize (int, optional): Font size for the title (Default: 14)
        xlabel_fontsize (int, optional): Font size for the x-axis label (Default: 12)
        ylabel_fontsize (int, optional): Font size for the y-axis label (Default: 12)
        xtick_fontsize (int, optional): Font size for the x-axis tick labels (Default: 10)
        value_label_fontsize (int, optional): Font size for the value labels on bars (Default: 9)
        grid_axis (str, optional): Axis for grid lines ('x', 'y', 'both', or None for no grid) (Default: "y")
        grid_linestyle (str, optional): Line style for grid lines (Default: "--")
        grid_alpha (float, optional): Alpha (transparency) for grid lines (Default: 0.3)
        is_semantic_clustering_enabled (bool, optional): Whether to enable semantic clustering of families (Default: True)
        similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)

    Returns:
        matplotlib.figure.Figure: The generated figure object

    Raises:
        ValueError: If neither data_sources nor folder_path is provided, or if no family data found
    """
    visualizer = DataDistributionVisualizer()
    fig = visualizer.plot_family_histogram(
        data_sources=data_sources,
        folder_path=folder_path,
        output_file=output_file,
        figsize=figsize,
        dpi=dpi,
        max_items=max_items,
        title=title,
        color_palette=color_palette,
        x_label=x_label,
        y_label=y_label,
        rotation=rotation,
        title_fontsize=title_fontsize,
        xlabel_fontsize=xlabel_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        xtick_fontsize=xtick_fontsize,
        value_label_fontsize=value_label_fontsize,
        grid_axis=grid_axis,
        grid_linestyle=grid_linestyle,
        grid_alpha=grid_alpha,
        is_semantic_clustering_enabled=is_semantic_clustering_enabled,
        similarity_threshold=similarity_threshold,
    )
    return fig


def plot_precursors_pie_chart(
    data_sources: Union[List[str], List[Dict], str] = None,
    folder_path: Optional[str] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    min_percentage: float = 1.0,
    title: str = "Distribution of Precursors in Materials Synthesis",
    color_palette: str = "Blues",
    title_fontsize: int = 14,
    label_fontsize: int = 10,
    legend_fontsize: int = 10,
    is_semantic_clustering_enabled: bool = True,
    similarity_threshold: float = 0.8,
):
    """
    Create a pie chart visualization of precursors distribution.

    Args:
        data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
        folder_path (str, optional): Path to folder containing JSON data files
        output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
        figsize (tuple, optional): Figure size as (width, height) in inches (Default: (10, 8))
        dpi (int, optional): DPI for output image (Default: 300)
        min_percentage (float, optional): Minimum percentage for a category to be shown separately (Default: 1.0)
        title (str, optional): Title for the plot (Default: "Distribution of Precursors in Materials Synthesis")
        color_palette (str, optional): Matplotlib colormap name for the pie sections (Default: "Blues")
        title_fontsize (int, optional): Font size for the title (Default: 14)
        label_fontsize (int, optional): Font size for the percentage labels (Default: 10)
        legend_fontsize (int, optional): Font size for the legend (Default: 10)
        is_semantic_clustering_enabled (bool, optional): Whether to use semantic similarity for clustering similar precursors (Default: True)
        similarity_threshold (float, optional): Threshold for similarity-based clustering when is_semantic_clustering_enabled is True (Default: 0.8)

    Returns:
        matplotlib.figure.Figure: The generated figure object

    Raises:
        ValueError: If neither data_sources nor folder_path is provided, or if no precursors data found
    """
    visualizer = DataDistributionVisualizer()
    fig = visualizer.plot_precursors_pie_chart(
        data_sources=data_sources,
        folder_path=folder_path,
        output_file=output_file,
        figsize=figsize,
        dpi=dpi,
        min_percentage=min_percentage,
        title=title,
        color_palette=color_palette,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        legend_fontsize=legend_fontsize,
        is_semantic_clustering_enabled=is_semantic_clustering_enabled,
        similarity_threshold=similarity_threshold,
    )
    return fig


def plot_precursors_histogram(
    data_sources: Union[List[str], List[Dict], str] = None,
    folder_path: Optional[str] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    max_items: int = 15,
    title: str = "Frequency Distribution of Precursors in Materials Synthesis",
    color_palette: str = "Blues",
    x_label: str = "Precursor",
    y_label: str = "Frequency",
    rotation: int = 45,
    title_fontsize: int = 14,
    xlabel_fontsize: int = 12,
    ylabel_fontsize: int = 12,
    xtick_fontsize: int = 10,
    value_label_fontsize: int = 9,
    grid_axis: str = "y",
    grid_linestyle: str = "--",
    grid_alpha: float = 0.3,
    is_semantic_clustering_enabled: bool = True,
    similarity_threshold: float = 0.8,
):
    """
    Create a histogram visualization of precursors distribution.

    Args:
        data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
        folder_path (str, optional): Path to folder containing JSON data files
        output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
        figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
        dpi (int, optional): DPI for output image (Default: 300)
        max_items (int, optional): Maximum number of items to display (Default: 15)
        title (str, optional): Title for the plot (Default: "Frequency Distribution of Precursors in Materials Synthesis")
        color_palette (str, optional): Matplotlib colormap name for the bars (Default: "Blues")
        x_label (str, optional): Label for the x-axis (Default: "Precursor")
        y_label (str, optional): Label for the y-axis (Default: "Frequency")
        rotation (int, optional): Rotation angle for x-axis labels (Default: 45)
        title_fontsize (int, optional): Font size for the title (Default: 14)
        xlabel_fontsize (int, optional): Font size for the x-axis label (Default: 12)
        ylabel_fontsize (int, optional): Font size for the y-axis label (Default: 12)
        xtick_fontsize (int, optional): Font size for the x-axis tick labels (Default: 10)
        value_label_fontsize (int, optional): Font size for the value labels on bars (Default: 9)
        grid_axis (str, optional): Axis for grid lines ('x', 'y', 'both', or None for no grid) (Default: "y")
        grid_linestyle (str, optional): Line style for grid lines (Default: "--")
        grid_alpha (float, optional): Alpha (transparency) for grid lines (Default: 0.3)
        is_semantic_clustering_enabled (bool, optional): Whether to enable semantic clustering of precursors (Default: True)
        similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)

    Returns:
        matplotlib.figure.Figure: The generated figure object

    Raises:
        ValueError: If neither data_sources nor folder_path is provided, or if no precursors data found
    """
    visualizer = DataDistributionVisualizer()
    fig = visualizer.plot_precursors_histogram(
        data_sources=data_sources,
        folder_path=folder_path,
        output_file=output_file,
        figsize=figsize,
        dpi=dpi,
        max_items=max_items,
        title=title,
        color_palette=color_palette,
        x_label=x_label,
        y_label=y_label,
        rotation=rotation,
        title_fontsize=title_fontsize,
        xlabel_fontsize=xlabel_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        xtick_fontsize=xtick_fontsize,
        value_label_fontsize=value_label_fontsize,
        grid_axis=grid_axis,
        grid_linestyle=grid_linestyle,
        grid_alpha=grid_alpha,
        is_semantic_clustering_enabled=is_semantic_clustering_enabled,
        similarity_threshold=similarity_threshold,
    )
    return fig


def plot_characterization_techniques_pie_chart(
    data_sources: Union[List[str], List[Dict], str] = None,
    folder_path: Optional[str] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    min_percentage: float = 1.0,
    title: str = "Distribution of Characterization Techniques",
    color_palette: str = "Blues",
    is_semantic_clustering_enabled: bool = True,
    similarity_threshold: float = 0.8,
    title_fontsize: int = 14,
    label_fontsize: int = 10,
    legend_fontsize: int = 10,
):
    """
    Create a pie chart visualization of characterization techniques distribution.

    Args:
        data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
        folder_path (str, optional): Path to folder containing JSON data files
        output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
        figsize (tuple, optional): Figure size as (width, height) in inches (Default: (10, 8))
        dpi (int, optional): DPI for output image (Default: 300)
        min_percentage (float, optional): Minimum percentage for a category to be shown separately (Default: 1.0)
        title (str, optional): Title for the plot (Default: "Distribution of Characterization Techniques")
        color_palette (str, optional): Matplotlib colormap name for the pie sections (Default: "Blues")
        is_semantic_clustering_enabled (bool, optional): Whether to use semantic similarity for clustering similar techniques (Default: True)
        similarity_threshold (float, optional): Threshold for similarity-based clustering when is_semantic_clustering_enabled is True (Default: 0.8)
        title_fontsize (int, optional): Font size for the title (Default: 14)
        label_fontsize (int, optional): Font size for the percentage labels (Default: 10)
        legend_fontsize (int, optional): Font size for the legend (Default: 10)

    Returns:
        matplotlib.figure.Figure: The generated figure object

    Raises:
        ValueError: If neither data_sources nor folder_path is provided, or if no characterization techniques data found
    """
    visualizer = DataDistributionVisualizer()
    fig = visualizer.plot_characterization_techniques_pie_chart(
        data_sources=data_sources,
        folder_path=folder_path,
        output_file=output_file,
        figsize=figsize,
        dpi=dpi,
        min_percentage=min_percentage,
        title=title,
        color_palette=color_palette,
        is_semantic_clustering_enabled=is_semantic_clustering_enabled,
        similarity_threshold=similarity_threshold,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        legend_fontsize=legend_fontsize,
    )
    return fig


def plot_characterization_techniques_histogram(
    data_sources: Union[List[str], List[Dict], str] = None,
    folder_path: Optional[str] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 300,
    max_items: int = 15,
    title: str = "Frequency Distribution of Characterization Techniques",
    color_palette: str = "Blues",
    x_label: str = "Characterization Technique",
    y_label: str = "Frequency",
    rotation: int = 45,
    is_semantic_clustering_enabled: bool = True,
    similarity_threshold: float = 0.8,
    title_fontsize: int = 14,
    xlabel_fontsize: int = 12,
    ylabel_fontsize: int = 12,
    xtick_fontsize: int = 10,
    value_label_fontsize: int = 9,
    grid_axis: str = "y",
    grid_linestyle: str = "--",
    grid_alpha: float = 0.3,
):
    """
    Create a histogram visualization of characterization techniques distribution.

    Args:
        data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
        folder_path (str, optional): Path to folder containing JSON data files
        output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
        figsize (tuple, optional): Figure size as (width, height) in inches (Default: (14, 8))
        dpi (int, optional): DPI for output image (Default: 300)
        max_items (int, optional): Maximum number of items to display (Default: 15)
        title (str, optional): Title for the plot (Default: "Frequency Distribution of Characterization Techniques")
        color_palette (str, optional): Matplotlib colormap name for the bars (Default: "Blues")
        x_label (str, optional): Label for the x-axis (Default: "Characterization Technique")
        y_label (str, optional): Label for the y-axis (Default: "Frequency")
        rotation (int, optional): Rotation angle for x-axis labels (Default: 45)
        is_semantic_clustering_enabled (bool, optional): Whether to use semantic similarity for clustering similar techniques (Default: True)
        similarity_threshold (float, optional): Threshold for similarity-based clustering when is_semantic_clustering_enabled is True (Default: 0.8)
        title_fontsize (int, optional): Font size for the title (Default: 14)
        xlabel_fontsize (int, optional): Font size for the x-axis label (Default: 12)
        ylabel_fontsize (int, optional): Font size for the y-axis label (Default: 12)
        xtick_fontsize (int, optional): Font size for the x-axis tick labels (Default: 10)
        value_label_fontsize (int, optional): Font size for the value labels on bars (Default: 9)
        grid_axis (str, optional): Axis for grid lines ('x', 'y', 'both', or None for no grid) (Default: "y")
        grid_linestyle (str, optional): Line style for grid lines (Default: "--")
        grid_alpha (float, optional): Alpha (transparency) for grid lines (Default: 0.3)

    Returns:
        matplotlib.figure.Figure: The generated figure object

    Raises:
        ValueError: If neither data_sources nor folder_path is provided, or if no characterization techniques data found
    """
    visualizer = DataDistributionVisualizer()
    fig = visualizer.plot_characterization_techniques_histogram(
        data_sources=data_sources,
        folder_path=folder_path,
        output_file=output_file,
        figsize=figsize,
        dpi=dpi,
        max_items=max_items,
        title=title,
        color_palette=color_palette,
        x_label=x_label,
        y_label=y_label,
        rotation=rotation,
        is_semantic_clustering_enabled=is_semantic_clustering_enabled,
        similarity_threshold=similarity_threshold,
        title_fontsize=title_fontsize,
        xlabel_fontsize=xlabel_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        xtick_fontsize=xtick_fontsize,
        value_label_fontsize=value_label_fontsize,
        grid_axis=grid_axis,
        grid_linestyle=grid_linestyle,
        grid_alpha=grid_alpha,
    )
    return fig


def create_knowledge_graph(
    result_file: str = None,
    is_semantic_clustering_enabled: bool = True,
    family_clustering_similarity_threshold: float = 0.9,
    method_clustering_similarity_threshold: float = 0.8,
    precursor_clustering_similarity_threshold: float = 0.9,
    technique_clustering_similarity_threshold: float = 0.8,
    keyword_clustering_similarity_threshold: float = 0.85,
):
    """
    Create a knowledge graph from extracted composition-property data directly in Neo4j database.

    Args:
        result_file (str, required): Path to the JSON file containing extracted results.
        is_semantic_clustering_enabled (bool, optional): Whether to enable clustering of similar compositions (Default: True)
        family_clustering_similarity_threshold (float, optional): Similarity threshold for family clustering (Default: 0.9)
        method_clustering_similarity_threshold (float, optional): Similarity threshold for method clustering (Default: 0.8)
        precursor_clustering_similarity_threshold (float, optional): Similarity threshold for precursor clustering (Default: 0.9)
        technique_clustering_similarity_threshold (float, optional): Similarity threshold for technique clustering (Default: 0.8)
        keyword_clustering_similarity_threshold (float, optional): Similarity threshold for keyword clustering (Default: 0.85)

    Returns:
        None (knowledge graph is created directly in Neo4j database).

    Raises:
        ValueErrorHandler: If result_file is not provided.
    """
    if result_file is None:
        logger.error(
            "result_file cannot be None. Please provide a valid file path. Exiting..."
        )
        raise ValueErrorHandler(
            message="Please provide result_file path to proceed for creating knowledge graph."
        )

    try:
        with CreateKG() as create_kg:
            return create_kg.create_knowledge_graph(
                result_file=result_file,
                is_semantic_clustering_enabled=is_semantic_clustering_enabled,
                method_clustering_similarity_threshold=method_clustering_similarity_threshold,
                technique_clustering_similarity_threshold=technique_clustering_similarity_threshold,
                keyword_clustering_similarity_threshold=keyword_clustering_similarity_threshold,
                family_clustering_similarity_threshold=family_clustering_similarity_threshold,
                precursor_clustering_similarity_threshold=precursor_clustering_similarity_threshold,
            )
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {e}")
        raise
