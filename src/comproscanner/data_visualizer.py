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


######## logger Configuration ########
logger = setup_logger("visualizer_logs.log")


def plot_family_pie_chart(
    data_sources: Union[List[str], List[Dict], str] = None,
    folder_path: Optional[str] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    min_percentage: float = 1.0,
    title: str = "Distribution of Material Families",
    colour_palette: Optional[str] = None,
):
    """
    Create a pie chart visualization of material families distribution.

    Args:
        data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
        folder_path (str, optional): Path to folder containing JSON data files
        output_file (str, optional): Path to save the output plot image
        figsize (tuple, optional): Figure size as (width, height) in inches (default: (10, 8))
        dpi (int, optional): DPI for output image (default: 300)
        min_percentage (float, optional): Minimum percentage for a category to be shown separately (default: 1.0)
        title (str, optional): Title for the plot (default: "Distribution of Material Families")
        colour_palette (str, optional): Matplotlib colormap name for the pie sections (default: None)

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
        colour_palette=colour_palette,
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
    colour_palette: Optional[str] = None,
    x_label: str = "Material Family",
    y_label: str = "Frequency",
    rotation: int = 45,
):
    """
    Create a histogram visualization of material families distribution.

    Args:
        data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
        folder_path (str, optional): Path to folder containing JSON data files
        output_file (str, optional): Path to save the output plot image
        figsize (tuple, optional): Figure size as (width, height) in inches (default: (12, 8))
        dpi (int, optional): DPI for output image (default: 300)
        max_items (int, optional): Maximum number of items to display (default: 15)
        title (str, optional): Title for the plot (default: "Frequency Distribution of Material Families")
        colour_palette (str, optional): Matplotlib colormap name for the bars (default: None)
        x_label (str, optional): Label for the x-axis (default: "Material Family")
        y_label (str, optional): Label for the y-axis (default: "Frequency")
        rotation (int, optional): Rotation angle for x-axis labels (default: 45)

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
        colour_palette=colour_palette,
        x_label=x_label,
        y_label=y_label,
        rotation=rotation,
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
    colour_palette: Optional[str] = None,
):
    """
    Create a pie chart visualization of precursors distribution.

    Args:
        data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
        folder_path (str, optional): Path to folder containing JSON data files
        output_file (str, optional): Path to save the output plot image
        figsize (tuple, optional): Figure size as (width, height) in inches (default: (10, 8))
        dpi (int, optional): DPI for output image (default: 300)
        min_percentage (float, optional): Minimum percentage for a category to be shown separately (default: 1.0)
        title (str, optional): Title for the plot (default: "Distribution of Precursors in Materials Synthesis")
        colour_palette (str, optional): Matplotlib colormap name for the pie sections (default: None)

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
        colour_palette=colour_palette,
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
    colour_palette: Optional[str] = None,
    x_label: str = "Precursor",
    y_label: str = "Frequency",
    rotation: int = 45,
):
    """
    Create a histogram visualization of precursors distribution.

    Args:
        data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
        folder_path (str, optional): Path to folder containing JSON data files
        output_file (str, optional): Path to save the output plot image
        figsize (tuple, optional): Figure size as (width, height) in inches (default: (12, 8))
        dpi (int, optional): DPI for output image (default: 300)
        max_items (int, optional): Maximum number of items to display (default: 15)
        title (str, optional): Title for the plot (default: "Frequency Distribution of Precursors in Materials Synthesis")
        colour_palette (str, optional): Matplotlib colormap name for the bars (default: None)
        x_label (str, optional): Label for the x-axis (default: "Precursor")
        y_label (str, optional): Label for the y-axis (default: "Frequency")
        rotation (int, optional): Rotation angle for x-axis labels (default: 45)

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
        colour_palette=colour_palette,
        x_label=x_label,
        y_label=y_label,
        rotation=rotation,
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
    colour_palette: Optional[str] = None,
):
    """
    Create a pie chart visualization of characterization techniques distribution.

    Args:
        data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
        folder_path (str, optional): Path to folder containing JSON data files
        output_file (str, optional): Path to save the output plot image
        figsize (tuple, optional): Figure size as (width, height) in inches (default: (10, 8))
        dpi (int, optional): DPI for output image (default: 300)
        min_percentage (float, optional): Minimum percentage for a category to be shown separately (default: 1.0)
        title (str, optional): Title for the plot (default: "Distribution of Characterization Techniques")
        colour_palette (str, optional): Matplotlib colormap name for the pie sections (default: None)

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
        colour_palette=colour_palette,
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
    colour_palette: Optional[str] = None,
    x_label: str = "Characterization Technique",
    y_label: str = "Frequency",
    rotation: int = 45,
):
    """
    Create a histogram visualization of characterization techniques distribution.

    Args:
        data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
        folder_path (str, optional): Path to folder containing JSON data files
        output_file (str, optional): Path to save the output plot image
        figsize (tuple, optional): Figure size as (width, height) in inches (default: (14, 8))
        dpi (int, optional): DPI for output image (default: 300)
        max_items (int, optional): Maximum number of items to display (default: 15)
        title (str, optional): Title for the plot (default: "Frequency Distribution of Characterization Techniques")
        colour_palette (str, optional): Matplotlib colormap name for the bars (default: None)
        x_label (str, optional): Label for the x-axis (default: "Characterization Technique")
        y_label (str, optional): Label for the y-axis (default: "Frequency")
        rotation (int, optional): Rotation angle for x-axis labels (default: 45)

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
        colour_palette=colour_palette,
        x_label=x_label,
        y_label=y_label,
        rotation=rotation,
    )
    return fig


def create_knowledge_graph(
    self,
    result_file: str = None,
):
    """
    Create a knowledge graph from extracted composition-property data directly in Neo4j database.

    Args:
        result_file (str, required): Path to the JSON file containing extracted results.

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
            )
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {e}")
        raise
