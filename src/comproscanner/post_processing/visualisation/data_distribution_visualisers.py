"""
data_visualiser.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 03-05-2025
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


class DataDistributionVisualiser:
    def __init__(self):
        """
        Initialize the DataDistributionVisualiser class for visualising material data distributions.
        """
        self.data = None
        self.colour_palettes = {
            "family": "viridis",
            "precursors": "plasma",
            "characterization_techniques": "mako",
        }

    def _load_data(self, data_sources=None, folder_path=None):
        """
        Load data from files or dictionaries for visualisation.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files.
                Either data_sources or folder_path must be provided.

        Returns:
            dict: Combined data dictionary

        Raises:
            ValueError: If neither data_sources nor folder_path is provided, or if no valid data found
        """
        if data_sources is None and folder_path is None:
            raise ValueError("Either data_sources or folder_path must be provided")

        all_data = {}

        # Process folder_path if provided
        if folder_path is not None:
            if not os.path.isdir(folder_path):
                raise ValueError(
                    f"The provided folder path does not exist: {folder_path}"
                )

            # Find all JSON files in the folder
            json_files = []
            for file in os.listdir(folder_path):
                if file.endswith(".json"):
                    json_files.append(os.path.join(folder_path, file))

            # If no data_sources were provided, use all JSON files from the folder
            if data_sources is None:
                data_sources = json_files
            # If data_sources is a list, append the JSON files from the folder
            elif isinstance(data_sources, list):
                data_sources.extend(json_files)

        # Handle the case when data_sources is a single string (filepath)
        if isinstance(data_sources, str):
            if os.path.isfile(data_sources):
                data_sources = [data_sources]
            else:
                raise ValueError(
                    f"The provided file path does not exist: {data_sources}"
                )

        # Process each data source
        for source in data_sources:
            # Load from dict or file
            if isinstance(source, dict):
                self._merge_data(all_data, source)
            elif isinstance(source, str):
                try:
                    with open(source, "r") as f:
                        data = json.load(f)
                        self._merge_data(all_data, data)
                except Exception as e:
                    print(f"Error loading {source}: {e}")
                    continue

        if not all_data:
            raise ValueError("No valid data found")

        self.data = all_data
        return all_data

    def _merge_data(self, target_dict, source_dict):
        """
        Merge source data dictionary into target data dictionary.

        Args:
            target_dict (dict): Target dictionary to merge data into
            source_dict (dict): Source dictionary containing data to merge
        """
        for doi, data in source_dict.items():
            target_dict[doi] = data

    def _extract_families(self):
        """
        Extract family information from the loaded data.

        Returns:
            dict: Counter object with family names and their frequencies
        """
        families = []

        for doi, item_data in self.data.items():
            if (
                "composition_data" in item_data
                and "family" in item_data["composition_data"]
            ):
                family = item_data["composition_data"]["family"]
                if family:  # Only add if not empty
                    families.append(family)

        return Counter(families)

    def _extract_precursors(self):
        """
        Extract precursors information from the loaded data.

        Returns:
            dict: Counter object with precursor names and their frequencies
        """
        all_precursors = []

        for doi, item_data in self.data.items():
            if (
                "synthesis_data" in item_data
                and "precursors" in item_data["synthesis_data"]
            ):
                precursors = item_data["synthesis_data"]["precursors"]
                if precursors:  # Only add if list is not empty
                    all_precursors.extend(precursors)

        return Counter(all_precursors)

    def _extract_characterization_techniques(self):
        """
        Extract characterization techniques information from the loaded data.

        Returns:
            dict: Counter object with technique names and their frequencies
        """
        all_techniques = []

        for doi, item_data in self.data.items():
            if (
                "synthesis_data" in item_data
                and "characterization_techniques" in item_data["synthesis_data"]
            ):
                techniques = item_data["synthesis_data"]["characterization_techniques"]
                if techniques:  # Only add if list is not empty
                    all_techniques.extend(techniques)

        return Counter(all_techniques)

    def _plot_pie_chart(
        self,
        data_counter,
        title,
        figsize=(10, 8),
        dpi=300,
        output_file=None,
        min_percentage=1.0,
        colour_palette=None,
    ):
        """
        Create a pie chart visualisation of data distribution.

        Args:
            data_counter (Counter): Counter object with data labels and frequencies
            title (str): Title for the plot
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            output_file (str, optional): Path to save the output plot image
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            colour_palette (str, optional): Matplotlib colormap name for the pie sections

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate percentages and create 'Others' category if needed
        total = sum(data_counter.values())
        percentages = {k: (v / total) * 100 for k, v in data_counter.items()}

        # Sort by percentage (descending)
        sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

        # Create 'Others' category for items below threshold
        main_items = [(k, v) for k, v in sorted_items if v >= min_percentage]
        others = [(k, v) for k, v in sorted_items if v < min_percentage]

        if others:
            others_sum = sum(v for _, v in others)
            plot_items = main_items + [("Others", others_sum)]
        else:
            plot_items = main_items

        labels = [f"{k} ({v:.1f}%)" for k, v in plot_items]
        values = [v for _, v in plot_items]

        # Generate colors using specified palette
        if colour_palette is None:
            cmap = plt.get_cmap("viridis")
        else:
            cmap = plt.get_cmap(colour_palette)

        colors = cmap(np.linspace(0, 0.9, len(plot_items)))

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops={"edgecolor": "w", "linewidth": 1},
        )

        # Adjust text properties
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight("bold")

        # Add legend
        ax.legend(
            wedges,
            labels,
            title="Categories",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )

        # Set title
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_aspect("equal")

        # Adjust layout
        plt.tight_layout()

        # Save figure if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Pie chart saved to {output_file}")

        return fig

    def _plot_histogram(
        self,
        data_counter,
        title,
        figsize=(12, 8),
        dpi=300,
        output_file=None,
        max_items=15,
        colour_palette=None,
        x_label=None,
        y_label="Frequency",
        rotation=45,
    ):
        """
        Create a histogram visualisation of data distribution.

        Args:
            data_counter (Counter): Counter object with data labels and frequencies
            title (str): Title for the plot
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            output_file (str, optional): Path to save the output plot image
            max_items (int, optional): Maximum number of items to display
            colour_palette (str, optional): Matplotlib colormap name for the bars
            x_label (str, optional): Label for the x-axis
            y_label (str, optional): Label for the y-axis
            rotation (int, optional): Rotation angle for x-axis labels

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by frequency (descending)
        sorted_items = sorted(data_counter.items(), key=lambda x: x[1], reverse=True)

        # Limit to max_items if needed
        if max_items and len(sorted_items) > max_items:
            display_items = sorted_items[: max_items - 1]

            # Add 'Others' category with sum of remaining items
            others_sum = sum(v for _, v in sorted_items[max_items - 1 :])
            display_items.append(("Others", others_sum))
        else:
            display_items = sorted_items

        labels = [k for k, _ in display_items]
        values = [v for _, v in display_items]

        # Generate colors using specified palette
        if colour_palette is None:
            cmap = plt.get_cmap("viridis")
        else:
            cmap = plt.get_cmap(colour_palette)

        colors = cmap(np.linspace(0, 0.9, len(display_items)))

        # Create bars
        bars = ax.bar(
            range(len(display_items)),
            values,
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Set x-axis labels
        ax.set_xticks(range(len(display_items)))
        ax.set_xticklabels(labels, rotation=rotation, ha="right")

        # Set axis labels
        if x_label:
            ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

        # Set title
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Add grid
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save figure if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Histogram saved to {output_file}")

        return fig

    def plot_family_pie_chart(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(10, 8),
        dpi=300,
        min_percentage=1.0,
        title="Distribution of Material Families",
        colour_palette=None,
    ):
        """
        Create a pie chart visualisation of material families distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            title (str, optional): Title for the plot
            colour_palette (str, optional): Matplotlib colormap name for the pie sections

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract families
        family_counter = self._extract_families()

        if not family_counter:
            raise ValueError("No family data found in the provided data sources")

        # Use default colour palette if not specified
        if colour_palette is None:
            colour_palette = self.colour_palettes["family"]

        # Create pie chart
        return self._plot_pie_chart(
            family_counter,
            title,
            figsize,
            dpi,
            output_file,
            min_percentage,
            colour_palette,
        )

    def plot_family_histogram(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        max_items=15,
        title="Frequency Distribution of Material Families",
        colour_palette=None,
        x_label="Material Family",
        y_label="Frequency",
        rotation=45,
    ):
        """
        Create a histogram visualisation of material families distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            max_items (int, optional): Maximum number of items to display
            title (str, optional): Title for the plot
            colour_palette (str, optional): Matplotlib colormap name for the bars
            x_label (str, optional): Label for the x-axis
            y_label (str, optional): Label for the y-axis
            rotation (int, optional): Rotation angle for x-axis labels

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract families
        family_counter = self._extract_families()

        if not family_counter:
            raise ValueError("No family data found in the provided data sources")

        # Use default colour palette if not specified
        if colour_palette is None:
            colour_palette = self.colour_palettes["family"]

        # Create histogram
        return self._plot_histogram(
            family_counter,
            title,
            figsize,
            dpi,
            output_file,
            max_items,
            colour_palette,
            x_label,
            y_label,
            rotation,
        )

    def plot_precursors_pie_chart(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(10, 8),
        dpi=300,
        min_percentage=1.0,
        title="Distribution of Precursors in Materials Synthesis",
        colour_palette=None,
    ):
        """
        Create a pie chart visualisation of precursors distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            title (str, optional): Title for the plot
            colour_palette (str, optional): Matplotlib colormap name for the pie sections

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract precursors
        precursors_counter = self._extract_precursors()

        if not precursors_counter:
            raise ValueError("No precursors data found in the provided data sources")

        # Use default colour palette if not specified
        if colour_palette is None:
            colour_palette = self.colour_palettes["precursors"]

        # Create pie chart
        return self._plot_pie_chart(
            precursors_counter,
            title,
            figsize,
            dpi,
            output_file,
            min_percentage,
            colour_palette,
        )

    def plot_precursors_histogram(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        max_items=15,
        title="Frequency Distribution of Precursors in Materials Synthesis",
        colour_palette=None,
        x_label="Precursor",
        y_label="Frequency",
        rotation=45,
    ):
        """
        Create a histogram visualisation of precursors distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            max_items (int, optional): Maximum number of items to display
            title (str, optional): Title for the plot
            colour_palette (str, optional): Matplotlib colormap name for the bars
            x_label (str, optional): Label for the x-axis
            y_label (str, optional): Label for the y-axis
            rotation (int, optional): Rotation angle for x-axis labels

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract precursors
        precursors_counter = self._extract_precursors()

        if not precursors_counter:
            raise ValueError("No precursors data found in the provided data sources")

        # Use default colour palette if not specified
        if colour_palette is None:
            colour_palette = self.colour_palettes["precursors"]

        # Create histogram
        return self._plot_histogram(
            precursors_counter,
            title,
            figsize,
            dpi,
            output_file,
            max_items,
            colour_palette,
            x_label,
            y_label,
            rotation,
        )

    def plot_characterization_techniques_pie_chart(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(10, 8),
        dpi=300,
        min_percentage=1.0,
        title="Distribution of Characterization Techniques",
        colour_palette=None,
    ):
        """
        Create a pie chart visualisation of characterization techniques distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            title (str, optional): Title for the plot
            colour_palette (str, optional): Matplotlib colormap name for the pie sections

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract characterization techniques
        techniques_counter = self._extract_characterization_techniques()

        if not techniques_counter:
            raise ValueError(
                "No characterization techniques data found in the provided data sources"
            )

        # Use default colour palette if not specified
        if colour_palette is None:
            colour_palette = self.colour_palettes["characterization_techniques"]

        # Create pie chart
        return self._plot_pie_chart(
            techniques_counter,
            title,
            figsize,
            dpi,
            output_file,
            min_percentage,
            colour_palette,
        )

    def plot_characterization_techniques_histogram(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(14, 8),
        dpi=300,
        max_items=15,
        title="Frequency Distribution of Characterization Techniques",
        colour_palette=None,
        x_label="Characterization Technique",
        y_label="Frequency",
        rotation=45,
    ):
        """
        Create a histogram visualisation of characterization techniques distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            max_items (int, optional): Maximum number of items to display
            title (str, optional): Title for the plot
            colour_palette (str, optional): Matplotlib colormap name for the bars
            x_label (str, optional): Label for the x-axis
            y_label (str, optional): Label for the y-axis
            rotation (int, optional): Rotation angle for x-axis labels

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract characterization techniques
        techniques_counter = self._extract_characterization_techniques()

        if not techniques_counter:
            raise ValueError(
                "No characterization techniques data found in the provided data sources"
            )

        # Use default colour palette if not specified
        if colour_palette is None:
            colour_palette = self.colour_palettes["characterization_techniques"]

        # Create histogram
        return self._plot_histogram(
            techniques_counter,
            title,
            figsize,
            dpi,
            output_file,
            max_items,
            colour_palette,
            x_label,
            y_label,
            rotation,
        )
