"""
data_distribution_visualizer.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 03-05-2025
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import difflib
from collections import Counter

try:
    from transformers import AutoTokenizer, AutoModel
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class DataDistributionVisualizer:
    def __init__(self):
        """
        Initialize the DataDistributionVisualizer class for visualizing material data distributions.
        """
        self.data = None
        self.color_palettes = {
            "family": "viridis",
            "precursors": "plasma",
            "characterization_techniques": "mako",
        }

    def _load_semantic_model(self, model_name="thellert/physbert_cased"):
        """
        Load the specified semantic model for similarity calculations.

        Args:
            model_name (str): Name of the model to load

        Returns:
            dict: Dictionary with model type and model/tokenizer objects
        """
        # Check if the model is already loaded
        if hasattr(self, "semantic_model") and self.semantic_model is not None:
            return self.semantic_model

        # Try loading the transformer model first
        if TRANSFORMERS_AVAILABLE:
            try:
                print(f"Attempting to load {model_name} transformer model...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                self.semantic_model = {
                    "type": "transformers",
                    "tokenizer": tokenizer,
                    "model": model,
                }
                print(f"Successfully loaded {model_name} transformer model")
                return self.semantic_model
            except Exception as e:
                print(f"Could not load {model_name}: {e}")

        # Try sentence-transformers as fallback
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print("Falling back to sentence-transformers model...")
                st_model = SentenceTransformer("all-mpnet-base-v2")
                self.semantic_model = {
                    "type": "sentence_transformer",
                    "model": st_model,
                }
                print("Successfully loaded sentence-transformers model")
                return self.semantic_model
            except Exception as e:
                print(f"Could not load sentence-transformers: {e}")

        # Final fallback to difflib
        print("Falling back to difflib.SequenceMatcher for similarity calculations")
        self.semantic_model = {"type": "difflib"}
        return self.semantic_model

    def calculate_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts using the best available method.

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        # Make sure a model is loaded
        if not hasattr(self, "semantic_model") or self.semantic_model is None:
            self._load_semantic_model()

        # Handle empty or None inputs
        if not text1 or not text2:
            return 0.0

        # Convert to strings if needed
        text1 = str(text1)
        text2 = str(text2)

        # Use the appropriate similarity calculation method
        if self.semantic_model["type"] == "transformers":
            return self._calculate_similarity_transformers(text1, text2)
        elif self.semantic_model["type"] == "sentence_transformer":
            return self._calculate_similarity_sentence_transformer(text1, text2)
        else:
            # Fallback to difflib
            return self._calculate_similarity_difflib(text1, text2)

    def _calculate_similarity_transformers(self, text1, text2):
        """
        Calculate similarity between two texts using transformers model.

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        tokenizer = self.semantic_model["tokenizer"]
        model = self.semantic_model["model"]

        # Tokenize and get embeddings
        inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
        inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

        # Get embeddings
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)

        # Use CLS token embedding (first token) as sentence representation
        emb1 = outputs1.last_hidden_state[:, 0, :]
        emb2 = outputs2.last_hidden_state[:, 0, :]

        # Normalize the embeddings
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

        # Calculate cosine similarity
        similarity = torch.mm(emb1, emb2.transpose(0, 1)).item()

        return similarity

    def _calculate_similarity_sentence_transformer(self, text1, text2):
        """
        Calculate similarity between two texts using sentence-transformers.

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        model = self.semantic_model["model"]

        # Get embeddings
        embedding1 = model.encode([text1])[0]
        embedding2 = model.encode([text2])[0]

        # Calculate cosine similarity
        import numpy as np

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Ensure result is in range [0, 1]
        return max(0.0, min(1.0, similarity))

    def _calculate_similarity_difflib(self, text1, text2):
        """
        Calculate similarity between two texts using difflib.SequenceMatcher.

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _cluster_characterization_techniques(
        self, techniques, similarity_threshold=0.8
    ):
        """
        Cluster similar characterization techniques using semantic similarity.

        Args:
            techniques (list): List of all characterization techniques
            similarity_threshold (float): Minimum similarity to consider two techniques as same

        Returns:
            dict: Dictionary mapping canonical names to lists of similar techniques
        """
        techniques_counter = Counter(techniques)
        sorted_techniques = sorted(
            techniques_counter.items(), key=lambda x: x[1], reverse=True
        )

        # Initialize clusters with the most frequent technique as first canonical form
        clusters = {}
        processed = set()

        # Process techniques from most to least frequent
        for technique, count in sorted_techniques:
            if technique in processed:
                continue

            # Create a new cluster with this technique as canonical form
            canonical = technique
            clusters[canonical] = [technique]
            processed.add(technique)

            # Compare to all remaining techniques
            for other_technique, other_count in sorted_techniques:
                if other_technique in processed:
                    continue

                # Calculate similarity
                similarity = self.calculate_similarity(canonical, other_technique)

                if similarity >= similarity_threshold:
                    clusters[canonical].append(other_technique)
                    processed.add(other_technique)

        return clusters

    def _extract_characterization_techniques_with_clustering(
        self, similarity_threshold=0.8
    ):
        """
        Extract characterization techniques with semantic clustering to merge similar techniques.

        Args:
            similarity_threshold (float): Threshold for similarity-based clustering

        Returns:
            Counter: Counter with canonicalized characterization techniques
        """
        all_techniques = []

        # Extract all raw techniques from the data
        for doi, item_data in self.data.items():
            if (
                "synthesis_data" in item_data
                and "characterization_techniques" in item_data["synthesis_data"]
            ):
                techniques = item_data["synthesis_data"]["characterization_techniques"]
                if techniques:  # Only add if list is not empty
                    all_techniques.extend(techniques)

        if not all_techniques:
            return Counter()

        # Cluster similar techniques
        clusters = self._cluster_characterization_techniques(
            all_techniques, similarity_threshold
        )

        # Count canonical techniques
        canonicalized_counts = Counter()
        raw_counts = Counter(all_techniques)

        for canonical, similar_techniques in clusters.items():
            canonicalized_counts[canonical] = sum(
                raw_counts[t] for t in similar_techniques
            )

        return canonicalized_counts

    def _load_data(self, data_sources=None, folder_path=None):
        """
        Load data from files or dictionaries for visualization.

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
                    with open(source, "r", encoding="utf-8") as f:
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
        color_palette=None,
    ):
        """
        Create a pie chart visualization of data distribution with percentage labels outside the chart.

        Args:
            data_counter (Counter): Counter object with data labels and frequencies
            title (str): Title for the plot
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image (Default: 300)
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved
            min_percentage (float, optional): Minimum percentage for a category to be shown separately (Default: 1.0)
            color_palette (str, optional): Matplotlib colormap name for the pie sections

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
        percentage_labels = [
            f"{v:.1f}%" for v in values
        ]  # Just the percentage for outside labels

        # Generate colors using specified palette
        if color_palette is None:
            cmap = plt.get_cmap("viridis")
        else:
            cmap = plt.get_cmap(color_palette)

        colors = cmap(np.linspace(0, 0.9, len(plot_items)))

        # Create pie chart with percentage labels outside
        wedges, texts = ax.pie(
            values,
            labels=percentage_labels,  # Use the percentage labels
            colors=colors,
            startangle=90,
            wedgeprops={"edgecolor": "w", "linewidth": 1},
            autopct=None,  # No internal percentage labels
            labeldistance=1.05,  # Position labels outside the pie
        )

        # Improve label positioning and styling
        for text in texts:
            text.set_fontweight("bold")  # Make labels bold

            # Adjust alignment based on position
            # Get the position of the text
            pos = text.get_position()
            x, y = pos

            # Calculate angle to determine which quadrant the text is in
            angle = np.arctan2(y, x)

            # Adjust alignment based on angle
            if -np.pi / 2 < angle < np.pi / 2:  # Right half
                text.set_horizontalalignment("left")
            else:  # Left half
                text.set_horizontalalignment("right")

            # Handle top and bottom special cases
            if angle > 3 * np.pi / 4 or angle < -3 * np.pi / 4:  # Bottom
                text.set_verticalalignment("top")
            elif -np.pi / 4 < angle < np.pi / 4:  # Right
                text.set_verticalalignment("center")
            elif np.pi / 4 < angle < 3 * np.pi / 4:  # Top
                text.set_verticalalignment("bottom")
            else:  # Left
                text.set_verticalalignment("center")

        # Add legend
        ax.legend(
            wedges,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )

        # Set title directly on the axes, centered
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_aspect("equal")

        # Adjust layout with title positioning
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
        color_palette=None,
        x_label=None,
        y_label="Frequency",
        rotation=45,
    ):
        """
        Create a histogram visualization of data distribution.

        Args:
            data_counter (Counter): Counter object with data labels and frequencies
            title (str): Title for the plot
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            output_file (str, optional): Path to save the output plot image
            max_items (int, optional): Maximum number of items to display
            color_palette (str, optional): Matplotlib colormap name for the bars
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
        if color_palette is None:
            cmap = plt.get_cmap("viridis")
        else:
            cmap = plt.get_cmap(color_palette)

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
        color_palette=None,
    ):
        """
        Create a pie chart visualization of material families distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the pie sections

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
        if color_palette is None:
            color_palette = self.color_palettes["family"]

        # Create pie chart
        return self._plot_pie_chart(
            family_counter,
            title,
            figsize,
            dpi,
            output_file,
            min_percentage,
            color_palette,
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
        color_palette=None,
        x_label="Material Family",
        y_label="Frequency",
        rotation=45,
    ):
        """
        Create a histogram visualization of material families distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            max_items (int, optional): Maximum number of items to display
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the bars
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
        if color_palette is None:
            color_palette = self.color_palettes["family"]

        # Create histogram
        return self._plot_histogram(
            family_counter,
            title,
            figsize,
            dpi,
            output_file,
            max_items,
            color_palette,
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
        color_palette=None,
    ):
        """
        Create a pie chart visualization of precursors distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the pie sections

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
        if color_palette is None:
            color_palette = self.color_palettes["precursors"]

        # Create pie chart
        return self._plot_pie_chart(
            precursors_counter,
            title,
            figsize,
            dpi,
            output_file,
            min_percentage,
            color_palette,
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
        color_palette=None,
        x_label="Precursor",
        y_label="Frequency",
        rotation=45,
    ):
        """
        Create a histogram visualization of precursors distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            max_items (int, optional): Maximum number of items to display
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the bars
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
        if color_palette is None:
            color_palette = self.color_palettes["precursors"]

        # Create histogram
        return self._plot_histogram(
            precursors_counter,
            title,
            figsize,
            dpi,
            output_file,
            max_items,
            color_palette,
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
        color_palette=None,
        use_semantic_clustering=True,
        similarity_threshold=0.8,
    ):
        """
        Create a pie chart visualization of characterization techniques distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the pie sections
            use_semantic_clustering (bool): Whether to use semantic similarity for clustering similar techniques
            similarity_threshold (float): Threshold for similarity-based clustering when use_semantic_clustering is True

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract characterization techniques
        if use_semantic_clustering:
            techniques_counter = (
                self._extract_characterization_techniques_with_clustering(
                    similarity_threshold
                )
            )
            # Add clustering info to title if semantic clustering is used
            title = f"{title} (Semantically Clustered)"
        else:
            techniques_counter = self._extract_characterization_techniques()

        if not techniques_counter:
            raise ValueError(
                "No characterization techniques data found in the provided data sources"
            )

        # Use default colour palette if not specified
        if color_palette is None:
            color_palette = self.color_palettes["characterization_techniques"]

        # Create pie chart
        return self._plot_pie_chart(
            techniques_counter,
            title,
            figsize,
            dpi,
            output_file,
            min_percentage,
            color_palette,
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
        color_palette=None,
        x_label="Characterization Technique",
        y_label="Frequency",
        rotation=45,
        use_semantic_clustering=True,
        similarity_threshold=0.8,
    ):
        """
        Create a histogram visualization of characterization techniques distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image
            figsize (tuple, optional): Figure size as (width, height) in inches
            dpi (int, optional): DPI for output image
            max_items (int, optional): Maximum number of items to display
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the bars
            x_label (str, optional): Label for the x-axis
            y_label (str, optional): Label for the y-axis
            rotation (int, optional): Rotation angle for x-axis labels
            use_semantic_clustering (bool): Whether to use semantic similarity for clustering similar techniques
            similarity_threshold (float): Threshold for similarity-based clustering when use_semantic_clustering is True

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract characterization techniques
        if use_semantic_clustering:
            techniques_counter = (
                self._extract_characterization_techniques_with_clustering(
                    similarity_threshold
                )
            )
            # Add clustering info to title if semantic clustering is used
            title = f"{title} (Semantically Clustered)"
        else:
            techniques_counter = self._extract_characterization_techniques()

        if not techniques_counter:
            raise ValueError(
                "No characterization techniques data found in the provided data sources"
            )

        # Use default colour palette if not specified
        if color_palette is None:
            color_palette = self.color_palettes["characterization_techniques"]

        # Create histogram
        return self._plot_histogram(
            techniques_counter,
            title,
            figsize,
            dpi,
            output_file,
            max_items,
            color_palette,
            x_label,
            y_label,
            rotation,
        )

    def get_technique_clusters(
        self, data_sources=None, folder_path=None, similarity_threshold=0.8
    ):
        """
        Get clusters of similar characterization techniques based on semantic similarity.

        This is a utility method to inspect how techniques are being clustered.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            similarity_threshold (float): Threshold for similarity-based clustering

        Returns:
            Dict[str, List[str]]: Dictionary mapping canonical names to lists of similar techniques
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Get all techniques
        all_techniques = []
        for doi, item_data in self.data.items():
            if (
                "synthesis_data" in item_data
                and "characterization_techniques" in item_data["synthesis_data"]
            ):
                techniques = item_data["synthesis_data"]["characterization_techniques"]
                if techniques:  # Only add if list is not empty
                    all_techniques.extend(techniques)

        if not all_techniques:
            return {}

        # Return clusters of similar techniques
        return self._cluster_characterization_techniques(
            all_techniques, similarity_threshold
        )
