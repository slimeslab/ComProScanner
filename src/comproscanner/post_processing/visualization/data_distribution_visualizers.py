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
from tqdm import tqdm
from ...utils.logger import setup_logger

logger = setup_logger("post-processing.log")

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
                logger.debug(
                    f"\n\nAttempting to load {model_name} transformer model..."
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                self.semantic_model = {
                    "type": "transformers",
                    "tokenizer": tokenizer,
                    "model": model,
                }
                logger.info(f"Successfully loaded {model_name} transformer model")
                return self.semantic_model
            except Exception as e:
                logger.error(f"Could not load {model_name}: {e}")

        # Try sentence-transformers as fallback
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.debug("Falling back to sentence-transformers model...")
                st_model = SentenceTransformer("all-mpnet-base-v2")
                self.semantic_model = {
                    "type": "sentence_transformer",
                    "model": st_model,
                }
                logger.info("Successfully loaded sentence-transformers model")
                return self.semantic_model
            except Exception as e:
                logger.error(f"Could not load sentence-transformers: {e}")

        # Final fallback to difflib
        logger.debug(
            "Falling back to difflib.SequenceMatcher for similarity calculations"
        )
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

    def cluster_items(self, items, similarity_threshold=0.8):
        """
        Cluster similar items using semantic similarity.

        Args:
            items (list): List of items to cluster
            similarity_threshold (float): Minimum similarity to consider two items as same

        Returns:
            dict: Dictionary mapping canonical names to lists of similar items
        """
        if not items:
            return {}

        items_counter = Counter(items)
        sorted_items = sorted(items_counter.items(), key=lambda x: x[1], reverse=True)

        # Initialize clusters with the most frequent item as first canonical form
        clusters = {}
        processed = set()

        logger.debug(
            f"\nStarting clustering of {len(sorted_items)} unique items with similarity threshold {similarity_threshold}"
        )

        # Use a progress bar that tracks clusters created, not total items
        pbar = tqdm(desc="Clustering progress", unit=" clusters")

        # Process items from most to least frequent
        try:
            for item, count in sorted_items:
                if item in processed:
                    continue

                # Create a new cluster with this item as canonical form
                canonical = item
                clusters[canonical] = [item]
                processed.add(item)

                # Compare to all remaining items
                for other_item, other_count in sorted_items:
                    if other_item in processed:
                        continue

                    # Calculate similarity
                    similarity = self.calculate_similarity(canonical, other_item)

                    if similarity >= similarity_threshold:
                        clusters[canonical].append(other_item)
                        processed.add(other_item)

                # Update progress: increment by 1 for each cluster created
                cluster_size = len(clusters[canonical])
                pbar.set_postfix_str(
                    f"Cluster size: {cluster_size} | Current: {canonical}"
                )
                pbar.update(1)

        finally:
            pbar.close()

        logger.info(
            f"Clustering completed: {len(clusters)} canonical clusters created from {len(sorted_items)} unique items"
        )
        return clusters

    def _extract_with_clustering(self, data_key, sub_key, similarity_threshold=0.8):
        """
        Generic function to extract and cluster data from loaded data.

        Args:
            data_key (str): Key in item_data to look for (e.g., "composition_data", "synthesis_data")
            sub_key (str): Sub-key within data_key (e.g., "family", "precursors", "characterization_techniques")
            similarity_threshold (float): Threshold for similarity-based clustering

        Returns:
            Counter: Counter with canonicalized items
        """
        all_items = []

        # Extract all raw items from the data
        for doi, item_data in self.data.items():
            if data_key in item_data and sub_key in item_data[data_key]:
                items = item_data[data_key][sub_key]
                if items:  # Only add if not empty
                    if isinstance(items, list):
                        all_items.extend(items)
                    else:
                        all_items.append(items)

        if not all_items:
            return Counter()

        # Cluster similar items
        clusters = self.cluster_items(all_items, similarity_threshold)

        # Count canonical items
        canonicalized_counts = Counter()
        raw_counts = Counter(all_items)

        for canonical, similar_items in clusters.items():
            canonicalized_counts[canonical] = sum(
                raw_counts[item] for item in similar_items
            )

        return canonicalized_counts

    def _extract_families_with_clustering(self, similarity_threshold=0.8):
        """Extract material families with semantic clustering."""
        return self._extract_with_clustering(
            "composition_data", "family", similarity_threshold
        )

    def _extract_precursors_with_clustering(self, similarity_threshold=0.8):
        """Extract precursors with semantic clustering."""
        return self._extract_with_clustering(
            "synthesis_data", "precursors", similarity_threshold
        )

    def _extract_characterization_techniques_with_clustering(
        self, similarity_threshold=0.8
    ):
        """Extract characterization techniques with semantic clustering."""
        return self._extract_with_clustering(
            "synthesis_data", "characterization_techniques", similarity_threshold
        )

    def get_clusters(
        self, data_type, data_sources=None, folder_path=None, similarity_threshold=0.8
    ):
        """
        Get clusters of similar items based on semantic similarity.

        Args:
            data_type (str): Type of data to cluster ("families", "precursors", "characterization_techniques")
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
            folder_path (str, optional): Path to folder containing JSON data files
            similarity_threshold (float): Threshold for similarity-based clustering

        Returns:
            Dict[str, List[str]]: Dictionary mapping canonical names to lists of similar items
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Get all items based on data type
        all_items = []

        if data_type == "families":
            data_key, sub_key = "composition_data", "family"
        elif data_type == "precursors":
            data_key, sub_key = "synthesis_data", "precursors"
        elif data_type == "characterization_techniques":
            data_key, sub_key = "synthesis_data", "characterization_techniques"
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        for doi, item_data in self.data.items():
            if data_key in item_data and sub_key in item_data[data_key]:
                items = item_data[data_key][sub_key]
                if items:  # Only add if not empty
                    if isinstance(items, list):
                        all_items.extend(items)
                    else:
                        all_items.append(items)

        if not all_items:
            return {}

        # Return clusters of similar items
        return self.cluster_items(all_items, similarity_threshold)

    # Keep existing methods for backward compatibility
    def get_family_clusters(
        self, data_sources=None, folder_path=None, similarity_threshold=0.8
    ):
        """Get clusters of similar material families."""
        return self.get_clusters(
            "families", data_sources, folder_path, similarity_threshold
        )

    def get_technique_clusters(
        self, data_sources=None, folder_path=None, similarity_threshold=0.8
    ):
        """Get clusters of similar characterization techniques."""
        return self.get_clusters(
            "characterization_techniques",
            data_sources,
            folder_path,
            similarity_threshold,
        )

    def get_precursor_clusters(
        self, data_sources=None, folder_path=None, similarity_threshold=0.8
    ):
        """Get clusters of similar precursors."""
        return self.get_clusters(
            "precursors", data_sources, folder_path, similarity_threshold
        )

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
                    logger.error(f"Error loading {source}: {e}")
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
        figsize=(12, 8),
        dpi=300,
        output_file=None,
        min_percentage=1.0,
        color_palette="Blues",
        title_fontsize=14,
        label_fontsize=10,
        legend_fontsize=10,
    ):
        """
        Create a pie chart visualization of data distribution with percentage labels outside the chart.

        Args:
            data_counter (Counter): Counter object with data labels and frequencies
            title (str): Title for the plot
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved
            min_percentage (float, optional): Minimum percentage for a category to be shown separately (Default: 1.0)
            color_palette (str, optional): Matplotlib colormap name for the pie sections (Default: "Blues")
            title_fontsize (int, optional): Font size for the title (Default: 14)
            label_fontsize (int, optional): Font size for the percentage labels (Default: 10)
            legend_fontsize (int, optional): Font size for the legend (Default: 10)

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
            text.set_fontsize(label_fontsize)

            # Get the position of the text
            pos = text.get_position()
            x, y = pos

            # Calculate angle to determine positioning
            angle = np.arctan2(y, x)
            # Convert to degrees for easier understanding
            angle_deg = np.degrees(angle)

            # Normalize angle to 0-360 range
            if angle_deg < 0:
                angle_deg += 360

            # More precise alignment based on angle ranges
            if 60 <= angle_deg <= 120:  # Top quadrant (60° to 120°)
                text.set_horizontalalignment("center")
                text.set_verticalalignment("bottom")
            elif 240 <= angle_deg <= 300:  # Bottom quadrant (240° to 300°)
                text.set_horizontalalignment("center")
                text.set_verticalalignment("top")
            elif angle_deg <= 30 or angle_deg >= 330:  # Right side (330° to 30°)
                text.set_horizontalalignment("left")
                text.set_verticalalignment("center")
            elif 150 <= angle_deg <= 210:  # Left side (150° to 210°)
                text.set_horizontalalignment("right")
                text.set_verticalalignment("center")
            else:  # Diagonal positions - use adaptive alignment
                if 30 < angle_deg < 60:  # Top-right
                    text.set_horizontalalignment("left")
                    text.set_verticalalignment("bottom")
                elif 120 < angle_deg < 150:  # Top-left
                    text.set_horizontalalignment("right")
                    text.set_verticalalignment("bottom")
                elif 210 < angle_deg < 240:  # Bottom-left
                    text.set_horizontalalignment("right")
                    text.set_verticalalignment("top")
                elif 300 < angle_deg < 330:  # Bottom-right
                    text.set_horizontalalignment("left")
                    text.set_verticalalignment("top")
                else:  # Fallback for any edge cases
                    if x >= 0:
                        text.set_horizontalalignment("left")
                    else:
                        text.set_horizontalalignment("right")
                    text.set_verticalalignment("center")

        # Add legend
        ax.legend(
            wedges,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=legend_fontsize,
        )

        # Set title directly on the axes, centered
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=20)

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_aspect("equal")

        # Adjust layout with title positioning
        plt.tight_layout()

        # Save figure if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            logger.info(f"\nPie chart saved to {output_file}")

        return fig

    def _plot_histogram(
        self,
        data_counter,
        title,
        figsize=(12, 8),
        dpi=300,
        output_file=None,
        max_items=15,
        color_palette="Blues",
        x_label=None,
        y_label="Frequency",
        rotation=45,
        title_fontsize=14,
        xlabel_fontsize=12,
        ylabel_fontsize=12,
        xtick_fontsize=10,
        ytick_fontsize=10,
        value_label_fontsize=9,
        grid_axis="y",
        grid_linestyle="--",
        grid_alpha=0.3,
    ):
        """
        Create a histogram visualization of data distribution.

        Args:
            data_counter (Counter): Counter object with data labels and frequencies
            title (str): Title for the plot
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            output_file (str, optional): Path to save the output plot image (Default: None)
            max_items (int, optional): Maximum number of items to display (Default: 15)
            color_palette (str, optional): Matplotlib colormap name for the bars (Default: "Blues")
            x_label (str, optional): Label for the x-axis (Default: None)
            y_label (str, optional): Label for the y-axis (Default: "Frequency")
            rotation (int, optional): Rotation angle for x-axis labels (Default: 45)
            title_fontsize (int, optional): Font size for the title (Default: 14)
            xlabel_fontsize (int, optional): Font size for the x-axis label (Default: 12)
            ylabel_fontsize (int, optional): Font size for the y-axis label (Default: 12)
            xtick_fontsize (int, optional): Font size for the x-axis tick labels (Default: 10)
            ytick_fontsize (int, optional): Font size for the y-axis tick labels (Default: 10)
            value_label_fontsize (int, optional): Font size for the value labels on bars (Default: 9)
            grid_axis (str, optional): Axis for grid lines ('x', 'y', 'both', or None for no grid) (Default: "y")
            grid_linestyle (str, optional): Line style for grid lines (Default: "--")
            grid_alpha (float, optional): Alpha (transparency) for grid lines (Default: 0.3)

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
                fontsize=value_label_fontsize,
            )

        # Set x-axis labels
        ax.set_xticks(range(len(display_items)))
        ax.set_xticklabels(
            labels, rotation=rotation, ha="right", fontsize=xtick_fontsize
        )

        # Set y-axis tick label font size
        ax.tick_params(axis="y", labelsize=ytick_fontsize)

        # Set axis labels
        if x_label:
            ax.set_xlabel(x_label, fontsize=xlabel_fontsize)
        ax.set_ylabel(y_label, fontsize=ylabel_fontsize)

        # Set title
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=20)

        # Add grid
        if grid_axis and grid_axis.lower() != "none":
            ax.grid(axis=grid_axis, linestyle=grid_linestyle, alpha=grid_alpha)

        # Adjust layout
        plt.tight_layout()

        # Save figure if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            logger.info(f"\nHistogram saved to {output_file}")

        return fig

    def plot_family_pie_chart(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        min_percentage=1.0,
        title="Distribution of Material Families",
        color_palette="Blues",
        title_fontsize=14,
        label_fontsize=10,
        legend_fontsize=10,
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
    ):
        """
        Create a pie chart visualization of material families distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the pie sections (Default: "Blues")
            title_fontsize (int, optional): Font size for the title (Default: 14)
            label_fontsize (int, optional): Font size for the percentage labels (Default: 10)
            legend_fontsize (int, optional): Font size for the legend (Default: 10)
            is_semantic_clustering_enabled (bool, optional): Whether to enable semantic clustering of families (Default: True)
            similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)


        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract families
        if is_semantic_clustering_enabled:
            family_counter = self._extract_families_with_clustering(
                similarity_threshold
            )
            title = f"{title} (Semantically Clustered)"
        else:
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
            title_fontsize,
            label_fontsize,
            legend_fontsize,
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
        color_palette="Blues",
        x_label="Material Family",
        y_label="Frequency",
        rotation=45,
        title_fontsize=14,
        xlabel_fontsize=12,
        ylabel_fontsize=12,
        xtick_fontsize=10,
        ytick_fontsize=10,
        value_label_fontsize=9,
        grid_axis="y",
        grid_linestyle="--",
        grid_alpha=0.3,
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
    ):
        """
        Create a histogram visualization of material families distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            max_items (int, optional): Maximum number of items to display (Default: 15)
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the bars (Default: "Blues")
            x_label (str, optional): Label for the x-axis (Default: "Material Family")
            y_label (str, optional): Label for the y-axis (Default: "Frequency")
            rotation (int, optional): Rotation angle for x-axis labels (Default: 45)
            title_fontsize (int, optional): Font size for the title (Default: 14)
            xlabel_fontsize (int, optional): Font size for the x-axis label (Default: 12)
            ylabel_fontsize (int, optional): Font size for the y-axis label (Default: 12)
            xtick_fontsize (int, optional): Font size for the x-axis tick labels (Default: 10)
            ytick_fontsize (int, optional): Font size for the y-axis tick labels (Default: 10)
            value_label_fontsize (int, optional): Font size for the value labels on bars (Default: 9)
            grid_axis (str, optional): Axis for grid lines ('x', 'y', 'both', or None for no grid) (Default: "y")
            grid_linestyle (str, optional): Line style for grid lines (Default: "--")
            grid_alpha (float, optional): Alpha (transparency) for grid lines (Default: 0.3)
            is_semantic_clustering_enabled (bool, optional): Whether to enable semantic clustering of families (Default: True)
            similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract families
        if is_semantic_clustering_enabled:
            family_counter = self._extract_families_with_clustering(
                similarity_threshold
            )
            title = f"{title} (Semantically Clustered)"
        else:
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
            title_fontsize,
            xlabel_fontsize,
            ylabel_fontsize,
            xtick_fontsize,
            ytick_fontsize,
            value_label_fontsize,
            grid_axis,
            grid_linestyle,
            grid_alpha,
        )

    def plot_precursors_pie_chart(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        min_percentage=1.0,
        title="Distribution of Precursors in Materials Synthesis",
        color_palette="Blues",
        title_fontsize=14,
        label_fontsize=10,
        legend_fontsize=10,
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
    ):
        """
        Create a pie chart visualization of precursors distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the pie sections (Default: "Blues")
            title_fontsize (int, optional): Font size for the title (Default: 14)
            label_fontsize (int, optional): Font size for the percentage labels (Default: 10)
            legend_fontsize (int, optional): Font size for the legend (Default: 10)
            is_semantic_clustering_enabled (bool, optional): Whether to enable semantic clustering of precursors (Default: True)
            similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract precursors
        if is_semantic_clustering_enabled:
            precursors_counter = self._extract_precursors_with_clustering(
                similarity_threshold
            )
            title = f"{title} (Semantically Clustered)"
        else:
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
            title_fontsize,
            label_fontsize,
            legend_fontsize,
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
        title_fontsize=14,
        xlabel_fontsize=12,
        ylabel_fontsize=12,
        xtick_fontsize=10,
        ytick_fontsize=10,
        value_label_fontsize=9,
        grid_axis="y",
        grid_linestyle="--",
        grid_alpha=0.3,
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
    ):
        """
        Create a histogram visualization of precursors distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            max_items (int, optional): Maximum number of items to display (Default: 15)
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the bars (Default: "Blues")
            x_label (str, optional): Label for the x-axis (Default: "Material Family")
            y_label (str, optional): Label for the y-axis (Default: "Frequency")
            rotation (int, optional): Rotation angle for x-axis labels (Default: 45)
            title_fontsize (int, optional): Font size for the title (Default: 14)
            xlabel_fontsize (int, optional): Font size for the x-axis label (Default: 12)
            ylabel_fontsize (int, optional): Font size for the y-axis label (Default: 12)
            xtick_fontsize (int, optional): Font size for the x-axis tick labels (Default: 10)
            ytick_fontsize (int, optional): Font size for the y-axis tick labels (Default: 10)
            value_label_fontsize (int, optional): Font size for the value labels on bars (Default: 9)
            grid_axis (str, optional): Axis for grid lines ('x', 'y', 'both', or None for no grid) (Default: "y")
            grid_linestyle (str, optional): Line style for grid lines (Default: "--")
            grid_alpha (float, optional): Alpha (transparency) for grid lines (Default: 0.3)
            is_semantic_clustering_enabled (bool, optional): Whether to enable semantic clustering of precursors (Default: True)
            similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract precursors
        if is_semantic_clustering_enabled:
            precursors_counter = self._extract_precursors_with_clustering(
                similarity_threshold
            )
            title = f"{title} (Semantically Clustered)"
        else:
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
            title_fontsize,
            xlabel_fontsize,
            ylabel_fontsize,
            xtick_fontsize,
            ytick_fontsize,
            value_label_fontsize,
            grid_axis,
            grid_linestyle,
            grid_alpha,
        )

    def plot_characterization_techniques_pie_chart(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        min_percentage=1.0,
        title="Distribution of Characterization Techniques",
        color_palette="Blues",
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
        title_fontsize=14,
        label_fontsize=10,
        legend_fontsize=10,
    ):
        """
        Create a pie chart visualization of characterization techniques distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the pie sections (Default: "Blues")
            is_semantic_clustering_enabled (bool): Whether to use semantic similarity for clustering similar techniques
            similarity_threshold (float): Threshold for similarity-based clustering when is_semantic_clustering_enabled is True
            title_fontsize (int, optional): Font size for the title (Default: 14)
            label_fontsize (int, optional): Font size for the percentage labels (Default: 10)
            legend_fontsize (int, optional): Font size for the legend labels (Default: 10)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract characterization techniques
        if is_semantic_clustering_enabled:
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
            title_fontsize,
            label_fontsize,
            legend_fontsize,
        )

    def plot_characterization_techniques_histogram(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        max_items=15,
        title="Frequency Distribution of Characterization Techniques",
        color_palette=None,
        x_label="Characterization Technique",
        y_label="Frequency",
        rotation=45,
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
        title_fontsize=14,
        xlabel_fontsize=12,
        ylabel_fontsize=12,
        xtick_fontsize=10,
        ytick_fontsize=10,
        value_label_fontsize=9,
        grid_axis="y",
        grid_linestyle="--",
        grid_alpha=0.3,
    ):
        """
        Create a histogram visualization of characterization techniques distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            max_items (int, optional): Maximum number of items to display (Default: 15)
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the bars (Default: "Blues")
            x_label (str, optional): Label for the x-axis (Default: "Characterization Technique")
            y_label (str, optional): Label for the y-axis (Default: "Frequency")
            rotation (int, optional): Rotation angle for x-axis labels (Default: 45)
            is_semantic_clustering_enabled (bool): Whether to use semantic similarity for clustering similar techniques
            similarity_threshold (float): Threshold for similarity-based clustering when is_semantic_clustering_enabled is True
            title_fontsize (int, optional): Font size for the title (Default: 14)
            xlabel_fontsize (int, optional): Font size for the x-axis label (Default: 12)
            ylabel_fontsize (int, optional): Font size for the y-axis label (Default: 12)
            xtick_fontsize (int, optional): Font size for the x-axis tick labels (Default: 10)
            ytick_fontsize (int, optional): Font size for the y-axis tick labels (Default: 10)
            value_label_fontsize (int, optional): Font size for the value labels on bars (Default: 9)
            grid_axis (str, optional): Axis for grid lines ('x', 'y', 'both', or None for no grid) (Default: "y")
            grid_linestyle (str, optional): Line style for grid lines (Default: "--")
            grid_alpha (float, optional): Alpha (transparency) for grid lines (Default: 0.3)


        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        # Extract characterization techniques
        if is_semantic_clustering_enabled:
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
            title_fontsize,
            xlabel_fontsize,
            ylabel_fontsize,
            xtick_fontsize,
            ytick_fontsize,
            value_label_fontsize,
            grid_axis,
            grid_linestyle,
            grid_alpha,
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
