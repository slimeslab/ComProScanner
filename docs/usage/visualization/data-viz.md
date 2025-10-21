# Data Visualization

Comprehensive visualization tools for extracted composition-property data, synthesis information, and material families.

## Basic Usage

```python
from comproscanner import data_visualizer

# Family pie chart
fig = data_visualizer.plot_family_pie_chart(
    data_sources=["results.json"],
    output_file="families.png"
)

# Knowledge graph
data_visualizer.create_knowledge_graph(
    result_file="results.json"
)
```

## Available Functions

### `plot_family_pie_chart()`

Create a pie chart visualization of material families distribution.

```python
from comproscanner import data_visualizer

fig = data_visualizer.plot_family_pie_chart(
    data_sources=["results.json"],
    output_file="families.png"
)
```

#### Required Parameters

Either `data_sources` OR `folder_path` must be provided.

#### :material-square-medium:`data_sources` _(Union[List[str], List[Dict], str])_

List of paths to JSON files or dictionaries containing materials data.

#### :material-square-medium:`folder_path` _(str)_

Path to folder containing JSON data files.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image. If None, the plot is not saved.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size as (width, height) in inches.

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

#### :material-square-medium:`min_percentage` _(float)_

Minimum percentage for a category to be shown separately. Categories below this threshold are grouped into "Others".

#### :material-square-medium:`title` _(str)_

Title for the plot.

#### :material-square-medium:`color_palette` _(str)_

Matplotlib colormap name for the pie sections.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title.

#### :material-square-medium:`label_fontsize` _(int)_

Font size for the percentage labels.

#### :material-square-medium:`legend_fontsize` _(int)_

Font size for the legend.

#### :material-square-medium:`is_semantic_clustering_enabled` _(bool)_

Whether to use semantic similarity for clustering similar families.

#### :material-square-medium:`similarity_threshold` _(float)_

Similarity threshold for clustering ranging between 0 and 1. Higher values require more similarity for grouping.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`figsize`** = (10, 8)<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`min_percentage`** = 1.0<br>:material-square-small:**`title`** = "Distribution of Material Families"<br>:material-square-small:**`color_palette`** = "Blues"<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`label_fontsize`** = 10<br>:material-square-small:**`legend_fontsize`** = 10<br>:material-square-small:**`is_semantic_clustering_enabled`** = True<br>:material-square-small:**`similarity_threshold`** = 0.8

---

### `plot_family_histogram()`

Create a histogram visualization of material families frequency distribution.

```python
from comproscanner import data_visualizer

fig = data_visualizer.plot_family_histogram(
    data_sources=["results.json"],
    output_file="families_hist.png"
)
```

#### Required Parameters

Either `data_sources` OR `folder_path` must be provided.

#### :material-square-medium:`data_sources` _(Union[List[str], List[Dict], str])_

List of paths to JSON files or dictionaries containing materials data.

#### :material-square-medium:`folder_path` _(str)_

Path to folder containing JSON data files.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image. If None, the plot is not saved.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size as (width, height) in inches.

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

#### :material-square-medium:`max_items` _(int)_

Maximum number of items to display. Shows top N most frequent items.

#### :material-square-medium:`title` _(str)_

Title for the plot.

#### :material-square-medium:`color_palette` _(str)_

Matplotlib colormap name for the bars.

#### :material-square-medium:`x_label` _(str)_

Label for the x-axis.

#### :material-square-medium:`y_label` _(str)_

Label for the y-axis.

#### :material-square-medium:`rotation` _(int)_

Rotation angle for x-axis labels in degrees.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title.

#### :material-square-medium:`xlabel_fontsize` _(int)_

Font size for the x-axis label.

#### :material-square-medium:`ylabel_fontsize` _(int)_

Font size for the y-axis label.

#### :material-square-medium:`xtick_fontsize` _(int)_

Font size for the x-axis tick labels.

#### :material-square-medium:`value_label_fontsize` _(int)_

Font size for the value labels displayed on top of bars.

#### :material-square-medium:`grid_axis` _(str)_

Axis for grid lines. Options: "x", "y", "both", or None for no grid.

#### :material-square-medium:`grid_linestyle` _(str)_

Line style for grid lines (e.g., "--", "-", ":", "-.").

#### :material-square-medium:`grid_alpha` _(float)_

Alpha (transparency) for grid lines ranging between 0 and 1.

#### :material-square-medium:`is_semantic_clustering_enabled` _(bool)_

Whether to enable semantic clustering of families.

#### :material-square-medium:`similarity_threshold` _(float)_

Similarity threshold for clustering which ranges between 0 and 1.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`figsize`** = (12, 8)<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`max_items`** = 15<br>:material-square-small:**`title`** = "Frequency Distribution of Material Families"<br>:material-square-small:**`color_palette`** = "Blues"<br>:material-square-small:**`x_label`** = "Material Family"<br>:material-square-small:**`y_label`** = "Frequency"<br>:material-square-small:**`rotation`** = 45<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`xlabel_fontsize`** = 12<br>:material-square-small:**`ylabel_fontsize`** = 12<br>:material-square-small:**`xtick_fontsize`** = 10<br>:material-square-small:**`value_label_fontsize`** = 9<br>:material-square-small:**`grid_axis`** = "y"<br>:material-square-small:**`grid_linestyle`** = "--"<br>:material-square-small:**`grid_alpha`** = 0.3<br>:material-square-small:**`is_semantic_clustering_enabled`** = True<br>:material-square-small:**`similarity_threshold`** = 0.8

---

### `plot_precursors_pie_chart()`

Create a pie chart visualization of precursors distribution in materials synthesis.

```python
from comproscanner import data_visualizer

fig = data_visualizer.plot_precursors_pie_chart(
    data_sources=["results.json"],
    output_file="precursors_pie.png"
)
```

#### Required Parameters

Either `data_sources` OR `folder_path` must be provided.

#### :material-square-medium:`data_sources` _(Union[List[str], List[Dict], str])_

List of paths to JSON files or dictionaries containing materials data.

#### :material-square-medium:`folder_path` _(str)_

Path to folder containing JSON data files.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image. If None, the plot is not saved.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size as (width, height) in inches.

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

#### :material-square-medium:`min_percentage` _(float)_

Minimum percentage for a category to be shown separately. Categories below this threshold are grouped into "Others".

#### :material-square-medium:`title` _(str)_

Title for the plot.

#### :material-square-medium:`color_palette` _(str)_

Matplotlib colormap name for the pie sections.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title.

#### :material-square-medium:`label_fontsize` _(int)_

Font size for the percentage labels.

#### :material-square-medium:`legend_fontsize` _(int)_

Font size for the legend.

#### :material-square-medium:`is_semantic_clustering_enabled` _(bool)_

Whether to use semantic similarity for clustering similar precursors.

#### :material-square-medium:`similarity_threshold` _(float)_

Threshold for similarity-based clustering when is_semantic_clustering_enabled is True (0-1).

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`figsize`** = (10, 8)<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`min_percentage`** = 1.0<br>:material-square-small:**`title`** = "Distribution of Precursors in Materials Synthesis"<br>:material-square-small:**`color_palette`** = "Blues"<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`label_fontsize`** = 10<br>:material-square-small:**`legend_fontsize`** = 10<br>:material-square-small:**`is_semantic_clustering_enabled`** = True<br>:material-square-small:**`similarity_threshold`** = 0.8

---

### `plot_precursors_histogram()`

Create a histogram visualization of precursors frequency distribution in materials synthesis.

```python
from comproscanner import data_visualizer

fig = data_visualizer.plot_precursors_histogram(
    data_sources=["results.json"],
    output_file="precursors_hist.png"
)
```

#### Required Parameters

Either `data_sources` OR `folder_path` must be provided.

#### :material-square-medium:`data_sources` _(Union[List[str], List[Dict], str])_

List of paths to JSON files or dictionaries containing materials data.

#### :material-square-medium:`folder_path` _(str)_

Path to folder containing JSON data files.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image. If None, the plot is not saved.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size as (width, height) in inches.

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

#### :material-square-medium:`max_items` _(int)_

Maximum number of items to display. Shows top N most frequent items.

#### :material-square-medium:`title` _(str)_

Title for the plot.

#### :material-square-medium:`color_palette` _(str)_

Matplotlib colormap name for the bars.

#### :material-square-medium:`x_label` _(str)_

Label for the x-axis.

#### :material-square-medium:`y_label` _(str)_

Label for the y-axis.

#### :material-square-medium:`rotation` _(int)_

Rotation angle for x-axis labels in degrees.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title.

#### :material-square-medium:`xlabel_fontsize` _(int)_

Font size for the x-axis label.

#### :material-square-medium:`ylabel_fontsize` _(int)_

Font size for the y-axis label.

#### :material-square-medium:`xtick_fontsize` _(int)_

Font size for the x-axis tick labels.

#### :material-square-medium:`value_label_fontsize` _(int)_

Font size for the value labels on bars.

#### :material-square-medium:`grid_axis` _(str)_

Axis for grid lines ('x', 'y', 'both', or None for no grid).

#### :material-square-medium:`grid_linestyle` _(str)_

Line style for grid lines.

#### :material-square-medium:`grid_alpha` _(float)_

Alpha (transparency) for grid lines ranging between 0 and 1.

#### :material-square-medium:`is_semantic_clustering_enabled` _(bool)_

Whether to enable semantic clustering of precursors.

#### :material-square-medium:`similarity_threshold` _(float)_

Similarity threshold for clustering which ranges between 0 and 1.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`figsize`** = (12, 8)<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`max_items`** = 15<br>:material-square-small:**`title`** = "Frequency Distribution of Precursors in Materials Synthesis"<br>:material-square-small:**`color_palette`** = "Blues"<br>:material-square-small:**`x_label`** = "Precursor"<br>:material-square-small:**`y_label`** = "Frequency"<br>:material-square-small:**`rotation`** = 45<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`xlabel_fontsize`** = 12<br>:material-square-small:**`ylabel_fontsize`** = 12<br>:material-square-small:**`xtick_fontsize`** = 10<br>:material-square-small:**`value_label_fontsize`** = 9<br>:material-square-small:**`grid_axis`** = "y"<br>:material-square-small:**`grid_linestyle`** = "--"<br>:material-square-small:**`grid_alpha`** = 0.3<br>:material-square-small:**`is_semantic_clustering_enabled`** = True<br>:material-square-small:**`similarity_threshold`** = 0.8

---

### `plot_characterization_techniques_pie_chart()`

Create a pie chart visualization of characterization techniques distribution.

```python
from comproscanner import data_visualizer

fig = data_visualizer.plot_characterization_techniques_pie_chart(
    data_sources=["results.json"],
    output_file="techniques_pie.png"
)
```

#### Required Parameters

Either `data_sources` OR `folder_path` must be provided.

#### :material-square-medium:`data_sources` _(Union[List[str], List[Dict], str])_

List of paths to JSON files or dictionaries containing materials data.

#### :material-square-medium:`folder_path` _(str)_

Path to folder containing JSON data files.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image. If None, the plot is not saved.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size as (width, height) in inches.

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

#### :material-square-medium:`min_percentage` _(float)_

Minimum percentage for a category to be shown separately.

#### :material-square-medium:`title` _(str)_

Title for the plot.

#### :material-square-medium:`color_palette` _(str)_

Matplotlib colormap name for the pie sections.

#### :material-square-medium:`is_semantic_clustering_enabled` _(bool)_

Whether to use semantic similarity for clustering similar techniques.

#### :material-square-medium:`similarity_threshold` _(float)_

Threshold for similarity-based clustering when is_semantic_clustering_enabled is True ranging between 0 and 1.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title.

#### :material-square-medium:`label_fontsize` _(int)_

Font size for the percentage labels.

#### :material-square-medium:`legend_fontsize` _(int)_

Font size for the legend.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`figsize`** = (10, 8)<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`min_percentage`** = 1.0<br>:material-square-small:**`title`** = "Distribution of Characterization Techniques"<br>:material-square-small:**`color_palette`** = "Blues"<br>:material-square-small:**`is_semantic_clustering_enabled`** = True<br>:material-square-small:**`similarity_threshold`** = 0.8<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`label_fontsize`** = 10<br>:material-square-small:**`legend_fontsize`** = 10

---

### `plot_characterization_techniques_histogram()`

Create a histogram visualization of characterization techniques frequency distribution.

```python
from comproscanner import data_visualizer

fig = data_visualizer.plot_characterization_techniques_histogram(
    data_sources=["results.json"],
    output_file="techniques_hist.png"
)
```

#### Required Parameters

Either `data_sources` OR `folder_path` must be provided.

#### :material-square-medium:`data_sources` _(Union[List[str], List[Dict], str])_

List of paths to JSON files or dictionaries containing materials data.

#### :material-square-medium:`folder_path` _(str)_

Path to folder containing JSON data files.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image. If None, the plot is not saved.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size as (width, height) in inches.

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

#### :material-square-medium:`max_items` _(int)_

Maximum number of items to display. Shows top N most frequent items.

#### :material-square-medium:`title` _(str)_

Title for the plot.

#### :material-square-medium:`color_palette` _(str)_

Matplotlib colormap name for the bars.

#### :material-square-medium:`x_label` _(str)_

Label for the x-axis.

#### :material-square-medium:`y_label` _(str)_

Label for the y-axis.

#### :material-square-medium:`rotation` _(int)_

Rotation angle for x-axis labels in degrees.

#### :material-square-medium:`is_semantic_clustering_enabled` _(bool)_

Whether to use semantic similarity for clustering similar techniques.

#### :material-square-medium:`similarity_threshold` _(float)_

Threshold for similarity-based clustering when is_semantic_clustering_enabled is True ranging between 0 and 1.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title.

#### :material-square-medium:`xlabel_fontsize` _(int)_

Font size for the x-axis label.

#### :material-square-medium:`ylabel_fontsize` _(int)_

Font size for the y-axis label.

#### :material-square-medium:`xtick_fontsize` _(int)_

Font size for the x-axis tick labels.

#### :material-square-medium:`value_label_fontsize` _(int)_

Font size for the value labels on bars.

#### :material-square-medium:`grid_axis` _(str)_

Axis for grid lines ('x', 'y', 'both', or None for no grid).

#### :material-square-medium:`grid_linestyle` _(str)_

Line style for grid lines.

#### :material-square-medium:`grid_alpha` _(float)_

Alpha (transparency) for grid lines ranging between 0 and 1.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`figsize`** = (14, 8)<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`max_items`** = 15<br>:material-square-small:**`title`** = "Frequency Distribution of Characterization Techniques"<br>:material-square-small:**`color_palette`** = "Blues"<br>:material-square-small:**`x_label`** = "Characterization Technique"<br>:material-square-small:**`y_label`** = "Frequency"<br>:material-square-small:**`rotation`** = 45<br>:material-square-small:**`is_semantic_clustering_enabled`** = True<br>:material-square-small:**`similarity_threshold`** = 0.8<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`xlabel_fontsize`** = 12<br>:material-square-small:**`ylabel_fontsize`** = 12<br>:material-square-small:**`xtick_fontsize`** = 10<br>:material-square-small:**`value_label_fontsize`** = 9<br>:material-square-small:**`grid_axis`** = "y"<br>:material-square-small:**`grid_linestyle`** = "--"<br>:material-square-small:**`grid_alpha`** = 0.3

---

### `create_knowledge_graph()`

Create a comprehensive knowledge graph from extracted composition-property data directly in Neo4j database. The knowledge graph visualizes relationships between materials, families, precursors, methods, techniques, and properties.

```python
from comproscanner import data_visualizer

data_visualizer.create_knowledge_graph(
    result_file="results.json"
)
```

#### Required Parameters

#### :material-square-medium:`result_file` _(str)_

Path to the JSON file containing extracted results.

#### Optional Parameters

#### :material-square-medium:`is_semantic_clustering_enabled` _(bool)_

Whether to enable clustering of similar items using semantic similarity.

#### :material-square-medium:`family_clustering_similarity_threshold` _(float)_

Similarity threshold specifically for clustering material families ranging between 0 and 1.

#### :material-square-medium:`method_clustering_similarity_threshold` _(float)_

Similarity threshold specifically for clustering synthesis methods ranging between 0 and 1.

#### :material-square-medium:`precursor_clustering_similarity_threshold` _(float)_

Similarity threshold specifically for clustering precursors ranging between 0 and 1.

#### :material-square-medium:`technique_clustering_similarity_threshold` _(float)_

Similarity threshold specifically for clustering characterization techniques ranging between 0 and 1.

#### :material-square-medium:`keyword_clustering_similarity_threshold` _(float)_

Similarity threshold specifically for clustering keywords ranging between 0 and 1.

!!! info "Default Values"

    :material-square-small:**`is_semantic_clustering_enabled`** = True<br>:material-square-small:**`family_clustering_similarity_threshold`** = 0.9<br>:material-square-small:**`method_clustering_similarity_threshold`** = 0.8<br>:material-square-small:**`precursor_clustering_similarity_threshold`** = 0.9<br>:material-square-small:**`technique_clustering_similarity_threshold`** = 0.8<br>:material-square-small:**`keyword_clustering_similarity_threshold`** = 0.85

!!! warning "Neo4j Database Required"
The knowledge graph is created directly in a Neo4j database. Ensure you have Neo4j running and configured with following credentials in your `.env` file before creating knowledge graphs.

```bash
# neo4j
NEO4J_URI=YOUR_NEO4J_URI # default URI for Neo4j is bolt://localhost:7687
NEO4J_USER=YOUR_NEO4J_USERNAME
NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD
NEO4J_DATABASE=YOUR_NEO4J_DATABASE_NAME
```

---

## Output Format

All visualization functions (except `create_knowledge_graph`) return a `matplotlib.figure.Figure` object that can be viewed interactively:

```python
from comproscanner import data_visualizer

fig = data_visualizer.plot_family_pie_chart(
    data_sources=["results.json"]
)

# Show the plot
fig.show()
```

## Next Steps

- Learn about [Evaluation Visualization](eval-viz.md)
- Explore [Data Extraction](../data-extraction.md)
- Learn about [RAG Configuration](../../rag-config.md)
