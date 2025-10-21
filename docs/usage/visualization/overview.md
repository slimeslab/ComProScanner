# Visualization Overview

ComProScanner provides comprehensive visualization tools for both extracted data and evaluation results.

## Two Visualization Modules

### 1. Data Visualizer

Visualize extracted composition-property data:

- Material family distributions
- Precursor analysis
- Characterization techniques
- Knowledge graphs

[Learn more →](data-viz.md)

### 2. Evaluation Visualizer

Visualize evaluation metrics for both single models and multiple model comparisons:

- Bar charts
- Radar charts
- Heatmaps
- Violin plots
- Histograms

[Learn more →](eval-viz.md)

## Quick Examples

### Data Visualization

```python
from comproscanner import data_visualizer

# Pie chart
fig = data_visualizer.plot_family_pie_chart(
    data_sources=["results.json"],
    output_file="families.png"
)

# Knowledge graph
data_visualizer.create_knowledge_graph(
    result_file="results.json"
)
```

### Evaluation Visualization

```python
from comproscanner import eval_visualizer

# Bar chart for single model
fig = eval_visualizer.plot_single_bar_chart(
    result_file="evaluation.json",
    output_file="metrics.png"
)

# Radar chart comparison for multiple models
fig = eval_visualizer.plot_multiple_radar_charts(
    result_sources=["eval1.json", "eval2.json"],
    model_names=["Model A", "Model B"],
    output_file="comparison.png"
)
```

## Supported Input Sources

=== "Single File"
`python
    plot_function(data_sources=["results.json"])
    `

=== "Multiple Files"
`python
    plot_function(data_sources=["r1.json", "r2.json", "r3.json"])
    `

=== "Folder"
`python
    plot_function(folder_path="results_folder/")
    `

=== "Dictionaries"
`python
    plot_function(data_sources=[data_dict1, data_dict2])
    `

## Next Steps

- [Data Visualization Guide](data-viz.md) - Visualize extracted data
- [Evaluation Visualization Guide](eval-viz.md) - Visualize evaluation results
- Learn about [RAG Configuration](../../rag-config.md)
