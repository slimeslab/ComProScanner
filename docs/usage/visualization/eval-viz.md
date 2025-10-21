# Evaluation Visualization

The evaluation visualization module provides comprehensive tools for visualizing model performance metrics through various chart types including bar charts, radar charts, heatmaps, histograms, and violin plots.

## Basic Usage

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

## Single Model Visualizations

### `plot_single_bar_chart()`

Create a bar chart visualization of evaluation metrics for a single model.

```python
fig = eval_visualizer.plot_single_bar_chart(
    result_file="evaluation.json",
    output_file="metrics.png"
)
```

#### Required Parameters

Either `result_file` OR `result_dict` must be provided.

#### :material-square-medium:`result_file` _(str)_

Path to JSON file containing evaluation results.

#### :material-square-medium:`result_dict` _(dict)_

Dictionary containing evaluation results.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image.

#### :material-square-medium:`model_name` _(str)_

Name of the model for display.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size (width, height) in inches.

#### :material-square-medium:`colormap` _(str)_

Matplotlib colormap name.

#### :material-square-medium:`display_values` _(bool)_

Whether to display metric values on bars.

#### :material-square-medium:`title` _(str)_

Custom title for the plot.

#### :material-square-medium:`typical_threshold` _(float)_

Threshold value to display as horizontal line.

#### :material-square-medium:`threashold_line_style` _(str)_

Style of the threshold line.

#### :material-square-medium:`threashold_tolerance_range` _(float)_

Tolerance range for threshold line to skip the bars for better visibility.

#### :material-square-medium:`threshold_color` _(str)_

Color for the threshold line.

#### :material-square-medium:`show_grid` _(bool)_

Whether to display grid lines.

#### :material-square-medium:`bar_width` _(float)_

Width of the bars.

#### :material-square-medium:`y_axis_label` _(str)_

Label for the y-axis.

#### :material-square-medium:`x_axis_label` _(str)_

Label for the x-axis.

#### :material-square-medium:`y_axis_range` _(Tuple[float, float])_

Range for the y-axis.

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

#### :material-square-medium:`metrics_to_include` _(List[str])_

List of metrics to include.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`model_name`** = None<br>:material-square-small:**`figsize`** = (12, 8)<br>:material-square-small:**`colormap`** = "Blues"<br>:material-square-small:**`display_values`** = True<br>:material-square-small:**`title`** = None<br>:material-square-small:**`typical_threshold`** = None<br>:material-square-small:**`threashold_line_style`** = "--"<br>:material-square-small:**`threashold_tolerance_range`** = 0.03<br>:material-square-small:**`threshold_color`** = "red"<br>:material-square-small:**`show_grid`** = True<br>:material-square-small:**`bar_width`** = 0.6<br>:material-square-small:**`y_axis_label`** = "Score"<br>:material-square-small:**`x_axis_label`** = None<br>:material-square-small:**`y_axis_range`** = (0, 1)<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`metrics_to_include`** = ["overall_accuracy", "overall_composition_accuracy", "overall_synthesis_accuracy", "absolute_precision", "absolute_recall", "absolute_f1_score", "normalized_precision", "normalized_recall", "normalized_f1_score"]

---

### `plot_single_radar_chart()`

Create a radar chart visualization for a single model's evaluation metrics.

```python
fig = eval_visualizer.plot_single_radar_chart(
    result_file="evaluation.json",
    output_file="radar.png"
)
```

#### Required Parameters

Either `result_file` OR `result_dict` must be provided.

#### :material-square-medium:`result_file` _(str)_

Path to JSON file containing evaluation results.

#### :material-square-medium:`result_dict` _(dict)_

Dictionary containing evaluation results.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image.

#### :material-square-medium:`model_name` _(str)_

Name of the model for display.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size (width, height) in inches.

#### :material-square-medium:`colormap` _(str)_

Matplotlib colormap name.

#### :material-square-medium:`display_values` _(bool)_

Whether to display metric values.

#### :material-square-medium:`title` _(str)_

Custom title for the plot.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title.

#### :material-square-medium:`title_pad` _(float)_

Padding for the title from plot.

#### :material-square-medium:`typical_threshold` _(float)_

Threshold value to display as circular line.

#### :material-square-medium:`threshold_color` _(str)_

Color for the threshold line.

#### :material-square-medium:`threshold_line_style` _(str)_

Style of the threshold line.

#### :material-square-medium:`label_fontsize` _(int)_

Font size for axis labels.

#### :material-square-medium:`value_fontsize` _(int)_

Font size for displayed values.

#### :material-square-medium:`legend_loc` _(str)_

Location for the legend box.

#### :material-square-medium:`legend_fontsize` _(int)_

Font size for the legend.

#### :material-square-medium:`bbox_to_anchor` _(Tuple[float, float])_

Bounding box for legend.

#### :material-square-medium:`show_grid` _(bool)_

Whether to display grid lines.

#### :material-square-medium:`show_grid_labels` _(bool)_

Whether to display grid line values/labels.

#### :material-square-medium:`grid_line_width` _(float)_

Width of the grid lines.

#### :material-square-medium:`grid_line_style` _(str)_

Style of the grid lines.

#### :material-square-medium:`grid_line_color` _(str)_

Color of the grid lines.

#### :material-square-medium:`grid_line_alpha` _(float)_

Alpha (transparency) of grid lines ranging from 0 to 1.

#### :material-square-medium:`fill_alpha` _(float)_

Alpha (transparency) of the filled area ranging from 0 to 1.

#### :material-square-medium:`marker_size` _(int)_

Size of the markers in the radar plot.

#### :material-square-medium:`line_width` _(float)_

Width of the lines in the radar plot.

#### :material-square-medium:`label_padding` _(float)_

Padding for the labels in the radar plot.

#### :material-square-medium:`clockwise` _(bool)_

Flag to indicate whether to draw the radar chart in a clockwise direction.

#### :material-square-medium:`start_angle` _(float)_

Starting angle for the radar chart.

#### :material-square-medium:`radar_range` _(Tuple[float, float])_

Range of axes for the radar chart.

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

#### :material-square-medium:`metrics_to_include` _(List[str])_

List of metrics to include.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`model_name`** = None<br>:material-square-small:**`figsize`** = (10, 8)<br>:material-square-small:**`colormap`** = "Blues"<br>:material-square-small:**`display_values`** = False<br>:material-square-small:**`title`** = None<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`title_pad`** = 50.0<br>:material-square-small:**`typical_threshold`** = None<br>:material-square-small:**`threshold_color`** = "red"<br>:material-square-small:**`threshold_line_style`** = "--"<br>:material-square-small:**`label_fontsize`** = 12<br>:material-square-small:**`value_fontsize`** = 10<br>:material-square-small:**`legend_loc`** = "best"<br>:material-square-small:**`legend_fontsize`** = 10<br>:material-square-small:**`bbox_to_anchor`** = None<br>:material-square-small:**`show_grid`** = True<br>:material-square-small:**`show_grid_labels`** = False<br>:material-square-small:**`grid_line_width`** = 1.0<br>:material-square-small:**`grid_line_style`** = "-"<br>:material-square-small:**`grid_line_color`** = "gray"<br>:material-square-small:**`grid_line_alpha`** = 0.2<br>:material-square-small:**`fill_alpha`** = 0.4<br>:material-square-small:**`marker_size`** = 7<br>:material-square-small:**`line_width`** = 2.0<br>:material-square-small:**`label_padding`** = 0.25<br>:material-square-small:**`clockwise`** = True<br>:material-square-small:**`start_angle`** = np.pi / 2<br>:material-square-small:**`radar_range`** = (0, 1)<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`metrics_to_include`** = ["overall_accuracy", "overall_composition_accuracy", "overall_synthesis_accuracy", "absolute_precision", "absolute_recall", "absolute_f1_score", "normalized_precision", "normalized_recall", "normalized_f1_score"]

---

### `plot_single_performance_heatmap()`

Create a heatmap showing the distribution of scores across metrics for a single model.

```python
fig = eval_visualizer.plot_single_performance_heatmap(
    result_file="evaluation.json",
    output_file="heatmap.png"
)
```

#### Required Parameters

Either `result_file` OR `result_dict` must be provided.

#### :material-square-medium:`result_file` _(str)_

Path to JSON file containing evaluation results.

#### :material-square-medium:`result_dict` _(dict)_

Dictionary containing evaluation results.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image.

#### :material-square-medium:`model_name` _(str)_

Name of the model for display.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size (width, height) in inches.

#### :material-square-medium:`colormap` _(str)_

Matplotlib colormap name for heatmap.

#### :material-square-medium:`bin_count` _(int)_

Number of bins to divide the score range into.

#### :material-square-medium:`score_range` _(Tuple[float, float])_

Min and max values for score bins.

#### :material-square-medium:`use_percentage` _(bool)_

Whether to show percentages (True) or counts (False).

#### :material-square-medium:`show_averages` _(bool)_

Whether to show average scores per metric.

#### :material-square-medium:`show_group_labels` _(bool)_

Whether to show metric group labels.

#### :material-square-medium:`show_annotations` _(bool)_

Whether to show value annotations in cells.

#### :material-square-medium:`annotation_format` _(str)_

Format string for annotations (e.g., '.1f' or 'd').

#### :material-square-medium:`title` _(str)_

Custom title for the plot.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title text.

#### :material-square-medium:`title_pad` _(float)_

Padding for the title from the top of the plot.

#### :material-square-medium:`labels` _(List[str])_

Labels for the x and y axes.

#### :material-square-medium:`label_fontsize` _(int)_

Font size for the axis labels.

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

#### :material-square-medium:`group_metrics` _(bool)_

Whether to visually group related metrics together.

#### :material-square-medium:`metric_groups` _(List[Dict])_

Custom metric groups definition for grouping metrics.

#### :material-square-medium:`group_colors` _(List[str])_

Colors for metric groups.

#### :material-square-medium:`metrics_to_include` _(List[str])_

Specific metrics to include in the heatmap.

#### :material-square-medium:`group_label_right_margin` _(int)_

Right margin for group labels.

#### :material-square-medium:`average_value_left_margin` _(int)_

Left margin for average values.

#### :material-square-medium:`plot_padding` _(float)_

Padding between heatmap and axes.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`model_name`** = None<br>:material-square-small:**`figsize`** = (12, 12)<br>:material-square-small:**`colormap`** = "YlGnBu"<br>:material-square-small:**`bin_count`** = 10<br>:material-square-small:**`score_range`** = (0, 1)<br>:material-square-small:**`use_percentage`** = True<br>:material-square-small:**`show_averages`** = False<br>:material-square-small:**`show_group_labels`** = False<br>:material-square-small:**`show_annotations`** = False<br>:material-square-small:**`annotation_format`** = None<br>:material-square-small:**`title`** = None<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`title_pad`** = None<br>:material-square-small:**`labels`** = ["Metrics", "Scores"]<br>:material-square-small:**`label_fontsize`** = 12<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`group_metrics`** = False<br>:material-square-small:**`metric_groups`** = None<br>:material-square-small:**`group_colors`** = None<br>:material-square-small:**`metrics_to_include`** = ["overall_accuracy", "overall_composition_accuracy", "overall_synthesis_accuracy", "precision", "recall", "f1_score", "normalized_precision", "normalized_recall", "normalized_f1_score"]<br>:material-square-small:**`group_label_right_margin`** = 1<br>:material-square-small:**`average_value_left_margin`** = 1<br>:material-square-small:**`plot_padding`** = 0.1

---

### `plot_single_histogram_chart()`

Create a histogram for a single metric from evaluation results.

```python
fig = eval_visualizer.plot_single_histogram_chart(
    result_file="evaluation.json",
    metric_name="overall_accuracy",
    output_file="histogram.png"
)
```

#### Required Parameters

Either `result_file` OR `result_dict` must be provided.

#### :material-square-medium:`result_file` _(str)_

Path to JSON file containing evaluation results.

#### :material-square-medium:`result_dict` _(dict)_

Dictionary containing evaluation results.

#### Optional Parameters

#### :material-square-medium:`metric_name` _(str)_

Name of the metric to plot.

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image.

#### :material-square-medium:`model_name` _(str)_

Name of the model for display in the plot title.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size (width, height) in inches.

#### :material-square-medium:`bins` _(int)_

Number of bins or bin edges for histogram.

#### :material-square-medium:`color` _(str)_

Color for the histogram bars.

#### :material-square-medium:`color_gradient` _(bool)_

Whether to use color gradient for histogram bars.

#### :material-square-medium:`gradient_colors` _(List[str])_

List of colors for gradient.

#### :material-square-medium:`show_kde` _(bool)_

Whether to show a Kernel Density Estimation (KDE) curve over the histogram.

#### :material-square-medium:`show_mean` _(bool)_

Whether to show a vertical line at the mean value.

#### :material-square-medium:`mean_color` _(str)_

Color for the mean line.

#### :material-square-medium:`mean_line_style` _(str)_

Line style for the mean line.

#### :material-square-medium:`show_median` _(bool)_

Whether to show a vertical line at the median value.

#### :material-square-medium:`median_color` _(str)_

Color for the median line.

#### :material-square-medium:`median_line_style` _(str)_

Line style for the median line.

#### :material-square-medium:`show_threshold` _(bool)_

Whether to show a threshold line.

#### :material-square-medium:`threshold_value` _(float)_

Value for the threshold line.

#### :material-square-medium:`threshold_color` _(str)_

Color for the threshold line.

#### :material-square-medium:`threshold_line_style` _(str)_

Line style for the threshold line.

#### :material-square-medium:`title` _(str)_

Custom title for the plot.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title.

#### :material-square-medium:`xlabel` _(str)_

Custom label for x-axis.

#### :material-square-medium:`ylabel` _(str)_

Label for y-axis.

#### :material-square-medium:`xlabel_fontsize` _(int)_

Font size for x-axis label.

#### :material-square-medium:`ylabel_fontsize` _(int)_

Font size for y-axis label.

#### :material-square-medium:`legend_loc` _(str)_

Location for the legend.

#### :material-square-medium:`bbox_to_anchor` _(Tuple[float, float])_

Bounding box for the legend.

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

!!! info "Default Values"

    :material-square-small:**`metric_name`** = "overall_accuracy"<br>:material-square-small:**`output_file`** = None<br>:material-square-small:**`model_name`** = None<br>:material-square-small:**`figsize`** = (8, 6)<br>:material-square-small:**`bins`** = 10<br>:material-square-small:**`color`** = "skyblue"<br>:material-square-small:**`color_gradient`** = False<br>:material-square-small:**`gradient_colors`** = None<br>:material-square-small:**`show_kde`** = False<br>:material-square-small:**`show_mean`** = False<br>:material-square-small:**`mean_color`** = "green"<br>:material-square-small:**`mean_line_style`** = "-"<br>:material-square-small:**`show_median`** = False<br>:material-square-small:**`median_color`** = "black"<br>:material-square-small:**`median_line_style`** = "-"<br>:material-square-small:**`show_threshold`** = False<br>:material-square-small:**`threshold_value`** = 0.8<br>:material-square-small:**`threshold_color`** = "red"<br>:material-square-small:**`threshold_line_style`** = "--"<br>:material-square-small:**`title`** = None<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`xlabel`** = None<br>:material-square-small:**`ylabel`** = "Count"<br>:material-square-small:**`xlabel_fontsize`** = 12<br>:material-square-small:**`ylabel_fontsize`** = 12<br>:material-square-small:**`legend_loc`** = "best"<br>:material-square-small:**`bbox_to_anchor`** = None<br>:material-square-small:**`dpi`** = 300

---

### `plot_single_violin_chart()`

Create a violin plot for all metrics from a single model's evaluation results.

```python
fig = eval_visualizer.plot_single_violin_chart(
    result_file="evaluation.json",
    output_file="violin.png"
)
```

#### Required Parameters

Either `result_file` OR `result_dict` must be provided.

#### :material-square-medium:`result_file` _(str)_

Path to JSON file containing evaluation results.

#### :material-square-medium:`result_dict` _(dict)_

Dictionary containing evaluation results.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image.

#### :material-square-medium:`model_name` _(str)_

Name of the model for display in the plot.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size (width, height) in inches.

#### :material-square-medium:`colormap` _(str)_

Matplotlib colormap name for the violins.

#### :material-square-medium:`title` _(str)_

Custom title for the plot.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title text.

#### :material-square-medium:`title_pad` _(float)_

Padding for the title from the top of the plot.

#### :material-square-medium:`show_box` _(bool)_

Whether to show a box plot inside the violin.

#### :material-square-medium:`show_mean` _(bool)_

Whether to show the mean marker.

#### :material-square-medium:`mean_marker` _(str)_

Marker style for the mean.

#### :material-square-medium:`mean_color` _(str)_

Color for the mean marker.

#### :material-square-medium:`show_median` _(bool)_

Whether to show the median line.

#### :material-square-medium:`median_color` _(str)_

Color for the median line.

#### :material-square-medium:`median_line_style` _(str)_

Line style for the median.

#### :material-square-medium:`show_grid` _(bool)_

Whether to display grid lines.

#### :material-square-medium:`show_threshold` _(bool)_

Whether to show a threshold line.

#### :material-square-medium:`threshold_value` _(float)_

Value for the threshold line.

#### :material-square-medium:`threshold_color` _(str)_

Color for the threshold line.

#### :material-square-medium:`threshold_line_style` _(str)_

Line style for the threshold line.

#### :material-square-medium:`violin_alpha` _(float)_

Alpha (transparency) of the violin plots (0-1).

#### :material-square-medium:`violin_width` _(float)_

Width of the violin plots.

#### :material-square-medium:`x_label` _(str)_

Label for the x-axis.

#### :material-square-medium:`y_label` _(str)_

Label for the y-axis.

#### :material-square-medium:`x_label_fontsize` _(int)_

Font size for x-axis label.

#### :material-square-medium:`y_label_fontsize` _(int)_

Font size for y-axis label.

#### :material-square-medium:`y_axis_range` _(Tuple[float, float])_

Range for the y-axis.

#### :material-square-medium:`label_rotation` _(int)_

Rotation angle for x-axis labels.

#### :material-square-medium:`inner` _(str)_

The representation of the data points inside the violin ('box', 'stick', 'point', or None).

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

#### :material-square-medium:`metrics_to_include` _(List[str])_

Specific metrics to include in the plot.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`model_name`** = None<br>:material-square-small:**`figsize`** = (14, 10)<br>:material-square-small:**`colormap`** = "Blues"<br>:material-square-small:**`title`** = None<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`title_pad`** = 10.0<br>:material-square-small:**`show_box`** = False<br>:material-square-small:**`show_mean`** = True<br>:material-square-small:**`mean_marker`** = "o"<br>:material-square-small:**`mean_color`** = "red"<br>:material-square-small:**`show_median`** = False<br>:material-square-small:**`median_color`** = "green"<br>:material-square-small:**`median_line_style`** = "-"<br>:material-square-small:**`show_grid`** = True<br>:material-square-small:**`show_threshold`** = False<br>:material-square-small:**`threshold_value`** = 0.8<br>:material-square-small:**`threshold_color`** = "red"<br>:material-square-small:**`threshold_line_style`** = "--"<br>:material-square-small:**`violin_alpha`** = 0.7<br>:material-square-small:**`violin_width`** = 0.8<br>:material-square-small:**`x_label`** = "Metrics"<br>:material-square-small:**`y_label`** = "Score"<br>:material-square-small:**`x_label_fontsize`** = 12<br>:material-square-small:**`y_label_fontsize`** = 12<br>:material-square-small:**`y_axis_range`** = (0, 1)<br>:material-square-small:**`label_rotation`** = 45<br>:material-square-small:**`inner`** = "box"<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`metrics_to_include`** = ["overall_accuracy", "overall_composition_accuracy", "overall_synthesis_accuracy", "precision", "recall", "f1_score", "normalized_precision", "normalized_recall", "normalized_f1_score"]

---

## Multi-Model Comparison Visualizations

### `plot_multiple_bar_charts()`

Plot evaluation metrics from multiple result files or dictionaries as grouped bar charts.

```python
fig = eval_visualizer.plot_multiple_bar_charts(
    result_sources=["model1.json", "model2.json"],
    output_file="comparison.png"
)
```

#### Required Parameters

Either `result_sources` OR `folder_path` must be provided.

#### :material-square-medium:`result_sources` _(Union[List[str], List[Dict], str])_

List of JSON file paths or dictionaries containing evaluation results for multiple models.

#### :material-square-medium:`folder_path` _(str)_

Path to folder containing JSON result files.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image.

#### :material-square-medium:`model_names` _(List[str])_

Names of models to display in the legend. Defaults to filename or agent_model_name from results.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size (width, height) in inches.

#### :material-square-medium:`colormap` _(str)_

Matplotlib colormap name for the bars.

#### :material-square-medium:`display_values` _(bool)_

Whether to display metric values on bars.

#### :material-square-medium:`title` _(str)_

Custom title for the plot.

#### :material-square-medium:`typical_threshold` _(float)_

Threshold value to display as horizontal line. If not provided, no line is drawn.

#### :material-square-medium:`threshold_line_style` _(str)_

Style of the threshold line.

#### :material-square-medium:`threashold_tolerance_range` _(float)_

Tolerance range for the threshold line.

#### :material-square-medium:`threshold_color` _(str)_

Color for the threshold line.

#### :material-square-medium:`show_grid` _(bool)_

Whether to display horizontal grid lines in the plot.

#### :material-square-medium:`y_label` _(str)_

Label for the y-axis.

#### :material-square-medium:`x_label` _(str)_

Label for the x-axis.

#### :material-square-medium:`group_width` _(float)_

Width allocated for each group of bars (0-1).

#### :material-square-medium:`bar_width` _(float)_

Width of individual bars. Calculated automatically if None.

#### :material-square-medium:`legend_loc` _(str)_

Location of the legend.

#### :material-square-medium:`legend_fontsize` _(int)_

Font size for the legend.

#### :material-square-medium:`y_axis_range` _(Tuple[float, float])_

Range for the y-axis.

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

#### :material-square-medium:`metrics_to_include` _(List[str])_

List of metrics to include in the plot.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`model_names`** = None<br>:material-square-small:**`figsize`** = (14, 10)<br>:material-square-small:**`colormap`** = "Blues"<br>:material-square-small:**`display_values`** = True<br>:material-square-small:**`title`** = None<br>:material-square-small:**`typical_threshold`** = None<br>:material-square-small:**`threshold_line_style`** = "--"<br>:material-square-small:**`threashold_tolerance_range`** = 0.03<br>:material-square-small:**`threshold_color`** = "red"<br>:material-square-small:**`show_grid`** = True<br>:material-square-small:**`y_label`** = "Score"<br>:material-square-small:**`x_label`** = None<br>:material-square-small:**`group_width`** = 0.8<br>:material-square-small:**`bar_width`** = None<br>:material-square-small:**`legend_loc`** = "best"<br>:material-square-small:**`legend_fontsize`** = 10<br>:material-square-small:**`y_axis_range`** = (0, 1)<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`metrics_to_include`** = ["overall_accuracy", "overall_composition_accuracy", "overall_synthesis_accuracy", "precision", "recall", "f1_score", "normalized_precision", "normalized_recall", "normalized_f1_score"]

---

### `plot_multiple_radar_charts()`

Plot evaluation metrics from multiple result files or dictionaries as a radar chart.

```python
fig = eval_visualizer.plot_multiple_radar_charts(
    result_sources=["model1.json", "model2.json"],
    output_file="radar_comparison.png"
)
```

#### Required Parameters

Either `result_sources` OR `folder_path` must be provided.

#### :material-square-medium:`result_sources` _(Union[List[str], List[Dict], str])_

List of JSON file paths or dictionaries containing evaluation results for multiple models.

#### :material-square-medium:`folder_path` _(str)_

Path to folder containing JSON result files.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image.

#### :material-square-medium:`model_names` _(List[str])_

Names of models to display in the legend.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size (width, height) in inches.

#### :material-square-medium:`colormap` _(str)_

Matplotlib colormap name for the plot lines and markers.

#### :material-square-medium:`display_values` _(bool)_

Whether to display metric values on the chart.

#### :material-square-medium:`title` _(str)_

Custom title for the plot.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title.

#### :material-square-medium:`title_pad` _(float)_

Padding for the title from the top of the plot.

#### :material-square-medium:`typical_threshold` _(float)_

Threshold value to display as circular line.

#### :material-square-medium:`threshold_color` _(str)_

Color for the threshold line.

#### :material-square-medium:`threshold_line_style` _(str)_

Style of the threshold line.

#### :material-square-medium:`label_fontsize` _(int)_

Font size for axis labels.

#### :material-square-medium:`value_fontsize` _(int)_

Font size for displayed values.

#### :material-square-medium:`legend_loc` _(str)_

Location of the legend.

#### :material-square-medium:`bbox_to_anchor` _(Tuple[float, float])_

Bounding box for the legend.

#### :material-square-medium:`legend_fontsize` _(int)_

Font size for the legend.

#### :material-square-medium:`show_grid` _(bool)_

Whether to display grid lines.

#### :material-square-medium:`show_grid_labels` _(bool)_

Whether to display grid line values/labels.

#### :material-square-medium:`grid_line_width` _(float)_

Width of the grid lines.

#### :material-square-medium:`grid_line_style` _(str)_

Style of the grid lines.

#### :material-square-medium:`grid_line_color` _(str)_

Color of the grid lines.

#### :material-square-medium:`grid_line_alpha` _(float)_

Alpha (transparency) of the grid lines (0-1).

#### :material-square-medium:`fill_alpha` _(float)_

Alpha (transparency) of the filled area (0-1).

#### :material-square-medium:`marker_size` _(int)_

Size of the data point markers.

#### :material-square-medium:`line_width` _(float)_

Width of the plot lines.

#### :material-square-medium:`label_padding` _(float)_

Distance padding for axis labels from plot.

#### :material-square-medium:`clockwise` _(bool)_

Direction of the radar chart.

#### :material-square-medium:`start_angle` _(float)_

Start angle in radians.

#### :material-square-medium:`radar_range` _(Tuple[float, float])_

Range for the radar axes.

#### :material-square-medium:`dpi` _(int)_

DPI for output image.

#### :material-square-medium:`metrics_to_include` _(List[str])_

List of metrics to include in the plot.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`model_names`** = None<br>:material-square-small:**`figsize`** = (12, 10)<br>:material-square-small:**`colormap`** = "viridis"<br>:material-square-small:**`display_values`** = False<br>:material-square-small:**`title`** = None<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`title_pad`** = 50.0<br>:material-square-small:**`typical_threshold`** = None<br>:material-square-small:**`threshold_color`** = "red"<br>:material-square-small:**`threshold_line_style`** = "--"<br>:material-square-small:**`label_fontsize`** = 12<br>:material-square-small:**`value_fontsize`** = 10<br>:material-square-small:**`legend_loc`** = "best"<br>:material-square-small:**`bbox_to_anchor`** = None<br>:material-square-small:**`legend_fontsize`** = 10<br>:material-square-small:**`show_grid`** = True<br>:material-square-small:**`show_grid_labels`** = False<br>:material-square-small:**`grid_line_width`** = 1.0<br>:material-square-small:**`grid_line_style`** = "-"<br>:material-square-small:**`grid_line_color`** = "gray"<br>:material-square-small:**`grid_line_alpha`** = 0.2<br>:material-square-small:**`fill_alpha`** = 0.25<br>:material-square-small:**`marker_size`** = 7<br>:material-square-small:**`line_width`** = 2<br>:material-square-small:**`label_padding`** = 0.25<br>:material-square-small:**`clockwise`** = True<br>:material-square-small:**`start_angle`** = np.pi / 2<br>:material-square-small:**`radar_range`** = (0, 1)<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`metrics_to_include`** = ["overall_accuracy", "overall_composition_accuracy", "overall_synthesis_accuracy", "precision", "recall", "f1_score", "normalized_precision", "normalized_recall", "normalized_f1_score"]

---

### `plot_multiple_performance_heatmaps()`

Create a heatmap showing the distribution of scores across metrics for multiple models.

```python
fig = eval_visualizer.plot_multiple_performance_heatmaps(
    result_sources=["model1.json", "model2.json"],
    output_file="heatmaps.png"
)
```

#### Required Parameters

Either `result_sources` OR `folder_path` must be provided.

#### :material-square-medium:`result_sources` _(Union[List[str], List[Dict], str])_

List of JSON file paths or dictionaries containing evaluation results for multiple models.

#### :material-square-medium:`folder_path` _(str)_

Path to folder containing JSON result files.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output visualization.

#### :material-square-medium:`model_names` _(List[str])_

Names to display for models in the plots.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size (width, height) in inches.

#### :material-square-medium:`colormap` _(str)_

Matplotlib colormap name for the heatmap.

#### :material-square-medium:`bin_count` _(int)_

Number of bins to divide the score range into.

#### :material-square-medium:`score_range` _(Tuple[float, float])_

Min and max values for score bins.

#### :material-square-medium:`use_percentage` _(bool)_

Whether to show percentages (True) or counts (False).

#### :material-square-medium:`show_averages` _(bool)_

Whether to show average scores per metric group and model.

#### :material-square-medium:`show_group_labels` _(bool)_

Whether to show metric group labels.

#### :material-square-medium:`show_annotations` _(bool)_

Whether to show value annotations in cells.

#### :material-square-medium:`annotation_format` _(str)_

Format string for annotations (e.g., '.1f' or 'd').

#### :material-square-medium:`title` _(str)_

Custom title for the plot.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title.

#### :material-square-medium:`labels` _(List[str])_

Labels for the x and y axes.

#### :material-square-medium:`label_fontsize` _(int)_

Font size for the axis labels.

#### :material-square-medium:`dpi` _(int)_

Resolution for saved image.

#### :material-square-medium:`group_metrics` _(bool)_

Whether to visually group related metrics.

#### :material-square-medium:`metric_groups` _(List[Dict])_

Custom metric groups definition.

#### :material-square-medium:`group_colors` _(List[str])_

Colors for metric groups.

#### :material-square-medium:`metrics_to_include` _(List[str])_

Specific metrics to include. If None, includes all available.

#### :material-square-medium:`sort_models_by` _(str)_

Metric to sort models by when displaying multiple models.

#### :material-square-medium:`combine_models` _(bool)_

Whether to combine all models into a single distribution plot.

#### :material-square-medium:`group_label_right_margin` _(int)_

Right margin for group labels.

#### :material-square-medium:`average_value_left_margin` _(int)_

Left margin for average values.

#### :material-square-medium:`plot_padding` _(float)_

Padding between heatmap and axes labels and title.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`model_names`** = None<br>:material-square-small:**`figsize`** = (14, 12)<br>:material-square-small:**`colormap`** = "YlGnBu"<br>:material-square-small:**`bin_count`** = 10<br>:material-square-small:**`score_range`** = (0, 1)<br>:material-square-small:**`use_percentage`** = True<br>:material-square-small:**`show_averages`** = False<br>:material-square-small:**`show_group_labels`** = False<br>:material-square-small:**`show_annotations`** = False<br>:material-square-small:**`annotation_format`** = None<br>:material-square-small:**`title`** = None<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`labels`** = ["Metrics", "Scores"]<br>:material-square-small:**`label_fontsize`** = 12<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`group_metrics`** = True<br>:material-square-small:**`metric_groups`** = None<br>:material-square-small:**`group_colors`** = None<br>:material-square-small:**`metrics_to_include`** = ["overall_accuracy", "overall_composition_accuracy", "overall_synthesis_accuracy", "precision", "recall", "f1_score", "normalized_precision", "normalized_recall", "normalized_f1_score"]<br>:material-square-small:**`sort_models_by`** = "overall_accuracy"<br>:material-square-small:**`combine_models`** = False<br>:material-square-small:**`group_label_right_margin`** = 1<br>:material-square-small:**`average_value_left_margin`** = 1<br>:material-square-small:**`plot_padding`** = 0.1

---

### `plot_multiple_confusion_matrices_combined()`

Create a confusion matrix-style heatmap showing all models vs all performance metrics in a single visualization.

```python
fig = eval_visualizer.plot_multiple_confusion_matrices_combined(
    result_sources=["model1.json", "model2.json"],
    output_file="confusion_matrices.png"
)
```

#### Required Parameters

Either `result_sources` OR `folder_path` must be provided.

#### :material-square-medium:`result_sources` _(Union[List[str], List[Dict], str])_

List of JSON file paths or dictionaries containing evaluation results for multiple models.

#### :material-square-medium:`folder_path` _(str)_

Path to folder containing JSON result files.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output visualization.

#### :material-square-medium:`model_names` _(List[str])_

Names to display for models in the plot.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size (width, height) in inches.

#### :material-square-medium:`colormap` _(str)_

Matplotlib colormap name for the heatmap.

#### :material-square-medium:`show_annotations` _(bool)_

Whether to show value annotations in cells.

#### :material-square-medium:`annotation_format` _(str)_

Format string for annotations (e.g., '.2f' or '.1f').

#### :material-square-medium:`annotation_fontsize` _(int)_

Font size for the annotation values inside cells.

#### :material-square-medium:`title` _(str)_

Custom title for the plot.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title.

#### :material-square-medium:`title_pad` _(float)_

Padding for the title from the top of the plot.

#### :material-square-medium:`labels` _(List[str])_

Labels for the x and y axes.

#### :material-square-medium:`label_fontsize` _(int)_

Font size for the axis labels.

#### :material-square-medium:`tick_label_fontsize` _(int)_

Font size for x and y tick labels.

#### :material-square-medium:`dpi` _(int)_

Resolution for saved image.

#### :material-square-medium:`metrics_to_include` _(List[str])_

Specific metrics to include. Default includes all 9 standard metrics.

#### :material-square-medium:`sort_models_by` _(str)_

Metric to sort models by, or "average" for average of all metrics.

#### :material-square-medium:`value_range` _(Tuple[float, float])_

Min and max values for color mapping.

#### :material-square-medium:`show_colorbar` _(bool)_

Whether to show the colorbar legend.

#### :material-square-medium:`colorbar_label` _(str)_

Label for the colorbar.

#### :material-square-medium:`colorbar_fontsize` _(int)_

Font size for colorbar labels.

#### :material-square-medium:`plot_padding` _(float)_

Padding between heatmap and axes labels and title.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`model_names`** = None<br>:material-square-small:**`figsize`** = (14, 10)<br>:material-square-small:**`colormap`** = "YlOrRd"<br>:material-square-small:**`show_annotations`** = True<br>:material-square-small:**`annotation_format`** = None<br>:material-square-small:**`annotation_fontsize`** = 10<br>:material-square-small:**`title`** = None<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`title_pad`** = 20.0<br>:material-square-small:**`labels`** = ["Models", "Metrics"]<br>:material-square-small:**`label_fontsize`** = 12<br>:material-square-small:**`tick_label_fontsize`** = 10<br>:material-square-small:**`dpi`** = 300<br>:material-square-small:**`metrics_to_include`** = ["overall_accuracy", "overall_composition_accuracy", "overall_synthesis_accuracy", "precision", "recall", "f1_score", "normalized_precision", "normalized_recall", "normalized_f1_score"]<br>:material-square-small:**`sort_models_by`** = "average"<br>:material-square-small:**`value_range`** = (0, 1)<br>:material-square-small:**`show_colorbar`** = True<br>:material-square-small:**`colorbar_label`** = "Score"<br>:material-square-small:**`colorbar_fontsize`** = 10<br>:material-square-small:**`plot_padding`** = 0.1

---

### `plot_multiple_histogram_charts()`

Create histograms for a single metric from evaluation results for multiple models.

```python
fig = eval_visualizer.plot_multiple_histogram_charts(
    result_sources=["model1.json", "model2.json"],
    metric_name="overall_accuracy",
    output_file="histograms.png"
)
```

#### Required Parameters

Either `result_sources` OR `folder_path` must be provided.

#### :material-square-medium:`result_sources` _(Union[List[str], List[Dict], str])_

List of JSON file paths or dictionaries containing evaluation results for multiple models.

#### :material-square-medium:`folder_path` _(str)_

Path to folder containing JSON result files.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output plot image.

#### :material-square-medium:`model_names` _(List[str])_

Names of the models for display in the plot titles.

#### :material-square-medium:`metric_name` _(str)_

Name of the metric to plot.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size (width, height) in inches.

#### :material-square-medium:`bins` _(int)_

Number of bins or bin edges for histogram.

#### :material-square-medium:`colormap` _(str)_

Matplotlib colormap name for the histogram colors.

#### :material-square-medium:`show_kde` _(bool)_

Whether to show a KDE curve over the histogram.

#### :material-square-medium:`kde_alpha` _(float)_

Alpha value for the KDE curve.

#### :material-square-medium:`show_mean` _(bool)_

Whether to show a vertical line at the mean value.

#### :material-square-medium:`mean_color` _(str)_

Color for the mean line.

#### :material-square-medium:`mean_line_style` _(str)_

Line style for the mean line.

#### :material-square-medium:`show_median` _(bool)_

Whether to show a vertical line at the median value.

#### :material-square-medium:`median_color` _(str)_

Color for the median line.

#### :material-square-medium:`median_line_style` _(str)_

Line style for the median line.

#### :material-square-medium:`show_threshold` _(bool)_

Whether to show a threshold line.

#### :material-square-medium:`threshold_value` _(float)_

Value for the threshold line.

#### :material-square-medium:`threshold_color` _(str)_

Color for the threshold line.

#### :material-square-medium:`threshold_line_style` _(str)_

Line style for the threshold line.

#### :material-square-medium:`show_grid` _(bool)_

Whether to show grid lines on the plot.

#### :material-square-medium:`title` _(str)_

Custom title for the plot.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title.

#### :material-square-medium:`xlabel` _(str)_

Custom label for x-axis.

#### :material-square-medium:`ylabel` _(str)_

Label for y-axis.

#### :material-square-medium:`xlabel_fontsize` _(int)_

Font size for x-axis label.

#### :material-square-medium:`ylabel_fontsize` _(int)_

Font size for y-axis label.

#### :material-square-medium:`legend_loc` _(str)_

Location for the legend.

#### :material-square-medium:`legend_fontsize` _(int)_

Font size for the legend.

#### :material-square-medium:`bbox_to_anchor` _(str)_

Bounding box for the legend.

#### :material-square-medium:`is_normalized` _(bool)_

Whether to normalize histograms to show percentages.

#### :material-square-medium:`shared_bins` _(bool)_

Whether to use shared bins across all histograms.

#### :material-square-medium:`dpi` _(int)_

DPI for the output image.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`model_names`** = None<br>:material-square-small:**`metric_name`** = "overall_accuracy"<br>:material-square-small:**`figsize`** = (14, 12)<br>:material-square-small:**`bins`** = 10<br>:material-square-small:**`colormap`** = "tab10"<br>:material-square-small:**`show_kde`** = False<br>:material-square-small:**`kde_alpha`** = 0.7<br>:material-square-small:**`show_mean`** = False<br>:material-square-small:**`mean_color`** = "green"<br>:material-square-small:**`mean_line_style`** = "-"<br>:material-square-small:**`show_median`** = False<br>:material-square-small:**`median_color`** = "black"<br>:material-square-small:**`median_line_style`** = "-"<br>:material-square-small:**`show_threshold`** = False<br>:material-square-small:**`threshold_value`** = 0.8<br>:material-square-small:**`threshold_color`** = "red"<br>:material-square-small:**`threshold_line_style`** = "--"<br>:material-square-small:**`show_grid`** = True<br>:material-square-small:**`title`** = None<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`xlabel`** = None<br>:material-square-small:**`ylabel`** = "Count"<br>:material-square-small:**`xlabel_fontsize`** = 12<br>:material-square-small:**`ylabel_fontsize`** = 12<br>:material-square-small:**`legend_loc`** = "best"<br>:material-square-small:**`legend_fontsize`** = 10<br>:material-square-small:**`bbox_to_anchor`** = None<br>:material-square-small:**`is_normalized`** = True<br>:material-square-small:**`shared_bins`** = True<br>:material-square-small:**`dpi`** = 300

---

### `plot_multiple_violin_charts()`

Create violin plots comparing multiple models on a single metric.

```python
fig = eval_visualizer.plot_multiple_violin_charts(
    result_sources=["model1.json", "model2.json"],
    metric_name="overall_accuracy",
    output_file="violins.png"
)
```

#### Required Parameters

Either `result_sources` OR `folder_path` must be provided.

#### :material-square-medium:`result_sources` _(Union[List[str], List[Dict], str])_

List of JSON file paths or dictionaries containing evaluation results for multiple models.

#### :material-square-medium:`folder_path` _(str)_

Path to folder containing JSON result files.

#### Optional Parameters

#### :material-square-medium:`output_file` _(str)_

Path to save the output visualization.

#### :material-square-medium:`model_names` _(List[str])_

Names to display for models in the plot.

#### :material-square-medium:`metric_name` _(str)_

Name of the metric to compare across models.

#### :material-square-medium:`figsize` _(Tuple[int, int])_

Figure size (width, height) in inches.

#### :material-square-medium:`colormap` _(str)_

Matplotlib colormap name for the violins.

#### :material-square-medium:`title` _(str)_

Custom title for the plot.

#### :material-square-medium:`title_fontsize` _(int)_

Font size for the title text.

#### :material-square-medium:`title_pad` _(float)_

Padding for the title from the top of the plot.

#### :material-square-medium:`show_box` _(bool)_

Whether to show a box plot inside the violin.

#### :material-square-medium:`show_mean` _(bool)_

Whether to show the mean marker.

#### :material-square-medium:`mean_marker` _(str)_

Marker style for the mean.

#### :material-square-medium:`mean_color` _(str)_

Color for the mean marker.

#### :material-square-medium:`show_median` _(bool)_

Whether to show the median line.

#### :material-square-medium:`median_color` _(str)_

Color for the median line.

#### :material-square-medium:`median_line_style` _(str)_

Line style for the median.

#### :material-square-medium:`show_grid` _(bool)_

Whether to display grid lines.

#### :material-square-medium:`show_threshold` _(bool)_

Whether to show a threshold line.

#### :material-square-medium:`threshold_value` _(float)_

Value for the threshold line.

#### :material-square-medium:`threshold_color` _(str)_

Color for the threshold line.

#### :material-square-medium:`threshold_line_style` _(str)_

Line style for the threshold line.

#### :material-square-medium:`violin_alpha` _(float)_

Alpha (transparency) of the violin plots (0-1).

#### :material-square-medium:`violin_width` _(float)_

Width of the violin plots.

#### :material-square-medium:`x_label` _(str)_

Label for the x-axis.

#### :material-square-medium:`y_label` _(str)_

Label for the y-axis.

#### :material-square-medium:`x_label_fontsize` _(int)_

Font size for x-axis label.

#### :material-square-medium:`y_label_fontsize` _(int)_

Font size for y-axis label.

#### :material-square-medium:`y_axis_range` _(Tuple[float, float])_

Range for the y-axis.

#### :material-square-medium:`label_rotation` _(int)_

Rotation angle for x-axis labels.

#### :material-square-medium:`inner` _(str)_

The representation of the data points inside the violin ('box', 'stick', 'point', or None).

#### :material-square-medium:`dpi` _(int)_

Resolution for saved image.

!!! info "Default Values"

    :material-square-small:**`output_file`** = None<br>:material-square-small:**`model_names`** = None<br>:material-square-small:**`metric_name`** = "overall_accuracy"<br>:material-square-small:**`figsize`** = (12, 8)<br>:material-square-small:**`colormap`** = "viridis"<br>:material-square-small:**`title`** = None<br>:material-square-small:**`title_fontsize`** = 14<br>:material-square-small:**`title_pad`** = 50.0<br>:material-square-small:**`show_box`** = False<br>:material-square-small:**`show_mean`** = True<br>:material-square-small:**`mean_marker`** = "o"<br>:material-square-small:**`mean_color`** = "red"<br>:material-square-small:**`show_median`** = False<br>:material-square-small:**`median_color`** = "green"<br>:material-square-small:**`median_line_style`** = "-"<br>:material-square-small:**`show_grid`** = True<br>:material-square-small:**`show_threshold`** = False<br>:material-square-small:**`threshold_value`** = 0.8<br>:material-square-small:**`threshold_color`** = "red"<br>:material-square-small:**`threshold_line_style`** = "--"<br>:material-square-small:**`violin_alpha`** = 0.7<br>:material-square-small:**`violin_width`** = 0.8<br>:material-square-small:**`x_label`** = "Models"<br>:material-square-small:**`y_label`** = "Score"<br>:material-square-small:**`x_label_fontsize`** = 12<br>:material-square-small:**`y_label_fontsize`** = 12<br>:material-square-small:**`y_axis_range`** = (0, 1)<br>:material-square-small:**`label_rotation`** = 45<br>:material-square-small:**`inner`** = "box"<br>:material-square-small:**`dpi`** = 300

---

## Next Steps

- Explore [Data Visualization](data-viz.md)
- Learn about [Semantic Evaluation](../evaluation/semantic.md)
- Learn about [RAG Configuration](../../rag-config.md)
