from comproscanner import data_visualizer

data_visualizer.plot_family_pie_chart(
    data_sources=[
        "../../../piezo_test/model-outputs/deepseek/deepseek-v3-piezo-ceramic-test-results.json"
    ],
    output_file="../plots-raw/best_model_family_distribution_pie_chart.png",
    figsize=(20, 15),
    color_palette="Reds",
    min_percentage=1.5,
    label_fontsize=22,
    legend_fontsize=28,
)

data_visualizer.plot_precursors_pie_chart(
    data_sources=[
        "../../../piezo_test/model-outputs/deepseek/deepseek-v3-piezo-ceramic-test-results.json"
    ],
    output_file="../plots-raw/best_model_precursors_distribution_pie_chart.png",
    figsize=(20, 15),
    color_palette="Blues",
    min_percentage=1.25,
    label_fontsize=22,
    legend_fontsize=28,
)

data_visualizer.plot_characterization_techniques_pie_chart(
    data_sources=[
        "../../../piezo_test/model-outputs/deepseek/deepseek-v3-piezo-ceramic-test-results.json"
    ],
    output_file="../plots-raw/best_model_characterization_techniques_distribution_pie_chart.png",
    figsize=(20, 15),
    color_palette="Greens",
    min_percentage=1,
    label_fontsize=22,
    legend_fontsize=28,
    similarity_threshold=0.78,
)
