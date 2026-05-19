from comproscanner import eval_visualizer

eval_visualizer.plot_multiple_confusion_matrices_combined(
    folder_path="../../vlm_piezo_test/eval-results",
    output_file="../plots-raw/vlm_model_comparison_confusion_matrix.png",
    model_names=[
        "Gemini-2.5-Pro",
        "Gemini-3-Flash-Preview",
        "GPT-5-Chat-Latest",
        "GPT-5.1",
    ],
    metrics_to_include=[
        "overall_composition_accuracy",
        "precision",
        "recall",
        "f1_score",
        "normalized_precision",
        "normalized_recall",
        "normalized_f1_score",
    ],
    colormap="viridis",
    value_range=(0.6, 1),
    label_fontsize=35,
    colorbar_fontsize=24,
    annotation_fontsize=24,
    tick_label_fontsize=24,
    title_pad=40,
    figsize=(24, 12),
)
