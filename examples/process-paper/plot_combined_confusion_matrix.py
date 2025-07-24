from comproscanner import eval_visualizer

eval_visualizer.plot_multiple_confusion_matrices_combined(
    folder_path="../piezo_test/eval-results/agentic-evaluation",
    output_file="plots-raw/model_comparison_confusion_matrix_agentic.png",
    model_names=[
        "DeepSeek-V3-0324",
        "Gemini-2.0-Flash",
        "Gemini-2.5-Flash-Preview",
        "Gemma-3-27B-Instruct",
        "GPT-4.1-Nano",
        "GPT-4o-Mini",
        "Llama-3.3-70B-Instruct",
        "Llama-4-Maverick-17B-Instruct",
        "Qwen-2.5-72B-Instruct",
        "Qwen3-235B-A22B",
    ],
    colormap="RdYlGn",
    label_fontsize=35,
    colorbar_fontsize=24,
    annotation_fontsize=24,
    tick_label_fontsize=24,
    title_pad=40,
    figsize=(24, 20),
)
