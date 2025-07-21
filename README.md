<p align="center">
  <img src="assets/comproscanner_logo.png" alt="ComProScanner Logo" width="500"/>
</p>

## ComProScanner: A python package for extracting composition-property data from scientific articles for building databases

[![Python Version](https://img.shields.io/badge/python-3.12-red.svg)](https://www.python.org/downloads/) [![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

ComProScanner is a comprehensive Python package designed to extract composition-property relationships from scientific articles, particularly focused on materials science. It provides tools for metadata collection, article processing from various publishers (Elsevier, Wiley, Springer, IOP) directly if Text and Data Mining (TDM) API keys are provided or from already collected PDFs provided as a folder path, extraction of composition-property data, evaluation of extraction performance, and visualization of results.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Core Modules](#core-modules)
- [Basic Usage](#basic-usage)
  - [Metadata Collection](#metadata-collection)
  - [Article Processing](#article-processing)
  - [Data Extraction](#data-extraction)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
- [Workflow Details](#workflow-details)
  - [Overview Workflow Diagram](#overview-workflow-diagram)
  - [Data Extraction Flow](#data-extraction-flow)
- [Advanced Configuration](#advanced-configuration)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

## Installation

```bash
pip install comproscanner
```

Or install from source:

```bash
git clone https://github.com/aritraroy24/comproscanner.git
cd comproscanner
pip install -e .
```

## Getting Started

The ComProScanner package follows a sequential workflow:

1. Collect and filter metadata from scientific articles related to user's property of interest
2. Process articles from various publishers to extract relevant text
3. Extract composition-property relationships from the processed text
4. Post Processing
   - Evaluate the quality of extraction if needed
   - Visualize the results (both extracted data and evaluation metrics)

#### Full Basic Usage Example:

This example demonstrates a complete workflow for extracting piezoelectric coefficient (d33) data from scientific literature:

```python
from comproscanner import ComProScanner

# Initialize with property of interest
scanner = ComProScanner(main_property_keyword="piezoelectric")

# Collect metadata
scanner.collect_metadata(
    base_queries=["piezoelectric", "piezoelectricity"],
    extra_queries=["materials", "applications"],
)

# Define property keywords for filtering
property_keywords = {
    "exact_keywords": ["d33"],
    "substring_keywords": [" d 33 "]
}

# Process articles from specific sources
scanner.process_articles(
    property_keywords=property_keywords,
    source_list=["elsevier", "springer"]
)

# Extract composition-property relationships
scanner.extract_composition_property_data(
    main_extraction_keyword="d33"
)

# Create knowledge graph from extracted data (optional)
from comproscanner import data_visualizer
data_visualizer.create_knowledge_graph(
    result_file="extracted_results.json"
)

# Evaluate extraction quality (optional)
from comproscanner import evaluate_semantic
evaluate_semantic(
    ground_truth_file="ground_truth.json",
    test_data_file="extracted_results.json",
    output_file="evaluation_results.json"
)

# Visualize results (optional)
from comproscanner import data_visualizer, eval_visualizer

# Plot material families distribution
fig = data_visualizer.plot_family_pie_chart(
    data_sources=["extracted_results.json"],
    output_file="family_distribution.png"
)

# Plot evaluation metrics
fig = eval_visualizer.plot_single_bar_chart(
    result_file="evaluation_results.json",
    output_file="evaluation_metrics.png"
)
```

## Core Modules

ComProScanner is organized into several core modules:

1. **metadata_extractor**: Tools for collecting and filtering article metadata
2. **article_processors**: Processors for different publishers (Elsevier, Wiley, IOP, Springer, PDFs)
3. **extract_flow**: Flow for extracting composition-property and synthesis data
4. **post_processing**: Evaluation and visualization tools
5. **utils**: Configuration, error handling, and logging utilities

## Basic Usage

### Metadata Collection

The `collect_metadata` function finds and filters relevant scientific articles based on the provided search criteria.

```python
from comproscanner import ComProScanner

# Initialize scanner with property keyword
scanner = ComProScanner(main_property_keyword="piezoelectric")

# Collect metadata for articles related to piezoelectric properties
scanner.collect_metadata(
    base_queries=["piezoelectric", "piezoelectricity"],
    extra_queries=["materials", "applications"],
    start_year=2022,
    end_year=2019
)
```

**Parameters:**

- `main_property_keyword` (str): The main property of interest (e.g., "piezoelectric")
- `base_queries` (list, optional): Primary search terms
- `extra_queries` (list, optional): Secondary search terms combined with base queries
- `start_year` (int, optional): Starting year for the search (default: current year)
- `end_year` (int, optional): Ending year for the search (default: current year - 2)

### Article Processing

The `process_articles` function processes articles from various sources to extract relevant text for further analysis.

```python
from comproscanner import ComProScanner

# Initialize scanner with property keyword
scanner = ComProScanner(main_property_keyword="piezoelectric")

# Define property keywords for filtering
property_keywords = {
    "exact_keywords": ["d33"],
    "substring_keywords": [" d 33 "]
}

# Process articles from specific sources
scanner.process_articles(
    property_keywords=property_keywords,
    source_list=["elsevier", "springer"]
)
```

**Parameters:**

- `property_keywords` (dict, required): Dictionary of keywords for filtering sentences
  - `exact_keywords`: List of keywords to match exactly
  - `substring_keywords`: List of keywords to match as substrings
- `source_list` (list, optional): List of sources to process (default: ["elsevier", "wiley", "iop", "springer", "pdfs"])
- `folder_path` (str, optional): Path to folder containing PDFs (required for "pdfs" source)
- `sql_batch_size` (int, optional): Batch size for SQL operations (default: 500)
- `csv_batch_size` (int, optional): Batch size for CSV operations (default: 1)
- `start_row` (int, optional): Start row for processing
- `end_row` (int, optional): End row for processing
- `doi_list` (list, optional): List of DOIs to process
- `is_sql_db` (bool, optional): Whether to use SQL database (default: False)
- `is_save_xml` (bool, optional): Whether to save XML files (default: False)
- `is_save_pdf` (bool, optional): Whether to save PDF files (default: False)
- `rag_db_path` (str, optional): Path to RAG database
- `chunk_size` (int, optional): Size of text chunks for RAG
- `chunk_overlap` (int, optional): Overlap between chunks for RAG
- `embedding_model` (str, optional): Name of embedding model for RAG

### Data Extraction

The `extract_composition_property_data` function extracts composition-property relationships and synthesis data from the processed articles.

```python
from comproscanner import ComProScanner

# Initialize scanner with property keyword
scanner = ComProScanner(main_property_keyword="piezoelectric")

# Extract composition-property relationships
scanner.extract_composition_property_data(
    main_extraction_keyword="d33"
)
```

**Parameters:**

- `main_property_keyword` (str): The main property keyword
- `main_extraction_keyword` (str): The specific property to extract (e.g., "d33")
- `start_row` (int, optional): Start row for processing (default: 0)
- `num_rows` (int, optional): Number of rows to process the articles for (default: all rows)
- `is_test_data_preparation` (bool, optional): A flag to indicate if the test data preparation is required. When True, the function will prepare test data by collecting DOIs with composition-property data (default: False)
- `test_doi_list_file` (str, optional): Path to the file containing the test DOIs. Required if _is_test_data_preparation_ is True. This file will store DOIs that contain composition-property data for testing purposes (default: None)
- `total_test_data` (int, optional): (int, optional): Total number of test data to collect when _is_test_data_preparation_ is True. The function will stop processing once this many DOIs with composition data are found (default: 50)
- `is_only_consider_test_doi_list` (bool, optional): A flag to indicate if only the test DOI list should be considered for processing. Should be set to True if the _test_doi_list_file_ already contains the required number of test DOIs and you want to process only those DOIs (default: False)
- `test_random_seed` (int, optional): Random seed for test data preparation to ensure same DOIs are selected (default: 42)
- `checked_doi_list_file` (str, optional): Path to file containing list of DOIs which have been processed already. Used to avoid reprocessing the same papers (default: "checked_dois.txt")
- `json_results_file` (str, optional): Path to the JSON results file where extracted data will be saved (default: "results.json")
- `csv_results_file` (str, optional): Path to the CSV results file where extracted data will be saved if _is_save_csv_ is True (default: "results.csv")
- `is_save_csv` (bool, optional): A flag to indicate if the results should be saved in CSV format in addition to JSON (default: False)
- `is_extract_synthesis_data` (bool, optional): A flag to indicate if the synthesis data (methods, precursors, characterization techniques) should be extracted along with composition-property data (default: True)
- `is_save_relevant` (bool, optional): A flag to indicate if only papers with composition-property data should be saved. If True, only saves papers that contain composition data. If False, saves all processed papers regardless of whether they contain composition data (default: True)
- `is_data_clean` (bool, optional): A flag to indicate if the extracted data should be cleaned after processing. When True, applies data cleaning strategies to improve data quality (default: False)
- `cleaning_strategy` (str, optional): The cleaning strategy to use when _is_data_clean_ is True. Options are "full" (with periodic element validation) or "basic" (without periodic element validation) (default: "full")
- `materials_data_identifier_query` (str, optional): Custom query to identify if materials data is present in the paper. Must be designed to expect a 'yes/no' answer. If not provided, defaults to a query asking about material chemical composition and the corresponding property value (default: "Is there any material chemical composition and corresponding {main_property_keyword} value mentioned in the paper? GIVE ONE WORD ANSWER. Either yes or no.")
- `model` (str, optional): The LLM model to use for extraction. Supports various providers (OpenAI, Anthropic, Google, etc.) (default: "gpt-4o-mini")
- `api_base` (str, optional): Base URL for standard API endpoints when using custom API services
- `base_url` (str, optional): Base URL for the model service, used for custom or local model deployments
- `api_key` (str, optional): API key for the model service. Can also be set via environment variables for specific providers
- `output_log_folder` (str, optional): Base folder path to save detailed logs for each processed paper. Logs will be saved in _{output_log_folder}/{doi}/_ subdirectory. Logs will be in JSON format if _is_log_json_ is True, otherwise plain text (default: None, no logs saved)
- `is_log_json` (bool, optional): Flag to save logs in JSON format instead of plain text when _output_log_folder_ is specified (default: False)
- `task_output_folder` (str, optional): Base folder path to save task outputs for each processed paper. Task outputs will be saved as .txt files in _{task_output_folder}/{doi}/_ subdirectory (default: None, no task outputs saved)
- `verbose` (bool, optional): Flag to enable verbose output in the terminal during processing (default: True)
- `temperature` (float, optional): Temperature parameter for text generation - controls randomness. Lower values (0.0-0.3) make output more deterministic, higher values (0.7-1.0) make it more creative (default: 0.1)
- `top_p` (float, optional): Nucleus sampling parameter for text generation - controls diversity by considering only the top p probability mass (default: 0.9)
- `timeout` (int, optional): Request timeout in seconds for API calls to the LLM (default: 60)
- `frequency_penalty` (float, optional): Frequency penalty for text generation to reduce repetition. Positive values penalize frequent tokens
- `max_tokens` (int, optional): Maximum number of tokens for LLM completion responses
- `rag_db_path` (str, optional): Path to the vector database used for Retrieval-Augmented Generation (RAG) tool (default: 'db')
- `embedding_model` (str, optional): Name of the embedding model for RAG vector database (default: 'huggingface:thellert/physbert_cased')
- `rag_chat_model` (str, optional): Name of the chat model for RAG responses during extraction (default: 'gpt-4o-mini')
- `rag_max_tokens` (int, optional): Maximum tokens for RAG completion responses (default: 512)
- `rag_top_k` (int, optional): Top k value for RAG retrieval - number of most relevant documents to retrieve (default: 3)
- `rag_base_url` (str, optional): (str, optional): Base URL for the RAG model service when using custom deployments
- `**flow_optional_args`: Optional arguments for the MaterialsFlow class to customize extraction behavior by giving additional notes, examples, and allowed methods/techniques

#### Flow Optional Arguments

The extraction flow can be further customized with these optional arguments:

```python
from comproscanner import ComProScanner

# Initialize scanner with property keyword
scanner = ComProScanner(main_property_keyword="piezoelectric")

# Define flow optional arguments
flow_optional_args = {
    "composition_property_extraction_task_notes": [
        "Write complete chemical formulas (e.g. Ba0.5Sr0.5TiO3)",
        "Use the abbreviation key-value pair to track the abbreviations while extracting composition-property keywords",
        # More notes...
    ],
    "synthesis_extraction_task_notes": [
        "For synthesis_methods, use the short name of the method if possible",
        # More notes...
    ],
    # Other optional arguments...
}

# Pass these to extract_composition_property_data
scanner.extract_composition_property_data(
    main_property_keyword="piezoelectric",
    main_extraction_keyword="d33",
    **flow_optional_args
)
```

Available flow optional arguments:

- `expected_composition_property_example` (str): Example of composition and property data format to guide the extraction agents
- `expected_variable_composition_property_example` (str): Example of variable composition and property data format
- `composition_property_extraction_agent_notes` (list): Additional notes and instructions for the composition extraction agent
- `composition_property_extraction_task_notes` (list): Specific task notes for composition extraction to improve accuracy
- `composition_property_formatting_agent_notes` (list): Notes for the agent responsible for formatting composition data
- `composition_property_formatting_task_notes` (list): Task-specific notes for composition data formatting
- `synthesis_extraction_agent_notes` (list): Additional notes and instructions for the synthesis extraction agent
- `synthesis_extraction_task_notes` (list): Specific task notes for synthesis data extraction
- `synthesis_formatting_agent_notes` (list): Notes for the agent responsible for formatting synthesis data
- `synthesis_formatting_task_notes` (list): Task-specific notes for synthesis data formatting
- `allowed_synthesis_methods` (list): List of allowed synthesis methods for knowledge-graph nodes to ensure consistency
- `allowed_characterization_techniques` (list): List of allowed characterization techniques for knowledge-graph nodes to ensure consistency

### Evaluation

The package provides two methods for evaluating extraction quality:

#### Semantic Evaluation

```python
from comproscanner import evaluate_semantic

results = evaluate_semantic(
    ground_truth_file="ground_truth.json",
    test_data_file="test_results.json",
    used_agent_model_name="gpt-4o-mini"
)
```

**Parameters:**

- `ground_truth_file` (str, required): Path to ground truth data file
- `test_data_file` (str, required): Path to test data file
- `weights` (dict, optional): Weights for evaluation metrics. If not provided, uses default weights:
  ```python
  {
    "compositions_property_values": 0.3,
    "property_unit": 0.1,
    "family": 0.1,
    "method": 0.1,
    "precursors": 0.15,
    "characterization_techniques": 0.15,
    "steps": 0.1
  }
  ```
- `output_file` (str, optional): Path to save evaluation results (default: "semantic_evaluation_result.json")
- `extraction_agent_model_name` (str, optional): Name of the agent model used for extraction (default: "gpt-4o-mini")
- `is_synthesis_evaluation` (bool, optional): Whether to evaluate synthesis extraction (default: True)
- `use_semantic_model` (bool, optional): Whether to use semantic model for evaluation. If False, will use the fallback SequenceMatcher class from difflib library (default: True)
- `primary_model_name` (str, optional): Primary semantic model (default: "thellert/physbert_cased")
- `fallback_model_name` (str, optional): (str, optional): Name of the fallback model for semantic evaluation (default: "all-mpnet-base-v2")
- `similarity_thresholds` (dict, optional): Similarity thresholds for evaluation (default: 0.8 for each metric)

#### Agentic Evaluation

```python
from comproscanner import evaluate_agentic

results = evaluate_agentic(
    ground_truth_file="ground_truth.json",
    test_data_file="test_results.json",
    used_agent_model_name="gpt-4o-mini"
)
```

**Parameters:**

- `ground_truth_file` (str, required): Path to ground truth data file
- `test_data_file` (str, required): Path to test data file
- `extraction_agent_model_name` (str, optional): Name of the agent model used for extraction (default: "gpt-4o-mini")
- `weights` (dict, optional): Weights for evaluation metrics. If not provided, uses default weights:
  ```python
  {
    "compositions_property_values": 0.3,
    "property_unit": 0.1,
    "family": 0.1,
    "method": 0.1,
    "precursors": 0.15,
    "characterization_techniques": 0.15,
    "steps": 0.1
  }
  ```
- `output_file` (str, optional): Path to save evaluation results (default: "agentic_evaluation_result.json")
- `is_synthesis_evaluation` (bool, optional): Whether to evaluate synthesis extraction (default: True)
- `llm` (LLM, optional): An instance of the LLM class. Defaults to instance of LLM with model="o3-mini"

### Visualization

The package provides two visualization modules for creating comprehensive visual analyses of extracted data and evaluation results.

- The data sources can be JSON files or dictionaries containing materials data or a folder containing JSON files. However, at least one of these parameters (`data_sources, folder_path`) must be provided.
- `create_knowledge_graph` takes only a single JSON file as input instead of a list or folder.
- Each function returns a matplotlib figure object except for `create_knowledge_graph` which creates a knowledge graph in Neo4j database.

#### Data Visualization

```python
from comproscanner import data_visualizer

# Plot material families distribution as pie chart
fig = data_visualizer.plot_family_pie_chart(
    data_sources=["results.json"],
    output_file="family_distribution.png"
)

# Create knowledge graph from extracted data
from comproscanner import data_visualizer
data_visualizer.create_knowledge_graph(result_file="results.json")
```

##### Data Visualization Functions

1. ###### plot_family_pie_chart

   Create a pie chart visualization of material families distribution

   **_Parameters_**:

   - `data_sources` (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
   - `folder_path` (str, optional): Path to folder containing JSON data files
   - `output_file` (str, optional): Path to save the output plot image
   - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (10, 8))
   - `dpi` (int, optional): DPI for output image (default: 300)
   - `min_percentage` (float, optional): Minimum percentage for a category to be shown separately (default: 1.0)
   - `title` (str, optional): Title for the plot (default: "Distribution of Material Families")
   - `color_palette` (str, optional): Matplotlib colormap name for the pie sections (default: None)

2. ###### plot_family_histogram

   Create a histogram visualization of material families distribution

   **_Parameters_**:

   - `data_sources` (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
   - `folder_path` (str, optional): Path to folder containing JSON data files
   - `output_file` (str, optional): Path to save the output plot image
   - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (12, 8))
   - `dpi` (int, optional): DPI for output image (default: 300)
   - `max_items` (int, optional): Maximum number of items to display (default: 15)
   - `title` (str, optional): Title for the plot (default: "Frequency Distribution of Material Families")
   - `color_palette` (str, optional): Matplotlib colormap name for the bars (default: None)
   - `x_label` (str, optional): Label for the x-axis (default: "Material Family")
   - `y_label` (str, optional): Label for the y-axis (default: "Frequency")
   - `rotation` (int, optional): Rotation angle for x-axis labels (default: 45)

3. ###### plot_precursors_pie_chart

   Create a pie chart visualization of precursors distribution

   **_Parameters_**:

   - `data_sources` (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
   - `folder_path` (str, optional): Path to folder containing JSON data files
   - `output_file` (str, optional): Path to save the output plot image
   - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (10, 8))
   - `dpi` (int, optional): DPI for output image (default: 300)
   - `min_percentage` (float, optional): Minimum percentage for a category to be shown separately (default: 1.0)
   - `title` (str, optional): Title for the plot (default: "Distribution of Precursors in Materials Synthesis")
   - `color_palette` (str, optional): Matplotlib colormap name for the pie sections (default: None)

4. ###### plot_precursors_histogram

   Create a histogram visualization of precursors distribution

   **_Parameters_**:

   - `data_sources` (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
   - `folder_path` (str, optional): Path to folder containing JSON data files
   - `output_file` (str, optional): Path to save the output plot image
   - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (12, 8))
   - `dpi` (int, optional): DPI for output image (default: 300)
   - `max_items` (int, optional): Maximum number of items to display (default: 15)
   - `title` (str, optional): Title for the plot (default: "Frequency Distribution of Precursors in Materials Synthesis")
   - `color_palette` (str, optional): Matplotlib colormap name for the bars (default: None)
   - `x_label` (str, optional): Label for the x-axis (default: "Precursor")
   - `y_label` (str, optional): Label for the y-axis (default: "Frequency")
   - `rotation` (int, optional): Rotation angle for x-axis labels (default: 45)

5. ###### plot_characterization_techniques_pie_chart

   Create a pie chart visualization of characterization techniques distribution

   **_Parameters_**:

   - `data_sources` (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
   - `folder_path` (str, optional): Path to folder containing JSON data files
   - `output_file` (str, optional): Path to save the output plot image
   - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (10, 8))
   - `dpi` (int, optional): DPI for output image (default: 300)
   - `min_percentage` (float, optional): Minimum percentage for a category to be shown separately (default: 1.0)
   - `title` (str, optional): Title for the plot (default: "Distribution of Characterization Techniques")
   - `color_palette` (str, optional): Matplotlib colormap name for the pie sections (default: None)

6. ###### plot_characterization_techniques_histogram

   Create a histogram visualization of characterization techniques distribution

   **_Parameters_**:

   - `data_sources` (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
   - `folder_path` (str, optional): Path to folder containing JSON data files
   - `output_file` (str, optional): Path to save the output plot image
   - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (14, 8))
   - `dpi` (int, optional): DPI for output image (default: 300)
   - `max_items` (int, optional): Maximum number of items to display (default: 15)
   - `title` (str, optional): Title for the plot (default: "Frequency Distribution of Characterization Techniques")
   - `color_palette` (str, optional): Matplotlib colormap name for the bars (default: None)
   - `x_label` (str, optional): Label for the x-axis (default: "Characterization Technique")
   - `y_label` (str, optional): Label for the y-axis (default: "Frequency")
   - `rotation` (int, optional): Rotation angle for x-axis labels (default: 45)

7. ###### create_knowledge_graph

   Create a knowledge graph from extracted composition-property data directly in Neo4j database

   **_Parameters_**:

   - `result_file` (str, required): Path to the JSON file containing extracted results

   **_Requirements_**:

   - Neo4j database setup with environment variables:
     - `NEO4J_URI`: Neo4j database URI
     - `NEO4J_USER`: Neo4j username
     - `NEO4J_PASSWORD`: Neo4j password
     - `NEO4J_DATABASE` (optional): Database name (default: "neo4j")

#### Evaluation Visualization

```python
from comproscanner import eval_visualizer

# Plot single bar chart for one evaluation result
fig = eval_visualizer.plot_single_bar_chart(
    result_file="evaluation_results.json",
    output_file="evaluation_metrics.png"
)

# Plot radar chart comparing multiple models
fig = eval_visualizer.plot_multiple_radar_charts(
    result_sources=["model1_results.json", "model2_results.json"],
    output_file="model_comparison_radar.png",
    model_names=["Model 1", "Model 2"]
)
```

##### Evaluation Visualization Functions

- For all evaluation functions, the default value for `metrics_to_include` is ['overall_accuracy', 'overall_composition_accuracy', 'overall_synthesis_accuracy', 'absolute_precision', 'absolute_recall', 'absolute_f1_score', 'normalized_precision', 'normalized_recall', 'normalized_f1_score']

1. ###### plot_single_bar_chart

   Create a bar chart visualization for evaluation metrics from a single model

   **_Parameters_**:

   - `result_file` (str, optional): Path to the JSON file containing evaluation results
   - `result_dict` (dict, optional): Dictionary containing evaluation results
   - `output_file` (str, optional): Path to save the output plot image
   - `model_name` (str, optional): Name of the model used for evaluation
   - `figsize` (tuple, optional): Figure size (width, height) in inches (default: (12, 8))
   - `colormap` (str, optional): Matplotlib colormap name (default: "Blues")
   - `display_values` (bool, optional): Whether to display metric values on bars (default: True)
   - `title` (str, optional): Custom title for the plot
   - `typical_threshold` (float, optional): Typical threshold value to display as a horizontal line
   - `threshold_line_style` (str, optional): Style of the threshold line (default: "--")
   - `threshold_tolerance_range` (float, optional): Tolerance range for the threshold line (default: 0.03)
   - `threshold_color` (str, optional): Color for the threshold line (default: "red")
   - `show_grid` (bool, optional): Whether to display horizontal grid lines (default: True)
   - `bar_width` (float, optional): Width of the bars (default: 0.6)
   - `y_axis_label` (str, optional): Label for the y-axis (default: "Score")
   - `x_axis_label` (str, optional): Label for the x-axis
   - `y_axis_range` (tuple, optional): Range for the y-axis (default: (0, 1))
   - `dpi` (int, optional): DPI for output image (default: 300)
   - `metrics_to_include` (list, optional): List of metrics to include in the plot

2. ###### plot_multiple_bar_charts

   Create grouped bar charts for evaluation metrics from multiple models

   **_Parameters_**:

   - `result_sources` (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
   - `folder_path` (str, optional): Path to folder containing JSON result files
   - `output_file` (str, optional): Path to save the output plot image
   - `model_names` (List[str], optional): Names of models to display in the legend
   - `figsize` (tuple, optional): Figure size (width, height) in inches (default: (14, 10))
   - `colormap` (str, optional): Matplotlib colormap name for the bars (default: "Blues")
   - `display_values` (bool, optional): Whether to display metric values on bars (default: True)
   - `title` (str, optional): Custom title for the plot
   - `typical_threshold` (float, optional): Typical threshold value to display as a horizontal line
   - `threshold_line_style` (str, optional): Style of the threshold line (default: "--")
   - `threshold_tolerance_range` (float, optional): Tolerance range for the threshold line (default: 0.03)
   - `threshold_color` (str, optional): Color for the threshold line (default: "red")
   - `show_grid` (bool, optional): Whether to display horizontal grid lines (default: True)
   - `y_label` (str, optional): Label for the y-axis (default: "Score")
   - `x_label` (str, optional): Label for the x-axis
   - `group_width` (float, optional): Width allocated for each group of bars (default: 0.8)
   - `bar_width` (float, optional): Width of individual bars
   - `legend_loc` (str, optional): Location of the legend (default: "best")
   - `legend_fontsize` (int, optional): Font size for the legend (default: 10)
   - `y_axis_range` (tuple, optional): Range for the y-axis (default: (0, 1))
   - `dpi` (int, optional): DPI for output image (default: 300)
   - `metrics_to_include` (list, optional): List of metrics to include in the plot

3. ###### plot_single_radar_chart

   Create a radar chart visualization for evaluation metrics from a single model

   **_Parameters_**:

   - `result_file` (str, optional): Path to JSON file containing evaluation results
   - `result_dict` (dict, optional): Dictionary containing evaluation results
   - `output_file` (str, optional): Path to save the output plot image
   - `model_name` (str, optional): Name of the model for display
   - `figsize` (tuple, optional): Figure size (width, height) in inches (default: (10, 8))
   - `colormap` (str, optional): Matplotlib colormap name (default: "Blues")
   - `display_values` (bool, optional): Whether to display metric values on chart (default: False)
   - `title` (str, optional): Custom title for the plot
   - `title_fontsize` (int, optional): Font size for the title (default: 14)
   - `title_pad` (float, optional): Padding for the title from the top (default: 50.0)
   - `typical_threshold` (float, optional): Threshold value to display as a circular line
   - `threshold_color` (str, optional): Color for the threshold line (default: "red")
   - `threshold_line_style` (str, optional): Style of the threshold line (default: "--")
   - `label_fontsize` (int, optional): Font size for axis labels (default: 12)
   - `value_fontsize` (int, optional): Font size for displayed values (default: 10)
   - `legend_loc` (str, optional): Location for the legend box (default: "best")
   - `legend_fontsize` (int, optional): Font size for the legend (default: 10)
   - `bbox_to_anchor` (tuple, optional): Bounding box for the legend box
   - `show_grid` (bool, optional): Whether to display the grid lines (default: True)
   - `show_grid_labels` (bool, optional): Whether to display grid line values/labels (default: False)
   - `grid_line_width` (float, optional): Width of the grid lines (default: 1.0)
   - `grid_line_style` (str, optional): Style of the grid lines (default: "-")
   - `grid_line_color` (str, optional): Color of the grid lines (default: "gray")
   - `grid_line_alpha` (float, optional): Alpha (transparency) of the grid lines (default: 0.2)
   - `fill_alpha` (float, optional): Alpha (transparency) of the filled area (default: 0.4)
   - `marker_size` (int, optional): Size of the data point markers (default: 7)
   - `line_width` (float, optional): Width of the plot lines (default: 2)
   - `label_padding` (float, optional): Distance padding for axis labels from plot (default: 0.25)
   - `clockwise` (bool, optional): Direction of the radar chart (default: True)
   - `start_angle` (float, optional): Start angle in radians (default: np.pi / 2)
   - `radar_range` (tuple, optional): Range for the radar axes (default: (0, 1))
   - `dpi` (int, optional): DPI for the output image (default: 300)
   - `metrics_to_include` (list, optional): List of metrics to include

4. ###### plot_multiple_radar_charts

   Create radar charts comparing evaluation metrics from multiple models

   **_Parameters_**:

   - `result_sources` (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
   - `folder_path` (str, optional): Path to folder containing JSON result files
   - `output_file` (str, optional): Path to save the output plot image
   - `model_names` (List[str], optional): Names of models to display in the legend
   - `figsize` (tuple, optional): Figure size (width, height) in inches (default: (12, 10))
   - `colormap` (str, optional): Matplotlib colormap name for the plot lines and markers (default: "viridis")
   - `display_values` (bool, optional): Whether to display metric values on the chart (default: False)
   - `title` (str, optional): Custom title for the plot
   - `title_fontsize` (int, optional): Font size for the title (default: 14)
   - `title_pad` (float, optional): Padding for the title from the top (default: 50.0)
   - `typical_threshold` (float, optional): Typical threshold value to display as a circular line
   - `threshold_color` (str, optional): Color for the threshold line (default: "red")
   - `threshold_line_style` (str, optional): Style of the threshold line (default: "--")
   - `label_fontsize` (int, optional): Font size for axis labels (default: 12)
   - `value_fontsize` (int, optional): Font size for displayed values (default: 10)
   - `legend_loc` (str, optional): Location of the legend (default: "best")
   - `bbox_to_anchor` (tuple, optional): Bounding box for the legend
   - `legend_fontsize` (int, optional): Font size for the legend (default: 10)
   - `show_grid` (bool, optional): Whether to display the grid lines (default: True)
   - `show_grid_labels` (bool, optional): Whether to display grid line values/labels (default: False)
   - `grid_line_width` (float, optional): Width of the grid lines (default: 1.0)
   - `grid_line_style` (str, optional): Style of the grid lines (default: "-")
   - `grid_line_color` (str, optional): Color of the grid lines (default: "gray")
   - `grid_line_alpha` (float, optional): Alpha (transparency) of the grid lines (default: 0.2)
   - `fill_alpha` (float, optional): Alpha (transparency) of the filled area (default: 0.25)
   - `marker_size` (int, optional): Size of the data point markers (default: 7)
   - `line_width` (float, optional): Width of the plot lines (default: 2)
   - `label_padding` (float, optional): Distance padding for axis labels from plot (default: 0.25)
   - `clockwise` (bool, optional): Direction of the radar chart (default: True)
   - `start_angle` (float, optional): Start angle in radians (default: np.pi / 2)
   - `radar_range` (tuple, optional): Range for the radar axes (default: (0, 1))
   - `dpi` (int, optional): DPI for the output image (default: 300)
   - `metrics_to_include` (list, optional): List of metrics to include in the plot

5. ###### plot_single_performance_heatmap

   Create a heatmap showing the distribution of scores across metrics for a single model

   **_Parameters_**:

   - `result_file` (str, optional): Path to JSON file containing evaluation results for the model
   - `result_dict` (dict, optional): Dictionary containing evaluation results for the model
   - `output_file` (str, optional): Path to save the output visualization
   - `model_name` (str, optional): Name to display for the model in the plot
   - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (12, 12))
   - `colormap` (str, optional): Matplotlib colormap name for the heatmap (default: "YlGnBu")
   - `bin_count` (int, optional): Number of bins to divide the score range into (default: 10)
   - `score_range` (tuple, optional): Min and max values for score bins (default: (0, 1))
   - `use_percentage` (bool, optional): Whether to show percentages (True) or counts (False) (default: True)
   - `show_averages` (bool, optional): Whether to show average scores per metric (default: False)
   - `show_group_labels` (bool, optional): Whether to show metric group labels (default: False)
   - `show_annotations` (bool, optional): Whether to show value annotations in cells (default: False)
   - `annotation_format` (str, optional): Format string for annotations (e.g., '.1f' or 'd')
   - `title` (str, optional): Custom title for the plot
   - `title_fontsize` (int, optional): Font size for the title text (default: 14)
   - `title_pad` (float, optional): Padding for the title from the top of the plot
   - `labels` (list, optional): Labels for the x and y axes (default: ["Metrics", "Scores"])
   - `label_fontsize` (int, optional): Font size for the axis labels (default: 12)
   - `dpi` (int, optional): Resolution for saved image (default: 300)
   - `group_metrics` (bool, optional): Whether to visually group related metrics together (default: False)
   - `metric_groups` (list, optional): Custom metric groups definition for grouping metrics
   - `group_colors` (list, optional): Colors for metric groups
   - `metrics_to_include` (list, optional): Specific metrics to include in the heatmap
   - `group_label_right_margin` (int, optional): Right margin for group labels (default: 1)
   - `average_value_left_margin` (int, optional): Left margin for average values (default: 1)
   - `plot_padding` (float, optional): Padding between heatmap and axes (default: 0.1)

6. ###### plot_multiple_performance_heatmaps

   Create a heatmap showing the distribution of scores across metrics for multiple models

   **_Parameters_**:

   - `result_sources` (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
   - `folder_path` (str, optional): Path to folder containing JSON result files
   - `output_file` (str, optional): Path to save the output visualization
   - `model_names` (List[str], optional): Names to display for models in the plots
   - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (14, 12))
   - `colormap` (str, optional): Matplotlib colormap name for the heatmap (default: "YlGnBu")
   - `bin_count` (int, optional): Number of bins to divide the score range into (default: 10)
   - `score_range` (tuple, optional): Min and max values for score bins (default: (0, 1))
   - `use_percentage` (bool, optional): Whether to show percentages (True) or counts (False) (default: True)
   - `show_averages` (bool, optional): Whether to show average scores per metric group and model (default: False)
   - `show_group_labels` (bool, optional): Whether to show metric group labels (default: False)
   - `show_annotations` (bool, optional): Whether to show value annotations in cells (default: False)
   - `annotation_format` (str, optional): Format string for annotations (e.g., '.1f' or 'd')
   - `title` (str, optional): Custom title for the plot
   - `title_fontsize` (int, optional): Font size for the title (default: 14)
   - `labels` (list, optional): Labels for the x and y axes (default: ["Metrics", "Scores"])
   - `label_fontsize` (int, optional): Font size for the axis labels (default: 12)
   - `dpi` (int, optional): Resolution for saved image (default: 300)
   - `group_metrics` (bool, optional): Whether to visually group related metrics (default: True)
   - `metric_groups` (list, optional): Custom metric groups definition
   - `group_colors` (list, optional): Colors for metric groups
   - `metrics_to_include` (list, optional): Specific metrics to include
   - `sort_models_by` (str, optional): Metric to sort models by when displaying multiple models (default: "overall_accuracy")
   - `combine_models` (bool, optional): Whether to combine all models into a single distribution plot (default: False)
   - `group_label_right_margin` (int, optional): Right margin for group labels (default: 1)
   - `average_value_left_margin` (int, optional): Left margin for average values (default: 1)
   - `plot_padding` (float, optional): Padding between heatmap and axes labels and title (default: 0.1)

7. ###### plot_multiple_confusion_matrices_combined

   Create a confusion matrix-style heatmap showing all models vs all performance metrics in a single visualization

   **_Parameters_**:

   - `result_sources` (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
   - `folder_path` (str, optional): Path to folder containing JSON result files
   - `output_file` (str, optional): Path to save the output visualization
   - `model_names` (List[str], optional): Names to display for models in the plot
   - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (14, 10))
   - `colormap` (str, optional): Matplotlib colormap name for the heatmap (default: "YlOrRd")
   - `show_annotations` (bool, optional): Whether to show value annotations in cells (default: True)
   - `annotation_format` (str, optional): Format string for annotations (e.g., '.2f' or '.1f')
   - `annotation_fontsize` (int, optional): Font size for the annotation values inside cells (default: 10)
   - `title` (str, optional): Custom title for the plot
   - `title_fontsize` (int, optional): Font size for the title (default: 14)
   - `title_pad` (float, optional): Padding for the title from the top of the plot (default: 20.0)
   - `labels` (list, optional): Labels for the x and y axes (default: ["Models", "Metrics"])
   - `label_fontsize` (int, optional): Font size for the axis labels (default: 12)
   - `tick_label_fontsize` (int, optional): Font size for x and y tick labels (default: 10)
   - `dpi` (int, optional): Resolution for saved image (default: 300)
   - `metrics_to_include` (list, optional): Specific metrics to include (default: all 9 standard metrics)
   - `sort_models_by` (str, optional): Metric to sort models by, or "average" for average of all metrics (default: "average")
   - `value_range` (tuple, optional): Min and max values for color mapping (default: (0, 1))
   - `show_colorbar` (bool, optional): Whether to show the colorbar legend (default: True)
   - `colorbar_label` (str, optional): Label for the colorbar (default: "Score")
   - `colorbar_fontsize` (int, optional): Font size for colorbar labels (default: 10)
   - `plot_padding` (float, optional): Padding between heatmap and axes labels and title (default: 0.1)

8. ###### plot_single_histogram_chart

   Create a histogram for a single metric from evaluation results

   **_Parameters_**:

   - `result_file` (str, optional): Path to JSON file containing evaluation results
   - `result_dict` (dict, optional): Dictionary containing evaluation results
   - `metric_name` (str, optional): Name of the metric to plot (default: "overall_accuracy")
   - `output_file` (str, optional): Path to save the output plot image
   - `model_name` (str, optional): Name of the model for display in the plot title
   - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (8, 6))
   - `bins` (int or sequence, optional): Number of bins or bin edges for histogram (default: 10)
   - `color` (str, optional): Color for the histogram bars (default: "skyblue")
   - `color_gradient` (bool, optional): Whether to use color gradient for histogram bars (default: False)
   - `gradient_colors` (list, optional): List of colors for gradient
   - `show_kde` (bool, optional): Whether to show a KDE curve over the histogram (default: False)
   - `show_mean` (bool, optional): Whether to show a vertical line at the mean value (default: False)
   - `mean_color` (str, optional): Color for the mean line (default: "green")
   - `mean_line_style` (str, optional): Line style for the mean line (default: "-")
   - `show_median` (bool, optional): Whether to show a vertical line at the median value (default: False)
   - `median_color` (str, optional): Color for the median line (default: "black")
   - `median_line_style` (str, optional): Line style for the median line (default: "-")
   - `show_threshold` (bool, optional): Whether to show a threshold line (default: False)
   - `threshold_value` (float, optional): Value for the threshold line (default: 0.8)
   - `threshold_color` (str, optional): Color for the threshold line (default: "red")
   - `threshold_line_style` (str, optional): Line style for the threshold line (default: "--")
   - `title` (str, optional): Custom title for the plot
   - `title_fontsize` (int, optional): Font size for the title (default: 14)
   - `xlabel` (str, optional): Custom label for x-axis
   - `ylabel` (str, optional): Label for y-axis (default: "Count")
   - `xlabel_fontsize` (int, optional): Font size for x-axis label (default: 12)
   - `ylabel_fontsize` (int, optional): Font size for y-axis label (default: 12)
   - `legend_loc` (str, optional): Location for the legend (default: "best")
   - `bbox_to_anchor` (tuple, optional): Bounding box for the legend
   - `dpi` (int, optional): DPI for the output image (default: 300)

9. ###### plot_multiple_histogram_charts

   Create histograms for a single metric from evaluation results for multiple models

   **_Parameters_**:

   - `result_sources` (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
   - `folder_path` (str, optional): Path to folder containing JSON result files
   - `output_file` (str, optional): Path to save the output plot image
   - `model_names` (List[str], optional): Names of the models for display in the plot titles
   - `metric_name` (str, optional): Name of the metric to plot (default: "overall_accuracy")
   - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (14, 12))
   - `bins` (int, optional): Number of bins or bin edges for histogram (default: 10)
   - `colormap` (str, optional): Matplotlib colormap name for the histogram colors (default: "tab10")
   - `show_kde` (bool, optional): Whether to show a KDE curve over the histogram (default: False)
   - `kde_alpha` (float, optional): Alpha value for the KDE curve (default: 0.7)
   - `show_mean` (bool, optional): Whether to show a vertical line at the mean value (default: False)
   - `mean_color` (str, optional): Color for the mean line (default: "green")
   - `mean_line_style` (str, optional): Line style for the mean line (default: "-")
   - `show_median` (bool, optional): Whether to show a vertical line at the median value (default: False)
   - `median_color` (str, optional): Color for the median line (default: "black")
   - `median_line_style` (str, optional): Line style for the median line (default: "-")
   - `show_threshold` (bool, optional): Whether to show a threshold line (default: False)
   - `threshold_value` (float, optional): Value for the threshold line (default: 0.8)
   - `threshold_color` (str, optional): Color for the threshold line (default: "red")
   - `threshold_line_style` (str, optional): Line style for the threshold line (default: "--")
   - `show_grid` (bool, optional): Whether to show grid lines on the plot (default: True)
   - `title` (str, optional): Custom title for the plot
   - `title_fontsize` (int, optional): Font size for the title (default: 14)
   - `xlabel` (str, optional): Custom label for x-axis
   - `ylabel` (str, optional): Label for y-axis (default: "Count")
   - `xlabel_fontsize` (int, optional): Font size for x-axis label (default: 12)
   - `ylabel_fontsize` (int, optional): Font size for y-axis label (default: 12)
   - `legend_loc` (str, optional): Location for the legend (default: "best")
   - `legend_fontsize` (int, optional): Font size for the legend (default: 10)
   - `bbox_to_anchor` (tuple, optional): Bounding box for the legend
   - `is_normalized` (bool, optional): Whether to normalize histograms to show percentages (default: True)
   - `shared_bins` (bool, optional): Whether to use shared bins across all histograms (default: True)
   - `dpi` (int, optional): DPI for the output image (default: 300)

10. ###### plot_single_violin_chart

    Create a violin plot for all metrics from a single model's evaluation results

    **_Parameters_**:

    - `result_file` (str, optional): Path to JSON file containing evaluation results
    - `result_dict` (dict, optional): Dictionary containing evaluation results
    - `output_file` (str, optional): Path to save the output visualization
    - `model_name` (str, optional): Name to display for the model in the plot
    - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (14, 10))
    - `colormap` (str, optional): Matplotlib colormap name for the violins (default: "Blues")
    - `title` (str, optional): Custom title for the plot
    - `title_fontsize` (int, optional): Font size for the title text (default: 14)
    - `title_pad` (float, optional): Padding for the title from the top of the plot (default: 10.0)
    - `show_box` (bool, optional): Whether to show a box plot inside the violin (default: False)
    - `show_mean` (bool, optional): Whether to show the mean marker (default: True)
    - `mean_marker` (str, optional): Marker style for the mean (default: "o")
    - `mean_color` (str, optional): Color for the mean marker (default: "red")
    - `show_median` (bool, optional): Whether to show the median line (default: False)
    - `median_color` (str, optional): Color for the median line (default: "green")
    - `median_line_style` (str, optional): Line style for the median (default: "-")
    - `show_grid` (bool, optional): Whether to display the grid lines (default: True)
    - `show_threshold` (bool, optional): Whether to show a threshold line (default: False)
    - `threshold_value` (float, optional): Value for the threshold line (default: 0.8)
    - `threshold_color` (str, optional): Color for the threshold line (default: "red")
    - `threshold_line_style` (str, optional): Line style for the threshold line (default: "--")
    - `violin_alpha` (float, optional): Alpha (transparency) of the violin plots (default: 0.7)
    - `violin_width` (float, optional): Width of the violin plots (default: 0.8)
    - `x_label` (str, optional): Label for the x-axis (default: "Models")
    - `y_label` (str, optional): Label for the y-axis (default: "Score")
    - `x_label_fontsize` (int, optional): Font size for x-axis label (default: 12)
    - `y_label_fontsize` (int, optional): Font size for y-axis label (default: 12)
    - `y_axis_range` (tuple, optional): Range for the y-axis (default: (0, 1))
    - `label_rotation` (int, optional): Rotation angle for x-axis labels (default: 45)
    - `inner` (str, optional): The representation of the data points inside the violin (default: "box")
    - `dpi` (int, optional): Resolution for saved image (default: 300)median_line_style` (str, optional): Line style for the median (default: "-")
    - `show_grid` (bool, optional): Whether to display the grid lines (default: True)
    - `show_threshold` (bool, optional): Whether to show a threshold line (default: False)
    - `threshold_value` (float, optional): Value for the threshold line (default: 0.8)
    - `threshold_color` (str, optional): Color for the threshold line (default: "red")
    - `threshold_line_style` (str, optional): Line style for the threshold line (default: "--")
    - `violin_alpha` (float, optional): Alpha (transparency) of the violin plots (default: 0.7)
    - `violin_width` (float, optional): Width of the violin plots (default: 0.8)
    - `x_label` (str, optional): Label for the x-axis (default: "Metrics")
    - `y_label` (str, optional): Label for the y-axis (default: "Score")
    - `x_label_fontsize` (int, optional): Font size for x-axis label (default: 12)
    - `y_label_fontsize` (int, optional): Font size for y-axis label (default: 12)
    - `y_axis_range` (tuple, optional): Range for the y-axis (default: (0, 1))
    - `label_rotation` (int, optional): Rotation angle for x-axis labels (default: 45)
    - `inner` (str, optional): The representation of the data points inside the violin (default: "box")
    - `dpi` (int, optional): Resolution for saved image (default: 300)
    - `metrics_to_include` (list, optional): Specific metrics to include in the plot

11. ###### plot_multiple_violin_charts

    Create violin plots comparing multiple models on a single metric

    **_Parameters_**:

    - `result_sources` (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing evaluation results
    - `folder_path` (str, optional): Path to folder containing JSON result files
    - `output_file` (str, optional): Path to save the output visualization
    - `model_names` (List[str], optional): Names to display for models in the plot
    - `metric_name` (str, optional): Name of the metric to compare across models (default: "overall_accuracy")
    - `figsize` (tuple, optional): Figure size as (width, height) in inches (default: (12, 8))
    - `colormap` (str, optional): Matplotlib colormap name for the violins (default: "viridis")
    - `title` (str, optional): Custom title for the plot
    - `title_fontsize` (int, optional): Font size for the title text (default: 14)
    - `title_pad` (float, optional): Padding for the title from the top of the plot (default: 50.0)
    - `show_box` (bool, optional): Whether to show a box plot inside the violin (default: False)
    - `show_mean` (bool, optional): Whether to show the mean marker (default: True)
    - `mean_marker` (str, optional): Marker style for the mean (default: "o")
    - `mean_color` (str, optional): Color for the mean marker (default: "red")
    - `show_median` (bool, optional): Whether to show the median line (default: False)
    - `median_color` (str, optional): Color for the median line (default: "green")
    - `median_line_style` (str, optional): Line style for the median (default: "-")
    - `show_grid` (bool, optional): Whether to display the grid lines (default: True)
    - `show_threshold` (bool, optional): Whether to show a threshold line (default: False)
    - `threshold_value` (float, optional): Value for the threshold line (default: 0.8)
    - `threshold_color` (str, optional): Color for the threshold line (default: "red")
    - `threshold_line_style` (str, optional): Line style for the threshold line (default: "--")
    - `violin_alpha` (float, optional): Alpha (transparency) of the violin plots (default: 0.7)
    - `violin_width` (float, optional): Width of the violin plots (default: 0.8)
    - `x_label` (str, optional): Label for the x-axis (default: "Models")
    - `y_label` (str, optional): Label for the y-axis (default: "Score")
    - `x_label_fontsize` (int, optional): Font size for x-axis label (default: 12)
    - `y_label_fontsize` (int, optional): Font size for y-axis label (default: 12)
    - `y_axis_range` (tuple, optional): Range for the y-axis (default: (0, 1))
    - `label_rotation` (int, optional): Rotation angle for x-axis labels (default: 45)
    - `inner` (str, optional): The representation of the data points inside the violin (default: "box")
    - `dpi` (int, optional): Resolution for saved image (default: 300)median_line_style` (str, optional): Line style for the median (default: "-")
    - `show_grid` (bool, optional): Whether to display the grid lines (default: True)
    - `show_threshold` (bool, optional): Whether to show a threshold line (default: False)
    - `threshold_value` (float, optional): Value for the threshold line (default: 0.8)
    - `threshold_color` (str, optional): Color for the threshold line (default: "red")
    - `threshold_line_style` (str, optional): Line style for the threshold line (default: "--")
    - `violin_alpha` (float, optional): Alpha (transparency) of the violin plots (default: 0.7)
    - `violin_width` (float, optional): Width of the violin plots (default: 0.8)
    - `x_label` (str, optional): Label for the x-axis (default: "Metrics")
    - `y_label` (str, optional): Label for the y-axis (default: "Score")
    - `x_label_fontsize` (int, optional): Font size for x-axis label (default: 12)
    - `y_label_fontsize` (int, optional): Font size for y-axis label (default: 12)
    - `y_axis_range` (tuple, optional): Range for the y-axis (default: (0, 1))
    - `label_rotation` (int, optional): Rotation angle for x-axis labels (default: 45)
    - `inner` (str, optional): The representation of the data points inside the violin (default: "box")
    - `dpi` (int, optional): Resolution for saved image (default: 300)

## Workflow Details

### Overview Workflow Diagram

Provides a clear explanation of the five main stages in the ComProScanner workflow:

Metadata Collection - finding relevant scientific articles
Article Processing - extracting relevant text from various publishers
Data Extraction - using LLMs to extract structured data
Post Processing - evaluation, cleaning and data visualization

<div align="center">
  <img src="assets/overall_workflow.png" alt="ComProScanner Logo" width="750"/>
  <p>Overall Workflow Diagram</p>
</div>

### Data Extraction Flow

Detailed explanation how the extraction process leverages CrewAI's agentic framework.

- Materials Data Identifier Agent - determines if articles contain relevant data
- Composition-Property Extraction Agent - extracts compositions, property values, and material family information
- Synthesis Extraction Agent - extracts synthesis methods, precursors, and characterization techniques (when enabled)
- Data Formatting Agents (x2) - ensure standardized JSON output

## Advanced Configuration

### RAG Configuration

Retrieval-Augmented Generation (RAG) can be configured for improved extraction:

```python
from comproscanner.utils.configs.rag_config import RAGConfig

rag_config = RAGConfig(
    rag_db_path="vector_db",
    chunk_size=500,
    chunk_overlap=30,
    embedding_model="huggingface:thellert/physbert_cased",
    rag_chat_model="gpt-4o-mini",
    rag_max_tokens=256,
    rag_top_k=5
)
```

#### RAG Embedding Model Configuration Options

Here's how to configure different embedding model types in the ComProScanner package for RAG:

**HuggingFace Models**

```python
from comproscanner.utils.configs.rag_config import RAGConfig

huggingface_config = RAGConfig(
    rag_db_path="vector_db",
    chunk_size=1000,
    chunk_overlap=25,
    embedding_model="huggingface:thellert/physbert_cased",  # Physics-specific BERT
    rag_chat_model="gpt-4o-mini",
    rag_max_tokens=512,
    rag_top_k=3
)
```

**Sentence Transformers Models**

```python
from comproscanner.utils.configs.rag_config import RAGConfig

sentence_transformers_config = RAGConfig(
    rag_db_path="vector_db",
    chunk_size=800,
    chunk_overlap=50,
    embedding_model="sentence-transformers:all-mpnet-base-v2",
    rag_chat_model="gpt-4o-mini",
    rag_max_tokens=512,
    rag_top_k=3
)
```

**OpenAI Models**

```python
from comproscanner.utils.configs.rag_config import RAGConfig

# Using OpenAI model with explicit prefix
openai_config = RAGConfig(
    rag_db_path="vector_db",
    chunk_size=1000,
    chunk_overlap=100,
    embedding_model="openai:text-embedding-3-small",
    rag_chat_model="gpt-4o-mini",
    rag_max_tokens=512,
    rag_top_k=3
)

# Using OpenAI model without prefix (default behavior)
default_openai_config = RAGConfig(
    rag_db_path="vector_db",
    chunk_size=1000,
    chunk_overlap=100,
    embedding_model="text-embedding-3-small",
    rag_chat_model="gpt-4o-mini",
    rag_max_tokens=512,
    rag_top_k=3
)
```

Note: For OpenAI models, ensure the organization has access to the specified embedding model.

#### RAG Tool Configuration Options

The ComProScanner package provides a flexible RAG tool that supports multiple language model providers. The package understands the model name prefix to determine the appropriate provider and configuration.:

##### OpenAI Models

```python
from comproscanner.utils.configs.rag_config import RAGConfig

openai_config = RAGConfig(
    rag_db_path="vector_db",
    rag_chat_model="gpt-4o-mini",  # No prefix needed for OpenAI models
    temperature=0.1,
    streaming=False,
    rag_max_tokens=512,
    rag_top_k=3
)
```

##### Google Gemini Models

```python
from comproscanner.utils.configs.rag_config import RAGConfig

gemini_config = RAGConfig(
    rag_db_path="vector_db",
    rag_chat_model="gemini-2.0-flash-thinking-exp",  # 'gemini' prefix indicates Google provided model
    temperature=0.2,
    streaming=True,
    rag_max_tokens=1024,
    rag_top_k=5
)
```

##### Anthropic Claude Models

```python
from comproscanner.utils.configs.rag_config import RAGConfig

claude_config = RAGConfig(
    rag_db_path="vector_db",
    rag_chat_model="claude-3-5-haiku-20241022",  # 'claude' prefix indicates Anthropic provided model
    temperature=0.0,
    streaming=False,
    rag_max_tokens=2048,
    rag_top_k=3
)
```

##### Ollama Models

```python
from comproscanner.utils.configs.rag_config import RAGConfig

ollama_config = RAGConfig(
    rag_db_path="vector_db",
    rag_chat_model="ollama/llama3",  # 'ollama/' prefix indicates Ollama provided model
    rag_base_url="http://localhost:11434",  # Custom Ollama server URL
    temperature=0.5,
    streaming=True,
    rag_max_tokens=512,
    rag_top_k=3
)
```

##### Together AI Models

```python
from comproscanner.utils.configs.rag_config import RAGConfig

together_config = RAGConfig(
    rag_db_path="vector_db",
    rag_chat_model="together/mistral-7b-instruct",  # 'together/' prefix indicates Together AI provided model
    temperature=0.7,
    streaming=False,
    rag_max_tokens=1024,
    rag_top_k=3
)
```

##### OpenRouter Models

```python
from comproscanner.utils.configs.rag_config import RAGConfig

openrouter_config = RAGConfig(
    rag_db_path="vector_db",
    rag_chat_model="openrouter/meta-llama/llama-3-70b-instruct",  # 'openrouter/ ' prefix indicates OpenRouter provided model
    temperature=0.2,
    streaming=False,
    rag_max_tokens=512,
    rag_top_k=3
)
```

##### Cohere Models

```python
from comproscanner.utils.configs.rag_config import RAGConfig

cohere_config = RAGConfig(
    rag_db_path="vector_db",
    rag_chat_model="cohere/command-r",  # 'cohere/' prefix indicates Cohere provided model
    temperature=0.3,
    streaming=False,
    rag_max_tokens=512,
    rag_top_k=3
)
```

##### Fireworks Models

```python
from comproscanner.utils.configs.rag_config import RAGConfig

fireworks_config = RAGConfig(
    rag_db_path="vector_db",
    rag_chat_model="fireworks/llama-v3-8b",  # 'fireworks/' prefix indicates Fireworks provided model
    temperature=0.1,
    streaming=False,
    rag_max_tokens=512,
    rag_top_k=3
)
```

**Dependencies**

The RAG tool dynamically checks for required package dependencies based on the selected model:

- OpenAI models require `langchain_openai`
- Google Gemini models require `langchain_google_genai`
- Anthropic Claude models require `langchain_anthropic`
- Ollama models require `langchain_ollama`
- Together AI models require `langchain_together`
- OpenRouter models require `langchain_openrouter`
- Cohere models require `langchain_cohere`
- Fireworks models require `langchain_fireworks`

If the required package is not installed, the tool will raise an `ImportErrorHandler` exception with appropriate guidance.

## Project Structure

```
comproscanner/
├── src/
│   └── comproscanner/
│       ├── article_processors/       # Processors for different publishers
│       ├── extract_flow/             # Data extraction flow components
│       ├── metadata_extractor/       # Metadata collection and filtering
│       ├── post_processing/          # Data cleaning and evaluation
│       │   ├── evaluation/           # Evaluation tools
│       │   └── visualization/        # Visualization tools
│       ├── utils/                    # Utility functions and configurations
│       ├── __init__.py               # Package initialization
│       ├── comproscanner.py          # Main class implementation
│       ├── data_visualizer.py        # Data visualization tools
│       └── eval_visualizer.py        # Evaluation visualization tools
├── tests/                            # Test suite
├── .env                              # Environment variables
├── pyproject.toml                    # Project metadata
└── README.md                         # This file
```

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](./LICENSE)

## Contact

Author: Aritra Roy  
Email: contact@aritraroy.live  
Website: https://aritraroy.live
