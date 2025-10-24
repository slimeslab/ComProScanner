# Quick Start Guide

This guide will help you get started with ComProScanner quickly.

## Complete Workflow Example

Here's a complete minimal example demonstrating the full workflow for extracting piezoelectric coefficient (d33) data:

```python
from comproscanner import ComProScanner

# Initialize with property of interest
scanner = ComProScanner(main_property_keyword="piezoelectric")

# Step 1: Collect metadata
scanner.collect_metadata()

# Step 2: Define property keywords for filtering
property_keywords = {
    "exact_keywords": ["d33"],
    "substring_keywords": [" d 33 "]
}

# Step 3: Process articles from specific sources
scanner.process_articles(
    property_keywords=property_keywords,
    source_list=["elsevier", "springer"]
)

# Step 4: Extract composition-property relationships
scanner.extract_composition_property_data(
    main_extraction_keyword="d33"
)
```

## Step-by-Step Breakdown

### 1. Initialize the Scanner

Create a ComProScanner instance with your main property keyword which helps the scanner to create associated files and directories for automated organization:

```python
from comproscanner import ComProScanner

scanner = ComProScanner(main_property_keyword="piezoelectric")
```

### 2. Collect Metadata

Find relevant scientific articles about piezoelectric materials from Scopus database for the last 2 years:

```python
scanner.collect_metadata(
    base_queries=["piezoelectric", "piezoelectricity"]
)
```

### 3. Process Articles

Extract relevant text from full-text articles for Elsevier, Wiley, and Springer Nature articles using their Text and Data Mining (TDM) APIs:

```python
property_keywords = {
    "exact_keywords": ["d33"],  # Exact matches
    "substring_keywords": [" d 33 "]  # Substring matches
}

scanner.process_articles(
    property_keywords=property_keywords,
    source_list=["elsevier", "wiley", "springer"]
)
```

<!-- !!! info "Publisher Support"
ComProScanner supports Elsevier, Wiley, and Springer Nature for automated full-text data extraction using publishers 'Text and Data Mining' (TDM) APIs, local folder path containing full-text XML IOP Science articles, and manually downloaded PDFs for all other publishers by providing a local path. -->

### 4. Extract Data

Use multiple [CrewAI](https://www.crewai.com/) agents to extract structured data from the processed articles using OpenAI's GPT-4o Mini model:

```python
scanner.extract_composition_property_data(
    main_extraction_keyword="d33",
    is_extract_synthesis_data=True,
    model="gpt-4o-mini"
)
```

## Optional

### Visualize Extracted Data

Create pie charts for material family distribution and knowledge graphs from the extracted results:

```python
from comproscanner import data_visualizer

# Plot material families distribution
fig = data_visualizer.plot_family_pie_chart(
    data_sources=["extracted_results.json"],
    output_file="family_distribution.png"
)

# Create knowledge graph
data_visualizer.create_knowledge_graph(
    result_file="extracted_results.json"
)
```

### Evaluate Extraction Quality

Evaluate the extraction result quality against ground truth data using semantic and agentic evaluation methods:

```python
from comproscanner import evaluate_semantic, evaluate_agentic

# Semantic evaluation
semantic_results = evaluate_semantic(
    ground_truth_file="ground_truth.json",
    test_data_file="extracted_results.json",
    output_file="semantic_evaluation.json"
)

# Agentic evaluation (more advanced)
agentic_results = evaluate_agentic(
    ground_truth_file="ground_truth.json",
    test_data_file="extracted_results.json",
    output_file="agentic_evaluation.json"
)
```

### Visualize Evaluation Results

Easily visualize evaluation metrics for both single and multiple model comparisons:

```python
from comproscanner import eval_visualizer

# Plot single model evaluation
fig = eval_visualizer.plot_single_bar_chart(
    result_file="semantic_evaluation.json",
    output_file="evaluation_metrics.png"
)

# Compare multiple models
fig = eval_visualizer.plot_multiple_radar_charts(
    result_sources=["model1_eval.json", "model2_eval.json"],
    model_names=["GPT-4", "Claude"],
    output_file="model_comparison.png"
)
```

## Understanding the Output

### Extracted Data Format

The extraction produces JSON files with structured data similar to the following example:

```json
"10.1016/j.apradiso.2024.111655": {
    "composition_data": {
      "compositions_property_values": {
        "Eu1.90Dy0.10Ge2O7": 0.66,
        "Eu1.90La0.10Ge2O7": 0.36,
        "Eu1.90Ho0.10Ge2O7": 0.62
      },
      "property_unit": "pC/N",
      "family": "RE2B2O7"
    },
    "synthesis_data": {
      "method": "solid-state reaction",
      "precursors": [
        "Eu2O3",
        "GeO2",
        "Dy2O3",
        "La2O3",
        "Ho2O3"
      ],
      "steps": [
        "Starting materials Eu2O3, GeO2, Dy2O3, La2O3 and Ho2O3 were combined in stoichiometric ratios with each dopant at 5 mol%.",
        "Samples were first heated at 800°C for 2 hours in pure alumina crucibles under open atmosphere.",
        "Materials were then heated to 1150°C for 10 hours followed by slow cooling.",
        "Resulting materials were ground into powder for further characterization.",
        "Ceramic discs were formed from obtained powder materials with 1 mm thickness and 10 mm diameter.",
        "Ceramic discs were compacted using uniaxial pressing under 250 MPa pressure with 2 wt% of 5 wt% PVA aqueous solution as binder.",
        "Samples were heated at 600°C for 30 minutes to eliminate organic additives.",
        "Sintering was conducted at 1400°C for 4 hours.",
        "Silver paste was applied to disc surfaces and fired at 650°C for 1 hour to form surface electrodes.",
        "Electric field of 9-18 kV/mm was applied in silicon oil bath at 120°C for 30 minutes followed by 24-hour aging."
      ],
      "characterization_techniques": [
        "TG/DTA",
        "XRD",
        "SEM",
        "EDX",
        "photoluminescence spectroscopy",
        "LCR meter",
        "d33 meter"
      ]
    },
    "article_metadata": {
      "doi": "10.1016/j.apradiso.2024.111655",
      "title": "Novel smart materials with high curie temperatures: Eu1.90Dy0.10Ge2O7, Eu1.90La0.10Ge2O7 and Eu1.90Ho0.10Ge2O7",
      "journal": "Applied Radiation and Isotopes",
      "year": "2025",
      "isOpenAccess": false,
      "authors": [
        {
          "name": "Esra Öztürk",
          "affiliation_id": "60020484",
          "affiliation_name": "Hacettepe Üniversitesi",
          "affiliation_country": "Turkey"
        },
        {
          "name": "Nilgun Kalaycioglu Ozpozan",
          "affiliation_id": "122321412",
          "affiliation_name": "Erciyes Ün.",
          "affiliation_country": "Türkiye"
        },
        {
          "name": "Volkan Kalem",
          "affiliation_id": "60193845",
          "affiliation_name": "Konya Technical University",
          "affiliation_country": "Turkey"
        }
      ],
      "keywords": [
        "Curie"
      ]
    }
  }
```

## Next Steps

Now that you understand the basics, explore:

- **[User Guide](../usage/metadata-collection.md)** - Detailed documentation for each module and functions
- **[Advanced Configuration](../rag-config.md)** - Configure RAG and custom flows

## Troubleshooting

### No Articles Found

- Check your search queries are relevant
- Verify the date range is appropriate

### Extraction Issues

- Ensure API keys are configured correctly
- Ensure sufficient API credits for LLM calls
- Check that articles contain relevant data
- Try adjusting temperature and model parameters or use a different model
- Try passing additional instructions to the extraction agents for better context

!!! question "Need Help?"
If you encounter issues, check the [GitHub Issues](https://github.com/slimeslab/ComProScanner/issues) or contact [Aritra Roy](mailto:contact@aritraroy.live).
