<p align="center">
  <img src="https://raw.githubusercontent.com/aritraroy24/ComProScanner/refs/heads/main/assets/comproscanner_logo.png" alt="ComProScanner Logo" width="500"/>
</p>

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/) [![License: MIT](https://custom-icon-badges.demolab.com/badge/license-MIT-yellow.svg?logo=law&logoColor=white)](https://opensource.org/licenses/MIT) [![PyPI](https://img.shields.io/pypi/v/comproscanner?logo=pypi&logoColor=white)](https://pypi.org/project/comproscanner/) [![Documentation](https://custom-icon-badges.demolab.com/badge/docs-latest-brightgreen.svg?logo=materialformkdocs&logoColor=white)](https://slimeslab.github.io/ComProScanner/) [![Coverage](https://img.shields.io/codecov/c/github/aritraroy24/ComProScanner?logo=codecov&logoColor=white&label=coverage&color=e62277)](https://codecov.io/gh/aritraroy24/ComProScanner) [![PyPI - Downloads](https://custom-icon-badges.demolab.com/pypi/dm/comproscanner?logo=download&logoColor=white&color=purple)](https://pypistats.org/packages/comproscanner) [![Ask DeepWiki](https://custom-icon-badges.demolab.com/badge/Ask%20DeepWiki-brightgreen.svg?logo=deepwikidevin&logoColor=white&labelColor=grey&color=5ab998)](https://deepwiki.com/slimeslab/ComProScanner) [![Digital Discovery](https://custom-icon-badges.demolab.com/badge/Digital_Discovery-10.1039/D5DD00521C-brightgreen.svg?logo=rsc&logoColor=white&color=c8c300)](https://doi.org/10.1039/D5DD00521C)

# ComProScanner

**A comprehensive Python package for extracting composition-property data from scientific articles for building databases**

## Overview

ComProScanner is a multi-agent framework designed to extract composition-property relationships from scientific articles in materials science. It automates the entire workflow from metadata collection to data extraction, evaluation, and visualization.

**Key Features:**

- 📚 Multi-publisher support (Elsevier, Springer, Wiley, IOP, local PDFs)
- 🤖 Agentic extraction using CrewAI framework
- 🔍 RAG-powered context retrieval for cost effective automation with accuracy
- 📊 Comprehensive evaluation and visualization tools
- 🎯 Customizable extraction workflows
- 🌐 Knowledge graph generation

## Installation

Install from PyPI:

```bash
pip install comproscanner
```

Or install from source:

```bash
git clone https://github.com/slimeslab/ComProScanner.git
cd comproscanner
pip install -e .
```

## Quick Start

Here's a complete example extracting piezoelectric coefficient ($d_{33}$) data:

```python
from comproscanner import ComProScanner

# Initialize scanner
scanner = ComProScanner(main_property_keyword="piezoelectric")

# Collect metadata
scanner.collect_metadata(
    base_queries=["piezoelectric", "piezoelectricity"],
    extra_queries=["ceramics", "applications"]
)

# Process articles
property_keywords = {
    "exact_keywords": ["d33"],
    "substring_keywords": [" d 33 "]
}

scanner.process_articles(
    property_keywords=property_keywords,
    source_list=["elsevier", "springer"]
)

# Extract composition-property data
scanner.extract_composition_property_data(
    main_extraction_keyword="d33"
)
```

## Workflow

<div align="center">
  <img src="https://raw.githubusercontent.com/aritraroy24/ComProScanner/refs/heads/main/assets/overall_workflow.png" alt="ComProScanner Workflow" width="750"/>
</div>

The ComProScanner workflow consists of four main stages:

1. **Metadata Retrieval** - Find relevant scientific articles
2. **Article Collection** - Extract full-text from various publishers
3. **Information Extraction** - Use LLM agents to extract structured data
4. **Post Processing & Dataset Creation** - Evaluate, clean, and visualize results

## Documentation

📖 **Full documentation is available at [slimeslab.github.io/ComProScanner](https://slimeslab.github.io/ComProScanner/)**

- [Installation Guide](https://slimeslab.github.io/ComProScanner/getting-started/installation/)
- [Quick Start Tutorial](https://slimeslab.github.io/ComProScanner/getting-started/quick-start/)
- [User Guide](https://slimeslab.github.io/ComProScanner/usage/metadata-collection/)
- [RAG Configuration](https://slimeslab.github.io/ComProScanner/rag-config/)

## Core Capabilities

### Supported Publishers

- **Elsevier** (via TDM API)
- **Springer Nature** (via TDM API)
- **Wiley** (via TDM API)
- **IOP Publishing** (via SFTP bulk access)
- **Local PDFs** (any publication)

### Data Extraction

- Composition-property relationships
- Material families
- Synthesis methods and precursors
- Characterization techniques
- Synthesis steps

### Evaluation Methods

- **Semantic Evaluation** - Using semantic similarity measures
- **Agentic Evaluation** - LLM-powered contextual analysis

### Visualization

- Data Visualization
- Evaluation Visualization

## Example Use Cases

### Extract Data from Multiple Sources

```python
scanner.process_articles(
    property_keywords=property_keywords,
    source_list=["elsevier", "springer", "wiley"]
)
```

### Process Local PDFs With Failed-Filename Report

```python
scanner.process_articles(
    property_keywords=property_keywords,
    source_list=["pdfs"],
    folder_path="/home/user/papers",
    save_failed_pdf_report=True,
    failed_pdf_report_path="/home/user/papers/failed_pdf_filenames.txt"
)
```

### Customize RAG Configuration

```python
scanner.extract_composition_property_data(
    main_extraction_keyword="d33",
    rag_chat_model="gemini-2.5-pro",
    rag_max_tokens=2048,
    rag_top_k=5
)
```

### Visualize Results

```python
from comproscanner import data_visualizer, eval_visualizer

# Create knowledge graph
data_visualizer.create_knowledge_graph(result_file="results.json")

# Plot evaluation metrics
eval_visualizer.plot_multiple_radar_charts(
    result_sources=["model1.json", "model2.json"],
    model_names=["GPT-4o", "Claude-3.5"]
)
```

## Requirements

- Python 3.12 or 3.13
- TDM API keys for desired publishers (Elsevier, Springer, Wiley)
- LLM API keys (OpenAI, Anthropic, Google, etc.)
- Optional: Neo4j for knowledge graph visualization

## Citation

If you use ComProScanner in your research, please cite:

```bibtex
@Article{roy2026comproscannermultiagentbasedframework,
author ="Roy, Aritra and Grisan, Enrico and Buckeridge, John and Gattinoni, Chiara",
title  ="ComProScanner: a multi-agent based framework for composition-property structured data extraction from scientific literature",
journal  ="Digital Discovery",
year  ="2026",
pages  ="Accepted",
publisher  ="RSC",
doi  ="10.1039/D5DD00521C",
url  ="https://doi.org/10.1039/D5DD00521C"
}
```

## Changelog

See the [CHANGELOG](CHANGELOG.md) for details on what has changed in each version.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](https://slimeslab.github.io/ComProScanner/about/contribution/) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright © 2025-2026 SLIMES Lab

## Contact

**Author:** Aritra Roy

- 🌐 Website: [aritraroy.live](https://aritraroy.live)
- 📧 Email: [contact@aritraroy.live](mailto:contact@aritraroy.live)
- 🐙 GitHub: [@aritraroy24](https://github.com/aritraroy24)
- 𝕏 Twitter: [@aritraroy24](https://twitter.com/aritraroy24)

**Project Links:**

- 📦 PyPI: [pypi.org/project/comproscanner](https://pypi.org/project/comproscanner/)
- 📖 Documentation: [slimeslab.github.io/ComProScanner](https://slimeslab.github.io/ComProScanner/)
- 🐛 Issues: [github.com/slimeslab/ComProScanner/issues](https://github.com/slimeslab/ComProScanner/issues)

---

Made with ❤️ by [SLIMES Lab](https://slimeslab.github.io)
