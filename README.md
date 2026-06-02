<p align="center">
  <img src="https://raw.githubusercontent.com/aritraroy24/ComProScanner/refs/heads/main/assets/comproscanner_logo.png" alt="ComProScanner Logo" width="500"/>
</p>

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/) [![License: MIT](https://custom-icon-badges.demolab.com/badge/license-MIT-brown.svg?logo=law&logoColor=white)](https://opensource.org/licenses/MIT) [![PyPI](https://img.shields.io/pypi/v/comproscanner?logo=pypi&logoColor=white)](https://pypi.org/project/comproscanner/) [![Documentation](https://custom-icon-badges.demolab.com/badge/docs-latest-brightgreen.svg?logo=materialformkdocs&logoColor=white)](https://slimeslab.github.io/ComProScanner/) [![Coverage](https://img.shields.io/codecov/c/github/aritraroy24/ComProScanner?logo=codecov&logoColor=white&label=coverage&color=e62277)](https://codecov.io/gh/aritraroy24/ComProScanner) [![PyPI - Downloads](https://custom-icon-badges.demolab.com/pypi/dm/comproscanner?logo=download&logoColor=white&color=purple)](https://pypistats.org/packages/comproscanner) [![Ask DeepWiki](https://custom-icon-badges.demolab.com/badge/Ask%20DeepWiki-brightgreen.svg?logo=deepwikidevin&logoColor=white&labelColor=grey&color=5ab998)](https://deepwiki.com/slimeslab/ComProScanner) [![Digital Discovery](https://custom-icon-badges.demolab.com/badge/Digital_Discovery-10.1039/D5DD00521C-brightgreen.svg?logo=rsc&logoColor=white&color=c8c300)](https://doi.org/10.1039/D5DD00521C) [![arXiv Preprint](https://custom-icon-badges.demolab.com/badge/arXiv-2606.00065-brightgreen.svg?logo=arxiv&logoColor=white&color=b22929)](https://arxiv.org/abs/2606.00065)

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

Here's a complete example extracting piezoelectric coefficient (d<sub>33</sub>) data:

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

## Requirements

- Python 3.12 or 3.13
- TDM API keys for desired publishers (Elsevier, Springer, Wiley)
- LLM API keys (OpenAI, Anthropic, Google, etc.)
- Optional: Neo4j for knowledge graph visualization

## Citation

If you use ComProScanner in your research, please cite the following papers:

```bibtex
@article{roy2026comproscanner,
      title={ComProScanner: a multi-agent based framework for composition-property structured data extraction from scientific literature},
      author={Roy, Aritra and Grisan, Enrico and Buckeridge, John and Gattinoni, Chiara},
      journal={Digital Discovery},
      volume={5},
      number={4},
      pages={1794--1808},
      year={2026},
      publisher={Royal Society of Chemistry},
      doi  ="10.1039/D5DD00521C",
      url  ="https://doi.org/10.1039/D5DD00521C"
}
@misc{roy2026comproscanner_vlm,
      title={Beyond Text and Tables: Vision-Language Model Integration in ComProScanner for Extracting Materials Data from Scientific Figures with High Accuracy}, 
      author={Aritra Roy and Enrico Grisan and Chiara Gattinoni and John Buckeridge},
      year={2026},
      eprint={2606.00065},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      doi={10.48550/arXiv.2606.00065},
      url={https://arxiv.org/abs/2606.00065}, 
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
