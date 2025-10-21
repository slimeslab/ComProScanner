# Installation

This guide will help you install ComProScanner and its dependencies.

## Requirements

- Python 3.12 or 3.13
- pip (Python package installer)

## Basic Installation

The simplest way to install ComProScanner is using pip:

```bash
pip install comproscanner
```

This will install the latest stable version from PyPI along with all required dependencies.

## Installation from Source

If you want to install from source or contribute to development:

### 1. Clone the Repository

```bash
git clone https://github.com/slimeslab/ComProScanner.git
cd comproscanner
```

### 2. Install in Development Mode

```bash
pip install -e .
```

The `-e` flag installs the package in editable mode, allowing you to make changes to the source code.

## Environment Variables

ComProScanner requires several API keys to function properly. Create a `.env` file in your project directory:

```bash
# Publisher TDM API Keys (for direct article access)
SCOPUS_API_KEY=your_scopus_api_key # for Elsevier as well as metadata retrieval
WILEY_API_KEY=your_wiley_api_key
SPRINGER_OPENACCESS_API_KEY=your_springer_openaccess_api_key # Springer provides two separate keys for open access and TDM API
SPRINGER_TDM_API_KEY=your_springer_tdm_api_key
IOP_papers_path=local_path_to_iop_papers # IOP Publishing provides XML articles in bulk through SFTP access

# API Keys for LLM Models (at least one is required which will be used for data extraction)
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_google_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# Hugging Face API Key (for accessing thellert/physbert_cased model for embeddings)
HF_TOKEN=your_huggingface_api_key

# Neo4j Configuration (for knowledge graph visualization)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

```

!!! warning "Keep your API keys secure and never commit them to version control!"

## Optional Dependencies

### For Additional LLM Providers

Depending on which LLM providers you want to use:

```bash
# For Anthropic Claude
pip install langchain-anthropic

# For Google Gemini
pip install langchain-google-genai

# For Ollama (local models)
pip install langchain-ollama

# For TogetherAI Model Integration
pip install langchain-together

# For OpenRouter Model Integration
pip install langchain-openrouter

# For Cohere Model Integration
pip install langchain-cohere
```

## Verification

Verify your installation by running:

```python
import comproscanner
print(comproscanner.__version__)
```

You should see the version number printed without any errors.

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade comproscanner
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'comproscanner'

Make sure you've installed the package correctly:

```bash
pip install comproscanner
```

#### API Key Errors

Ensure your `.env` file is in the correct location and contains valid API keys.

#### Dependency Conflicts

If you encounter dependency conflicts, try creating a fresh virtual environment:

```bash
python -m venv compro_env
source compro_env/bin/activate  # On Windows: compro_env\Scripts\activate
pip install comproscanner
```

## Next Steps

Now that you have ComProScanner installed, check out the [Quick Start Guide](quick-start.md) to begin extracting data from scientific articles.
