# RAG Configuration

Customize Retrieval-Augmented Generation (RAG) settings for improved data extraction accuracy specific to your use case.

!!! note "RAG is Used by Materials Data Identifier Agent"

    RAG is automatically used during data extraction to retrieve relevant context from article text before querying the LLM to understand whether the article has material compositions and corresponding property values for screening. The parameters below allow you to customize RAG behavior based on your specific requirements, such as article length, domain-specific models, or extraction complexity.

## Configuration Parameters

### Chunking Parameters

These parameters control how articles are split into chunks for vector storage.

#### :material-square-medium:`chunk_size` _(int)_

Size of text chunks in characters for creating vector database embeddings.

#### :material-square-medium:`chunk_overlap` _(int)_

Number of overlapping characters between consecutive chunks to maintain context continuity.

!!! info "Default Values"

    :material-square-small:**`chunk_size`** = 1000<br>:material-square-small:**`chunk_overlap`** = 25

### Embedding Model Parameters

#### :material-square-medium:`embedding_model` _(str)_

Name of the embedding model to use for creating vector databases for RAG.

**Supported Providers:**

| Provider              | Format                                                     | Example                                                          |
| --------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| HuggingFace           | `huggingface:model-name`                                   | `huggingface:thellert/physbert_cased`                            |
| Sentence Transformers | `sentence-transformers:model-name`                         | `sentence-transformers:all-mpnet-base-v2`                        |
| OpenAI                | • `openai:model-name`<br>• `model-name` (default behavior) | • `openai:text-embedding-3-small` <br>• `text-embedding-3-small` |

!!! info "Default Value"

    :material-square-small:**`embedding_model`** = "huggingface:thellert/physbert_cased"

!!! tip "Recommended Models for Scientific Text"

    - **PhysBERT**: `huggingface:thellert/physbert_cased` - A specialized text embedding model for physics, designed to improve information retrieval, citation classification, and clustering of physics literature
    - **MatBERT**: `huggingface:pranav-s/MaterialsBERT` - A fine-tuned version of PubMedBERT on a dataset of 2.4 million materials science abstracts
    - **MatSciBERT**: `huggingface:m3rg-iitd/matscibert` - A material domain language model for text mining and information extraction

### Retrieval Parameters

These parameters control how relevant context is retrieved during extraction.

#### :material-square-medium:`rag_db_path` _(str)_

Custom path to store or load the vector databases of property-mentioned articles for RAG processing.

#### :material-square-medium:`rag_top_k` _(int)_

Number of most relevant text chunks to retrieve from the vector database for context.

#### :material-square-medium:`rag_max_tokens` _(int)_

Maximum number of tokens for RAG model responses.

!!! info "Default Values"

    :material-square-small:**`rag_db_path`** = "db"<br>:material-square-small:**`rag_top_k`** = 3<br>:material-square-small:**`rag_max_tokens`** = 512

### RAG Chat Model Parameters

#### :material-square-medium:`rag_chat_model` _(str)_

Chat model to use for RAG-based context retrieval and synthesis.

#### :material-square-medium:`rag_base_url` _(str)_

Custom base URL for RAG chat model API (useful for local or custom deployments).

!!! info "Default Values"

    :material-square-small:**`rag_chat_model`** = "gpt-4o-mini"<br>:material-square-small:**`rag_base_url`** = None

## Configuration Examples

### Using OpenAI

**API Key Required:** Set `OPENAI_API_KEY` in your `.env` file.

```python
from comproscanner import ComProScanner

scanner = ComProScanner(output_dir="output")

# Process articles with custom chunking
scanner.process_articles(
    property_keywords={
        "exact_keywords": ["d33"],
        "substring_keywords": [" d 33 "]
    },
    rag_db_path="embeddings/piezo",
    chunk_size=800,
    chunk_overlap=50,
    embedding_model="openai:text-embedding-3-small"
)

# Extract with GPT-4o
scanner.extract_composition_property_data(
    main_extraction_keyword="d33",
    rag_db_path="embeddings/piezo",
    rag_chat_model="gpt-4o",
    rag_max_tokens=1024,
    rag_top_k=5,
)
```

### Using Google Gemini

**API Key Required:** Set `GEMINI_API_KEY` in your `.env` file.

```python
scanner.extract_composition_property_data(
    main_extraction_keyword="d33",
    rag_db_path="embeddings/piezo",
    rag_chat_model="gemini-2.0-flash",
    rag_max_tokens=1024,
    rag_top_k=4,
)
```

### Using Anthropic Claude

**API Key Required:** Set `ANTHROPIC_API_KEY` in your `.env` file.

```python
scanner.extract_composition_property_data(
    main_extraction_keyword="d33",
    rag_db_path="embeddings/piezo",
    rag_chat_model="claude-3-5-sonnet-20241022",
    rag_max_tokens=2048,
    rag_top_k=4,
)
```

### Using Local Ollama

```python
scanner.extract_composition_property_data(
    main_extraction_keyword="d33",
    rag_db_path="embeddings/piezo",
    rag_chat_model="ollama/llama3.1",
    rag_base_url="http://localhost:11434",
    rag_max_tokens=512,
    rag_top_k=3,
)
```

### Using Together AI

**API Key Required:** Set `TOGETHER_API_KEY` in your `.env` file.

```python
scanner.extract_composition_property_data(
    main_extraction_keyword="d33",
    rag_db_path="embeddings/piezo",
    rag_chat_model="together_ai/meta-llama/Llama-3-70b-chat-hf",
    rag_max_tokens=1024,
    rag_top_k=4,
)
```

### Using OpenRouter

**API Key Required:** Set `OPENROUTER_API_KEY` in your `.env` file.

```python
scanner.extract_composition_property_data(
    main_extraction_keyword="d33",
    rag_db_path="embeddings/piezo",
    rag_chat_model="openrouter/meta-llama/llama-3-70b-instruct",
    rag_max_tokens=1024,
    rag_top_k=4,
)
```

### Using Cohere

**API Key Required:** Set `COHERE_API_KEY` in your `.env` file.

```python
scanner.extract_composition_property_data(
    main_extraction_keyword="d33",
    rag_db_path="embeddings/piezo",
    rag_chat_model="cohere/command-r-plus",
    rag_max_tokens=1024,
    rag_top_k=4,
)
```

### Using Fireworks AI

**API Key Required:** Set `FIREWORKS_API_KEY` in your `.env` file.

```python
scanner.extract_composition_property_data(
    main_extraction_keyword="d33",
    rag_db_path="embeddings/piezo",
    rag_chat_model="fireworks_ai/accounts/fireworks/models/llama-v3-8b-instruct",
    rag_max_tokens=1024,
    rag_top_k=4,
)
```

### Using Domain-Specific Embeddings

```python
# PhysBERT for physics/materials science
scanner.process_articles(
    property_keywords=property_keywords,
    embedding_model="huggingface:thellert/physbert_cased",
    chunk_size=1000,
    chunk_overlap=50
)

# MatBERT for materials science
scanner.process_articles(
    property_keywords=property_keywords,
    embedding_model="huggingface:pranav-s/MaterialsBERT",
    chunk_size=1000,
    chunk_overlap=50
)
```

## Dependencies

Install required packages based on your chosen providers:

### OpenAI

```bash
pip install langchain-openai
```

### Google Gemini

```bash
pip install langchain-google-genai
```

### Anthropic Claude

```bash
pip install langchain-anthropic
```

### Ollama

```bash
pip install langchain-ollama

# Install Ollama locally
# Visit: https://ollama.ai/download
```

### Other Providers

```bash
# Together AI
pip install langchain-together

# Cohere
pip install langchain-cohere

# For HuggingFace embeddings
pip install sentence-transformers
```

## Next Steps

- Explore [Data Extraction](../usage/data-extraction.md)
- Review [Evaluation Methods](../usage/evaluation/agentic.md)
