# API Key Guide

This page explains which provider credentials ComProScanner can use, what each one is used for, and how to generate or obtain them.

## Overview

ComProScanner can work with three groups of external providers:

!!! important "Which credentials do you actually need?"

    | Provider group | Requirement level |
    | --- | --- |
    | **Publisher/content providers for article access** | **Optional** for manual or local workflows, but **required** for automated article retrieval. |
    | **LLM providers for extraction, vision models and RAG** | **At least one required** for extraction, vision models and RAG workflows. However, default models are different for extraction/RAG and vision-language models. |
    | **Default embedding provider for vector database creation** | **Required** unless you configure a custom embedding provider. |

Use only the providers relevant to your workflow. You do not need every key.

## Publisher Providers

### Elsevier / Scopus

Environment variable: `SCOPUS_API_KEY`

Used for:

- Scopus-based metadata retrieval
- Elsevier article retrieval in XML format

How to get it:

1. Create or sign in to your [Elsevier developer account](https://dev.elsevier.com/).
2. Open the [API key management area](https://dev.elsevier.com/apikey/manage).
3. Create a key for Scopus or content APIs.
4. Copy the generated key into your `.env` file as `SCOPUS_API_KEY`.

```bash
SCOPUS_API_KEY=your_scopus_api_key
```

### Springer Nature Open Access API

Environment variable: `SPRINGER_OPENACCESS_API_KEY`

Used for:

- Springer Open Access article retrieval in XML format

How to get it:

1. Create or sign in to your [Springer Nature account](https://dev.springernature.com/).
2. Fill up the form to request an Open Access API key at [https://dev.springernature.com/register/](https://dev.springernature.com/register/).
3. Get the Open Access API key from the [Springer Nature API management page](https://datasolutions.springernature.com/account/api-management/).
4. Copy the key into your `.env` file.

```bash
SPRINGER_OPENACCESS_API_KEY=your_springer_openaccess_api_key
```

### Springer Nature TDM API

Environment variable: `SPRINGER_TDM_API_KEY`

Used for:

- Springer subscription article retrieval in XML format

How to get it:

1. Subscribe to the Springer Nature TDM service via [https://dev.springernature.com/subscription/](https://dev.springernature.com/subscription/) and select the appropriate access level based on your institution and use case.
2. Copy the issued TDM key or token into your `.env` file.

```bash
SPRINGER_TDM_API_KEY=your_springer_tdm_api_key
```

### Wiley TDM API

Environment variable: `WILEY_API_KEY`

Used for:

- Wiley full-text article download as PDF

How to get it:

1. Create your [Wiley account](https://onlinelibrary.wiley.com/action/registration).
2. Login to your Wiley account at [https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining](https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining) under the "**Get a Text and Data Mining Token**" section.
3. Accept the terms and conditions to generate your API token.
4. Copy the API token into your `.env` file.

```bash
WILEY_API_KEY=your_wiley_api_key
```

### IOP Publishing

Environment variable: `IOP_papers_path` (*not an API key but a required path variable for processing IOP Science XML files*)

Used for:

- Local processing of IOP Science XML files downloaded in bulk

How to get it:

1. Email [contentsupport@ioppublishing.org](mailto:contentsupport@ioppublishing.org) to request bulk access to the IOP Science XML files, typically through SFTP as IOP Publishing does not provide direct API access for bulk downloads.
2. Once you have access, download the XML files to a local directory.
3. Set `IOP_papers_path` to the absolute local folder path containing all the downloaded files.

```bash
IOP_papers_path=/absolute/path/to/iop_papers
```

## LLM Providers

These providers can be used for extraction models, RAG chat models, and vision-language models where supported by your configuration.

### OpenAI

Environment variable: `OPENAI_API_KEY`

Typical model prefixes: `openai/...` or OpenAI model names directly

How to get it:

1. Create or sign in to your [OpenAI account](https://platform.openai.com/).
2. Open the [API keys section](https://platform.openai.com/api-keys).
3. Create a new secret key.
4. Store it in `.env`.

```bash
OPENAI_API_KEY=your_openai_api_key
```

### Google Gemini

Environment variable: `GEMINI_API_KEY`

Typical model prefixes: `gemini/...`

How to get it:

1. Create or sign in to your [Google AI Studio account](https://aistudio.google.com/).
2. Generate an API key from the [Gemini API key page](https://aistudio.google.com/app/apikey).
3. Store it in `.env` as `GEMINI_API_KEY`.

```bash
GEMINI_API_KEY=your_gemini_api_key
```

### Anthropic

Environment variable: `ANTHROPIC_API_KEY`

Typical model prefixes: `anthropic/...`

How to get it:

1. Create or sign in to your [Anthropic Console account](https://console.anthropic.com/).
2. Create a new API key from the [Anthropic keys page](https://console.anthropic.com/settings/keys).
3. Store it in `.env`.

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### DeepSeek

Environment variable: `DEEPSEEK_API_KEY`

Typical model prefixes: `deepseek/...`

How to get it:

1. Create or sign in to your [DeepSeek platform account](https://platform.deepseek.com/).
2. Generate an API key from the [DeepSeek API keys page](https://platform.deepseek.com/api_keys).
3. Store it in `.env`.

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### OpenRouter

Environment variable: `OPENROUTER_API_KEY`

Typical model prefixes: `openrouter/...`

How to get it:

1. Create or sign in to your [OpenRouter account](https://openrouter.ai/).
2. Generate an API key from the [OpenRouter keys page](https://openrouter.ai/keys).
3. Store it in `.env`.

```bash
OPENROUTER_API_KEY=your_openrouter_api_key
```

### Together AI

Environment variable: `TOGETHER_API_KEY`

Typical model prefixes: `together/...`

How to get it:

1. Create or sign in to your [Together AI account](https://www.together.ai/).
2. Generate an API key from the [Together AI API keys page](https://api.together.ai/settings/api-keys).
3. Store it in `.env`.

```bash
TOGETHER_API_KEY=your_together_api_key
```

### Cohere

Environment variable: `COHERE_API_KEY`

Typical model prefixes: `cohere/...`

How to get it:

1. Create or sign in to your [Cohere account](https://dashboard.cohere.com/).
2. Create an API key from the [Cohere API keys page](https://dashboard.cohere.com/api-keys).
3. Store it in `.env`.

```bash
COHERE_API_KEY=your_cohere_api_key
```

### Fireworks AI

Environment variable: `FIREWORKS_API_KEY`

Typical model prefixes: `fireworks/...`

How to get it:

1. Create or sign in to your [Fireworks AI account](https://fireworks.ai/).
2. Generate an API key from the [Fireworks AI API keys page](https://app.fireworks.ai/settings/users/api-keys).
3. Store it in `.env`.

```bash
FIREWORKS_API_KEY=your_fireworks_api_key
```

### Ollama

Environment variable: none required

Used for:

- Local model inference

How to set it up:

1. Install Ollama from the [main Ollama website](https://ollama.com/).
2. Pull the model you want to use by following the [Ollama library and setup docs](https://ollama.com/library).
3. Set `base_url` or `rag_base_url` if needed, such as `http://localhost:11434`.

## Default Embedding Provider

### Hugging Face

Environment variable: `HF_TOKEN`

> **Optional.** Only required for downloading gated or private Hugging Face models. Public models work without a token.

Used for:

- Accessing gated or private Hugging Face models
- Rate-limited API access

How to get it:

1. Create or sign in to your [Hugging Face account](https://huggingface.co/).
2. Open the [access tokens page](https://huggingface.co/settings/tokens).
3. Create a new token with the required permissions.
4. Store it in `.env`.

```bash
HF_TOKEN=your_huggingface_token
```

## Recommended `.env` Template

Use the subset you need:

```bash
# Publisher providers
SCOPUS_API_KEY=your_scopus_api_key
SPRINGER_OPENACCESS_API_KEY=your_springer_openaccess_api_key
SPRINGER_TDM_API_KEY=your_springer_tdm_api_key
WILEY_API_KEY=your_wiley_api_key
IOP_papers_path=/absolute/path/to/iop_papers

# LLM providers
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
TOGETHER_API_KEY=your_together_api_key
COHERE_API_KEY=your_cohere_api_key
FIREWORKS_API_KEY=your_fireworks_api_key

# Model and embedding access
HF_TOKEN=your_huggingface_token
```

## Notes

- Keep all keys in your local `.env` file and never commit them to version control.
- For most users, the minimum setup is one publisher source plus one LLM provider.
- If you use Gemini models, use `GEMINI_API_KEY`.
- If you use the default embedding setup, make sure `HF_TOKEN` is available.

## Related Pages

- [Installation](installation.md)
- [Article Processing](../usage/article-processing.md)
- [Data Extraction](../usage/data-extraction.md)
- [RAG Configuration](../rag-config.md)
