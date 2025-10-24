# Evaluation Overview

ComProScanner provides both semantic and agentic methods for evaluating extraction quality. The following sections outline the approaches, advantages, and use cases for each method.

## Evaluation Methods

### Semantic Evaluation

**Approach**: Uses embedding models to compute similarity between extracted and ground truth data.

**Advantages**:

- Fast and cost-effective
- Physics-aware with PhysBERT embeddings (default)
- Consistent results

**Best For**:

- Quick quality assessment
- Large-scale evaluations
- Budget-conscious projects

### Agentic Evaluation

**Approach**: Uses specialized LLM agents to evaluate extraction quality with nuanced understanding.

**Advantages**:

- More accurate assessment
- Better context understanding
- Handles edge cases

**Best For**:

- High-stakes evaluations
- Complex compositions
- Detailed analysis

## Comparison

| Feature               | Semantic | Agentic   |
| --------------------- | -------- | --------- |
| Speed                 | Fast     | Slower    |
| Cost                  | Low      | Higher    |
| Accuracy              | Good     | Excellent |
| Context Understanding | Limited  | Advanced  |
| Reproducibility       | High     | Moderate  |

## Evaluation Metrics

Both methods provide:

- **Overall Accuracy**: Combined accuracy across both composition accuracy and synthesis accuracy
- **Composition Accuracy**: Custom weight-based accuracy of extracted composition-property based data (compositions_property_values, property_unit, family)
- **Synthesis Accuracy**: Custom weight-based accuracy of synthesis related data (method, precursors, steps, characterization_techniques)
- **Classification Metrics**: Standard Precision/Recall/F1 metrics for detailed performance analysis
- **Normalized Classification Metrics**: Classification metrics normalized to ensure an equitable comparison between articles with significant disparities in the quantity of extractable information.

To read more about the evaluation metrics, please refer the journal article [here](https://arxiv.org/abs/2510.20362).

## Quick Start

### Semantic Evaluation

```python
from comproscanner import evaluate_semantic

results = evaluate_semantic(
    ground_truth_file="ground_truth.json",
    test_data_file="extracted_results.json",
    output_file="semantic_eval.json"
)
```

### Agentic Evaluation

```python
from comproscanner import evaluate_agentic

results = evaluate_agentic(
    ground_truth_file="ground_truth.json",
    test_data_file="extracted_results.json",
    output_file="agentic_eval.json"
)
```

## Detailed Guides

- [Semantic Evaluation](semantic.md) - Semantic-based evaluation
- [Agentic Evaluation](agentic.md) - LLM agent-based evaluation

## Next Steps

- Continue to [Semantic Evaluation](semantic.md) or [Agentic Evaluation](agentic.md)
- Explore [Visualization](../visualization/overview.md)
- Learn about [RAG Configuration](../../rag-config.md)
