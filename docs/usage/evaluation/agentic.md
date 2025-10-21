# Agentic Evaluation

This method leverages the reasoning capabilities of LLMs to evaluate extraction quality with human-like understanding.

## Basic Usage

```python
from comproscanner import evaluate_agentic

results = evaluate_agentic(
    ground_truth_file="ground_truth.json",
    test_data_file="extracted_results.json",
    output_file="agentic_eval.json"
)
```

## Parameters

### Required Parameters

#### :material-square-medium:`ground_truth_file` _(str)_

Path to ground truth data created by human experts.

#### :material-square-medium:`test_data_file` _(str)_

Path to extracted results to be evaluated.

### Optional Parameters

#### :material-square-medium:`weights` _(dict)_

Dictionary specifying weights for each metric during evaluation for ensuring scoring based on importance. The total should sum to 1.0.

#### :material-square-medium:`output_file` _(str)_

Path to save the evaluation results.

#### :material-square-medium:`extraction_agent_model_name` _(str)_

Name of the LLM model used for data extraction (e.g., "gpt-4o-mini").

#### :material-square-medium:`is_synthesis_evaluation` _(bool)_

Whether to evaluate synthesis-related information.

#### :material-square-medium:`llm` _(LLM)_

An instance of the LiteLLM class. Read more about LiteLLM instance from CrewAI [here](https://docs.crewai.com/en/concepts/llms).

!!! info "Default Values"

    :material-square-small:**`weights`** = {
        "compositions_property_values": 0.3,
        "property_unit": 0.1,
        "family": 0.1,
        "method": 0.1,
        "precursors": 0.15,
        "characterization_techniques": 0.15,
        "steps": 0.1
    }<br>:material-square-small:**`output_file`** = "agentic_evaluation_result.json"<br>:material-square-small:**`extraction_agent_model_name`** = "gpt-4o-mini"<br>:material-square-small:**`is_synthesis_evaluation`** = True<br>:material-square-small:**`llm`** = LLM(model="o3-mini")

## How It Works

```mermaid
graph TB
    A[Ground Truth] --> C[LLM Agent]
    B[Test Data] --> C
    C --> D[Contextual Analysis]
    D --> E[Reasoning]
    E --> F[Scoring]
    F --> G[Evaluation Results]
```

### Custom LLM

```python
from comproscanner import evaluate_agentic
from crewai import LLM

# Use GPT-4
gpt4_llm = LLM(
    model="gpt-4-turbo",
    temperature=0.5,
    max_tokens=4096
)

results = evaluate_agentic(
    ground_truth_file="ground_truth.json",
    test_data_file="test_data.json",
    llm=gpt4_llm
)
```

## Output Format

```json
{
  "agent_model_name": "DeepSeek-V3-0324",
  "overall_accuracy": 0.8238965626040629,
  "overall_composition_accuracy": 0.8998672619047621,
  "overall_synthesis_accuracy": 0.7479258633033636,
  "total_items": 100,
  "absolute_classification_metrics": {
    "true_positives": 1763,
    "false_positives": 326,
    "false_negatives": 352,
    "precision": 0.8439444710387746,
    "recall": 0.8335697399527187,
    "f1_score": 0.8387250237868696
  },
  "normalized_classification_metrics": {
    "true_positives": 77.77361988011987,
    "false_positives": 31.98932808857808,
    "false_negatives": 14.577932609057608,
    "precision": 0.7085598676003058,
    "recall": 0.8421474007081151,
    "f1_score": 0.7695996052131786
  },
  "item_results": {
    "10.1016/j.jeurceramsoc.2025.117193": {
      "overall_match": true,
      "field_scores": {
        "composition_data": 1.0,
        "synthesis_data": 0.982
      },
      "overall_score": 0.991,
      "absolute_classification_metrics": {
        "true_positives": 23,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 1.0
      },
      "normalized_classification_metrics": {
        "true_positives": 0.9819999999999998,
        "false_positives": 0.009000000000000003,
        "false_negatives": 0.009000000000000003,
        "precision": 0.9909182643794148,
        "recall": 0.9909182643794148,
        "f1_score": 0.9909182643794148
      },
      "details": {
        "composition_data": {
          "property_unit": {
            "match_value": 1,
            "reference": "pC/N",
            "test": "pC/N"
          },
          "family": {
            "match_value": 1,
            "reference": "PbNb2O6",
            "test": "PbNb2O6-based"
          },
          "compositions_property_values": {
            "reference": {
              "Pb0.95K0.1[Nb0.96Ta0.04]2O6": 44,
              "Pb0.9K0.2[Nb0.96Ta0.04]2O6": 54,
              "Pb0.85K0.3[Nb0.96Ta0.04]2O6": 93,
              "Pb0.8K0.4[Nb0.96Ta0.04]2O6": 141
            },
            "test": {
              "Pb0.95K0.1[Nb0.96Ta0.04]2O6": 44,
              "Pb0.9K0.2[Nb0.96Ta0.04]2O6": 54,
              "Pb0.85K0.3[Nb0.96Ta0.04]2O6": 93,
              "Pb0.8K0.4[Nb0.96Ta0.04]2O6": 141
            },
            "key_matches": [
              {
                "reference_key": "Pb0.95K0.1[Nb0.96Ta0.04]2O6",
                "test_key": "Pb0.95K0.1[Nb0.96Ta0.04]2O6",
                "match_value": 1
              },
              {
                "reference_key": "Pb0.9K0.2[Nb0.96Ta0.04]2O6",
                "test_key": "Pb0.9K0.2[Nb0.96Ta0.04]2O6",
                "match_value": 1
              },
              {
                "reference_key": "Pb0.85K0.3[Nb0.96Ta0.04]2O6",
                "test_key": "Pb0.85K0.3[Nb0.96Ta0.04]2O6",
                "match_value": 1
              },
              {
                "reference_key": "Pb0.8K0.4[Nb0.96Ta0.04]2O6",
                "test_key": "Pb0.8K0.4[Nb0.96Ta0.04]2O6",
                "match_value": 1
              }
            ],
            "value_matches": [
              {
                "key": "Pb0.95K0.1[Nb0.96Ta0.04]2O6",
                "reference_value": 44,
                "test_value": 44,
                "match_value": 1
              },
              {
                "key": "Pb0.9K0.2[Nb0.96Ta0.04]2O6",
                "reference_value": 54,
                "test_value": 54,
                "match_value": 1
              },
              {
                "key": "Pb0.85K0.3[Nb0.96Ta0.04]2O6",
                "reference_value": 93,
                "test_value": 93,
                "match_value": 1
              },
              {
                "key": "Pb0.8K0.4[Nb0.96Ta0.04]2O6",
                "reference_value": 141,
                "test_value": 141,
                "match_value": 1
              }
            ],
            "pair_matches": [
              {
                "reference_pair": {
                  "Pb0.95K0.1[Nb0.96Ta0.04]2O6": 44
                },
                "test_pair": {
                  "Pb0.95K0.1[Nb0.96Ta0.04]2O6": 44
                },
                "match_value": 1
              },
              {
                "reference_pair": {
                  "Pb0.9K0.2[Nb0.96Ta0.04]2O6": 54
                },
                "test_pair": {
                  "Pb0.9K0.2[Nb0.96Ta0.04]2O6": 54
                },
                "match_value": 1
              },
              {
                "reference_pair": {
                  "Pb0.85K0.3[Nb0.96Ta0.04]2O6": 93
                },
                "test_pair": {
                  "Pb0.85K0.3[Nb0.96Ta0.04]2O6": 93
                },
                "match_value": 1
              },
              {
                "reference_pair": {
                  "Pb0.8K0.4[Nb0.96Ta0.04]2O6": 141
                },
                "test_pair": {
                  "Pb0.8K0.4[Nb0.96Ta0.04]2O6": 141
                },
                "match_value": 1
              }
            ],
            "total_ground_truth_keys": 4,
            "total_match": 4,
            "missing_keys": [],
            "extra_keys": [],
            "key_match_ratio": 1.0,
            "value_match_ratio": 1.0,
            "pair_match_ratio": 1.0,
            "overall_match_ratio": 1.0,
            "similarity_score": 1.0,
            "match": true
          }
        },
        "synthesis_data": {
          "method": {
            "match_value": 1,
            "reference": "solid-state reaction",
            "test": "solid-state reaction",
            "similarity": 1.0,
            "match": true
          },
          "precursors": {
            "reference": ["PbO", "K2CO3", "Nb2O5", "Ta2O5"],
            "test": ["PbO", "K2CO3", "Nb2O5", "Ta2O5"],
            "matches": [
              {
                "reference_item": "PbO",
                "test_item": "PbO",
                "match_value": 1
              },
              {
                "reference_item": "K2CO3",
                "test_item": "K2CO3",
                "match_value": 1
              },
              {
                "reference_item": "Nb2O5",
                "test_item": "Nb2O5",
                "match_value": 1
              },
              {
                "reference_item": "Ta2O5",
                "test_item": "Ta2O5",
                "match_value": 1
              }
            ],
            "total_ground_truth_items": 4,
            "total_match": 4,
            "missing_items": [],
            "extra_items": [],
            "similarity": 1.0,
            "match": true
          },
          "characterization_techniques": {
            "reference": [
              "XRD",
              "SEM",
              "EDS",
              "XPS",
              "Raman spectroscopy",
              "d33 tester",
              "LCR meter"
            ],
            "test": [
              "XRD",
              "SEM",
              "EDS",
              "XPS",
              "Raman spectroscopy",
              "Quasi-static d33 tester",
              "LCR meter"
            ],
            "matches": [
              {
                "reference_item": "XRD",
                "test_item": "XRD",
                "match_value": 1
              },
              {
                "reference_item": "SEM",
                "test_item": "SEM",
                "match_value": 1
              },
              {
                "reference_item": "EDS",
                "test_item": "EDS",
                "match_value": 1
              },
              {
                "reference_item": "XPS",
                "test_item": "XPS",
                "match_value": 1
              },
              {
                "reference_item": "Raman spectroscopy",
                "test_item": "Raman spectroscopy",
                "match_value": 1
              },
              {
                "reference_item": "d33 tester",
                "test_item": "Quasi-static d33 tester",
                "match_value": 1
              },
              {
                "reference_item": "LCR meter",
                "test_item": "LCR meter",
                "match_value": 1
              }
            ],
            "total_ground_truth_items": 7,
            "total_match": 7,
            "missing_items": [],
            "extra_items": [],
            "similarity": 1.0,
            "match": true
          },
          "steps": {
            "match_value": 0.82,
            "reference_steps": [
              "Raw materials were weighed according to stoichiometric ratio and ball-milled with zirconia balls in nylon jars for 12 hours.",
              "Additional 2 wt% PbO was introduced for compensating volatilization during sintering process.",
              "After drying at 120\u00b0C, powders were calcined at 900\u00b0C for 4 hours and re-milled for another 6 hours.",
              "Dried powders were mixed with polyvinyl alcohol binder and sieved through screen mesh.",
              "Particles were pressed into pellets with diameter of 12 mm and thickness around 1.2 mm under uniaxial pressure of 15 MPa.",
              "Green pellets were fired at 500\u00b0C for 2 hours for binder burn-out.",
              "Specimens were sintered at 1190-1250\u00b0C for 1 hour.",
              "Cooling process was controlled at 3\u00b0C/min till 600\u00b0C to prevent ceramics from cracking.",
              "Natural cooling without quenching was performed below 600\u00b0C.",
              "Sintered pellets were coated with silver paste on both surfaces and dried at 120\u00b0C for 30 minutes.",
              "Samples were polarized in silicon oil at 150\u00b0C with DC electric field up to 4 kV/mm for 30 minutes."
            ],
            "test_steps": [
              "Weigh raw materials according to stoichiometric ratio",
              "Ball-mill with zirconia balls in nylon jars for 12 h",
              "Add 2 wt% PbO to compensate volatilization",
              "Dry at 120 \u00b0C",
              "Calcinate at 900 \u00b0C for 4 h",
              "Re-mill for 6 h",
              "Mix with PVA binder and sieve",
              "Press into pellets (12 mm diameter, 1.2 mm thickness) under 15 MPa",
              "Fire at 500 \u00b0C for 2 h for binder burn-out",
              "Sinter at 1190-1250 \u00b0C for 1 h",
              "Control cooling at 3 \u00b0C/min till 600 \u00b0C",
              "Natural cooling below 600 \u00b0C"
            ],
            "steps_match": "true"
          }
        }
      }
    }
  }
  // More items...
}
```

## Next Steps

- Compare with [Semantic Evaluation](semantic.md)
- Visualize results with [Evaluation Visualizer](../visualization/eval-viz.md)
- Learn about [RAG Configuration](../../rag-config.md)
