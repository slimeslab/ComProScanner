# Data Cleaning

The data cleaning module helps remove entries based on abbreviations, periodic elements and resolve arithmetic expressions, fractional compositions, etc. along with bracket standardization in extracted chemical formulas.

## Basic Usage

```python
from comproscanner import ComProScanner

# Initialize scanner
scanner = ComProScanner(main_property_keyword="piezoelectric")

# Clean extracted data
scanner.clean_data(
    json_results_file="extracted_results.json"
)
```

## Parameters

### Required Parameters

#### :material-square-medium:`json_results_file` _(str)_

Path to the JSON results file containing extracted data that needs to be cleaned.

### Optional Parameters

#### :material-square-medium:`is_save_separate_results` _(bool)_

Whether to save separate cleaned results files.

#### :material-square-medium:`cleaned_json_results_file` _(str)_

Path to the cleaned JSON results file with articles having relevant composition-property data.

#### :material-square-medium:`is_save_composition_property_file` _(bool)_

Whether to save composition-property values to a separate file as a dictionary.

#### :material-square-medium:`composition_property_file` _(str)_

Path to the cleaned composition-property file containing a dictionary of composition-property data.

#### :material-square-medium:`cleaning_strategy` _(str)_

The cleaning strategy to be used. It can be either `full` or `basic`. While comprehensive cleaning including abbreviation removal, arithmetic resolution, bracket standardization, etc., are done for both strategies, the `full` strategy ensures entries with only periodic elements in the composition.
!!! info "Default Values"

    :material-square-small:**`is_save_separate_results`** = True<br>:material-square-small:**`cleaned_json_results_file`** = "cleaned_results.json"<br>:material-square-small:**`is_save_composition_property_file`** = True<br>:material-square-small:**`composition_property_file`** = "composition_property.json"<br>:material-square-small:**`cleaning_strategy`** = "full"

## Next Steps

- Learn about [Evaluation](evaluation/overview.md)
- Explore [Visualization](visualization/overview.md)
- Configure [Advanced RAG](../rag-config.md)
