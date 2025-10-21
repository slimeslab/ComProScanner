# Metadata Collection

The metadata collection module helps you find and filter metadata for relevant scientific articles from Scopus database using Scopus Search API based on query property keywords.

## Basic Usage

```python
from comproscanner import ComProScanner

# Initialize scanner
scanner = ComProScanner(main_property_keyword="piezoelectric")

# Collect metadata
scanner.collect_metadata()
```

## Parameters

### Required Parameters

#### :material-square-medium:`main_property_keyword` _(str)_

The main property of interest for your research. This keyword will be used to generate search queries for metadata collection.

### Optional Parameters

#### :material-square-medium:`base_queries` _(list)_

List of base search queries related to the main property. If not provided, the main property keyword will be used as the sole base query.

#### :material-square-medium:`extra_queries` _(list)_

List of additional search queries to expand the search scope.

#### :material-square-medium:`start_year` _(int)_

Starting publication year for filtering articles. It must be bigger than `end_year` as the search is performed backwards in time.

#### :material-square-medium:`end_year` _(int)_

Ending publication year for filtering articles.

!!! info "Default Values"

    :material-square-small:**`base_queries`** = None<br>:material-square-small:**`extra_queries`** = None<br>:material-square-small:**`start_year`** = current year<br>:material-square-small:**`end_year`** = current year - 2

## Advanced Examples

### Example 1: Broad Property Search

```python
scanner = ComProScanner(main_property_keyword="magnetic")

scanner.collect_metadata(
    base_queries=[
        "magnetic",
        "magnetism",
        "ferromagnetic",
        "antiferromagnetic"
    ],
    extra_queries=[
        "materials",
        "thin films",
        "nanoparticles"
    ]
)
```

### Example 2: Recent Publications Only

```python
from datetime import datetime

current_year = datetime.now().year

scanner.collect_metadata(
    base_queries=["superconductivity"],
    start_year=current_year,
    end_year=current_year - 1  # Last year only
)
```

## Output Format

Similar to the following example, metadata for all relevant articles is stored in a CSV file:

```csv title="piezoelectric_metadata.csv" linenums="1"
doi,publication_name,issn,scopus_id,article_title,article_type,metadata_publisher,general_publisher
10.1016/j.seppur.2024.130955,Separation and Purification Technology,13835866,SCOPUS_ID:85211235836,Multistage gradient crystallization study towards lithium carbonate crystal growth,Article,Elsevier B.V.,elsevier
// More articles...
```

## Next Steps

- Learn about [Article Processing](article-processing.md)
- Understand the [complete workflow](../workflow/overview.md)
- Explore [RAG configuration](../rag-config.md)
