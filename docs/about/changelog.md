## Unreleased

### Added

- New parameter `apply_advanced_cleaning` added to data cleaning methods in `data_cleaner.py`. When set to `True`, it triggers the advanced cleaning pipeline.

- Advanced composition cleaning methods in `data_cleaner.py`:

  - `_remove_miller_indices()` - Removes crystal plane notations from chemical formulas
  - `_remove_zero_coefficient_elements()` - Removes elements with zero coefficients
  - `_normalize_coefficients()` - Removes trailing zeros from coefficients
  - `_expand_leading_and_trailing_coefficients()` - Expands leading/trailing coefficient patterns
  - `_expand_parenthetical_coefficients()` - Expands nested bracket coefficients

- Enhanced documentation in `docs/usage/data-cleaning.md`:

  - Added `apply_advanced_cleaning` parameter documentation
  - Added Mermaid process flow diagram showing cleaning stages
  - Added advanced cleaning examples with tables for each transformation type

- Template for GitHub issues added to [.github/ISSUE_TEMPLATE](https://github.com/slimeslab/ComProScanner/tree/main/.github/ISSUE_TEMPLATE) for the following topics:

  - bug reports
  - feature requests
  - documentation improvements
  - support questions

- [Changelog page](https://slimeslab.github.io/ComProScanner/about/changelog/) added in the documentation. Also, [CHANGELOG.md](https://github.com/slimeslab/ComProScanner/blob/main/CHANGELOG.md) linked in [README.md](https://github.com/slimeslab/ComProScanner/blob/main/README.md).

- DeepWiki integration badge added to README.md for community Q&A support:

  - [Ask DeepWiki](https://deepwiki.com/slimeslab/ComProScanner)

- arXiv preprint badge added to README.md:

  - [arXiv:2510.20362](https://arxiv.org/abs/2510.20362)

- [CITATION.cff](https://github.com/slimeslab/ComProScanner/blob/main/CITATION.cff) added for standardized citation information based on the latest release and arXiv preprint.

### Fixed

- CSV progress tracking in `elsevier_processor.py`:

  - DtypeWarning resolved by adding `dtype=str, low_memory=False` to `pd.read_csv()`
  - Data loss issue fixed with immediate CSV persistence for processed articles
  - Sleep delays optimized for batch writes

- Type annotation warnings in documentation build (griffe/mkdocstrings):
  - Added return type annotations to function signatures in `comproscanner.py`
  - Added return type annotations to all visualization functions in `data_visualizer.py` and `eval_visualizer.py`
  - Fixed parameter type format in docstrings from colon to comma notation
  - Added `TYPE_CHECKING` conditional imports for matplotlib Figure type
  - Fixed `**kwargs` type annotations across multiple modules

- Numbered list formatting in `docs/about/contribution.md`:
  - Fixed list continuation by using 4-space indentation for code blocks and nested lists
  - Disabled format on save for Markdown files in `.vscode/settings.json`

- GitHub Actions CI disk space issue:
  - Added `--no-cache-dir` flag to pip install to reduce disk usage

### Changed

- README badges section converted from HTML to markdown format for better compatibility across platforms.

## [0.1.4] - 02-12-2025

### Added

- New function `clean_data()` added for improved data cleaning and preprocessing instead of integrating it into data extraction function.

- New documentation page for Data Cleaning added:

  - docs/usage/data-cleaning.md
  - Added to mkdocs.yml navigation.

- New API overview documentation page added:

  - docs/api.md
  - Added to mkdocs.yml navigation.
  - New mkdocstrings configuration added to mkdocs.yml for automatic API documentation generation.

- New tests added for remaining utils functions.

- Added pytest coverage tracking (50%) using `pytest-cov` and coverage report generation using _codecov_.

### Fixed

- Tests updated to reflect changes in data cleaning process.

### Removed

- Arguments related to data cleaning removed from data extraction function.

### Changed

- README images updated with raw GitHub links for better reliability: [ComProScanner Logo](https://raw.githubusercontent.com/aritraroy24/ComProScanner/main/assets/comproscanner_logo.png), [ComProScanner Workflow](https://raw.githubusercontent.com/aritraroy24/ComProScanner/main/assets/overall_workflow.png)

## [0.1.3] - 04-11-2025

### Fixed

- **RecursiveCharacterTextSplitter** importing updated for latest _langchain_ version to avoid import errors:
  - Changed from `from langchain.text_splitter import RecursiveCharacterTextSplitter`
  - To `from langchain.text_splitter.recursive_character import RecursiveCharacterTextSplitter`

## [0.1.2] - 24-10-2025

### Added

- Link to ComProScanner preprint on arXiv in the documentation index page and README.md: [arXiv:2510.20362](https://arxiv.org/abs/2510.20362)

## [0.1.1] - 22-10-2025

### Fixed

- README images updated with external image link to fix PyPI rendering issue. [ComProScanner Logo](https://i.ibb.co/whHSbGvT/comproscanner-logo.png), [ComProScanner Workflow](https://i.ibb.co/QWd2qd3/overall-workflow.png)

## [0.1.0] - 22-10-2025

### Added

- Initial release of ComProScanner.
