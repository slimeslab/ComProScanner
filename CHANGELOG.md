## Unreleased

### Added

- Added DeepWiki integration badge to README.md for community Q&A support:
  - [Ask DeepWiki](https://deepwiki.com/slimeslab/ComProScanner)
- Added arXiv preprint badge to README.md:

  - [arXiv:2510.20362](https://arxiv.org/abs/2510.20362)

- [CITATION.cff](https://github.com/slimeslab/ComProScanner/blob/main/CITATION.cff) added for standardized citation information based on the latest release and arXiv preprint.

### Changed

- Moved the README badges section from html `p` tags to markdown format for better compatibility across platforms.

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

- README images updated with raw GitHub links for better reliability:
  - [ComProScanner Logo](https://raw.githubusercontent.com/aritraroy24/ComProScanner/main/assets/comproscanner_logo.png)
  - [ComProScanner Workflow](https://raw.githubusercontent.com/aritraroy24/ComProScanner/main/assets/overall_workflow.png)

## [0.1.3] - 04-11-2025

### Fixed

- **RecursiveCharacterTextSplitter** importing updated for latest _langchain_ version to avoid import errors:
  - Changed from `from langchain.text_splitter import RecursiveCharacterTextSplitter`
  - To `from langchain.text_splitter.recursive_character import RecursiveCharacterTextSplitter`

## [0.1.2] - 24-10-2025

### Added

- Link to ComProScanner preprint on arXiv in the documentation index page and README.md:
  - [arXiv:2510.20362](https://arxiv.org/abs/2510.20362)

## [0.1.1] - 22-10-2025

### Fixed

- README images updated with external image link to fix PyPI rendering issue.
  - [ComProScanner Logo](https://i.ibb.co/whHSbGvT/comproscanner-logo.png)
  - [ComProScanner Workflow](https://i.ibb.co/QWd2qd3/overall-workflow.png)

## [0.1.0] - 22-10-2025

### Added

- Initial release of ComProScanner.
