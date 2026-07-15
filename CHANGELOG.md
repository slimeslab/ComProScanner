# Unreleased

### Added

- Added `is_track_pdfs` and `track_pdfs_report_path` to `process_articles()` for local PDF workflows. When enabled (default), each processed PDF is recorded as a `filename<TAB>doi` entry in `logs/{keyword}_pdf_processed_dois.txt`, allowing re-runs to skip already-processed PDFs before any conversion or API calls. Falls back to scanning the output CSV when the tracking file does not yet exist.

- Centralised non-keyword default file paths (`results/failed_automated_articles.txt`, `agentic_evaluation_result.json`, `detailed_evaluation.json`) as class-level constants on `DefaultPaths` so they can be changed in one place.


### Changed

- Replaced the `cleaning_strategy` (`"full"`/`"basic"`) and `apply_advanced_cleaning` parameters on `DataCleaner`, `ComProScanner.clean_data()`, and the top-level `clean_data()` function with a single `cleaning_steps` parameter accepting either `"all"` (default) or a list of individually selectable step names: `abbreviation_filtering`, `element_validation`, `text_normalization`, `miller_indices`, `coefficient_expansion` (exposed via the new `CleaningStep` enum). Unicode subscript conversion and arithmetic/fraction resolution remain always-on, since the other steps depend on their output. `normalization` and `zero_coefficient` are no longer separate steps — both are folded into `coefficient_expansion`, which already performs them internally.


### Fixed

- Replaced the blind space-stripping behaviour (`key.replace(" ", "")`, which mangled descriptive composition text such as `"Bi4Ti3O12 ultrathin with oxygen vacancies"` into `"Bi4Ti3O12ultrathinwithoxygenvacancies"`) with a new optional `text_normalization` step that strips leading/trailing whitespace, collapses runs of multiple spaces down to one, and title-cases descriptive word tokens, while leaving formula segments and element-symbol sequences untouched.

- Fixed Miller-index handling in data cleaning: a bare 3-digit parenthetical crystal-plane notation (e.g. `"AlN (002)"`) was previously fed straight into the mandatory arithmetic/bracket resolver, which misread `(002)` as a bare-number coefficient bracket and merged its digits into the preceding element — silently producing `"AlN2"` instead of recognising it as a Miller index at all. The `miller_indices` cleaning step now detects this notation before arithmetic/bracket resolution runs, and — rather than stripping the notation and keeping the bare formula, which would collapse distinct surface-orientation entries for the same material onto the same dict key (e.g. `"AlN (002)"` and `"AlN (110)"` both becoming `"AlN"` and silently overwriting one property value with the other when merged) — drops the whole composition entry, tracking it in `filtered_compositions` the same way `abbreviation_filtering`/`element_validation` drops are tracked. When `miller_indices` is *not* selected, the same 3-digit-bracket detection now also stops both the arithmetic resolver and `coefficient_expansion` from silently merging the digits into a wrong formula; the composition is instead dropped later as an unresolved composition, the same as any other leftover bracket.

- Handled multi-word property keywords (e.g.,  _thermal conductivity_) for accurate Scopus search, uniform filename handling (`thermal conductivity` resolves to `thermal_conductivity_metadata.csv` or similar) and restoring the original form `thermal conductivity` in the data extraction RAG search query instead of `thermal_conductivity`. This fix is associated with [#5](https://github.com/slimeslab/ComProScanner/pull/5) and contributed by [@WilmerGaspar](https://github.com/WilmerGaspar).

- Fixed DOI-to-folder-name conversion across `extract_flow` (`RAGTool`, `GraphExtractorTool`, `EquationTool`, `DataExtractionFlow`, and all crew log/output folder paths) to also replace `:` with `_` (not just `/`), so DOIs like `10.1023/A:1015522900295` no longer raise `WinError 267: The directory name is invalid` on Windows and correctly resolve to their saved figure/vector-DB/log directories.

- Previously, a new `MultiModelEmbeddings` instance (and thus a fresh copy of the PhysBERT model) was loaded onto the GPU for every paper processed, because `RAGTool → VectorDatabaseManager → MultiModelEmbeddings` were all re-instantiated per paper. After certain number of papers this exhausted VRAM with `cudaErrorMemoryAllocation` (Refer to issue [#6](https://github.com/slimeslab/ComProScanner/issues/6)). This fix introduces a class-level `_hf_model_cache` dict on MultiModelEmbeddings so the tokenizer and model are loaded onto the GPU exactly once and shared as references across all subsequent instances. Also explicitly delete intermediate CUDA tensors and call `torch.cuda.empty_cache()` after each embedding call to prevent activation memory from accumulating within a paper's processing. Added the same cache flush in `VectorDatabaseManager.create_database` and `query_database` after `gc.collect()`. This fix is associated with PR [#7](https://github.com/slimeslab/ComProScanner/pull/7).


---

# 2026.05.19

### Added

- Added `SCIENCEDIRECT_INSTTOKEN` environment variable support in `ElsevierArticleProcessor` for off-campus remote access to subscription-based Elsevier articles and figures. When set, the token is sent as the `X-ELS-Insttoken` header in all ScienceDirect API requests and figure downloads. The variable is optional; omitting it does not affect on-campus access.

- New `value_error_thresholds` parameter added to both `evaluate_semantic()` and `evaluate_agentic()` for range-based absolute error tolerances on numeric property value comparisons:

  - Accepts a dict mapping `(min, max)` tuples to absolute error thresholds. Ranges are interpreted as **layers**: the narrowest range containing the ground-truth value determines the tolerance. For example, `(-150, 150): 1` applies only to values in (-150, -50) and (50, 150) when `(-50, 50): 0.5` is also present — no need for separate positive/negative sub-ranges. Tuple element order is irrelevant: `(-150, 150)` and `(150, -150)` are equivalent. Values outside all configured ranges fall back to exact comparison.

  - **Semantic evaluation**: handled inside `_is_value_in_range()` via the new `_get_error_threshold()` helper in `MaterialsDataSemanticEvaluator`.

  - **Agentic evaluation**: a new `GetValueErrorThresholdTool` (CrewAI `BaseTool`) is added to the composition evaluator agent when thresholds are configured. The agent calls this tool with the reference value to retrieve the tolerance before deciding on each numeric match. No tool is added and no prompt changes are made when no thresholds are provided.

- Exposed `value_error_thresholds` in public evaluation methods: `ComProScanner.evaluate_semantic()`, `ComProScanner.evaluate_agentic()`, `comproscanner.evaluate_semantic()`, and `comproscanner.evaluate_agentic()`.

- VLM-based graph data extraction added across all publishers and PDF processors:

  - New `GraphExtractorTool` — a CrewAI agent tool that reads saved figures for a given DOI and uses a vision LLM to extract composition-property value pairs from graphs and charts. Default VLM: `gemini/gemini-3-flash-preview`.

  - New `FigureExtractor` utility — shared helper for caption keyword-based figure filtering and saving, used by all article processors.

  - New `main_figure_keywords` parameter in `process_articles()` and `extract_composition_property_data()`, and new `vlm_model` and `related_figures_base_path` parameters in `extract_composition_property_data()`.

- New unit tests added for all three agent tools in `tests/test_agent_tools/`.

- Added `save_failed_pdf_report` and `failed_pdf_report_path` to `process_articles()`, with filename-derived DOI validation and failed-PDF reporting for local PDF workflows.

- Added `save_failed_automated_report` and `failed_automated_report_path` to `process_articles()` for automated publisher sources (Elsevier, Springer Nature, IOP, Wiley), mirroring the existing PDF failure report. Failed articles are written as tab-separated `doi`, `publisher`, `reason` entries to `results/failed_automated_articles.txt` by default.

- Added image-aware fallback in `DataExtractionFlow.identify_materials_data_presence()`:

  - The Materials Data Identifier still runs text RAG first.
  - If RAG returns `no`, the flow now checks saved DOI figures with VLM and upgrades the decision to `yes` when relevant graph/figure evidence is found (including doping concentration vs property plots where full formulas are absent).

- Added `is_store_unresolved_compositions` and `unresolved_compositions_file` parameters to `clean_data()` to optionally log split composition-property resolution statistics (`source`, `filtered`, `unresolved`, `resolved` counts) and persist filtered and unresolved composition keys in a JSON file keyed by DOI under `"filtered"` and `"unresolved"` top-level keys.

- Added explicit Equation Tool model control:

  - New `equation_model` parameter in `extract_composition_property_data()` (threaded through `DataExtractionFlow` and `CompositionExtractionCrew` into `EquationTool`).
  - EquationTool model precedence is now: `equation_model` argument -> API-key-based auto-selection.

- Clarified Equation Tool instruction customization in extraction docs and API:

  - `formula_instruction` remains available in `extract_composition_property_data()` for domain-specific formula-derivation guidance, while preserving the built-in default instruction when unset.

### Changed

- Versioning scheme migrated from [Semantic Versioning](https://semver.org/) (SemVer) to [Calendar Versioning](https://calver.org/) (CalVer) using the `YYYY.MM.DD` format. Starting from this release, version numbers reflect the release date rather than an incrementing major/minor/patch scheme.

### Fixed

- `_parse_json_output()` now recovers JSON from mixed-text crew outputs (e.g. `Thought: … { "json": "here" }`) by scanning for the first `{` / `[` and last `}` / `]` and retrying `json.loads()` on the extracted substring, before falling back to `ast.literal_eval()`.

- Composition formatter agent now verifies `MaterialParserTool` output for incomplete variable substitution (e.g. `(1-x-y)` partially resolved as `(0.9-0.010)`) and overrides with the correct fully-substituted BODMAS expression when the tool is wrong.

- `process_articles()` now routes user-provided `doi_list` by `general_publisher` from metadata and sends each DOI only to its matching source processor.

- PNG, GIF, and WEBP figures now convert correctly to JPEG: transparent images are composited onto a white background, animated GIFs are pinned to frame 0, and two additional Springer Nature CDN URL patterns are tried to improve download success for these formats.

- Added and updated tests for new extraction-flow behavior:

  - EquationTool model selection tests now cover explicit arg override, env override, and updated model defaults.
  - DataExtractionFlow tests now cover figure-based materials-data fallback and `equation_model` forwarding into `CompositionExtractionCrew`.

---
## [0.1.6] - 2026-04-02
### Changed
- Updated [README.md](README.md), [CITATION.cff](CITATION.cff) and docs with the published version (advance article) of the ComProScanner paper in _Digital Discovery_ as fully open access:
  - [ComProScanner: a multi-agent based framework for composition-property structured data extraction from scientific literature](https://doi.org/10.1039/D5DD00521C) 

### Added
- Guide for API key creation for various LLM providers and publisher APIs added to the documentation at `docs/getting-started/api-key-guide.md` with detailed instructions for each provider.

### Fixed
- Model prefix handling in `rag_tool.py` standardized to reflect the docs.
- `HF_TOKEN` documentation clarified as optional — only required for gated or private Hugging Face models.

---

## [0.1.5] - 2026-02-08

### Added

- Data related to comparison with other agentic data extraction frameworks added for the ComProScanner paper in the `examples/piezo_test/comparing_existing_frameworks` folder.

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

- OAWorks API is replaced with OpenAlex API as OAWorks is no longer available.

- Empty/corrupted PDF handled in `pdf_processor.py` and `wiley_processor.py` to avoid having GLYPH errors during text extraction.

- Data extraction failures fixed if composition-property text data is empty.

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

---
## [0.1.4] - 2025-12-02

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

---
## [0.1.3] - 2025-11-04

### Fixed

- **RecursiveCharacterTextSplitter** importing updated for latest _langchain_ version to avoid import errors:
  - Changed from `from langchain.text_splitter import RecursiveCharacterTextSplitter`
  - To `from langchain.text_splitter.recursive_character import RecursiveCharacterTextSplitter`

---
## [0.1.2] - 2025-10-24

### Added

- Link to ComProScanner preprint on arXiv in the documentation index page and README.md:
  - [arXiv:2510.20362](https://arxiv.org/abs/2510.20362)

---
## [0.1.1] - 2025-10-22

### Fixed

- README images updated with external image link to fix PyPI rendering issue.
  - [ComProScanner Logo](https://i.ibb.co/whHSbGvT/comproscanner-logo.png)
  - [ComProScanner Workflow](https://i.ibb.co/QWd2qd3/overall-workflow.png)

---
## [0.1.0] - 2025-10-22

### Added

- Initial release of ComProScanner.
