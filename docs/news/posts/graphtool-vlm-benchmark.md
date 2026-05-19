---
date:
  created: 2026-05-08
authors:
  - aritraroy24
cover_image: https://i.ibb.co/jPJf86Qg/comproscanner-vlm-integration.png
description: >-
  ComProScanner 2026.05.19 introduces GraphExtractorTool — VLM-based graph
  data extraction benchmarked across 4 models on 50 piezoelectric articles.
  Gemini-3-Flash-Preview leads with 0.97 composition accuracy.
---

# ComProScanner Now Has Graph Extraction: Benchmarking VLMs on Piezoelectric Data

<!-- more -->

---
<div align="center">
<img src="https://i.ibb.co/jPJf86Qg/comproscanner-vlm-integration.png" id="vlm-integration" style="width:100%; max-width:700px;" alt="ComProScanner VLM Integration">
<p style="font-size: 10px;"><i><b>Image Credit:</b> ChatGPT Images 2.0</i></p>
</div>

## Overview

The [2026.05.19](https://pypi.org/project/comproscanner/2026.05.19) release of ComProScanner introduces several substantive additions to the framework, the most architecturally significant being the **GraphExtractorTool**: a vision language model (VLM) based agent tool that reads saved scientific figures and extracts composition–property value pairs directly from graphs and charts embedded in research articles. This post documents the rationale for including graph-based extraction, the methodology by which VLM candidates were selected and the results of a structured benchmark conducted across 50 piezoelectric articles from the existing test corpus. **Gemini-3-Flash-Preview** outperformed all other models with an average score of **0.96** in all metrics. Furthermore, an **EquationTool** has been added to the framework for generating chemical compositions by understanding the element replacement logic and XRD patterns. Finally, additional changes accompanying this release are summarised briefly.

---

## 1. Motivation: The Graph Extraction Gap

Prior versions of ComProScanner extracted composition–property data exclusively from the textual and tabular content of articles, using a retrieval-augmented generation (RAG) pipeline followed by a multi-agent composition extraction crew in an automated fashion with the help of Text and Data Mining (TDM) API keys provided by the journal publishers, described in detail in our [ComProScanner paper](https://doi.org/10.1039/D5DD00521C) published in _**Digital Discovery, RSC**_<sup>1</sup>. While this approach performs effectively when data are reported in tabular form or stated explicitly in prose, a substantial proportion of piezoelectric and related materials literature reports key property values _only_ in graphical form. Although such figures have been attempted to be processed<sup>2,3,4,5,6</sup>, until now, no automated publisher-to-dataset creating single framework existed for handling composition–property data with high accuracy.

Two complementary mechanisms have been introduced to address this:

1. **`GraphExtractorTool`** — a CrewAI `BaseTool` that, given a DOI, reads all saved figures for that article and passes them to a VLM with a structured extraction prompt, returning composition–property value pairs in the standard ComProScanner JSON schema.
2. **Image-aware fallback in `DataExtractionFlow`** — the Materials Data Identifier now runs text RAG first; if RAG returns `no`, the flow checks saved DOI figures via VLM and upgrades the decision to `yes` when relevant graphical evidence is found. This prevents articles with graph-only data from being silently discarded before extraction even begins.

A companion **`FigureExtractor`** utility handles caption-keyword–based filtering (if specified, otherwise saves all) and JPEG conversion (including transparent-PNG compositing onto white, animated-GIF frame pinning) and is shared across all publisher processors. Current ComProScanner overall architecture is represented in [Fig. 1](#overall-workflow).
<div align="center">
<img src="https://i.ibb.co/FG0XK9j/overall-workflow.png" alt="overall workflow graphtool" style="width:100%; max-width:700px;" id="overall-workflow" alt="ComProScanner Overall Workflow">
</div>
<p style="font-size: 10px;">
<b>Fig. 1:</b> Overall workflow diagram of ComProScanner framework incorporating the GraphExtractorTool and EquationTool. The flow is separated in four distinct operational phases, distinguished with four different colour regions: (a) metadata retrieval (yellow), (b) article collection (purple), (c) information extraction (green) and (d) evaluation, post-processing and dataset creation (brown).
</p>

The CrewAI-based multi-agentic information extraction phase flowchart is represented in detail in [Fig. 2](#flow-diagram). The _Materials Data Identifier_ agent now includes image-aware RAG, which is invoked when the initial text-based RAG step fails to find relevant data. `Composition-Property Data Extractor` is the agent responsible for invoking the `GraphExtractorTool` when the flow reaches the graph extraction step. Moreover, the agent has access to the `EquationTool` for generating chemical compound formulas including solid-solutions, alloys and doped systems by understanding the text regarding element replacement logic and XRD pattern if XRD images are available.
<div align="center">
<img src="https://i.ibb.co/tTYHqtxP/flow-diagram.png" alt="flow diagram" style="width:100%; max-width:500px;" id="flow-diagram" alt="ComProScanner Flow Diagram">
</div>
<p style="font-size: 10px;">
<b>Fig. 2:</b> Flow diagram of ComProScanner framework's information extraction process incorporating the image-aware RAGTool, GraphExtractorTool, EquationTool and Material-ParserTool. The CrewAI-based extraction system is comprised of five specialised agents. The process begins with a property identifier agent ((a) RAG Crew) that leverages Retrieval-Augmented Generation (RAG) technology to filter relevant articles. The remaining four agents are strategically organised into two parallel functional subgroups: one dedicated to composition data extraction ((b) composition crew set) and the other focused on synthesis information collection ((c) synthesis crew set). Each subgroup implements a sequential two-agent architecture—the first agent extracts raw data while the second performs formatting and standardisation.
</p>

---

## 2. VLM Model Selection: LMArena Diagram Leaderboard

Because the primary use case is reading scientific charts, not general image captioning, model selection was grounded in the **LMArena VLM Leaderboard (Diagram category)**<sup>7</sup>, which ranks models by human preference votes on diagram-understanding tasks and reports Arena ELO scores. Another critical factor was cost: the input token price for processing images with the VLM, as this directly impacts the scalability of the tool for building large datasets. Therefore, the selection process involved applying simultaneous filters on both ELO score and input cost to identify models that are not only effective at diagram comprehension but also economically viable for large-scale use. We selected models with an Arena ELO score of at least 1,250 and an input cost of less than $1.50 per million tokens, ensuring a balance between performance and affordability. As of 15 April 2026, this yielded four models for evaluation: Gemini-3-Flash-Preview<sup>8</sup>, Gemini-2.5-Pro<sup>9</sup>, GPT-5-Chat-Latest<sup>10</sup> and GPT-5.1<sup>11</sup>  ([Fig. 3](#lmarena-leaderboard)).

<div align="center">
<img src="https://i.ibb.co/HT1yLBfx/lmarena-vlm-leaderboard.png" alt="lmarena diagram leaderboard" style="width:100%; max-width:700px;" id="lmarena-leaderboard" alt="LMArena Leaderboard for VLMs">
</div>
<p style="font-size: 10px;">
<b>Fig. 3:</b> LMArena Leaderboard for VLMs (Diagram category) as of April 2026. The region highlighted in pink indicates the models that were selected for evaluation based on the criteria of having an Arena ELO score of at least 1,250 and an input cost of less than $1.50 per 1 million tokens.
</p>

---

## 3. Results and Discussion

The benchmark was conducted on 50 articles randomly selected from 73 DOIs in the existing piezoelectric ceramic article set that contained related figures. Along with GraphExtractorTool, an **EquationTool** has been added to the _Composition-Property Data Extractor agent_ (refer to the [ComProScanner paper](https://doi.org/10.1039/D5DD00521C) for details) for generating chemical compositions by understanding the element replacement logic and XRD patterns. `claude-sonnet-4-6` has been used for generating the formulas based on the text and XRD patterns. Other settings have been kept as default mentioned in the original paper. For saving the figures, a set of keywords related to piezoelectric coefficient (d<sub>33</sub>) and XRD patterns were used to filter the figures (refer to the `vlm_test_example.py` script on [GitHub](https://github.com/slimeslab/ComProScanner/tree/main/examples/vlm_piezo_test/vlm_test_example.py)) to reduce the API costs. However, it should be noted that the filtering process is not perfect and some relevant figures may have been missed and users should be aware of this limitation. The evaluation was performed on the `composition_property_values` field only, using the standard ComProScanner semantic evaluator. Synthesis data (synthesis methods, precursors, characterisation techniques) was excluded from this evaluation as these fields were already evaluated and reported in the ComProScanner paper and are not affected by graph extraction.


Of the 50 selected articles, 48 yielded evaluable composition-property data after extraction and cleaning. One of the remaining two articles, provided by Wiley, was not retrievable even as a PDF. The other article contained environment-dependent d<sub>33</sub> values which were extracted with '**--**' and were removed during data cleaning. The evaluation was performed at a strict semantic threshold of **1.0 (exact match)** to ensure that the reported metrics reflect precise extraction performance without partial-credit inflation. For d<sub>33</sub> values, error thresholds of <span>&#177;</span>0.5, <span>&#177;</span>1 and <span>&#177;</span>2 have been applied for different value ranges (refer to the `vlm_test_example.py` script on [GitHub](https://github.com/slimeslab/ComProScanner/tree/main/examples/vlm_piezo_test/vlm_test_example.py)). Two complementary classification metric sets are reported, along with weight-based composition accuracy as discussed in the ComProScanner paper. The model performance is summarised in confusion matrix illustrated below in [Fig. 4](#confusion-matrix).
<div align="center">
<img src="https://i.ibb.co/7dzx8s1b/confusion-matrix.png" alt="confusion matrix" style="width:100%; max-width:700px;" id="confusion-matrix" alt="Confusion Matrix for VLM Benchmark">
</div>
<p style="font-size: 10px;">
<b>Fig. 4:</b> Confusion matrix from semantic evaluation with 1.0 threshold for composition-property data, showcasing all 7 evaluation parameters, such as weight-based composition accuracy, classification metrics (precision, recall and F1-score) and normalised classification metrics (normalised precision, normalised recall and normalised F1-score), across 4 different VLMs used in this study.
</p>

**Gemini-3-Flash-Preview** is the strongest performer across all evaluation dimensions. It achieves a composition accuracy of 0.97, with absolute precision, recall, and F1 of 0.96, 0.95, and 0.96 respectively, and normalised precision, recall, and F1 of 0.97, 0.96, and 0.97 respectively. This outcome is entirely consistent with its standing on the LMArena Diagram leaderboard, where Gemini-3-Flash-Preview carries a higher Arena ELO score than Gemini-2.5-Pro whilst commanding a substantially lower input cost per million tokens, making it simultaneously the highest-performing and most economical model in this evaluation. **Gemini-2.5-Pro** performs respectably, with a composition accuracy of 0.86, absolute precision, recall, and F1 of 0.84, 0.76, and 0.80 respectively, and normalised precision, recall, and F1 of 0.88, 0.80, and 0.84 respectively. The notably lower recall relative to precision, a gap of approximately 0.08 in both absolute and normalised settings suggests the model is more conservative in proposing data points than the Flash variant, consistent with the Pro model's tendency towards cautious reasoning in ambiguous figure layouts. **GPT-5-Chat-Latest** and **GPT-5.1** perform broadly comparably to one another and can be considered together. Both yield a composition accuracy of 0.78. In terms of absolute classification metrics, both models achieve a precision of 0.71, with recall of 0.62 and 0.63 and F1 of 0.66 and 0.67 respectively. Normalised metrics follow a similar pattern: precision of 0.75 and 0.76, recall of 0.68 and 0.69, and F1 of 0.71 and 0.72 respectively. Both fall approximately 0.12–0.13 below Gemini-2.5-Pro on normalised F1, indicating difficulty with the full diversity of graphical representations across the corpus. Given this performance gap at similar cost, Gemini-3-Flash-Preview is adopted as the default VLM for the `GraphExtractorTool`, whilst the `vlm_model` parameter remains available for users to override with any LiteLLM-compatible model identifier (refer to the [ComProScanner documentation](../../usage/data-extraction.md)).

---

## 4. Additional Changes in This Release

The following improvements accompany the graph extraction feature but are documented here only briefly, as they are covered in full in the [changelog](../../about/changelog.md):

- **`value_error_thresholds` parameter** — both `evaluate_semantic()` and `evaluate_agentic()` now accept a layered dict mapping `(min, max)` tuples to absolute error tolerances. The narrowest enclosing range wins; tuple element order is irrelevant. 

- **`SCIENCEDIRECT_INSTTOKEN` support** — the `ElsevierArticleProcessor` now accepts an institutional token for off-campus remote access to subscription-based Elsevier articles; the token is forwarded as the `X-ELS-Insttoken` header.

- **Improved JSON parsing** — `_parse_json_output()` now recovers JSON from mixed-text crew outputs (e.g. `Thought: … {"json": "here"}`) via first-brace/last-brace scanning before falling back to `ast.literal_eval()`.

- **`MaterialParserTool` substitution verification** — the composition formatter agent now detects and corrects wrong variable substitution artefacts.

- **Failed-article reporting** — `save_failed_pdf_report` / `save_failed_automated_report` parameters write tab-separated failure logs for both local PDF and automated publisher workflows.

- **Unresolved composition tracking** — `clean_data()` handles better cleaning and can now log split-composition resolution statistics and persist filtered/unresolved composition keys to JSON.

- **CalVer migration** — the versioning scheme has been switched from [SemVer](https://semver.org/) (`MAJOR.MINOR.PATCH`) to [CalVer](https://calver.org/) (`YYYY.MM.DD`), with the first release under the new scheme being [`2026.05.19`](https://pypi.org/project/comproscanner/2026.05.19/) for easier tracking of feature development over time.


---

## 5. Conclusion

This release substantially extends ComProScanner's extraction capability to graph-resident data, a class of information that is pervasive in materials science literature but was previously inaccessible to the pipeline. Benchmarking across 50 articles at an exact-match threshold demonstrates that **Gemini-3-Flash-Preview** is the most effective VLM for this task among the four candidates selected via the LMArena Diagram leaderboard, achieving a composition accuracy of 0.97. The accompanying infrastructure improvements such as robust JSON recovery, institutional token support, layered error thresholds and explicit model-control parameters further harden the framework for large-scale use. 

---

## 6. Data and Code Availability
The benchmark data, model outputs, and evaluation scripts regarding the VLM tests are available in `examples/vlm_piezo_test` folder on the [ComProScanner GitHub repository](https://github.com/slimeslab/ComProScanner/tree/main/examples/vlm_piezo_test).

---
## References
1. A. Roy, E. Grisan, J. Buckeridge and C. Gattinoni, Digital Discovery, 2026, 5, 1794-1808.
2. Y. Han, J. Xia, R. Zhang, B. Wang, Y. Liu, D. Pan, Y. Wang, J. Zhang and Q. Chen, International Journal of Machine Learning and Cybernetics, 2025, 16, 7277–7292. 
3. D. Circi, M. Bradley, S. Blouir, B. Wilthan, A. Anastasopoulos, A. Shehu, L. Catherine and B. Dhingra, in LLM for Scientific Discovery: Reasoning, Assistance, and Collaboration, 2025, https://openreview.net/forum?id=vj8dqNrzEe.
4. Z. Zheng, Z. He, O. Khattab, N. Rampal, M. A. Zaharia, C. Borgs, J. T. Chayes and O. M. Yaghi, Digital Discovery, 2024, 3, 491–501. 
5. M. P. Polak and D. Morgan, Leveraging Vision Capabilities of Multimodal LLMs for Automated Data Extraction from Plots, arXiv, 2025, preprint, arXiv:2503.12326, 10.48550/arXiv.2503.12326, https://arxiv.org/abs/2503.12326.
6. Y. Wu, T. Su, S. Hu and D. Pan, Skill-Based Autonomous Agents for Material Creep Database Construction, arXiv, 2026, preprint, arXiv:2602.03069, 10.48550/arXiv.2602.03069, https://arxiv.org/abs/2602.03069.
7.  W. L. Chiang, L. Zheng, Y. Sheng, A. N. Angelopoulos, T. Li, D. Li, H. Zhang, B. Zhu, M. Jordan, J. E. Gonzalez and I. Stoica, Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference, 2024.
8. S. Pichai, D. Hassabis and K. Kavukcuoglu, A new era of intelligence with Gemini 3, https://blog.google/products-and-platforms/products/gemini/gemini-3 (accessed 07 May 2026).
9. G. Comanici, E. Bieber, M. Schaekermann, et al., Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities, arXiv, 2025, preprint, arXiv:2507.06261, 10.48550/arXiv.2507.06261, https://arxiv.org/abs/2507.06261.
10. A. Singh, A. Fry, A. Perelman, et al., OpenAI GPT-5 System Card, arXiv, 2026, preprint, arXiv:2601.03267, 10.48550/arXiv.2601.03267, https://arxiv.org/abs/2601.03267.
11. OpenAI, GPT‑5.1: A smarter, more conversational ChatGPT, https://openai.com/index/gpt-5-1 (accessed 07 May 2026).