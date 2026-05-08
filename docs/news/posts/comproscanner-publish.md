---
date:
  created: 2026-04-05
authors:
  - aritraroy24
cover_image: https://pubs.rsc.org/en/Image/Get?imageInfo.ImageType=GA&imageInfo.ImageIdentifier.ManuscriptID=D5DD00521C&imageInfo.ImageIdentifier.Year=2026
description: >-
  ComProScanner is published in Digital Discovery (RSC) — a multi-agent
  framework for autonomous extraction of composition–property data from
  scientific literature into ML-ready databases.
---
# ComProScanner Published in Digital Discovery
<!-- more -->

---
<div align="center">
<img src="https://pubs.rsc.org/en/Image/Get?imageInfo.ImageType=GA&imageInfo.ImageIdentifier.ManuscriptID=D5DD00521C&imageInfo.ImageIdentifier.Year=2026" style="width:100%; max-width:700px;" id="toc-image" alt="ComProScanner TOC Graphic">
</div>

## Overview

**ComProScanner** has been published in [_Digital Discovery_](https://www.rsc.org/publishing/journals/digital-discovery), a journal of the Royal Society of Chemistry. The paper, titled **"ComProScanner: a multi-agent based framework for composition-property structured data extraction from scientific literature"**, presents an autonomous multi-agent platform for extracting, validating, and visualising machine-readable chemical compositions and properties from journal articles, creating structured, ML-ready databases with no human intervention.

---

## 1. The Research

ComProScanner establishes a scalable, user-friendly pathway for building large experimental datasets from journal articles, directly addressing one of the core bottlenecks in data-driven materials discovery. By combining LLM agents with RAG, domain-specific embeddings, and publisher TDM APIs, it bridges the gap between unstructured scientific literature and machine learning pipelines.

### Key Features

- **Publisher-to-database pipeline** via Text and Data Mining (TDM) APIs from Elsevier, Springer Nature, IOP Publishing, and Wiley — no manual PDF uploads required
- **Five specialised AI agents** organised into a RAG-based filtering crew, a composition extraction crew, and a synthesis information crew
- **Variable composition resolution** using a deep learning tool to expand formulas like Na(1−x)Li<sub>x</sub>TiO₃ into distinct compounds
- **Built-in evaluation framework** with weight-based accuracy, classification metrics, and novel normalised classification metrics
- **Implementable in fewer than 20 lines of Python code** via `pip install comproscanner`

---

## 2. Results

The framework was benchmarked on 100 journal articles across 10 LLMs for extracting piezoelectric material compositions and d<sub>33</sub> coefficient values. **DeepSeek-V3-0324** achieved the best overall accuracy of **0.82**, with a compositional accuracy of **0.90**. Strikingly, over 99% of the extracted piezoelectric materials are absent from the Materials Project database, highlighting the untapped wealth of data buried in the literature.

---

## 3. About the Authors

This work was led by **Aritra Roy**, a PhD student in the Energy, Materials and Environment Research Centre at LSBU, under the supervision of **Prof. John Buckeridge** (LSBU) and **Dr. Chiara Gattinoni** (King's College London), with contributions from **Prof. Enrico Grisan** (LSBU). Financial and legal support for publisher TDM licences was provided by LSBU and KCL. Dr. Gattinoni's work was supported by the EPSRC New Investigator Award [grant number UKRI132].

---

## 4. Publication Details

**Title:** ComProScanner: a multi-agent based framework for composition-property structured data extraction from scientific literature  
**Journal:** Digital Discovery  
**Authors:** Aritra Roy, Enrico Grisan, John Buckeridge, and Chiara Gattinoni  
**DOI:** 10.1039/D5DD00521C  
**Link:** [Read the full paper here](https://doi.org/10.1039/D5DD00521C)

**Citation (BibTeX):**

```bibtex
@Article{roy2026comproscannermultiagentbasedframework,
  author    = {Roy, Aritra and Grisan, Enrico and Buckeridge, John and Gattinoni, Chiara},
  title     = {ComProScanner: a multi-agent based framework for composition-property structured data extraction from scientific literature},
  journal   = {Digital Discovery},
  year      = {2026},
  pages     = {Accepted},
  publisher = {RSC},
  doi       = {10.1039/D5DD00521C},
  url       = {https://doi.org/10.1039/D5DD00521C}
}
```

---

## 5. Access ComProScanner

- **PyPI:** [pypi.org/project/comproscanner](https://pypi.org/project/comproscanner)
- **Documentation:** [slimeslab.github.io/ComProScanner](https://slimeslab.github.io/ComProScanner)
- **GitHub:** [github.com/slimeslab/ComProScanner](https://github.com/slimeslab/ComProScanner)

---

## 6. Looking Ahead

ComProScanner is fully open source and available for the community to use, extend, and adapt to new properties and material systems. Future directions include integrating OCR or vision-language models (VLMs) to extract data from figures and graphs, and extending the framework to support extraction of multiple material properties simultaneously.

---
