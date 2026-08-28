# VLM_crea
Exploring how nature in everyday images, assessed by vision-language models, relates to self-perceived creativity in daily life.
VLM_crea is a compact framework for (1) image evaluation, (2) extraction and categorization of visually detectable psychological factors from scientific papers, (3) large-scale VLM scoring across 997 categories, and (4) statistical analysis.

---

## Overview

VLM_crea consists of three main pipelines:

1. **Step 1 — Image Evaluation (Greeness Pipeline)**  
   Extracts and downsizes images, creates multimodal LLM/VLM jobs, evaluates them, and exports CSV/Excel/JSON outputs.

2. **Step 2 — Factor Extraction & Categorization (EPMC Pipeline)**  
   Retrieves ~65k papers from Europe PMC, extracts visual factors linked to affect/emotion/stress/well-being, cleans and filters categories, splits them by outcome × direction, deduplicates them using an LLM, and exports final grouped category lists.

3. **Step 3 — 997-Category VLM Evaluation**  
   Processes images against 997 psychological categories using state-of-the-art vision-language models, generates structured outputs, logs, and token reports.

## Repository Structure

```plaintext
AmbuVision/
├─ Step1_VLM_Greeness/            # Step 1: Image evaluation
├─ Step2_/Extract_categories      # Step 2: Factor Extraction & Categorization
├─ Step3_VLM_categories/          # Step 3: 997-category VLM scoring
├─ baseline_comparison/           # Supplemental baseline comparison pipelines
├─ artifacts/
│  ├─ epmc_fulltext/              # raw Europe PMC fulltexts
│  └─ visual_factors/
│     ├─ all_visual_associations.csv
│     └─ output/                  # cleaned, split & deduplicated factor tables
├─ retrieve.py                    # Step 2.1 — EPMC retrieval
├─ llm_extraction.py              # Step 2.2 — factor→outcome extraction
├─ factor_categorization.py       # Step 2.3 — cleaning
├─ remove_none.py                 # Step 2.3 — drop None categories
├─ divide_six.py                  # Step 2.4 — split into six bins
├─ deduplicator.py                # Step 2.5 — LLM grouping
├─ group_lists.py                 # Step 2.6 — per-group CSVs
└─ unique_group_lists.py          # Step 2.6 — master category list
```

## Requirements
Python 3.10+ for EPMC pipeline
Python 3.13+ for image pipelines
OpenAI-compatible LLM/VLM server
Disk space for ~65k papers
