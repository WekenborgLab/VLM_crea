# VLM_crea
Exploring how nature in everyday images, assessed by vision-language models, relates to self-perceived creativity in daily life.
VLM_crea is a compact framework for image evaluation.

---

## Overview

VLM_crea consists of three main pipelines:

## Image Evaluation (Greeness Pipeline)
   Extracts and downsizes images, creates multimodal LLM/VLM jobs, evaluates them, and exports CSV/Excel/JSON outputs.


## Repository Structure

```plaintext
VLM_crea/
├─ VLM_Greeness/            # Image evaluation
```

## Requirements
Python 3.10+ for EPMC pipeline
Python 3.13+ for image pipelines
OpenAI-compatible LLM/VLM server
Disk space for ~65k papers
