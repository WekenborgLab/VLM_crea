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


# VLM Greenness Evaluation

Evaluates images using Vision Language Models (VLM) across the following categories: SocialDensity, NatureScore, InsideOutside, Animals, FoodDrinks, Greenness, DigitalGadget, GadgetsMedia, PlantPresence, NaturalLightExposure, Smartphone, PC, Laptop, TV, Flatscreen, Smartwatch, Tablet, Monitor, Display, Camera, ElectronicWearableDevice, eBookReader, Screen, Temperature.

## Prerequisites

- **Python 3.13+**
- **uv** (Python package manager) - install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **OpenAI-compatible API endpoint** (e.g., OpenAI, local LLM server, or other provider)

## Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set the required variables:
   - `FOLDERPATH` - Path to folder containing subfolders with images (JPG/PNG)
   - `BASE_URL` - Your LLM API endpoint (e.g., `https://api.openai.com/v1`)
   - `MODEL` - Model name (e.g., `gpt-4o-mini`)
   - `OPENAI_API_KEY` - Your API key

   Optional settings (see `.env.example` for defaults):
   - `OUTPUT_DIR` - Where to save results (default: current directory)
   - `MAX_CONCURRENCY` - Number of parallel workers (default: 12)
   - `MAX_IMAGE_SIZE` - Longest side in pixels, 0 = no resize (default: 1280)
   - `CONVERT_TO_EXCEL` - Generate Excel files alongside CSV (default: True)

## Running the Experiments

```bash
uv run python main.py
```

The script will:
1. Scan all subfolders in `FOLDERPATH` for images
2. Downscale images to reduce API costs (if `MAX_IMAGE_SIZE` > 0)
3. Send each image to the LLM with 5 evaluation prompts
4. Stream results to CSV files in real-time
5. Convert CSVs to Excel (if enabled)
6. Generate token usage and failure reports

## Input Structure

```
FOLDERPATH/
├── subfolder1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── subfolder2/
│   └── ...
└── ...
```

## Output

A timestamped folder is created in `OUTPUT_DIR` containing:

- **Evaluation results** (one per category):
  - `nature_score.csv` / `nature_score.xlsx`
  - `plant_presence.csv` / `plant_presence.xlsx`
  - `natural_light.csv` / `natural_light.xlsx`
  - `greenness.csv` / `greenness.xlsx`
  - `insideoutside.csv` / `insideoutside.xlsx`
  - `socialdensity.csv`/ `socialdensitiy.xlsx`
  - `animals.csv` / `animals.xlsx`
  - `food_drinks.csv` / `food_drinks.xlsx`
  - `digital_gadget.csv` / `digital_gadget.xlsx`
  - `gadgets_media.csv` / `gadgets_media.xlsx`
  - `smartphone.csv` / `smartphone.xlsx`
  - `pc.csv` / `pc.xlsx`
  - `laptop.csv` / `laptop.xlsx`
  - `tv.csv` / `tv.xlsx`
  - `flatscreen.csv` / `flatscreen.xlsx`
  - `smartwatch.csv` / `smartwatch.xlsx`
  - `tablet.csv` / `tablet.xlsx`
  - `monitor.csv` / `monitor.xlsx`
  - `display.csv` / `display.xlsx`
  - `camera.csv` / `camera.xlsx`
  - `electronic_wearable_device.csv` / `electronic_wearable_device.xlsx`
  - `ebook_reader.csv` / `ebook_reader.xlsx`
  - `screen.csv` / `screen.xlsx`
  - `temperature.csv` / `temperature.xlsx`

- **Metadata and logs**:
  - `tokens.json` / `tokens.csv` - Token usage per job and overall
  - `failures.csv` - Failed requests with error details
  - `run.log` - Detailed execution log
  - `experiment.json` / `experiment_summary.csv` - Run metadata (if `WRITE_EXPERIMENT_METADATA=True`)

Each CSV contains: `PictureId`, `Folder`, `TimeCreated`, `TimeDigitized`, `TimeModified`, plus the evaluation scores and confidence values for that category.

## Evaluation Categories

All categories return scores 1-10 (except InsideOutside which uses 1=Inside, 2=Outside), plus a confidence score (1-10) and image description:

- **NatureScore**: Overall naturalness of the scene
- **PlantPresence**: Amount of vegetation visible
- **NaturalLightExposure**: Natural light availability
- **Greenness**: Green color intensity
- **InsideOutside**: Indoor vs outdoor classification
- **SocialDensity**:
- **Animals**:
- **FoodDrinks**:
- **DigitalGadget**:
- **GadgetsMedia**:
- **NaturalLightExposure**:
- **Smartphone**:
- **PC**:
- **Laptop**:
- **TV**:
- **Flatscreen**:
- **Smartwatch**:
- **Tablet**:
- **Monitor**:
- **Display**:
- **Camera**:
- **ElectronicWearableDevice**:
- **eBookReader**:
- **Screen**:
- **Temperature**:
