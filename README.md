# VLM_crea

Exploring how nature in everyday images, assessed by vision-language models, relates to self-perceived creativity in daily life.

VLM_crea is a compact framework for image evaluation. It rates a corpus of images on a set of visual categories using any OpenAI-compatible VLM endpoint, and streams the ratings to one CSV/Excel file per category.

---

## Overview

For every image and every category in `CATEGORIES`, the pipeline sends one request to the VLM with the following prompt (`create_prompt` in `main.py`):

```text

            Analyze the image regarding the presence of the following category: "{category}".
            Return a JSON with three fields:
            "presence": 1 to 10 value indicating the presence (1=absolute absence, 10=dominance/presence),
            "confidence": 1 to 10 value indicating confidence regarding the presence of the category,
            "description": a brief description of what is seen in the image.
            Return JSON only.
    
```

`{category}` is the only substitution — no definition of the category is given, so its semantics are whatever the VLM infers from the label alone. The reply is parsed against a Pydantic schema (`presence: int`, `confidence: int`, `description: str`) via the OpenAI SDK's structured-output path.

`run_experiment.sh` reproduces the run reported in the paper: the full corpus rated on all 24 categories, five independent times under identical settings.

`main.py` is the entry point, `Middle.py` the VLM client, `Imager.py` the image handling and output.

## Evaluation categories

All 24 categories use the same 1–10 presence scale and the same prompt.

- **SocialDensity**
- **NatureScore**: Overall naturalness of the scene
- **InsideOutside**: Indoor vs outdoor classification
- **Animals**
- **FoodDrinks**
- **Greenness**: Green color intensity
- **DigitalGadget**
- **GadgetsMedia**
- **PlantPresence**: Amount of vegetation visible
- **NaturalLightExposure**: Natural light availability
- **Smartphone**
- **PC**
- **Laptop**
- **TV**
- **Flatscreen**
- **Smartwatch**
- **Tablet**
- **Monitor**
- **Display**
- **Camera**
- **ElectronicWearableDevice**
- **eBookReader**
- **Screen**
- **Temperature**

The list is supplied at runtime — see `CATEGORIES` in `.env.example`.

## Setup

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and an OpenAI-compatible VLM endpoint with image input and structured outputs.

```bash
uv sync
cp .env.example .env
```

Set `FOLDERPATH`, `BASE_URL`, `MODEL` and `OPENAI_API_KEY` in `.env`; all other options are documented there.

## Running

```bash
./run_experiment.sh     # the reported run: 5 repeats
uv run python main.py   # a single pass with your own .env settings
```

`FOLDERPATH` is expected to contain one subfolder per participant, each holding JPG/PNG images. Each run scans every subfolder, downscales each image once in memory, sends it to the VLM once per category, and streams results to CSV as they arrive.

## Output

A timestamped folder is created in `OUTPUT_DIR` holding one CSV and Excel file per category (`PictureId`, `Folder`, EXIF timestamps, `<Category>`, `<Category>Confidence`, `ImageDescription`), plus `tokens.json`/`.csv`, `failures.csv`, `run.log`, and `experiment.json`/`experiment_summary.csv`.

## Reported run

| | |
| --- | --- |
| Model | `Qwen3.5-397B-A17B-FP8` |
| Serving | vLLM 0.17.0 on 4× NVIDIA H200 |
| Seed | 42 |
| Reasoning | disabled (`enable_thinking: false`) |
| Temperature / max tokens | not set — server defaults |
| Image preprocessing | longest side ≤ 1024 px, JPEG quality 85 |
| Concurrency | 5 images in parallel × 20 categories in parallel |
| Corpus | 8,068 images × 24 categories = 193,632 requests per repeat |
| Repeats | 5 |
| Runtime | ≈ 6.5 h per repeat |

`SEED` is passed through to the endpoint, but bit-exact reproducibility is not guaranteed: with vLLM's continuous batching, batch composition depends on request arrival timing, which changes reduction order in FP8 kernels.

## Data availability

The image corpus and the resulting ratings are not included in this repository. The images are participants' personal photographs and the `ImageDescription` fields are derived from them, so neither can be shared publicly. See the data availability statement in the paper.
