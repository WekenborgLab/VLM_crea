#!/usr/bin/env bash
#
# Repeated-run stability of VLM category ratings.
#
# Rates every image in $FOLDERPATH on 24 categories, five independent times with
# identical settings (SEED=42, thinking disabled). Each invocation writes its own
# timestamped subfolder under $OUTPUT_DIR, so the five repeats can be compared
# against each other.
#
# Set FOLDERPATH, BASE_URL, MODEL and OPENAI_API_KEY in .env before running.
# Usage:  ./run_experiment.sh
#
set -euo pipefail

cd "$(dirname "$0")"

export CATEGORIES="SocialDensity,NatureScore,InsideOutside,Animals,FoodDrinks,Greenness,DigitalGadget,GadgetsMedia,PlantPresence,NaturalLightExposure,Smartphone,PC,Laptop,TV,Flatscreen,Smartwatch,Tablet,Monitor,Display,Camera,ElectronicWearableDevice,eBookReader,Screen,Temperature"

# These exports take precedence over .env (load_dotenv does not override the
# environment), so they define the experimental condition regardless of .env.
export SEED=42
export MAX_IMAGE_SIZE=1024
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/results}"
export MAX_CONCURRENCY=5
export PER_IMAGE_JOBS_CONCURRENCY=20
export DISABLE_THINKING=true
export WRITE_EXPERIMENT_METADATA=True

mkdir -p "$OUTPUT_DIR"

# Five independent repeats under identical settings.
for run in 1 2 3 4 5; do
  export EXPERIMENT_NAME="repeat_${run}"
  echo "=== Starting ${EXPERIMENT_NAME} ==="
  uv run python main.py
done

echo "=== All 5 repeats complete. Results in ${OUTPUT_DIR} ==="
