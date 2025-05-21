#!/bin/bash
# Load environment variables and helper functions
source launch_script_constants.sh
source local_job_setup.sh

# Default parameters
TASK="nodeclassification"
GENERATOR="sbm"

while getopts t:g:s:l: flag
do
    case "${flag}" in
        t) TASK=${OPTARG};;
        g) GENERATOR=${OPTARG};;
        s) SAVE_RESULTS=${OPTARG};;
        l) TUNING_METRIC_IS_LOSS=${OPTARG};;
    esac
done

# Get the directory where this script resides
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Ensure src/ is on Python’s path
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Get the parent directory (assumed to contain the scratch folder)
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
# Set output path in the scratch folder
OUTPUT_PATH="${PARENT_DIR}/scratch/dgoodwin/dev_stuff/adding_hgcn"

rm -rf "${OUTPUT_PATH}"
mkdir -p "${OUTPUT_PATH}"

# Build gin files string; adjust paths if your directory structure differs.
GIN_FILES="${SCRIPT_DIR}/src/configs/${TASK}.gin \
${SCRIPT_DIR}/src/configs/${TASK}_generators/${GENERATOR}/default_setup.gin \
${SCRIPT_DIR}/src/configs/common_hparams/${TASK}.gin"
if [ "${RUN_MODE2}" = true ]; then
  GIN_FILES="${GIN_FILES} ${SCRIPT_DIR}/src/configs/${TASK}_generators/${GENERATOR}/optimal_model_hparams.gin"
fi

# Build gin parameter string using a helper function to get task class name
TASK_CLASS_NAME=$(get_task_class_name ${TASK})
GIN_PARAMS="GeneratorBeamHandlerWrapper.nsamples=${NUM_SAMPLES} \
${TASK_CLASS_NAME}BeamHandler.num_tuning_rounds=${NUM_TUNING_ROUNDS} \
${TASK_CLASS_NAME}BeamHandler.save_tuning_results=${SAVE_TUNING_RESULTS} \
${TASK_CLASS_NAME}BeamHandler.tuning_metric_is_loss=${TUNING_METRIC_IS_LOSS}"

# Execute the pipeline with DirectRunner
python3 src/beam_benchmark_main.py \
  --runner DirectRunner \
  --gin_files ${GIN_FILES} \
  --gin_params ${GIN_PARAMS} \
  --output ${OUTPUT_PATH} \
  --write_intermediate ${SAVE_RESULTS}
