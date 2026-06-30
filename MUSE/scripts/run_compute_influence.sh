#!/bin/bash

# ========== MUSE Data Configuration ==========
CORPUS="news"
FORGET="./data/$CORPUS/raw/forget.txt"
RETAIN="./data/$CORPUS/raw/retain1.txt"

# ========== Model Configuration ==========
MODEL_NAME="muse-bench/MUSE-News_target"  # Fine-tuned model path
TOKENIZER_NAME="NousResearch/Llama-2-7b-chat-hf"

# ========== Dataset Configuration ==========
MAX_LENGTH=4096

# ========== Factor Configuration ==========
FACTOR_STRATEGY="ekfac"  # "ekfac", "kfac", or "diagonal"
FACTORS_PATH="./kronfluence_factors"
FACTORS_NAME="ekfac"
ANALYSIS_NAME="if_results"

# ========== Computation Configuration ==========
QUERY_BATCH_SIZE=1
TRAIN_BATCH_SIZE=1
USE_HALF_PRECISION="--use_half_precision"
USE_COMPILE="--use_compile"

# ========== Output Configuration ==========
SAVE_DIR="./influence_results"
SAVE_ID="muse_news"

# ========== Script Path Setup ==========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"  # MUSE directory
cd "${PROJECT_ROOT}"

echo "Working directory: $(pwd)"
echo ""

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate tofu
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo ""

# ========== GPU Configuration ==========
export CUDA_VISIBLE_DEVICES=1

echo "Starting MUSE Influence Score Computation..."
echo "Model: ${MODEL_NAME}"
echo "Tokenizer: ${TOKENIZER_NAME}"
echo "Forget File: ${FORGET}"
echo "Retain File: ${RETAIN}"
echo "Max Length: ${MAX_LENGTH}"
echo ""

# Build command
cmd="python if/compute_influence.py \
    --model_name ${MODEL_NAME} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --forget_file ${FORGET} \
    --retain_file ${RETAIN} \
    --max_length ${MAX_LENGTH} \
    --factors_path ${FACTORS_PATH} \
    --factors_name ${FACTORS_NAME} \
    --analysis_name ${ANALYSIS_NAME} \
    --factor_strategy ${FACTOR_STRATEGY} \
    --query_batch_size ${QUERY_BATCH_SIZE} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    ${USE_HALF_PRECISION} \
    ${USE_COMPILE} \
    --save_dir ${SAVE_DIR}"

# Add optional save_id if specified
if [ ! -z "$SAVE_ID" ]; then
    cmd="${cmd} --save_id ${SAVE_ID}"
fi

echo "Running command:"
echo "$cmd"
echo ""

# Execute
eval $cmd
