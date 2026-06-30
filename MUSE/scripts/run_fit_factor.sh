#!/bin/bash
set -e

# ========== MUSE Data Configuration ==========
CORPUS="news"
DATA_FILE="./data/$CORPUS/raw/forget.txt"  # or retain1.txt

# ========== Model Configuration ==========
MODEL_NAME="muse-bench/MUSE-news_target"  # Fine-tuned model path
TOKENIZER_NAME="NousResearch/Llama-2-7b-chat-hf"

# ========== Dataset Configuration ==========
MAX_LENGTH=4096

# ========== Factor Configuration ==========
FACTOR_STRATEGY="ekfac"  # "ekfac", "kfac", or "diagonal"
TRAIN_BATCH_SIZE=1
OUTPUT_DIR="./kronfluence_factors"

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
export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=18765
export NPROC_PER_NODE=1

echo "Starting MUSE Kronfluence Factor Fitting..."
echo "Model: ${MODEL_NAME}"
echo "Tokenizer: ${TOKENIZER_NAME}"
echo "Data File: ${DATA_FILE}"
echo "Max Length: ${MAX_LENGTH}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} | Processes: ${NPROC_PER_NODE} | Port: ${MASTER_PORT}"
echo ""

# Build command
CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} \
  if/fit_factor.py \
  --model_name \"${MODEL_NAME}\" \
  --tokenizer_name \"${TOKENIZER_NAME}\" \
  --data_file \"${DATA_FILE}\" \
  --max_length ${MAX_LENGTH} \
  --factor_strategy \"${FACTOR_STRATEGY}\" \
  --train_batch_size ${TRAIN_BATCH_SIZE} \
  --output_dir \"${OUTPUT_DIR}\" \
  --covariance_module_partitions 4 \
  --lambda_module_partitions 4 \
  --covariance_data_partitions 4 \
  --lambda_data_partitions 4"

echo "Running command:"
echo "$CMD"
echo ""

# Execute
eval $CMD
