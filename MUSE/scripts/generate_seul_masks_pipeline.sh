#!/bin/bash
# =============================================================================
# SEUL Mask Generation Pipeline for MUSE
# =============================================================================
# This script generates SEUL forget masks for MUSE data.
#
# Steps:
#   1. Extract sensitive spans with GPT-4o at the character level.
#   2. Convert character-level spans to token-level masks.
#   3. Use the generated mask in the seul_offline algorithm.
#
# Usage:
#   bash scripts/generate_seul_masks_pipeline.sh [corpus] [max_samples]
#
# Examples:
#   bash scripts/generate_seul_masks_pipeline.sh news 100
#   bash scripts/generate_seul_masks_pipeline.sh books 50
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
CORPUS=${1:-"news"}  # news or books
MAX_SAMPLES=${2:-""}  # Empty string means process the full dataset.
MODEL_NAME="meta-llama/Llama-2-7b-hf"
MAX_LENGTH=2048

# MUSE project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MUSE_ROOT="$SCRIPT_DIR/.."

# Data paths
DATA_DIR="$MUSE_ROOT/data/${CORPUS}/raw"
FORGET_FILE="$DATA_DIR/forget.txt"

# Output paths
OUTPUT_DIR="$MUSE_ROOT/data/${CORPUS}/seul_masks"
SPANS_JSON="$OUTPUT_DIR/gpt_spans_forget.json"
MASK_PT="$OUTPUT_DIR/forget_mask.pt"

# OpenAI API key, loaded from the environment or repository root .env file.
REPO_ROOT="$(cd "$MUSE_ROOT/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
fi

# =============================================================================
# Validation
# =============================================================================
if [ "$CORPUS" != "news" ] && [ "$CORPUS" != "books" ]; then
    echo "Error: Corpus must be 'news' or 'books'"
    exit 1
fi

if [ ! -f "$FORGET_FILE" ]; then
    echo "Error: Forget file not found: $FORGET_FILE"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    echo "Please set it in environment or $REPO_ROOT/.env file"
    exit 1
fi

# Create output directory.
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "       SEUL Mask Generation Pipeline"
echo "=============================================="
echo "Corpus: $CORPUS"
echo "Forget file: $FORGET_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Max length: $MAX_LENGTH"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples: $MAX_SAMPLES"
else
    echo "Max samples: ALL"
fi
echo "=============================================="

# =============================================================================
# Step 1: Extract sensitive spans with GPT-4o.
# =============================================================================
echo ""
echo "=============================================="
echo "Step 1: Extracting sensitive spans with GPT-4o"
echo "=============================================="

if [ -f "$SPANS_JSON" ]; then
    echo "Spans file already exists: $SPANS_JSON"
    read -p "Re-run GPT span extraction? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping span extraction."
    else
        rm "$SPANS_JSON"
    fi
fi

if [ ! -f "$SPANS_JSON" ]; then
    cd "$MUSE_ROOT/scripts"

    if [ -n "$MAX_SAMPLES" ]; then
        python3 generate_gpt_spans_muse.py \
            --data_path "$FORGET_FILE" \
            --corpus "$CORPUS" \
            --output_path "$SPANS_JSON" \
            --model "gpt-4o" \
            --max_samples "$MAX_SAMPLES" \
            --delay 0.5
    else
        python3 generate_gpt_spans_muse.py \
            --data_path "$FORGET_FILE" \
            --corpus "$CORPUS" \
            --output_path "$SPANS_JSON" \
            --model "gpt-4o" \
            --delay 0.5
    fi

    echo "Spans extracted to: $SPANS_JSON"
fi

# =============================================================================
# Step 2: Convert character-level spans to token-level masks.
# =============================================================================
echo ""
echo "=============================================="
echo "Step 2: Converting spans to token-level masks"
echo "=============================================="

if [ -f "$MASK_PT" ]; then
    echo "Mask file already exists: $MASK_PT"
    read -p "Re-run mask conversion? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping mask conversion."
    else
        rm "$MASK_PT"
    fi
fi

if [ ! -f "$MASK_PT" ]; then
    cd "$MUSE_ROOT/scripts"

    python3 convert_spans_to_mask_muse.py \
        --spans_path "$SPANS_JSON" \
        --data_path "$FORGET_FILE" \
        --output_path "$MASK_PT" \
        --model_name "$MODEL_NAME" \
        --max_length "$MAX_LENGTH"

    echo "Masks converted to: $MASK_PT"
fi

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=============================================="
echo "       SEUL Mask Generation Completed!"
echo "=============================================="
echo "Spans JSON: $SPANS_JSON"
echo "Mask PT: $MASK_PT"
echo ""
echo "Use this mask as input to the training/evaluation pipeline that consumes token-level forget masks:"
echo "  $MASK_PT"
echo "=============================================="
