#!/bin/bash
# Evaluate selected tokens against MUSE ground-truth token labels.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SELECTED_TOKENS="${1:-}"
GT_CSV="${2:-$PROJECT_DIR/if/gt_token.csv}"
EVALUATE_SCRIPT="$SCRIPT_DIR/evaluate_metrics.py"

if [ -z "$SELECTED_TOKENS" ]; then
    echo "Usage: bash scripts/run_evaluation.sh <selected_tokens.json> [gt_token.csv]"
    echo ""
    echo "Example:"
    echo "  bash scripts/run_evaluation.sh \\"
    echo "    influence_results/influence_masks/news_forget_half_forget10/mask_sample=0.1_word_selected_words.json"
    exit 1
fi

if [[ "$SELECTED_TOKENS" != /* ]]; then
    SELECTED_TOKENS="$PROJECT_DIR/$SELECTED_TOKENS"
fi

if [[ "$GT_CSV" != /* ]]; then
    GT_CSV="$PROJECT_DIR/$GT_CSV"
fi

SELECTED_DIR="$(dirname "$SELECTED_TOKENS")"
SELECTED_BASENAME="$(basename "$SELECTED_TOKENS" .json)"
OUTPUT_FILE="$SELECTED_DIR/${SELECTED_BASENAME}_evaluation.txt"

echo "================================================"
echo "Running Evaluation Metrics"
echo "================================================"
echo "Selected tokens: $SELECTED_TOKENS"
echo "GT labels CSV: $GT_CSV"
echo "Output file: $OUTPUT_FILE"
echo ""

# Check if files exist
if [ ! -f "$SELECTED_TOKENS" ]; then
    echo "Error: selected_tokens.json not found at $SELECTED_TOKENS"
    exit 1
fi

if [ ! -f "$GT_CSV" ]; then
    echo "Error: gt_token.csv not found at $GT_CSV"
    exit 1
fi

# Run evaluation and save to both console and file
python3 "$EVALUATE_SCRIPT" "$SELECTED_TOKENS" "$GT_CSV" | tee "$OUTPUT_FILE"

echo ""
echo "================================================"
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "================================================"
