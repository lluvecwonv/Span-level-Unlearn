import logging
import os
import sys
import argparse
from pathlib import Path

import torch

# Add kronfluence to path
KRONFLUENCE_DIR = Path(__file__).parent.parent.parent / "kronfluence" / "src"
sys.path.insert(0, str(KRONFLUENCE_DIR))

from transformers import AutoTokenizer
from tqdm import tqdm

# Add parent TOFU directory to path for imports
TOFU_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(TOFU_DIR))

from data_module import TextDatasetQA
import yaml
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Build binary token mask from GT spans.")

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="qwen2.5-7b",
        help="Model family (e.g., llama2, phi).",
    )

    # Dataset configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default="locuslab/TOFU",
        help="Path to TOFU dataset.",
    )
    parser.add_argument(
        "--forget_split",
        type=str,
        default="forget10",
        help="Forget split name (e.g., forget10).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--question_key",
        type=str,
        default="question",
        help="Key for question in dataset.",
    )
    parser.add_argument(
        "--answer_key",
        type=str,
        default="answer",
        help="Key for answer in dataset.",
    )

    # GT spans configuration
    parser.add_argument(
        "--gt_spans_path",
        type=str,
        required=True,
        help="Path to GT spans JSON file (e.g., selected_result_v.1.json).",
    )

    # Output configuration
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./influence_results",
        help="Directory to save the mask.",
    )
    parser.add_argument(
        "--save_id",
        type=str,
        default=None,
        help="ID to append to the output file names.",
    )

    return parser.parse_args()


def load_gt_spans(gt_spans_path: str) -> dict:
    """
    Load GT spans from JSON file.

    Supports two formats:
    1. selected_span: [{"text": score}, ...]  (list of dicts)
    2. total_selected_span: {"text": score, ...}  (single dict)

    Returns:
        Dictionary mapping sample_index -> list of span_text
    """
    with open(gt_spans_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    gt_spans_dict = {}
    for item in gt_data:
        sample_idx = item['sample_index']

        span_list = []

        # Try format 1: selected_span (list of dicts)
        selected_spans = item.get('selected_span', [])
        if selected_spans:
            for span_dict in selected_spans:
                for span_text, score in span_dict.items():
                    span_list.append(span_text)

        # Try format 2: total_selected_span (single dict)
        total_spans = item.get('total_selected_span', {})
        if total_spans and not span_list:
            for span_text, score in total_spans.items():
                span_list.append(span_text)

        gt_spans_dict[sample_idx] = span_list

    return gt_spans_dict


def find_span_in_tokens(span_text: str, tokens: list) -> list:
    """
    Find all token ranges that match the given span text.

    Returns:
        List of (start_idx, end_idx) tuples for matching token ranges
    """
    matches = []
    span_normalized = span_text.strip().lower()

    # Try to find the span by sliding window
    for start_idx in range(len(tokens)):
        for end_idx in range(start_idx + 1, min(start_idx + 80, len(tokens) + 1)):
            token_slice = tokens[start_idx:end_idx]
            decoded_text = ''.join([t.replace('▁', ' ').replace('Ġ', ' ') for t in token_slice]).strip()

            if decoded_text.lower() == span_normalized:
                matches.append((start_idx, end_idx))
                break

    return matches


def build_binary_mask_from_gt_spans(
    dataset,
    tokenizer,
    gt_spans_dict: dict,
    num_samples: int,
    max_length: int,
) -> tuple:
    """
    Build a binary mask where tokens in GT spans are True, others are False.

    Args:
        dataset: Dataset containing input_ids
        tokenizer: Tokenizer for converting tokens to strings
        gt_spans_dict: Dictionary mapping sample_index -> list of span_text
        num_samples: Number of samples in the dataset
        max_length: Maximum sequence length

    Returns:
        mask: Boolean tensor where True = token in GT span
        mask_info: List of dicts with masked token info for each sample
    """
    mask = torch.zeros((num_samples, max_length), dtype=torch.bool)
    mask_info = []  # Store info for JSON output

    # Select 3 random samples for debugging
    import random
    random.seed(42)
    debug_samples = random.sample(range(num_samples), min(3, num_samples))
    logging.info(f"Debug mode enabled for samples: {debug_samples}")
    logging.info("")

    logging.info(f"Building binary mask from GT spans for {num_samples} samples...")

    for i in tqdm(range(num_samples), desc="Processing samples"):
        # Get tokens for this sample
        dataset_item = dataset[i]
        if isinstance(dataset_item, dict):
            input_ids = dataset_item["input_ids"]
            labels = dataset_item.get("labels", None)
        else:
            input_ids = dataset_item[0]
            labels = dataset_item[1] if len(dataset_item) > 1 else None

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # ==== DEBUG: Check if this is a debug sample ====
        is_debug = i in debug_samples

        if is_debug:
            logging.info("=" * 80)
            logging.info(f"DEBUG: Processing sample index {i}")
            logging.info("=" * 80)

            # Decode full text
            full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            logging.info(f"Full text: {full_text}")
            logging.info("")

            # Show all tokens with indices
            logging.info("All tokens:")
            for tidx, tok in enumerate(tokens):
                label_val = str(labels[tidx]) if labels else "N/A"
                is_answer = "ANSWER" if labels and labels[tidx] != -100 else "QUESTION"
                logging.info(f"  [{tidx:3d}] {tok:20s} | label={label_val:>6s} | {is_answer}")
            logging.info("")

        # Get answer-only mask (label != -100)
        if labels is not None:
            answer_mask = torch.tensor([label != -100 for label in labels], dtype=torch.bool)
        else:
            answer_mask = torch.ones(len(tokens), dtype=torch.bool)

        # Skip if no GT spans for this sample
        if i not in gt_spans_dict or len(gt_spans_dict[i]) == 0:
            if is_debug:
                logging.info(f"No GT spans for sample {i}")
                logging.info("=" * 80)
            continue

        # ==== DEBUG: GT spans ====
        if is_debug:
            logging.info(f"GT spans for sample {i}: {gt_spans_dict[i]}")
            logging.info("")

        # Find and mark tokens for each GT span
        for span_text in gt_spans_dict[i]:
            matches = find_span_in_tokens(span_text, tokens)

            if len(matches) == 0:
                logging.warning(f"Sample {i}: Could not find span '{span_text}' in tokens")
                continue

            # ==== DEBUG: Matches ====
            if is_debug:
                logging.info(f"Span: '{span_text}'")
                logging.info(f"  Found {len(matches)} match(es):")
                for match_idx, (start_idx, end_idx) in enumerate(matches):
                    matched_tokens = tokens[start_idx:end_idx]
                    matched_text = ''.join([t.replace('▁', ' ').replace('Ġ', ' ') for t in matched_tokens]).strip()
                    logging.info(f"    Match {match_idx}: tokens[{start_idx}:{end_idx}] = {matched_tokens}")
                    logging.info(f"              Decoded: '{matched_text}'")
                logging.info("")

            # Mark all tokens in all occurrences of this span
            for start_idx, end_idx in matches:
                for tok_idx in range(start_idx, end_idx):
                    # Only mark tokens in answer part
                    if answer_mask[tok_idx]:
                        mask[i, tok_idx] = True
                        # ==== DEBUG: Masking ====
                        if is_debug:
                            logging.info(f"  ✓ Masked token[{tok_idx}]: {tokens[tok_idx]}")

        # Collect mask info for this sample
        masked_indices = torch.nonzero(mask[i], as_tuple=True)[0].tolist()
        masked_tokens_list = [tokens[idx] for idx in masked_indices]
        masked_text = ''.join([t.replace('▁', ' ').replace('Ġ', ' ') for t in masked_tokens_list]).strip()

        # Get full answer text
        answer_indices = [idx for idx, label in enumerate(labels) if label != -100] if labels else []
        answer_tokens = [tokens[idx] for idx in answer_indices]
        full_answer = ''.join([t.replace('▁', ' ').replace('Ġ', ' ') for t in answer_tokens]).strip()

        mask_info.append({
            "sample_index": i,
            "gt_spans": gt_spans_dict.get(i, []),
            "masked_token_indices": masked_indices,
            "masked_tokens": masked_tokens_list,
            "masked_text": masked_text,
            "full_answer": full_answer,
            "num_masked_tokens": len(masked_indices),
        })

        # ==== DEBUG: Final mask ====
        if is_debug:
            logging.info("")
            logging.info(f"Final mask for sample {i}:")
            logging.info(f"  Total masked tokens: {len(masked_indices)}")
            logging.info(f"  Masked token indices: {masked_indices}")
            logging.info("")

            # Show full True/False mask for all tokens
            logging.info("  Full mask (True/False for all tokens):")
            for tidx, tok in enumerate(tokens):
                is_masked = mask[i, tidx].item()
                mask_status = "✓ TRUE " if is_masked else "✗ FALSE"
                label_val = str(labels[tidx]) if labels else "N/A"
                is_answer = "ANSWER" if labels and labels[tidx] != -100 else "QUESTION"
                logging.info(f"    [{tidx:3d}] {mask_status} | {tok:20s} | label={label_val:>6s} | {is_answer}")
            logging.info("")

            if len(masked_indices) > 0:
                logging.info("  Masked tokens (TRUE only):")
                for idx in masked_indices:
                    logging.info(f"    [{idx:3d}] {tokens[idx]}")

            logging.info(f"\n  Masked text: '{masked_text}'")
            logging.info("=" * 80)
            logging.info("")

    total_true = mask.sum().item()
    logging.info(f"Total tokens marked as True: {total_true}")

    return mask, mask_info


def load_dataset(args, tokenizer):
    """Load forget dataset."""
    logging.info(f"Using Q+A full sequence (Answer tokens only will be masked)")
    logging.info(f"Loading forget dataset: {args.forget_split}")

    forget_dataset = TextDatasetQA(
        data_path=args.data_path,
        tokenizer=tokenizer,
        model_family=args.model_family,
        max_length=args.max_length,
        split=args.forget_split,
        question_key=args.question_key,
        answer_key=args.answer_key,
        answer_only=False,  # Use Q+A full sequence to match training format
    )

    logging.info(f"Forget dataset size: {len(forget_dataset)}")
    return forget_dataset


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get model config from yaml
    config_path = TOFU_DIR / "config" / "model_config.yaml"
    with open(config_path, "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    model_cfg = model_configs[args.model_family]
    model_id = model_cfg["hf_key"]

    logging.info(f"Building binary mask from GT spans for model family: {args.model_family}")
    logging.info(f"Using HuggingFace model: {model_id}")
    logging.info(f"Using checkpoint: {args.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

    # Load dataset
    forget_dataset = load_dataset(args, tokenizer)

    # Load GT spans
    logging.info(f"Loading GT spans from {args.gt_spans_path}")
    gt_spans_dict = load_gt_spans(args.gt_spans_path)
    logging.info(f"Loaded GT spans for {len(gt_spans_dict)} samples")

    # Build mask
    mask, mask_info = build_binary_mask_from_gt_spans(
        dataset=forget_dataset,
        tokenizer=tokenizer,
        gt_spans_dict=gt_spans_dict,
        num_samples=len(forget_dataset),
        max_length=args.max_length,
    )

    # Save mask
    save_id_str = f"_{args.save_id}" if args.save_id else ""
    save_dir = os.path.join(
        args.save_dir,
        "influence_masks",
        f"gt_spans_{args.forget_split}{save_id_str}",
    )
    os.makedirs(save_dir, exist_ok=True)

    # Save binary mask
    mask_path = os.path.join(save_dir, "mask_binary.pt")
    torch.save(mask, mask_path)

    # Save JSON with masked token info
    json_path = os.path.join(save_dir, "mask_info.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mask_info, f, indent=2, ensure_ascii=False)

    logging.info(f"Binary mask built successfully!")
    logging.info(f"Mask shape: {mask.shape}")
    logging.info(f"Total True tokens: {mask.sum().item()}")
    logging.info(f"Total False tokens: {(~mask).sum().item()}")
    logging.info(f"Mask saved to: {mask_path}")
    logging.info(f"Mask info JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
