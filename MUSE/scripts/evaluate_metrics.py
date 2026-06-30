#!/usr/bin/env python3
"""
Evaluate Precision, Recall, and F1 Score for selected tokens vs ground truth.
Uses substring matching with minimum length constraint (≥3 chars).
"""

import json
import sys
import csv
from collections import Counter


def load_gt_labels_from_csv(csv_path):
    """
    Load ground truth labels from CSV file.

    Args:
        csv_path: Path to gt_token.csv

    Returns:
        dict mapping full_context to list of GT tokens
    """
    gt_mapping = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_label_tokens = row.get('gt_label_tokens', '').strip()
            full_context = row.get('full_context', '').strip()
            if gt_label_tokens and full_context:
                # Split by comma and clean up tokens
                tokens = [t.strip().lower() for t in gt_label_tokens.split(',') if t.strip()]
                # Use full_context as key for matching
                gt_mapping[full_context] = tokens
    return gt_mapping


def calculate_metrics(selected_tokens_path, gt_csv_path=None, min_length=2):
    """
    Calculate Precision, Recall, F1 with substring matching.

    Args:
        selected_tokens_path: Path to selected_tokens.json
        gt_csv_path: Optional path to gt_token.csv (if None, uses gt_token from JSON)
        min_length: Minimum length for substring matching (default: 2)

    Returns:
        dict with metrics
    """
    # Load selected tokens
    with open(selected_tokens_path, 'r') as f:
        selected = json.load(f)

    # Load GT labels from CSV if provided
    gt_from_csv = None
    if gt_csv_path:
        gt_from_csv = load_gt_labels_from_csv(gt_csv_path)
        print(f"Loaded GT labels from CSV: {len(gt_from_csv)} samples")
    else:
        print("Using GT labels from selected_tokens.json")

    # Calculate metrics with SUBSTRING MATCHING
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    samples_with_gt = 0
    samples_with_match = 0

    sample_stats = {}

    for entry in selected:
        sample_idx = entry['sample_index']
        word = entry['word'].lower().strip()
        full_context = entry.get('full_context', '').strip()

        # Remove special tokens for matching with CSV
        context_for_matching = full_context.replace('<s> ', '').replace(' </s>', '').replace('</s>', '').strip()

        # Get GT tokens from CSV or JSON
        if gt_from_csv is not None and context_for_matching in gt_from_csv:
            gt_tokens = gt_from_csv[context_for_matching]
        else:
            gt_tokens_str = entry.get('gt_token', '').strip()
            gt_tokens = [t.strip().lower() for t in gt_tokens_str.split('|')] if gt_tokens_str else []

        if sample_idx not in sample_stats:
            sample_stats[sample_idx] = {
                'selected': [],
                'gt_tokens': [],
                'matches': [],
                'full_context': full_context
            }

        sample_stats[sample_idx]['selected'].append(word)

        if gt_tokens:
            sample_stats[sample_idx]['gt_tokens'] = gt_tokens

            # SUBSTRING MATCHING with minimum length constraint
            is_match = False
            is_duplicate = word in sample_stats[sample_idx]['matches']  # Check if already matched

            for gt_token in gt_tokens:
                if len(word) >= min_length and len(gt_token) >= min_length:
                    if word in gt_token or gt_token in word:
                        is_match = True
                        if not is_duplicate:
                            sample_stats[sample_idx]['matches'].append(word)
                        break

            # Only count as TP if it's a match AND not a duplicate
            if is_match and not is_duplicate:
                tp += 1
            else:
                fp += 1
        else:
            fp += 1

    # Count false negatives and calculate per-sample metrics
    sample_details = []
    for sample_idx, stats in sample_stats.items():
        if stats['gt_tokens']:
            samples_with_gt += 1
            if stats['matches']:
                samples_with_match += 1

            # Count FN for this sample
            sample_fn = 0
            for gt_token in stats['gt_tokens']:
                found = False
                for selected_word in stats['selected']:
                    if len(gt_token) >= min_length and len(selected_word) >= min_length:
                        if gt_token in selected_word or selected_word in gt_token:
                            found = True
                            break
                if not found:
                    fn += 1
                    sample_fn += 1

            # Calculate per-sample metrics
            sample_tp = len(stats['matches'])
            sample_fp = len(stats['selected']) - sample_tp
            sample_precision = sample_tp / len(stats['selected']) * 100 if len(stats['selected']) > 0 else 0
            sample_recall = sample_tp / (sample_tp + sample_fn) * 100 if (sample_tp + sample_fn) > 0 else 0

            sample_details.append({
                'sample_index': sample_idx,
                'full_context': stats['full_context'],
                'selected_words': stats['selected'],
                'gt_tokens': stats['gt_tokens'],
                'matched_words': stats['matches'],
                'precision': round(sample_precision, 2),
                'recall': round(sample_recall, 2),
                'tp': sample_tp,
                'fp': sample_fp,
                'fn': sample_fn
            })

    # Calculate overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Count unique words
    unique_words = set()
    for stats in sample_stats.values():
        unique_words.update(stats['selected'])

    return {
        'total_tokens': len(selected),
        'unique_words': len(unique_words),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'samples_with_gt': samples_with_gt,
        'samples_with_match': samples_with_match,
        'sample_level_recall': samples_with_match / samples_with_gt * 100 if samples_with_gt > 0 else 0,
        'sample_stats': sample_stats,
        'sample_details': sample_details
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_metrics.py <selected_tokens.json> [gt_token.csv]")
        sys.exit(1)

    selected_tokens_path = sys.argv[1]
    gt_csv_path = sys.argv[2] if len(sys.argv) > 2 else None

    print("="*80)
    print("EVALUATING PRECISION & RECALL (Substring matching ≥2 chars)")
    print("="*80)
    if gt_csv_path:
        print(f"Using GT labels from CSV: {gt_csv_path}")
    print()

    metrics = calculate_metrics(selected_tokens_path, gt_csv_path)

    print(f"\nTotal selected tokens: {metrics['total_tokens']}")
    print(f"Unique words: {metrics['unique_words']}")
    print(f"\nTrue Positives (TP): {metrics['tp']}")
    print(f"False Positives (FP): {metrics['fp']}")
    print(f"False Negatives (FN): {metrics['fn']}")
    print(f"\nPrecision: {metrics['precision']:.2f}%")
    print(f"Recall: {metrics['recall']:.2f}%")
    print(f"F1 Score: {metrics['f1']:.2f}%")
    print(f"\nSamples with GT: {metrics['samples_with_gt']}/400")
    print(f"Samples with at least one match: {metrics['samples_with_match']}/{metrics['samples_with_gt']}")
    print(f"Sample-level recall: {metrics['sample_level_recall']:.2f}%")

    # Show unique words distribution
    word_counts = Counter([len(set(stats['selected'])) for stats in metrics['sample_stats'].values()])
    print("\n" + "="*80)
    print("Unique words per sample distribution:")
    print("="*80)
    for count in sorted(word_counts.keys()):
        print(f"  {count} unique words: {word_counts[count]} samples")

    # Show some matching examples
    print("\n" + "="*80)
    print("Examples of MATCHING tokens:")
    print("="*80)
    match_count = 0
    for sample_idx, stats in sorted(metrics['sample_stats'].items()):
        if stats['matches'] and match_count < 10:
            print(f"Sample {sample_idx}: {stats['matches'][:3]} ✓ (GT: {', '.join(stats['gt_tokens'][:3])})")
            match_count += 1

    # Save per-sample details to JSON
    import os
    output_dir = os.path.dirname(selected_tokens_path)
    output_basename = os.path.basename(selected_tokens_path).replace('.json', '')
    sample_details_path = os.path.join(output_dir, f"{output_basename}_sample_details.json")

    with open(sample_details_path, 'w', encoding='utf-8') as f:
        json.dump(metrics['sample_details'], f, ensure_ascii=False, indent=2)

    print(f"\nPer-sample details saved to: {sample_details_path}")


if __name__ == '__main__':
    main()
