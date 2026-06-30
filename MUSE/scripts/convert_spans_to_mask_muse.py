#!/usr/bin/env python3
"""
Convert MUSE GPT span JSON files into forget masks as torch tensors.

Usage:
    python scripts/convert_spans_to_mask_muse.py \
        --spans_path data/gpt_spans_news_forget.json \
        --data_path ../data/news/raw/forget.txt \
        --output_path data/news_forget_mask.pt \
        --model_name meta-llama/Llama-2-7b-hf \
        --max_length 2048

Output: a .pt file containing List[torch.Tensor].
        Each tensor is a boolean mask of shape [max_length].

Important: MUSE splits data at the paragraph level, so indices in the
span JSON may differ from indices in the final mask list. This script reloads
the original text and builds the correct mapping.
"""

import argparse
import json
import torch
from transformers import AutoTokenizer
from pathlib import Path
from typing import List


def load_text_data(file_path: str) -> List[str]:
    """
    Load text files using the same logic as generate_gpt_spans_muse.py.
    """
    path = Path(file_path)

    if path.suffix == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        # Split into paragraphs using blank lines.
        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
        return paragraphs

    elif path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data[0], str):
            return data
        elif isinstance(data[0], dict) and 'text' in data[0]:
            return [item['text'] for item in data]
        else:
            raise ValueError("JSON format not recognized")

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def text_to_chunks(tokenizer, text: str, max_length: int, add_bos_token: bool = True) -> List[torch.Tensor]:
    """
    Split text into chunks using the same logic as MUSE DefaultDataset.
    """
    # Tokenize the full text without a BOS token.
    tokens = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids[0]

    chunks = []
    if add_bos_token:
        # Add BOS by chunking at max_length - 1 and prepending the BOS token.
        for i in range(0, len(tokens), max_length - 1):
            chunk = tokens[i : i + max_length - 1]
            # Add the BOS token.
            chunk_with_bos = torch.cat([torch.tensor([tokenizer.bos_token_id]), chunk])
            chunks.append(chunk_with_bos)
    else:
        # Simple chunking without BOS.
        for i in range(0, len(tokens), max_length):
            chunks.append(tokens[i : i + max_length])

    # If the final chunk is shorter than max_length, rotate in tokens from the first chunk.
    if len(chunks) > 0 and len(chunks[-1]) < max_length:
        chunks[-1] = torch.cat([chunks[-1], chunks[0]], dim=-1)[:max_length]

    return chunks


def spans_to_token_mask(
    tokenizer,
    text: str,
    spans: List[List[int]],
    max_length: int = 2048,
    add_bos_token: bool = True
) -> List[torch.Tensor]:
    """
    Convert character-level spans into token-level masks with MUSE chunking.

    Returns:
        List[torch.Tensor]: a boolean mask for each chunk.
    """
    if not spans:
        # Return all-False masks when no spans are available.
        chunks = text_to_chunks(tokenizer, text, max_length, add_bos_token)
        return [torch.zeros(max_length, dtype=torch.bool) for _ in chunks]

    # Tokenize the full text without BOS first to obtain offset mappings.
    encoding_full = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors='pt'
    )

    full_tokens = encoding_full['input_ids'][0]
    full_offsets = encoding_full['offset_mapping'][0]

    # Mark which tokens overlap with any span.
    token_in_span = torch.zeros(len(full_tokens), dtype=torch.bool)

    for span_start, span_end in spans:
        for i, (tok_start, tok_end) in enumerate(full_offsets):
            if tok_start is None or tok_end is None:
                continue
            # Mark tokens that overlap with the span.
            tok_start = tok_start.item() if isinstance(tok_start, torch.Tensor) else tok_start
            tok_end = tok_end.item() if isinstance(tok_end, torch.Tensor) else tok_end
            if tok_start < span_end and tok_end > span_start:
                token_in_span[i] = True

    # Now build chunk-level masks, including BOS when requested.
    chunk_masks = []

    if add_bos_token:
        chunk_size = max_length - 1
        for i in range(0, len(full_tokens), chunk_size):
            # Mask for tokens in this chunk.
            chunk_token_mask = token_in_span[i : i + chunk_size]

            # The BOS token is never masked.
            chunk_mask = torch.cat([
                torch.tensor([False]),  # BOS token
                chunk_token_mask
            ])

            # Padding
            if len(chunk_mask) < max_length:
                pad_length = max_length - len(chunk_mask)
                chunk_mask = torch.cat([
                    chunk_mask,
                    torch.zeros(pad_length, dtype=torch.bool)
                ])

            chunk_masks.append(chunk_mask[:max_length])
    else:
        for i in range(0, len(full_tokens), max_length):
            chunk_token_mask = token_in_span[i : i + max_length]

            # Padding
            if len(chunk_token_mask) < max_length:
                pad_length = max_length - len(chunk_token_mask)
                chunk_token_mask = torch.cat([
                    chunk_token_mask,
                    torch.zeros(pad_length, dtype=torch.bool)
                ])

            chunk_masks.append(chunk_token_mask[:max_length])

    # Rotate the final chunk in the same way as DefaultDataset.
    if len(chunk_masks) > 1 and len(chunk_masks[-1]) < max_length:
        # Rotate the final chunk with the first chunk.
        last_chunk_len = len(chunk_masks[-1])
        rotated_mask = torch.cat([
            chunk_masks[-1],
            chunk_masks[0]
        ])[:max_length]
        chunk_masks[-1] = rotated_mask

    return chunk_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spans_path", type=str, required=True, help="GPT span JSON file")
    parser.add_argument("--data_path", type=str, required=True, help="Original data file (.txt or .json)")
    parser.add_argument("--output_path", type=str, required=True, help="Output .pt file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--no_bos_token", action="store_true", help="Do not add a BOS token")
    args = parser.parse_args()

    add_bos_token = not args.no_bos_token

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load span file.
    print(f"Loading spans: {args.spans_path}")
    with open(args.spans_path, 'r', encoding='utf-8') as f:
        spans_data = json.load(f)

    # Load original data while preserving order.
    print(f"Loading original data: {args.data_path}")
    texts = load_text_data(args.data_path)

    if len(texts) != len(spans_data):
        print(f"Warning: Text count ({len(texts)}) != Span count ({len(spans_data)})")
        print("Using minimum of the two...")
        min_len = min(len(texts), len(spans_data))
        texts = texts[:min_len]

    all_masks = []
    total_span_tokens = 0
    total_chunks = 0

    for idx, text in enumerate(texts):
        if str(idx) not in spans_data:
            print(f"Warning: No span data for index {idx}, using empty spans")
            spans = []
        else:
            item = spans_data[str(idx)]

            # Check text alignment.
            if item['text'] != text:
                print(f"Warning: Text mismatch at index {idx}")
                print(f"  Expected: {text[:100]}...")
                print(f"  Got: {item['text'][:100]}...")

            spans = item['spans']

        # Convert the text into chunks and build a mask for each chunk.
        chunk_masks = spans_to_token_mask(
            tokenizer,
            text,
            spans,
            args.max_length,
            add_bos_token
        )

        all_masks.extend(chunk_masks)

        # Statistics.
        chunk_span_tokens = sum(mask.sum().item() for mask in chunk_masks)
        total_span_tokens += chunk_span_tokens
        total_chunks += len(chunk_masks)

        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"[{idx+1}/{len(texts)}] Processed {len(chunk_masks)} chunks, {chunk_span_tokens} span tokens")

    print(f"\nTotal paragraphs: {len(texts)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total masks: {len(all_masks)}")
    print(f"Total span tokens: {total_span_tokens}")
    print(f"Avg span tokens per chunk: {total_span_tokens / total_chunks:.2f}")
    print(f"Avg chunks per paragraph: {total_chunks / len(texts):.2f}")

    # Save.
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(all_masks, args.output_path)
    print(f"\nSaved {len(all_masks)} masks to {args.output_path}")


if __name__ == "__main__":
    main()
