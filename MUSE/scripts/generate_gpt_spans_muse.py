#!/usr/bin/env python3
"""
SEUL Offline for MUSE: label PII and sensitive-information spans in news
articles with GPT.

Usage:
    python scripts/generate_gpt_spans_muse.py \
        --data_path ../data/news/raw/forget.txt \
        --output_path data/gpt_spans_news_forget.json \
        --corpus news

Output format:
    {
        "0": {"text": "...", "spans": [[start, end], ...]},
        "1": {...},
        ...
    }
"""

import argparse
import json
import os
import time
from typing import List, Tuple
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai with: pip install openai")
    OpenAI = None


NEWS_SYSTEM_PROMPT = """You are a privacy expert for machine unlearning. Given a news article, identify ALL personally identifiable information (PII) and sensitive information spans that should be unlearned to protect privacy.

BE AGGRESSIVE - mark anything that could identify specific individuals or sensitive information:
- Full names of people (first name + last name combinations)
- Specific addresses (street addresses, house numbers)
- Phone numbers, email addresses, social security numbers
- Credit card numbers, bank account numbers
- Medical record numbers, health insurance IDs
- Passport numbers, driver's license numbers
- Biometric identifiers (fingerprints, facial recognition data)
- IP addresses when tied to individuals
- Dates of birth when combined with other identifiers
- Exact GPS coordinates of private residences
- Unique identifiers that can directly identify individuals

DO NOT mark general information:
- City names, country names (unless part of a specific address)
- Job titles or professions alone
- Generic dates or years
- Company/organization names
- General statistics or numbers
- Common first names without surnames

Return ONLY a JSON array of spans. Each span is [start_char, end_char] (0-indexed, end exclusive).
Mark ONLY information that could directly identify specific individuals.

Example:
Article: John Smith, living at 123 Oak Street, New York, was arrested yesterday. Police said the 45-year-old teacher had been under investigation.
Output: [[0, 10], [22, 39]]
(Spans: "John Smith", "123 Oak Street")

Example 2:
Article: The mayor of London announced new policies. Residents of the city welcomed the changes.
Output: []
(No PII - "mayor of London" is a title, "London" is just a city name)
"""


BOOKS_SYSTEM_PROMPT = """You are a content filter for machine unlearning. Given a book excerpt, identify ALL spans containing copyrighted creative content that should be unlearned.

Mark the following:
- Direct quotes from copyrighted works
- Unique character names and descriptions
- Plot details and story elements
- Distinctive dialogue
- Creative descriptions and prose
- Song lyrics or poetry
- Unique fictional terminology

DO NOT mark:
- Common words and phrases
- Generic descriptions
- Factual information
- Historical facts
- Common names

Return ONLY a JSON array of spans. Each span is [start_char, end_char] (0-indexed, end exclusive).

Example:
Text: Harry Potter picked up his wand and said "Expelliarmus!" The spell shot from his wand with a bright light.
Output: [[0, 12], [36, 50]]
(Spans: "Harry Potter", "Expelliarmus!")
"""


def get_system_prompt(corpus: str) -> str:
    """Return the system prompt for the selected corpus."""
    if corpus == "news":
        return NEWS_SYSTEM_PROMPT
    elif corpus == "books":
        return BOOKS_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown corpus: {corpus}")


def get_spans_from_gpt(client, text: str, system_prompt: str, model: str = "gpt-4o") -> List[Tuple[int, int]]:
    """Ask GPT to identify sensitive spans."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Article: {text}"}
            ],
            temperature=0,
            max_tokens=1000
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON and remove code fences when present.
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        # Handle empty lists.
        content = content.strip()
        if not content or content == "[]":
            return []

        spans = json.loads(content)
        return spans
    except Exception as e:
        print(f"Error: {e}")
        print(f"Content: {content if 'content' in locals() else 'N/A'}")
        return []


def backward_verify(client, text: str, spans: List[Tuple[int, int]], model: str = "gpt-4o") -> List[Tuple[int, int]]:
    """Backward verification: score sensitivity using only the span text."""
    if not spans:
        return []

    verified = []
    for start, end in spans:
        span_text = text[start:end]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Rate how sensitive this text span is for privacy (0=not sensitive, 1=somewhat, 2=very sensitive). Return only the number."},
                    {"role": "user", "content": f"Span: \"{span_text}\""}
                ],
                temperature=0,
                max_tokens=10
            )
            score = int(response.choices[0].message.content.strip())
            if score >= 1:  # Keep spans with score >= 1.
                verified.append([start, end])
        except:
            verified.append([start, end])  # Keep the span on errors.

    return verified


def load_text_data(file_path: str) -> List[str]:
    """
    Load text data.
    - .txt: split the full text into paragraphs using blank lines.
    - .json: support [{"text": "..."}, ...] or ["...", ...].
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to .txt or .json file")
    parser.add_argument("--corpus", type=str, choices=["news", "books"], required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--verify", action="store_true", help="Run backward verification")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls in seconds")
    args = parser.parse_args()

    if OpenAI is None:
        print("Please install openai first with: pip install openai")
        return

    # Load API key.
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Try loading from a repository-level .env file.
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        break

    if not api_key:
        print("API key required: pass --api_key, set OPENAI_API_KEY, or add it to the repository root .env file")
        return

    client = OpenAI(api_key=api_key)
    system_prompt = get_system_prompt(args.corpus)

    # Load data.
    print(f"Loading {args.data_path}...")
    texts = load_text_data(args.data_path)

    if args.max_samples:
        texts = texts[:args.max_samples]

    print(f"Loaded {len(texts)} text samples")

    results = {}

    for idx, text in enumerate(texts):
        print(f"[{idx+1}/{len(texts)}] Processing ({len(text)} chars): {text[:80]}...")

        # Forward pass: extract spans with GPT.
        spans = get_spans_from_gpt(client, text, system_prompt, args.model)

        # Backward verification (optional)
        if args.verify and spans:
            print(f"  Verifying {len(spans)} spans...")
            spans = backward_verify(client, text, spans, args.model)

        results[str(idx)] = {
            "text": text,
            "spans": spans
        }

        print(f"  Found {len(spans)} spans")

        time.sleep(args.delay)

    # Save.
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {args.output_path}")
    print(f"Total: {len(results)} samples, {sum(len(v['spans']) for v in results.values())} spans")

    # Print statistics.
    span_counts = [len(v['spans']) for v in results.values()]
    if span_counts:
        print(f"Avg spans per sample: {sum(span_counts) / len(span_counts):.2f}")
        print(f"Max spans: {max(span_counts)}, Min spans: {min(span_counts)}")
        print(f"Samples with spans: {sum(1 for c in span_counts if c > 0)} / {len(span_counts)}")


if __name__ == "__main__":
    main()
