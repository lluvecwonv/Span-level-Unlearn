import json
import csv
import ast
from collections import Counter, defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import re
import argparse
import os
from tqdm import tqdm

# python preprocess.py --file news --name news


def load_model(model_name, tokenizer_name="NousResearch/Llama-2-7b-chat-hf"):
    print(f"Loading model: {model_name}")
    print(f"Loading tokenizer: {tokenizer_name}")
    # Load tokenizer from base LLaMA 2 model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # Use GPU 1 (second GPU)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto"
    )
    model.eval()
    print("Model loaded successfully")
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_new_tokens, temperature):
    """Generate response using the loaded model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated part (exclude input prompt)
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

def generate_batch_responses(tokenizer, model, prompts, max_new_tokens, temperature):
    """
    Generate multiple responses in parallel using batch processing.
    For LLaMA 3, prompts are already formatted with special tokens.
    """

    # 1) For LLaMA 3, use the full prompt as-is (already formatted with special tokens)
    input_strings = prompts

    # 2) Left padding (recommended for decoder-only batching)
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = "left"
    if left_pad_tokenizer.pad_token is None:
        left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    # 3) Tokenize
    inputs = left_pad_tokenizer(
        input_strings,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 4) Generate - greedy decoding
    # The JSON output should be short, so limit generation and stop near the closing marker.

    # Stop tokens include "]}", "]\n}", and EOS.
    stop_token_ids = [
        left_pad_tokenizer.encode("]}", add_special_tokens=False),
        left_pad_tokenizer.encode("]\n}", add_special_tokens=False),
        [left_pad_tokenizer.eos_token_id],
    ]
    # Flatten and get unique stop token ids
    eos_ids = [left_pad_tokenizer.eos_token_id]

    # 4) Generate (deterministic by default; sampling only if temperature > 0)
    do_sample = temperature is not None and temperature > 0
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
        pad_token_id=left_pad_tokenizer.pad_token_id,
        eos_token_id=left_pad_tokenizer.eos_token_id
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    # 5) Decode ONLY the continuation after the input prompt tokens
    # Because we left-pad, "input length" is the number of non-pad tokens (attention_mask sum)
    results = []
    attn = inputs.get("attention_mask", None)
    for i, out_ids in enumerate(outputs):
        if attn is not None:
            prompt_len = int(attn[i].sum().item())
        else:
            # fallback: assume full length (less reliable without attention_mask)
            prompt_len = inputs["input_ids"].shape[1]

        completion_ids = out_ids[prompt_len:]
        generated_text = left_pad_tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # Post-process JSON output by removing text after the closing marker.
        # Handle several possible termination patterns.
        end_patterns = ['"]}\n', '"]}', "']}"]
        for end_pattern in end_patterns:
            if end_pattern in generated_text:
                generated_text = generated_text[:generated_text.index(end_pattern) + len(end_pattern)]
                break

        # Remove Chinese and special characters, keeping only characters needed for JSON.
        # Remove Korean, Chinese, and similar non-JSON text.
        cleaned = ""
        for char in generated_text:
            # Keep only JSON-compatible characters: ASCII plus selected punctuation.
            if ord(char) < 128 or char in '[]{}",:\' ':
                cleaned += char
            elif '\u4e00' <= char <= '\u9fff':  # Chinese character range.
                break  # Stop when Chinese text begins.
        generated_text = cleaned.strip()

        results.append(generated_text)

    return results


# STEP1
def extract_high_si_score_word_list_word(word_file,text_file,threshold):
    with open(word_file, "r", encoding = "utf-8") as f:
        word_data = json.load(f)

    # Collect scores for each sample_index
    scores_by_index = defaultdict(list)
    word_score_map = defaultdict(dict)

    for item in word_data:
        sample_index = item.get("sample_index", None)
        word = item.get("word",None)
        score = item.get("normalized_si_score",None)

        if sample_index is None or word is None or score is None:
            continue

        idx = int(sample_index)
        score_float = float(score)
        scores_by_index[idx].append(score_float)
        word_score_map[idx][str(word)] = score_float

    # Compute percentiles for each index
    thresholds_by_index = {}
    for idx, scores in scores_by_index.items():
        q3 = np.percentile(scores, threshold)  # Third quartile (75th percentile)
        thresholds_by_index[idx] = q3

    result = defaultdict(dict)

    # Select only words above the threshold for each index
    for idx, word_scores in word_score_map.items():
        threshold_for_idx = thresholds_by_index.get(idx, 0)
        result[idx]["words"] = []
        result[idx]["all_words"] = []

        for word, score in word_scores.items():
            # Store all words
            if word not in result[idx]["all_words"]:
                result[idx]["all_words"].append(word)

            # Select only words above the threshold
            if score >= threshold_for_idx:
                if word not in result[idx]["words"]:
                    result[idx]["words"].append(word)

    with open(text_file, "r", encoding = "utf-8") as f:
        data = json.load(f)

        # MUSE format: list of strings
        if isinstance(data, list):
            for idx, text in enumerate(data):
                if idx in result:
                    result[idx]["text"] = text if isinstance(text, str) else str(text)
        # TOFU format: line-by-line JSON objects
        else:
            f.seek(0)
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if idx in result:
                        answer = obj.get("answer", "")
                        result[idx]["text"] = answer
                except json.JSONDecodeError:
                    continue

    return dict(result)

def load_json_object_from_text(r: str, prefix_used: bool = True):
    """
    Parse JSON from model output.
    If prefix_used=True, the prompt already includes '{"spans": ' so we need to prepend it.
    """
    s = r.strip()

    # The prompt starts with '{"spans": ', so prepend that prefix to the model output.
    if prefix_used:
        # Check whether the model output is already complete JSON.
        if not s.startswith("{"):
            s = '{"spans": ' + s

    # 1) If model outputs double braces, normalize them.
    if s.startswith("{{") and s.endswith("}}"):
        s = s[1:-1].strip()

    # 2) Extract the first JSON object substring from first '{' to last '}'
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]

    # 2.5) Try to find valid JSON by looking for {"spans": [...]} pattern
    # and cutting off everything after the first complete array
    import re
    match = re.search(r'\{"spans"\s*:\s*\[(.*?)\]', s, re.DOTALL)
    if match:
        # Found a complete spans array, use it
        s = '{"spans": [' + match.group(1) + ']}'

    # 2.6) Extract spans more aggressively
    # Look for any quoted strings after {"spans": [ pattern
    def extract_spans_aggressive(s):
        # Try to find {"spans": [ and extract everything after
        spans_match = re.search(r'\{"spans"\s*:\s*\[', s)
        if not spans_match:
            return s

        # Get everything after the opening bracket
        after_bracket = s[spans_match.end():]

        # Extract all quoted strings (both " and ')
        quoted_strings = re.findall(r'["\']([^"\']+)["\']', after_bracket[:1000])  # Limit to first 1000 chars

        if quoted_strings:
            # Also try to extract capitalized phrases (names) even without quotes
            # Look for patterns like: Word Word, Another Word
            words_without_quotes = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', after_bracket[:500])

            # Combine and deduplicate
            all_spans = list(dict.fromkeys(quoted_strings + words_without_quotes))

            # Filter out common words
            common_words = {'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of', 'And', 'Or', 'But'}
            all_spans = [span for span in all_spans if span not in common_words and len(span) > 1]

            if all_spans:
                return '{"spans": [' + ', '.join(f'"{span}"' for span in all_spans) + ']}'

        return s

    s = extract_spans_aggressive(s)

    # 3) Convert single quotes to double quotes for JSON compatibility.
    # Note: single quotes inside strings should ideally be preserved.
    # Simple fallback: try replacing all single quotes with double quotes.
    s_fixed = s.replace("'", '"')

    # 4) Try to repair incomplete JSON, such as missing closing brackets.
    for attempt_str in [s, s_fixed]:
        try:
            return json.loads(attempt_str)
        except json.JSONDecodeError:
            pass

    # Final fallback: add closing brackets.
    for attempt_str in [s, s_fixed]:
        temp = attempt_str
        if temp.count("[") > temp.count("]"):
            temp = temp + "]" * (temp.count("[") - temp.count("]"))
        if temp.count("{") > temp.count("}"):
            temp = temp + "}" * (temp.count("{") - temp.count("}"))
        try:
            return json.loads(temp)
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("Failed to parse", s, 0)


def normalize_for_match(s: str) -> str:
    # Lowercase, remove quotes/commas, and normalize whitespace
    s = s.lower()
    s = re.sub(r"[\"'`,]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_entities_from_natural_language(response: str, text: str, keywords: list) -> list:
    """
    Fallback function to extract entities from natural language response.
    When model fails to generate JSON, extract entities from the text.
    CONSERVATIVE approach: only extract high-quality named entities.
    """
    entities = []

    # Stopwords to filter out
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been',
                 'he', 'she', 'it', 'they', 'that', 'this', 'these', 'those'}

    # Strategy 1: Extract person names (2+ capitalized words, no stopwords)
    # Pattern: First Last, or Title First Last
    name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    name_matches = re.findall(name_pattern, response)
    for name in name_matches:
        # Filter: must be in text, 2+ words, no single letters
        words = name.split()
        if (name in text and
            len(words) >= 2 and
            all(len(w) > 1 for w in words) and
            name not in entities):
            entities.append(name)

    # Strategy 2: Extract with title patterns (high confidence)
    # "Prime Minister X", "President Y", "CEO Z", etc.
    title_patterns = [
        r'(?:Prime Minister|President|CEO|Minister|Director|Chairman|Dr|Professor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(?:Mr|Ms|Mrs|Miss)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    ]
    for pattern in title_patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            full_name = match.strip()
            if full_name in text and full_name not in entities and len(full_name) > 2:
                entities.append(full_name)

    # Strategy 3: Extract organization/location names (capitalized, 2+ words)
    # Only if they appear with certain keywords
    org_keywords = ['company', 'corporation', 'police', 'government', 'court', 'university',
                    'hospital', 'school', 'committee', 'council', 'agency']
    for keyword in org_keywords:
        # Find phrases before the keyword
        pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+' + keyword
        matches = re.findall(pattern, response, re.IGNORECASE)
        for match in matches:
            if match in text and match not in entities and len(match.split()) >= 2:
                entities.append(match)

    # Filter: Remove overly long sentences (likely extraction errors)
    # Keep spans with 2-8 words
    filtered_entities = []
    for e in entities:
        word_count = len(e.split())
        if not (2 <= word_count <= 8):
            continue

        # Check if contains keyword (case-insensitive)
        e_lower = e.lower()
        has_keyword = any(kw.lower() in e_lower for kw in keywords if len(kw) > 2)

        # Check if it's a high-quality name (proper noun pattern)
        # Title + Name pattern (e.g., "Sir Jeffrey Donaldson", "Mr Murphy")
        is_titled_name = bool(re.match(r'^(?:Sir|Mr|Ms|Mrs|Dr|President|Minister|CEO)\s+[A-Z]', e))

        # Full name pattern (2+ capitalized words, e.g., "Jeffrey Donaldson", "Conor Murphy")
        is_full_name = bool(re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', e)) and word_count >= 2

        # Accept if: has keyword OR is high-quality name
        if has_keyword or is_titled_name or is_full_name:
            filtered_entities.append(e)

    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for e in filtered_entities:
        if e not in seen:
            seen.add(e)
            unique_entities.append(e)

    return unique_entities


def process_sample_with_self_consistency(tokenizer, model, sample_index, high_si_score_word_list, text, all_words, iteration):
    """Process a single sample using self-consistency approach."""
    print(f"\nProcessing sample {sample_index}...")
    print(high_si_score_word_list)
    print(text)


    # Continuation-style prompt for base model (not instruction-tuned)
    # Provide few-shot examples and let the model continue the pattern

    # Limit keywords to avoid too long prompt
    keywords_str = ', '.join(high_si_score_word_list[:20])  # Use only first 20 keywords
    if len(high_si_score_word_list) > 20:
        keywords_str += f' (and {len(high_si_score_word_list)-20} more)'

    prompt = f"""Extract person names and important phrases from text that match keywords.

Example 1:
TEXT: "John Doe works at Microsoft in Seattle. CEO Jane Smith announced the merger."
KEYWORDS: John, CEO, announced
Output: {{"spans": ["John Doe", "CEO Jane Smith", "announced the merger"]}}

Example 2:
TEXT: "Prime Minister visited Athens yesterday. The Greek government responded."
KEYWORDS: Prime, Minister, Athens, government
Output: {{"spans": ["Prime Minister", "Athens", "Greek government"]}}

Example 3:
TEXT: "The train crash in Greece killed 57 victims. Minister Kostas Karamanlis resigned."
KEYWORDS: train, crash, victims, Minister, Kostas
Output: {{"spans": ["train crash", "57 victims", "Minister Kostas Karamanlis"]}}

Example 4:
TEXT: "Madonna thanked her fans for their positive energy and support."
KEYWORDS: Madonna, thanked, fans, positive
Output: {{"spans": ["Madonna", "thanked her fans", "positive energy"]}}

Example 5:
TEXT: "The weather is sunny today with clear skies."
KEYWORDS: weather, sunny
Output: {{"spans": []}}

Rules:
- Extract spans EXACTLY as they appear in TEXT
- Focus on person names (2+ capitalized words) and key phrases
- Each span must contain at least one keyword
- Return empty list if no matching spans found

TEXT: "{text[:1000]}"
KEYWORDS: {keywords_str}
Output: {{"spans": ["""

    prompts = [prompt] * iteration

    iteration_results = generate_batch_responses(tokenizer, model, prompts, max_new_tokens=256, temperature=None)  # Greedy decoding to avoid sampling errors

    entity_counter = Counter()
    itr_selected_span = []
    
    print(f'[TEXT] : {text}')
    print(f' [HIGH_SI] : {high_si_score_word_list}')
    print(iteration_results)

    for iter_idx, r in enumerate(iteration_results):
        try:
            obj = load_json_object_from_text(r)
            entities = obj.get("spans",[])
        except json.JSONDecodeError:
            try:
                entities = ast.literal_eval(r)
            except (SyntaxError, ValueError):
                print(f"######[Warning]#######: Failed to parse JSON response")
                print(f"  Attempting to extract entities from natural language...")
                # Fallback: Extract entities from natural language response
                entities = extract_entities_from_natural_language(r, text, high_si_score_word_list)
                if entities:
                    print(f"  ✓ Extracted {len(entities)} entities via fallback: {entities[:5]}")
                else:
                    print(f"  ✗ No entities extracted via fallback")

        # Handle cases where entities is not a list
        if not isinstance(entities, list):
            entities = [entities] if entities else []
        
        entities = [e for e in entities if isinstance(e, str) and e in text] # Verify that model-generated span candidates appear exactly in the text.

        # Normalize list elements by extracting strings only
        cleaned_entities = []
        for entity in entities:
            if isinstance(entity, str):
                cleaned_entities.append(entity.strip())
            elif isinstance(entity, dict) and 'text' in entity:
                # Extract the text value for {"text": "..."} items
                cleaned_entities.append(entity['text'].strip())
            else:
                # Convert other formats to strings
                cleaned_entities.append(str(entity).strip())

        # **Modified Rule**: More flexible filtering
        # Keep spans that either:
        # 1. Contain at least one word from high_si_score (case-insensitive)
        # 2. Look like high-quality proper nouns (2+ capitalized words)
        verified_entities = []
        for span in cleaned_entities:
            span_norm = normalize_for_match(span)
            contains_keyword = False
            matched_keywords = []

            for keyword in high_si_score_word_list:
                keyword_norm = normalize_for_match(keyword)
                if keyword_norm and keyword_norm in span_norm:
                    contains_keyword = True
                    matched_keywords.append(keyword)

            # Check if it's a high-quality proper noun
            # Pattern: 2+ consecutive capitalized words (e.g., "Jeffrey Donaldson", "Conor Murphy")
            is_quality_name = False
            words = span.split()
            if (len(words) >= 2 and
                span[0].isupper() and
                all(len(w) > 1 for w in words)):  # No single-letter words
                # Check if it's a name pattern (First Last or Title First Last)
                cap_words = sum(1 for w in words if w[0].isupper())
                if cap_words >= 2:
                    is_quality_name = True

            if contains_keyword or is_quality_name:
                verified_entities.append(span)


        # Save each iteration result before and after post-processing
        itr_result = {
            f"iteration_{iter_idx}": verified_entities
        }
        itr_selected_span.append(itr_result)

        # Add only validated entities to the global counter
        for entity in verified_entities:
            entity_counter[entity] += 1

    # Calculate SC score (Selection Consistency): count / iteration
    sc_scores = {entity: count / iteration for entity, count in entity_counter.items()}

    result_dict = {
        "sample_index": sample_index,
        "high_si_score_word_list_words": high_si_score_word_list,
        "text": text,
        "itr_selected_span": itr_selected_span,
        "total_selected_span": sc_scores
    }

    return result_dict

def load_gt_tokens_from_csv(csv_file):
    """Load ground truth tokens from CSV file."""
    gt_tokens_dict = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            gt_label = row.get('gt_label_tokens', '').strip()
            full_context = row.get('full_context', '').strip()

            # Parse gt_label_tokens (handle comma-separated values)
            if gt_label:
                # Split by comma and clean each token
                gt_tokens = [token.strip() for token in gt_label.split(',')]
            else:
                gt_tokens = []

            gt_tokens_dict[idx] = {
                'gt_tokens': gt_tokens,
                'full_context': full_context
            }

    return gt_tokens_dict

def normalize_span(s):
    """Normalize span for matching: lowercase, remove punctuation at edges."""
    s = s.lower().strip()
    # Remove leading/trailing punctuation
    s = re.sub(r'^[^\w]+', '', s)  # Remove leading special characters
    s = re.sub(r'[^\w]+$', '', s)  # Remove trailing special characters
    return s

def calculate_metrics(selected_spans, gt_tokens, text):
    """
    Calculate precision, recall, f1, tp, fp, fn.

    Matching logic:
    - Normalize both spans (lowercase, remove trailing punctuation)
    - Check if one contains the other or exact match
    """
    # Normalize for matching
    selected_norm = [(span, normalize_span(span)) for span in selected_spans]
    gt_norm = [(token, normalize_span(token)) for token in gt_tokens]

    # Find matched words using flexible matching
    matched_words = []
    matched_gt_indices = set()
    matched_sel_indices = set()

    for gt_idx, (gt_orig, gt_n) in enumerate(gt_norm):
        for sel_idx, (sel_orig, sel_n) in enumerate(selected_norm):
            if sel_idx in matched_sel_indices:
                continue
            # Flexible match: exact, or one contains the other
            if gt_n == sel_n or gt_n in sel_n or sel_n in gt_n:
                if gt_idx not in matched_gt_indices:
                    matched_words.append(gt_tokens[gt_idx])
                    matched_gt_indices.add(gt_idx)
                    matched_sel_indices.add(sel_idx)
                    break

    tp = len(matched_words)
    fp = len(selected_spans) - tp
    fn = len(gt_tokens) - tp

    precision = (tp / len(selected_spans) * 100) if len(selected_spans) > 0 else 0.0
    recall = (tp / len(gt_tokens) * 100) if len(gt_tokens) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1': round(f1, 2),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'matched_words': matched_words
    }

def evaluate_results(selected_output, gt_tokens_dict, output_file):
    """Evaluate selected results against ground truth and save to JSON."""
    evaluation_results = []

    for result in selected_output:
        sample_idx = result['sample_index']
        selected_spans_dicts = result['selected_span']

        # Extract span names from list of dicts
        selected_spans = [list(span_dict.keys())[0] for span_dict in selected_spans_dicts]

        # Get ground truth for this sample
        if sample_idx in gt_tokens_dict:
            gt_info = gt_tokens_dict[sample_idx]
            gt_tokens = gt_info['gt_tokens']
            full_context = gt_info['full_context']
        else:
            gt_tokens = []
            full_context = ""

        # Calculate metrics
        metrics = calculate_metrics(selected_spans, gt_tokens, full_context)

        # Create evaluation result
        eval_result = {
            'sample_index': sample_idx,
            'full_context': full_context,
            'selected_words': selected_spans,
            'gt_tokens': gt_tokens,
            'matched_words': metrics['matched_words'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn']
        }

        evaluation_results.append(eval_result)

    # Save evaluation results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    return evaluation_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TOFU span selection with self-consistency')
    parser.add_argument('--file', type=str, default='', help='Subfolder name for output files')
    parser.add_argument('--name', type=str, default='', help='Name suffix for output files')
    args = parser.parse_args()

    muse_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    word_file = os.path.join(
        muse_root,
        "influence_results",
        "influence_masks",
        "news_forget_half_forget10",
        "mask_sample=0.1_word_all_words.json"
    )
    text_file = os.path.join(muse_root, "data", "news", "raw", "forget.json")

    # Determine output directory
    base_result_dir = os.path.join(muse_root, "self_conf", "results")
    if args.file:
        output_dir = os.path.join(base_result_dir, args.file)
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = base_result_dir

    # Set output file names with optional suffix
    if args.name:
        output_filename = f"result_{args.name}.json"
        selected_filename = f"selected_result_{args.name}.json"
        evaluation_filename = f"evaluation_result_{args.name}.json"
        params_filename = f"{args.name}.txt"
    else:
        output_filename = "result_ours_chatml3.json"
        selected_filename = "selected_result_test.json"
        evaluation_filename = "evaluation_result.json"
        params_filename = "parameters.txt"

    # Set output file paths
    output_file = os.path.join(output_dir, output_filename)
    selected_output_file = os.path.join(output_dir, selected_filename)
    evaluation_output_file = os.path.join(output_dir, evaluation_filename)
    params_file = os.path.join(output_dir, params_filename)

    # Use the fine-tuned model before unlearning for span extraction.
    # Use the Hugging Face MUSE target model.
    MODEL_NAME = "muse-bench/MUSE-news_target"
    # Load models
    tokenizer, model = load_model(MODEL_NAME)

    iteration = 10  # Number of self-consistency iterations
    sc_threshold = 0.5  # SC score threshold (0~1) - balanced for quality filtering (5+ out of 10 iterations)

    # STEP1: Extract high SI score words
    si_threshold = 85  # 85th percentile
    x = extract_high_si_score_word_list_word(word_file, text_file, si_threshold)

    output = []


    # STEP2: Process each sample with self-consistency
    print(f"\nProcessing {len(x)} samples with {iteration} iterations each...")
    print(f"Results will be saved incrementally to:")
    print(f"  - {output_file}")
    print(f"  - {selected_output_file}\n")

    selected_output = []

    for i in tqdm(range(len(x)), desc="Processing samples"):
        result_dict = process_sample_with_self_consistency(
            tokenizer, model, i, x[i]["words"], x[i]["text"], x[i]["all_words"],iteration
        )
        output.append(result_dict)

        # Filter and add to selected output
        selected_spans = [
            {span: score}
            for span, score in result_dict["total_selected_span"].items()
            if score >= sc_threshold
        ]
        selected_output.append({
            "sample_index": result_dict["sample_index"],
            "selected_span": selected_spans
        })

        # Save incrementally after each sample (real-time save)
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        with open(selected_output_file, "w", encoding='utf-8') as f:
            json.dump(selected_output, f, ensure_ascii=False, indent=2)

    num_samples = len(selected_output)

    # Save parameters to text file
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("EXPERIMENT PARAMETERS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Model Configuration:\n")
        f.write(f"  - Model Name: {MODEL_NAME}\n\n")

        f.write("Hyperparameters:\n")
        f.write(f"  - Iteration: {iteration}\n")
        f.write(f"  - SC Threshold: {sc_threshold}\n")
        f.write(f"  - SI Threshold (percentile): {si_threshold}\n")
        f.write(f"  - Number of Samples: {num_samples}\n\n")

    print(f"\n Full results saved to: {output_file}")
    print(f" Selected results (SC >= {sc_threshold}) saved to: {selected_output_file}")
    print(f" Parameters saved to: {params_file}")
    print("Finish")

if __name__ == "__main__":
    main()
