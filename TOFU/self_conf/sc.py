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

# python preprocess_llama3.py --file preprocess_llama3 --name v.1

TOFU_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_model(model_name):
    print(f"Loading model: {model_name}")
    # Load tokenizer from base Llama 3 model since fine-tuned model may not have tokenizer files
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True, token=hf_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

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
        padding=True,
        add_special_tokens=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

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

        # Post-process: truncate after closing brace (model doesn't generate EOS properly)
        if '}' in generated_text:
            # Find the last closing brace (for JSON output)
            last_brace = generated_text.rfind('}')
            generated_text = generated_text[:last_brace + 1]

        results.append(generated_text)

    return results


# STEP1
def extract_high_si_score_word(word_file,text_file,threshold):
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
        for idx, text in enumerate(f):
            text = text.strip()
            if not text:
                continue

            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue

            if idx in result:
                answer = obj.get("answer","")
                result[idx]["text"] = answer

    return dict(result)

def load_json_object_from_text(r: str):
    s = r.strip()

    # 1) If model outputs double braces, normalize them.
    #    Example: '{{"spans": []}}' -> '{"spans": []}'
    if s.startswith("{{") and s.endswith("}}"):
        s = s[1:-1].strip()

    # 2) Extract the first JSON object substring from first '{' to last '}'
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]

    return json.loads(s)


def normalize_for_match(s: str) -> str:
    # Lowercase, remove quotes/commas, and normalize whitespace
    s = s.lower()
    s = re.sub(r"[\"'`,]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def process_sample_with_self_consistency(tokenizer, model, sample_index, high_si_score, text, all_words, iteration):
    """Process a single sample using self-consistency approach."""
    print(f"\nProcessing sample {sample_index}...")
    print(high_si_score)
    print(text)

    hi_si_json = json.dumps(high_si_score, ensure_ascii = False)

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a high-precision data forensics agent specializing in Machine Learning Unlearning. Your goal is to identify the "Forget Set"—specific text segments within a document that contribute to a high Self-Influence (SI) score.

A high SI score indicates that these specific spans contain unique, sensitive, or highly influential information (such as PII, proprietary facts, or distinctive phrasing) that the model has memorized too strongly.

[CRITICAL RULES]
1. EXTRACT VERBATIM: Spans must be exact substrings from the provided [TEXT].
2. GRANULARITY: Focus on the specific names, dates, unique identifiers, or rare facts.
3. OUTPUT FORMAT: You must output ONLY a valid JSON object. No markdown blocks (```json), no conversational text, no preamble.

[JSON SCHEMA]
{{"spans": ["exact_span_1", "exact_span_2"]}}

[EXAMPLE]
Input Text: "Contact representative John Doe at 555-0199 regarding the Alpha project."
Output: {{"spans": ["John Doe", "555-0199", "Alpha project"]}}

<|eot_id|><|start_header_id|>user<|end_header_id|>

[TEXT]: {text}
[HIGH_SI_SCORE]: {high_si_score}

Based on the SI score provided, identify the most influential/sensitive spans for unlearning. 
Remember: Output only the JSON object.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    prompts = [prompt] * iteration

    iteration_results = generate_batch_responses(tokenizer, model, prompts, max_new_tokens=256, temperature=0.7)

    entity_counter = Counter()
    itr_selected_span = []

    print(iteration_results)

    for iter_idx, r in enumerate(iteration_results):
        try:
            obj = load_json_object_from_text(r)
            entities = obj.get("spans",[])
        except json.JSONDecodeError:
            try:
                entities = ast.literal_eval(r)
            except (SyntaxError, ValueError):
                print(f"######[Warning]#######: Failed to parse response: {r}")
                entities = []

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

        # **Rule #1 Enforcement**: Deterministic post-filter
        # Only keep spans that contain at least one word from high_si_score (case-insensitive)
        verified_entities = []
        for span in cleaned_entities:
            span_norm = normalize_for_match(span)
            contains_keyword = False
            matched_keywords = []

            for keyword in high_si_score:
                keyword_norm = normalize_for_match(keyword)
                if keyword_norm and keyword_norm in span_norm:
                    contains_keyword = True
                    matched_keywords.append(keyword)

            if contains_keyword:
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
        "high_si_score_words": high_si_score,
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

def calculate_metrics(selected_spans, gt_tokens, text):
    """
    Calculate precision, recall, f1, tp, fp, fn.

    Matching logic:
    - A selected span matches a gt_token only if they are exactly the same
      (case-insensitive comparison).
    """
    # Normalize to lowercase for matching
    selected_lower = [span.lower() for span in selected_spans]
    gt_lower = [token.lower() for token in gt_tokens]

    # Find matched words using exact matching
    matched_words = []
    matched_gt_indices = set()

    for gt_idx, gt_token in enumerate(gt_lower):
        for sel_span in selected_lower:
            # Exact match only (case-insensitive)
            if gt_token == sel_span:
                if gt_idx not in matched_gt_indices:
                    matched_words.append(gt_tokens[gt_idx])
                    matched_gt_indices.add(gt_idx)
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

def generate_dummy_si_data(text_file):
    """Generate dummy SI score data from text file for testing."""
    import random

    result = {}

    with open(text_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            answer = obj.get("answer", "")
            if not answer:
                continue

            # Extract words from answer
            words = re.findall(r'\b\w+\b', answer)

            # Randomly select some words as "high SI score" words (simulate)
            # In real scenario, these would come from actual SI analysis
            num_high_si = max(1, len(words) // 4)  # ~25% of words
            high_si_words = random.sample(words, min(num_high_si, len(words)))

            result[idx] = {
                "words": high_si_words,
                "all_words": words,
                "text": answer
            }

    return result


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TOFU span selection with self-consistency')
    parser.add_argument('--file', type=str, default='', help='Subfolder name for output files')
    parser.add_argument('--name', type=str, default='', help='Name suffix for output files')
    parser.add_argument('--use_dummy_si', action='store_true', help='Use dummy SI data for testing')
    parser.add_argument('--gpu', type=int, default=1, help='GPU device ID')
    parser.add_argument('--model_name', type=str, default='lluvecwonv/llama3-8b-tofu-ft-5epochs', help='Model checkpoint or Hugging Face ID')
    parser.add_argument('--word_file', type=str, default=os.path.join(TOFU_DIR, 'self_conf', 'llama3.json'), help='Influence word JSON file')
    parser.add_argument('--text_file', type=str, default=os.path.join(TOFU_DIR, 'TOFU_data', 'forget10.json'), help='TOFU forget split JSON file')
    parser.add_argument('--gt_csv_file', type=str, default=os.path.join(TOFU_DIR, 'self_conf', 'gt_token_llama3.csv'), help='Ground-truth token CSV file')
    parser.add_argument('--result_dir', type=str, default=os.path.join(TOFU_DIR, 'self_conf', 'result'), help='Directory for self-consistency outputs')
    args = parser.parse_args()

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    word_file = args.word_file
    text_file = args.text_file
    gt_csv_file = args.gt_csv_file

    # Determine output directory
    base_result_dir = args.result_dir
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

    # Load models
    tokenizer, model = load_model(args.model_name)

    iteration = 10
    sc_threshold = 0.4  # SC score threshold (0~1)

    # STEP1: Extract high SI score words
    si_threshold = 85

    if args.use_dummy_si or not os.path.exists(word_file):
        print("Using dummy SI data for testing...")
        x = generate_dummy_si_data(text_file)
    else:
        x = extract_high_si_score_word(word_file, text_file, si_threshold)

    output = []


    # STEP2: Process each sample with self-consistency
    for i in range(30): # sorted(x.keys())
        result_dict = process_sample_with_self_consistency(
            tokenizer, model, i, x[i]["words"], x[i]["text"], x[i]["all_words"],iteration
        )
        output.append(result_dict)

    # Save full results
    with open(output_file,"w", encoding = 'utf-8') as f:
        json.dump(output, f, ensure_ascii = False, indent = 2)

    # STEP3: Filter spans by SC threshold and save
    selected_output = []
    for result in output:
        selected_spans = [
            {span: score}
            for span, score in result["total_selected_span"].items()
            if score >= sc_threshold
        ]
        selected_output.append({
            "sample_index": result["sample_index"],
            "selected_span": selected_spans
        })

    with open(selected_output_file, "w", encoding='utf-8') as f:
        json.dump(selected_output, f, ensure_ascii=False, indent=2)

    # STEP4: Load ground truth and evaluate (optional)
    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0
    num_samples = len(selected_output)

    if os.path.exists(gt_csv_file):
        print("\n Loading ground truth tokens from CSV...")
        gt_tokens_dict = load_gt_tokens_from_csv(gt_csv_file)

        print(" Evaluating results against ground truth...")
        evaluation_results = evaluate_results(selected_output, gt_tokens_dict, evaluation_output_file)

        # Calculate average metrics
        total_precision = sum(r['precision'] for r in evaluation_results)
        total_recall = sum(r['recall'] for r in evaluation_results)
        total_f1 = sum(r['f1'] for r in evaluation_results)
        num_samples = len(evaluation_results)

        avg_precision = total_precision / num_samples if num_samples > 0 else 0
        avg_recall = total_recall / num_samples if num_samples > 0 else 0
        avg_f1 = total_f1 / num_samples if num_samples > 0 else 0
    else:
        print("\n Ground truth CSV not found, skipping evaluation...")

    # STEP5: Save parameters to text file
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

        f.write("=" * 50 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Average Metrics:\n")
        f.write(f"  - Precision: {avg_precision:.2f}%\n")
        f.write(f"  - Recall: {avg_recall:.2f}%\n")
        f.write(f"  - F1 Score: {avg_f1:.2f}%\n")

    print(f"\n Full results saved to: {output_file}")
    print(f" Selected results (SC >= {sc_threshold}) saved to: {selected_output_file}")
    print(f" Evaluation results saved to: {evaluation_output_file}")
    print(f" Parameters saved to: {params_file}")
    print(f"\n Average Metrics:")
    print(f"   Precision: {avg_precision:.2f}%")
    print(f"   Recall: {avg_recall:.2f}%")
    print(f"   F1 Score: {avg_f1:.2f}%")
    print("Finish")

if __name__ == "__main__":
    main()
