# Selective Span-Level Unlearning for Large Language Models

Official code for:

**Selective Span-Level Unlearning for Large Language Models**

Chaewon Yoon, Dongjun Kim, and Hyun-Je Song

ACL 2026, Short Papers

This repository provides the experiment utilities for selective span-level
unlearning. Instead of treating every token in a forget example as an
unlearning target, the method first identifies important tokens from
model-intrinsic signals and then turns those tokens into coherent span-level
targets with self-consistency.

The cleaned repository focuses on selective target construction and benchmark
evaluation. Full model-training scripts, baseline implementations, temporary
outputs, logs, and cached experiment artifacts are intentionally not included.

## Method Overview

The pipeline has two main stages:

1. **Token-level importance scoring**
   Estimate which forget-set tokens most strongly contribute to information
   targeted for unlearning by contrasting forget and retain gradients with
   influence-function tooling.

2. **Span-level target selection**
   Use high-importance tokens as anchors, then apply a self-consistency
   generation procedure to select stable text spans. The resulting spans are
   converted into token masks for selective unlearning.

The generated masks and scores can be passed to an external training pipeline
that supports token-level forget masks.

Benchmark-specific documentation:

- `TOFU/README.md`
- `MUSE/README.md`

## Installation

Install the top-level requirements:

```bash
pip install -r requirements.txt
```

For isolated benchmark environments:

```bash
# TOFU
cd TOFU
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

# MUSE
cd ../MUSE
conda env create -f environment.yml
conda activate muse
```

Some experiments require gated Hugging Face models. For LLaMA-based
self-consistency scripts, set:

```bash
export HF_TOKEN=your_huggingface_token
```

GPT-based MUSE span generation requires:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

## TOFU

Typical TOFU workflow:

```bash
cd TOFU

python if/fit_factor.py \
  --model_name /path/to/model \
  --model_family llama2-7b \
  --data_path locuslab/TOFU \
  --retain_split retain90 \
  --output_dir ./factors

python if/compute_influence.py \
  --model_name /path/to/model \
  --model_family llama2-7b \
  --data_path locuslab/TOFU \
  --forget_split forget10 \
  --retain_split retain90 \
  --factors_path ./factors \
  --factors_name ekfac \
  --save_dir ./influence_results

python self_conf/sc.py \
  --model_name lluvecwonv/llama3-8b-tofu-ft-5epochs \
  --word_file self_conf/llama3.json \
  --text_file TOFU_data/forget10.json \
  --gt_csv_file self_conf/gt_token_llama3.csv \
  --result_dir self_conf/result \
  --name forget10

python if/build_forget_token_mask.py \
  --model_name /path/to/model \
  --model_family qwen2.5-7b \
  --data_path locuslab/TOFU \
  --forget_split forget10 \
  --gt_spans_path self_conf/result/selected_result_forget10.json \
  --save_dir ./influence_results \
  --save_id forget10
```

Evaluate a checkpoint:

```bash
python evaluate_util.py \
  model_path=/path/to/checkpoint \
  model_family=llama2-7b \
  save_dir=./eval_results \
  split=forget10_perturbed
```

Aggregate TOFU metrics:

```bash
python aggregate_eval_stat.py \
  retain_result=/path/to/retain/eval_results \
  ckpt_result=./eval_results \
  method_name=selective_span_unlearning \
  save_file=./eval_results/aggr_result.json
```

## MUSE

Prepare MUSE data:

```bash
cd MUSE
python load_data.py
```

Run influence scoring:

```bash
bash scripts/run_fit_factor.sh
bash scripts/run_compute_influence.sh
```

Generate GPT-based span masks:

```bash
bash scripts/generate_seul_masks_pipeline.sh news 100
```

Evaluate one or more checkpoints:

```bash
python eval.py \
  --model_dirs /path/to/model \
  --names model_name \
  --corpus news \
  --out_file results.csv
```

Available MUSE metrics include `verbmem_f`, `privleak`, `knowmem_f`, and
`knowmem_r`.

## Notes

- This repository keeps benchmark data utilities, influence scoring,
  self-consistency span selection, mask construction, and evaluation code.
- Training and baseline code has been removed from this cleaned release.
- Generated outputs such as logs, cached factors, masks, checkpoints, and
  evaluation result folders should be regenerated locally and kept out of the
  repository unless explicitly needed for release.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{yoon-etal-2026-selective,
  title = "Selective Span-Level Unlearning for Large Language Models",
  author = "Yoon, Chaewon  and
    Kim, Dongjun  and
    Song, Hyun-Je",
  editor = "Liakata, Maria  and
    Moreira, Viviane P.  and
    Zhang, Jiajun  and
    Jurgens, David",
  booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 2: Short Papers)",
  month = jul,
  year = "2026",
  address = "San Diego, California, United States",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2026.acl-short.35/",
  pages = "423--431",
  ISBN = "979-8-89176-391-3"
}
```

## License

Please check the licenses of the benchmark datasets, base models, and included
third-party code before redistribution or commercial use.
