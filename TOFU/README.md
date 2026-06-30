# TOFU Experiments

This directory contains the TOFU-side implementation used for selective
span-level unlearning experiments.

The code is organized around the paper pipeline:

1. Fit influence factors on retain examples.
2. Compute differential token-level importance scores by contrasting forget
   and retain behavior.
3. Convert important tokens into stable span-level targets with
   self-consistency.
4. Build token masks from the selected spans.
5. Evaluate forget quality and retain utility on TOFU.

Training scripts and baseline implementations are intentionally not included
here. This folder keeps the experiment utilities needed to reproduce the
selective target construction and evaluation workflow.

## Structure

```text
TOFU/
|-- TOFU_data/              # TOFU forget/retain/evaluation splits
|-- config/                 # Minimal model and evaluation configs
|-- data/                   # Auxiliary evaluation data
|-- if/                     # Influence-factor, influence-score, and mask code
|-- self_conf/              # Self-consistency span selection
|-- aggregate_eval_stat.py  # Aggregate TOFU evaluation metrics
|-- data_module.py          # TOFU dataset and collator utilities
|-- evaluate_util.py        # TOFU evaluation loop
|-- utils.py                # Shared model/evaluation helpers
`-- requirements.txt
```

## Setup

```bash
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

Some models require Hugging Face access. Set `HF_TOKEN` before running LLaMA
3 based self-consistency scripts:

```bash
export HF_TOKEN=your_huggingface_token
```

## Influence Scoring

Fit factors on the retain split:

```bash
python if/fit_factor.py \
  --model_name /path/to/model \
  --model_family llama2-7b \
  --data_path locuslab/TOFU \
  --retain_split retain90 \
  --output_dir ./factors
```

Compute differential influence scores for the forget split:

```bash
python if/compute_influence.py \
  --model_name /path/to/model \
  --model_family llama2-7b \
  --data_path locuslab/TOFU \
  --forget_split forget10 \
  --retain_split retain90 \
  --factors_path ./factors \
  --factors_name ekfac \
  --save_dir ./influence_results
```

## Span Selection

Run the self-consistency stage to select stable span-level targets:

```bash
python self_conf/sc.py \
  --model_name lluvecwonv/llama3-8b-tofu-ft-5epochs \
  --word_file self_conf/llama3.json \
  --text_file TOFU_data/forget10.json \
  --gt_csv_file self_conf/gt_token_llama3.csv \
  --result_dir self_conf/result \
  --name forget10
```

The selected spans can then be converted into token masks:

```bash
python if/build_forget_token_mask.py \
  --model_name /path/to/model \
  --model_family qwen2.5-7b \
  --data_path locuslab/TOFU \
  --forget_split forget10 \
  --gt_spans_path self_conf/result/selected_result_forget10.json \
  --save_dir ./influence_results \
  --save_id forget10
```

## Evaluation

Evaluate a checkpoint on the standard TOFU tasks:

```bash
python evaluate_util.py \
  model_path=/path/to/checkpoint \
  model_family=llama2-7b \
  save_dir=./eval_results \
  split=forget10_perturbed
```

Aggregate the evaluation outputs:

```bash
python aggregate_eval_stat.py \
  retain_result=/path/to/retain/eval_results \
  ckpt_result=./eval_results \
  method_name=selective_span_unlearning \
  save_file=./eval_results/aggr_result.json
```
