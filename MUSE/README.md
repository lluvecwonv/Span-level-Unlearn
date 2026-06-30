# MUSE Experiments

This folder keeps the MUSE-side assets needed for the selective span-level
unlearning experiments:

- data loading for the News and Books corpora
- influence-factor fitting and influence scoring
- span-to-token mask construction
- MUSE benchmark evaluation metrics
- self-consistency span selection utilities

Model training and baseline algorithm implementations are intentionally not
included here. The generated masks and scores can be passed to an external
training pipeline.

## Setup

```bash
conda env create -f environment.yml
conda activate muse
```

## Data

Download the MUSE data and target model metadata:

```bash
python load_data.py
```

The main corpus files are expected under:

```text
data/news/raw/
data/books/raw/
```

## Influence Scoring

Fit Kronfluence factors:

```bash
bash scripts/run_fit_factor.sh
```

Compute forget-vs-retain influence scores:

```bash
bash scripts/run_compute_influence.sh
```

The implementation lives in:

```text
if/fit_factor.py
if/compute_influence.py
if/utils/
```

## Mask Construction

Generate GPT-based span masks:

```bash
bash scripts/generate_seul_masks_pipeline.sh news 100
```

The resulting `forget_mask.pt` files are the token-level masks consumed by the
unlearning/training pipeline.

## Evaluation

Evaluate one or more model checkpoints with the MUSE benchmark metrics:

```bash
python eval.py \
  --model_dirs /path/to/model \
  --names model_name \
  --corpus news \
  --out_file results.csv
```

Available metrics include `verbmem_f`, `privleak`, `knowmem_f`, and
`knowmem_r`.
