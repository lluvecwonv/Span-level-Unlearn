"""
Script to compute differential influence scores for MUSE forget vs retain datasets.
Formula: S_j = (g_forget_query_avg - g_retain_query_avg)^T * H^{-1} * g_forget_j
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent MUSE directory to path for imports
MUSE_DIR = Path(__file__).parent.parent.resolve()
IF_DIR = Path(__file__).parent.resolve()

# Add local Kronfluence to path (HIGHEST PRIORITY)
KRONFLUENCE_DIR = MUSE_DIR.parent / "kronfluence" / "src"
sys.path.insert(0, str(KRONFLUENCE_DIR))  # FIRST: local kronfluence
sys.path.insert(1, str(MUSE_DIR))         # Second: for MUSE modules

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import ScoreArguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

# Debug: Check which Kronfluence is being imported
import kronfluence
print("="*80)
print(f"DEBUG: Kronfluence location: {kronfluence.__file__}")
print(f"DEBUG: sys.path[0]: {sys.path[0]}")
print(f"DEBUG: Expected path: {KRONFLUENCE_DIR}")
print("="*80)

# Import MUSE-specific modules from if/utils
sys.path.insert(0, str(MUSE_DIR / "if" / "utils"))
from dataset import DefaultDataset
from task import LanguageModelingTask


def parse_args():
    parser = argparse.ArgumentParser(description="Compute differential influence scores for MUSE.")

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Path or name of tokenizer (e.g., NousResearch/Llama-2-7b-chat-hf).",
    )

    # Dataset configuration
    parser.add_argument(
        "--forget_file",
        type=str,
        required=True,
        help="Path to forget dataset file (.txt or .json).",
    )
    parser.add_argument(
        "--retain_file",
        type=str,
        required=True,
        help="Path to retain dataset file (.txt or .json).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--add_bos_token",
        action="store_true",
        default=True,
        help="Add BOS token to sequences.",
    )

    # Factor configuration
    parser.add_argument(
        "--factors_name",
        type=str,
        default="ekfac",
        help="Name of factors directory.",
    )
    parser.add_argument(
        "--factors_path",
        type=str,
        required=True,
        help="Path to the factors directory.",
    )
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors (ekfac, kfac, diagonal).",
    )
    parser.add_argument(
        "--analysis_name",
        type=str,
        default="if_results",
        help="Subdirectory inside factors_path where factors are stored.",
    )

    # Computation configuration
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for computing train gradients.",
    )
    parser.add_argument(
        "--use_half_precision",
        action="store_true",
        default=False,
        help="Whether to use half precision for computing scores.",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        default=False,
        help="Whether to use torch compile.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )

    # Output configuration
    parser.add_argument(
        "--save_id",
        type=str,
        default=None,
        help="ID to append to the output names.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./influence_results",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, tokenizer_name: str):
    """Load model and tokenizer from checkpoint."""
    logging.info(f"Loading model from {model_name}")
    logging.info(f"Loading tokenizer from {tokenizer_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    # Debug: Check model device placement
    logging.info("="*60)
    logging.info("Model Device Placement:")
    for name, param in list(model.named_parameters())[:5]:  # First 5 params
        logging.info(f"  {name}: {param.device}")
    logging.info(f"  ... (showing first 5 parameters)")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    logging.info("="*60)

    # Get number of layers for task configuration
    num_layers = model.config.num_hidden_layers

    return model, tokenizer, num_layers


def load_datasets(args, tokenizer):
    """Load forget and retain datasets using MUSE DefaultDataset."""
    logging.info(f"Loading forget dataset: {args.forget_file}")
    forget_dataset = DefaultDataset(
        file_path=args.forget_file,
        tokenizer=tokenizer,
        max_len=args.max_length,
        add_bos_token=args.add_bos_token,
    )

    logging.info(f"Loading retain dataset: {args.retain_file}")
    retain_dataset = DefaultDataset(
        file_path=args.retain_file,
        tokenizer=tokenizer,
        max_len=args.max_length,
        add_bos_token=args.add_bos_token,
    )

    logging.info(f"Forget dataset size: {len(forget_dataset)}")
    logging.info(f"Retain dataset size: {len(retain_dataset)}")

    return forget_dataset, retain_dataset


def configure_score_args(args) -> tuple:
    """Configure score arguments based on command line arguments."""
    score_args = ScoreArguments()
    scores_name = f"differential_{args.factor_strategy}"

    if args.use_half_precision:
        score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
        scores_name += "_half"

    if args.use_compile:
        scores_name += "_compile"

    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    if rank is not None:
        score_args.query_gradient_low_rank = rank
        score_args.query_gradient_accumulation_steps = 10
        scores_name += f"_qlr{rank}"

    if args.save_id:
        scores_name += f"_{args.save_id}"

    # Configure for differential influence
    score_args.compute_per_token_scores = True
    score_args.aggregate_query_gradients = True

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        scores_name = os.path.join(args.save_dir, scores_name)

    return score_args, scores_name


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info("="*60)
    logging.info("MUSE Differential Influence Score Computation")
    logging.info("Formula: S_j = (g_forget_avg - g_retain_avg)^T * H^{-1} * g_forget_j")
    logging.info("="*60)

    # Load model and tokenizer
    model, tokenizer, num_layers = load_model_and_tokenizer(args.model_name, args.tokenizer_name)

    # Load datasets
    forget_dataset, retain_dataset = load_datasets(args, tokenizer)

    # Define task with model config
    task_config = {
        'model': {
            'family': 'llama2-7b',  # Auto-detection is possible.
            'num_layers': num_layers
        }
    }
    task = LanguageModelingTask(config=task_config)

    # Setup accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    logging.info(f"Using device: {accelerator.device}")
    logging.info(f"Number of GPUs: {torch.cuda.device_count()}")

    # Prepare model with Kronfluence wrapper
    model = prepare_model(model, task)

    if args.use_compile:
        logging.info("Compiling model with torch.compile")
        model = torch.compile(model)

    # Initialize analyzer
    try:
        logging.info("Initializing Analyzer")
        analyzer = Analyzer(
            analysis_name=args.analysis_name,
            model=model,
            task=task,
            profile=args.profile,
            output_dir=args.factors_path,
        )
        logging.info("Analyzer initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Analyzer: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

    # Configure DataLoader with MUSE collate function
    try:
        logging.info("Setting DataLoader kwargs")
        dataloader_kwargs = DataLoaderKwargs(collate_fn=forget_dataset.get_collate_fn())
        analyzer.set_dataloader_kwargs(dataloader_kwargs)
        logging.info("DataLoader kwargs set")
    except Exception as e:
        logging.error(f"Failed to set DataLoader kwargs: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

    # Configure score arguments
    try:
        logging.info("Configuring score arguments")
        score_args, scores_name = configure_score_args(args)
        logging.info(f"Score arguments configured: {scores_name}")
    except Exception as e:
        logging.error(f"Failed to configure score arguments: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

    logging.info("=" * 60)
    logging.info("Starting pairwise score computation...")
    logging.info(f"scores_name: {scores_name}")
    logging.info(f"factors_name: {args.factors_name}")
    logging.info(f"analysis_name: {args.analysis_name}")
    logging.info(f"Query batch size: {args.query_batch_size}")
    logging.info(f"Train batch size: {args.train_batch_size}")
    logging.info(f"Forget dataset size: {len(forget_dataset)}")
    logging.info(f"Retain dataset size: {len(retain_dataset)}")
    logging.info("=" * 60)

    # Compute pairwise scores
    try:
        analyzer.compute_pairwise_scores(
            scores_name=scores_name,
            score_args=score_args,
            factors_name=args.factors_name,
            forget_dataset=forget_dataset,       # Positive query
            retain_dataset=retain_dataset,       # Negative query
            train_dataset=forget_dataset,        # Target samples
            per_device_query_batch_size=args.query_batch_size,
            per_device_train_batch_size=args.train_batch_size,
            overwrite_output_dir=False,
        )
        logging.info("compute_pairwise_scores completed!")
    except Exception as e:
        logging.error(f"compute_pairwise_scores failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise
    logging.info("=" * 60)

    # Load and log results
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    logging.info("="*60)
    logging.info(f"Differential influence scores computed successfully!")
    logging.info(f"Scores shape: {scores.shape}")
    logging.info(f"Scores saved to: {scores_name}")
    logging.info("="*60)


if __name__ == "__main__":
    main()
