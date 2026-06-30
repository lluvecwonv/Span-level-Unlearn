import argparse
import logging
import os
import sys
import torch
import time
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from typing import Optional, List, Tuple
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Add grandparent directory for kronfluence using a relative path.
grandparent_dir = os.path.dirname(parent_dir)
tnpo_dir = os.path.dirname(grandparent_dir)  # tnpo directory
kronfluence_path = os.path.join(tnpo_dir, "kronfluence")
sys.path.insert(0, kronfluence_path)

# Import kronfluence
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

# Import MUSE dataset utilities from utils
if_utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils')
sys.path.insert(0, if_utils_dir)
from dataset import DefaultDataset, ForgetRetainDataset, muse_data_collator
from task import LanguageModelingTask

# Configure CUDA memory
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True",
)


def load_model_and_tokenizer(model_name, tokenizer_name=None):
    """Load model and tokenizer (MUSE style - no model_family dependency)."""
    # Use model_name for tokenizer if not specified
    if tokenizer_name is None:
        tokenizer_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Get num_layers from model config
    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'n_layer'):
        num_layers = model.config.n_layer
    else:
        raise ValueError("Cannot determine number of layers from model config")

    # Auto-detect model family from model type
    model_type = model.config.model_type.lower()
    if 'llama' in model_type:
        family = 'llama2-7b'
    elif 'pythia' in model_type or 'gpt_neox' in model_type:
        family = 'pythia'
    elif 'phi' in model_type:
        family = 'phi'
    else:
        # Default to llama structure for most modern models
        logging.warning(f"Unknown model type '{model_type}', defaulting to llama2-7b structure")
        family = 'llama2-7b'

    return model, tokenizer, num_layers, family


def parse_factor_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fit Kronfluence factors (MUSE style)")

    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Tokenizer name or path (default: same as model_name)")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max sequence length (default: 4096)")

    # Data arguments (MUSE style)
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to data file (.json or .txt)")
    parser.add_argument("--add_bos_token", action="store_true", default=True,
                        help="Add BOS token to sequences")
    
    # Factor computation
    parser.add_argument("--factor_strategy", type=str, default="ekfac", choices=["ekfac", "kfac", "diagonal"])
    parser.add_argument("--use_half_precision", action="store_true", default=False)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./kronfluence_factors")
    
    # Partition arguments (for backward compatibility)
    parser.add_argument("--covariance_module_partitions", type=int, default=4)
    parser.add_argument("--lambda_module_partitions", type=int, default=4)
    parser.add_argument("--covariance_data_partitions", type=int, default=4)
    parser.add_argument("--lambda_data_partitions", type=int, default=4)
    
    return parser.parse_args()


def main():
    """Main function (MUSE style)."""
    args = parse_factor_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Starting Kronfluence factor fitting (MUSE style)")

    # Load model and tokenizer
    model, tokenizer, num_layers, family = load_model_and_tokenizer(args.model_name, args.tokenizer_name)
    logging.info(f"Loaded model: {args.model_name} with {num_layers} layers")
    logging.info(f"Auto-detected model family: {family}")
    logging.info(f"Using max_length: {args.max_length}")

    # Load MUSE dataset
    train_dataset = DefaultDataset(
        file_path=args.data_file,
        tokenizer=tokenizer,
        max_len=args.max_length,
        add_bos_token=args.add_bos_token
    )
    logging.info(f"Loaded MUSE dataset with {len(train_dataset)} samples from {args.data_file}")

    # Setup task with model config (auto-detected family)
    task_config = {
        'model': {
            'num_layers': num_layers,
            'family': family
        }
    }
    task = LanguageModelingTask(config=task_config)
    model = prepare_model(model, task)

    # Setup accelerator
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[init_kwargs, ddp_kwargs])
    model = accelerator.prepare_model(model)

    os.makedirs(args.output_dir, exist_ok=True)

    # Create analyzer
    analyzer = Analyzer(
        analysis_name="if_results",
        model=model,
        task=task,
        output_dir=args.output_dir,
    )

    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4,
        collate_fn=muse_data_collator,
        pin_memory=True
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Configure factors
    factor_args = FactorArguments(strategy=args.factor_strategy)
    factors_name = args.factor_strategy

    if args.use_half_precision:
        factor_args = all_low_precision_factor_arguments(strategy=args.factor_strategy, dtype=torch.bfloat16)
        factors_name += "_half"

    # Set partition parameters
    factor_args.covariance_module_partitions = args.covariance_module_partitions
    factor_args.lambda_module_partitions = args.lambda_module_partitions
    factor_args.covariance_data_partitions = args.covariance_data_partitions
    factor_args.lambda_data_partitions = args.lambda_data_partitions

    # No limit on examples
    factor_args.covariance_max_examples = None
    factor_args.lambda_max_examples = None

    # Fit all factors using standard Kronfluence
    logging.info("Fitting all factors (covariance, eigendecomposition, lambda)...")
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.train_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )

    logging.info("Factor fitting completed!")


if __name__ == "__main__":
    main()
