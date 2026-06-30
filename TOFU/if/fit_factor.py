import argparse
import logging
import os
import sys
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs

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

from typing import Optional
from tqdm import tqdm
from torch import nn

# Import the existing TOFU utilities from TOFU/utils.py.
from data_module import TextDatasetQA, custom_data_collator
from utils import get_model_identifiers_from_yaml

# Import local task definition
# Handle both directory names (utils.py or utils) for compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_py_dir = os.path.join(current_dir, "utils.py")
utils_dir = os.path.join(current_dir, "utils")

# Check which directory exists and import accordingly
if os.path.exists(utils_py_dir) and os.path.isdir(utils_py_dir):
    # utils.py directory exists in the server environment.
    task_module_path = os.path.join(utils_py_dir, "task.py")
    if os.path.exists(task_module_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("task", task_module_path)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        LanguageModelingTask = task_module.LanguageModelingTask
    else:
        raise ImportError(f"Cannot find task.py in {utils_py_dir}")
elif os.path.exists(utils_dir) and os.path.isdir(utils_dir):
    # utils directory exists in the local environment.
    task_module_path = os.path.join(utils_dir, "task.py")
    if os.path.exists(task_module_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("task", task_module_path)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        LanguageModelingTask = task_module.LanguageModelingTask
    else:
        raise ImportError(f"Cannot find task.py in {utils_dir}")
else:
    raise ImportError("Cannot find utils or utils.py directory")

# Configure CUDA memory
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True",
)


def kronfluence_data_collator(batch):
    """Custom data collator for Kronfluence."""
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            input_ids.append(item[0])
            labels.append(item[1])
            attention_mask.append(item[2])
        else:
            raise ValueError(f"Expected tuple/list with at least 3 items, got {type(item)}")

    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask)
    }


def load_model_and_tokenizer(model_name, model_family):
    """Load model and tokenizer."""
    model_configs = get_model_identifiers_from_yaml(model_family)

    # Try to load tokenizer from model_name first (fine-tuned model),
    # fall back to hf_key if not available
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_configs['hf_key'])
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

    # Get max_length from model_configs
    max_length = model_configs.get('max_length', 256)  # default: 256

    return model, tokenizer, num_layers, max_length


def parse_factor_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fit Kronfluence factors with embed_out diagonal approximation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_family", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None,
                        help="Max sequence length (default: read from model_config.yaml)")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="locuslab/TOFU")
    parser.add_argument("--split", type=str, default="full")
    parser.add_argument("--question_key", type=str, default="question")
    parser.add_argument("--forget_split", type=str, default="forget10")
    parser.add_argument("--retain_split", type=str, default="retain90")
    parser.add_argument("--answer_only", action="store_true", default=False,
                        help="Use only answer (no question) for influence computation")
    
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
    """Main function."""
    args = parse_factor_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Starting Kronfluence factor fitting with embed_out diagonal approximation")

    # Load model and tokenizer
    model, tokenizer, num_layers, max_length_from_config = load_model_and_tokenizer(args.model_name, args.model_family)
    logging.info(f"Loaded model: {args.model_name} with {num_layers} layers")

    # Use max_length from args if provided, otherwise use from model_config
    max_length = args.max_length if args.max_length is not None else max_length_from_config
    logging.info(f"Using max_length: {max_length}")

    # Load dataset
    train_dataset = TextDatasetQA(
        data_path=args.data_path,
        tokenizer=tokenizer,
        model_family=args.model_family,
        max_length=max_length,
        split=args.split,
        question_key=args.question_key,
        answer_only=args.answer_only  # Use answer only.
    )
    if args.answer_only:
        logging.info(f"Using ANSWER ONLY mode (Question removed)")
    logging.info(f"Loaded dataset with {len(train_dataset)} samples")

    # Setup task with model config
    task_config = {
        'model': {
            'family': args.model_family,
            'num_layers': num_layers
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
        collate_fn=kronfluence_data_collator,
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
