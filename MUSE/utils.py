import json
import pandas as pd
import os
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


# def read_json(fpath: str) -> Dict | List:
def read_json(fpath: str):
    with open(fpath, 'r') as f:
        return json.load(f)


def read_text(fpath: str) -> str:
    with open(fpath, 'r') as f:
        return f.read()


def write_json(obj: Dict | List, fpath: str):
# def write_json(obj, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return json.dump(obj, f)


def write_text(obj: str, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return f.write(obj)


def write_csv(obj, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    pd.DataFrame(obj).to_csv(fpath, index=False)


def load_model(model_dir: str, **kwargs):
    return AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)


def load_tokenizer(tokenizer_dir: str, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, **kwargs)
    # Llama models don't have pad_token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
    

def get_model_identifiers_from_yaml(model_family):
    """
    Load model configuration from model_config.yaml

    Args:
        model_family: Model family name (e.g., 'llama2-7b', 'phi')

    Returns:
        dict: Model configuration including hf_key, tags, etc.
    """
    # Get absolute path to MUSE directory (where utils.py is located)
    MUSE_DIR = Path(__file__).parent.resolve()
    config_path = MUSE_DIR / "config" / "model_config.yaml"

    # If config doesn't exist in MUSE, try TOFU
    if not config_path.exists():
        config_path = MUSE_DIR.parent / "TOFU" / "config" / "model_config.yaml"

    model_configs = {}
    with open(config_path, "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]


def add_dataset_index(dataset):
    """
    Add index column to dataset

    Args:
        dataset: HuggingFace dataset

    Returns:
        dataset with index column added
    """
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset
