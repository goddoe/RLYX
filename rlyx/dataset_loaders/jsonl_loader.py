"""
JSONL (JSON Lines) dataset loader for custom datasets
"""
import os
import json
from datasets import Dataset
from rlyx.registries import DATASET_LOADER_REGISTRY


@DATASET_LOADER_REGISTRY.register("jsonl_loader")
def load_data(dataset_name_or_path, **kwargs):
    """
    Load dataset from JSONL files
    
    Expected directory structure:
    dataset_name_or_path/
    ├── train.jsonl
    └── test.jsonl
    
    Each line in JSONL should be a JSON object with a 'text' field
    
    Args:
        dataset_name_or_path: Path to directory containing train.jsonl and test.jsonl
        **kwargs: Additional arguments (unused)
    
    Returns:
        Dataset dict with 'train' and 'test' splits
    """
    train_path = os.path.join(dataset_name_or_path, "train.jsonl")
    test_path = os.path.join(dataset_name_or_path, "test.jsonl")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.jsonl not found in {dataset_name_or_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.jsonl not found in {dataset_name_or_path}")
    
    def read_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        return data
    
    train_data = read_jsonl(train_path)
    test_data = read_jsonl(test_path)
    
    # Validate data
    if not train_data:
        raise ValueError("train.jsonl is empty")
    if not test_data:
        raise ValueError("test.jsonl is empty")
    
    # Check for 'text' field
    if 'text' not in train_data[0]:
        raise ValueError("Each line in train.jsonl must have a 'text' field")
    if 'text' not in test_data[0]:
        raise ValueError("Each line in test.jsonl must have a 'text' field")
    
    return {
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    }