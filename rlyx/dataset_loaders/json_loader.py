"""
JSON dataset loader for custom datasets
"""
import json
from datasets import Dataset
from rlyx.registries import DATASET_LOADER_REGISTRY


@DATASET_LOADER_REGISTRY.register("json_loader")
def load_data(dataset_name_or_path, **kwargs):
    """
    Load dataset from JSON file
    
    Expected JSON format:
    {
        "train": [
            {"text": "example text 1"},
            {"text": "example text 2"},
            ...
        ],
        "test": [
            {"text": "test text 1"},
            ...
        ]
    }
    
    Args:
        dataset_name_or_path: Path to JSON file
        **kwargs: Additional arguments (unused)
    
    Returns:
        Dataset dict with 'train' and 'test' splits
    """
    with open(dataset_name_or_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure we have train and test splits
    if "train" not in data:
        raise ValueError("JSON file must contain 'train' key")
    if "test" not in data:
        raise ValueError("JSON file must contain 'test' key")
    
    return {
        "train": Dataset.from_list(data["train"]),
        "test": Dataset.from_list(data["test"])
    }