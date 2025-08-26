"""
Plain text file dataset loader
"""
import os
from datasets import Dataset
from rlyx.registries import DATASET_LOADER_REGISTRY


@DATASET_LOADER_REGISTRY.register("text_file_loader")
def load_data(dataset_name_or_path, **kwargs):
    """
    Load dataset from plain text files
    
    Expected directory structure:
    dataset_name_or_path/
    ├── train.txt
    └── test.txt
    
    Each line in the text file becomes one data sample
    
    Args:
        dataset_name_or_path: Path to directory containing train.txt and test.txt
        **kwargs: Additional arguments (unused)
    
    Returns:
        Dataset dict with 'train' and 'test' splits
    """
    train_path = os.path.join(dataset_name_or_path, "train.txt")
    test_path = os.path.join(dataset_name_or_path, "test.txt")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.txt not found in {dataset_name_or_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.txt not found in {dataset_name_or_path}")
    
    def read_text_file(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data.append({"text": line})
        return data
    
    train_data = read_text_file(train_path)
    test_data = read_text_file(test_path)
    
    if not train_data:
        raise ValueError("train.txt is empty")
    if not test_data:
        raise ValueError("test.txt is empty")
    
    return {
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    }