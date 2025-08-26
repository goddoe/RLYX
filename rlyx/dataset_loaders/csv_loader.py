"""
CSV dataset loader for custom datasets
"""
import os
import pandas as pd
from datasets import Dataset
from rlyx.registries import DATASET_LOADER_REGISTRY


@DATASET_LOADER_REGISTRY.register("csv_loader")
def load_data(dataset_name_or_path, **kwargs):
    """
    Load dataset from CSV files
    
    Expected directory structure:
    dataset_name_or_path/
    ├── train.csv
    └── test.csv
    
    CSV format should have a 'text' column
    
    Args:
        dataset_name_or_path: Path to directory containing train.csv and test.csv
        **kwargs: Additional arguments (unused)
    
    Returns:
        Dataset dict with 'train' and 'test' splits
    """
    train_path = os.path.join(dataset_name_or_path, "train.csv")
    test_path = os.path.join(dataset_name_or_path, "test.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found in {dataset_name_or_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found in {dataset_name_or_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Ensure 'text' column exists
    if 'text' not in train_df.columns:
        raise ValueError("train.csv must have a 'text' column")
    if 'text' not in test_df.columns:
        raise ValueError("test.csv must have a 'text' column")
    
    return {
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    }