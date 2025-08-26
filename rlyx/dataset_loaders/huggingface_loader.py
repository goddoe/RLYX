"""
Default HuggingFace datasets loader
"""
from datasets import load_dataset
from rlyx.registries import DATASET_LOADER_REGISTRY


@DATASET_LOADER_REGISTRY.register("huggingface_loader")
def load_data(dataset_name_or_path, **kwargs):
    """
    Load dataset from HuggingFace datasets library
    
    Args:
        dataset_name_or_path: HuggingFace dataset name or path
        **kwargs: Additional arguments passed to load_dataset
    
    Returns:
        Dataset dict with 'train' and 'test' splits
    """
    return load_dataset(dataset_name_or_path, **kwargs)