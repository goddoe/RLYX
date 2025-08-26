"""
All registries for the RLYX framework
"""
from rlyx.utils.registry import Registry

# Create all global registries with lazy loading support
CHAT_TEMPLATE_REGISTRY = Registry("chat_templates", "rlyx.chat_templates")
DATASET_LOADER_REGISTRY = Registry("dataset_loaders", "rlyx.dataset_loaders")
EVALUATOR_REGISTRY = Registry("evaluators", "rlyx.evaluators")
REWARD_REGISTRY = Registry("rewards", "rlyx.rewards")
TOKENIZER_REGISTRY = Registry("tokenizers", "rlyx.tokenizers")

__all__ = [
    "CHAT_TEMPLATE_REGISTRY",
    "DATASET_LOADER_REGISTRY",
    "EVALUATOR_REGISTRY",
    "REWARD_REGISTRY",
    "TOKENIZER_REGISTRY",
]