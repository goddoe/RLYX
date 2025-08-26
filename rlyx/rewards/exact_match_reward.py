import re
from rlyx.utils.math_utils import extract_answer
from rlyx.registries import REWARD_REGISTRY
 

@REWARD_REGISTRY.register("exact_match_reward")
def exact_match_reward_func(completion, gold_text, **kwargs):
    answer_block = extract_answer(completion)

    if isinstance(answer_block, str):
        if answer_block.strip() == gold_text.strip():
            return 1.0
    return 0.

