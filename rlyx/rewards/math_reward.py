import re
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig

from rlyx.utils.math_utils import extract_answer, extract_numbers, compare_numbers
from rlyx.registries import REWARD_REGISTRY
 

@REWARD_REGISTRY.register("math_reward")
def math_reward_func(completion, gold_text, **kwargs):
    answer_block = extract_answer(completion)
    answer_number = extract_numbers(answer_block)

    if answer_number:
        result = compare_numbers(answer_number[0], gold_text, tolerance=1e-5)
        if result["within_tolerance"]:
            return 1.0

    # Reference : https://github.com/huggingface/open-r1/blob/1fc8d425a995ddf8dbc6f8ef239d8161acdb7fc1/src/open_r1/grpo.py#L53-L82C1
    gold_parsed = parse(gold_text, extraction_mode="first_match",
                        extraction_config=[LatexExtractionConfig()])

    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            completion,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        reward = float(verify(answer_parsed, gold_parsed))
        return reward

    reward = 0.0
    return reward
