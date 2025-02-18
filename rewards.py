import re
import numpy as np
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from utils import compare_numbers, extract_answer, extract_numbers

def format_reward_func(completion, end_of_turn_token="<im_end>", **kwargs):
    def count_substring_and_calc_reward(substring, completion):
        count = completion.count(substring)
        if count == 0:
            return 0.
        elif count  == 1:
            return 1.
        return max((10 - count) * 0.1, -1.)

    reward = 0.

    keywords = ["<think>", "</think>", "<answer>", "</answer>", end_of_turn_token]
    for keyword in keywords:
        reward += count_substring_and_calc_reward(keyword, completion)

    # if reward == 0.:
    #     return -1.

    # for keyword in keywords:
    #     if completion.count(keyword) != 1:
    #         return 0.

    if completion.startswith("<think>"):
        reward += 1.

    if completion.endswith(end_of_turn_token):
        reward += 1.

    pattern = r"^<think>(.*?)</think>\n<answer>(.*?)</answer>" + end_of_turn_token + r"$"
    if re.match(pattern, completion, re.DOTALL):
        reward += 3.0

    # possible max value is 10
    scale = 1./10
    # scale = 1./5
    reward = reward * scale
    return reward

 
def math_reward_func(completion, solution, **kwargs):
    answer_block = extract_answer(completion)
    answer_number = extract_numbers(answer_block)

    if answer_number:
        result = compare_numbers(answer_number[0], solution, tolerance=1e-5)
        if result["within_tolerance"]:
            return 1.0

    # Reference : https://github.com/huggingface/open-r1/blob/1fc8d425a995ddf8dbc6f8ef239d8161acdb7fc1/src/open_r1/grpo.py#L53-L82C1
    gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])

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
