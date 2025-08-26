import re
from rlyx.registries import REWARD_REGISTRY


@REWARD_REGISTRY.register("default_format_reward")
def format_reward_func(completion, end_of_turn_token="<|im_end|>", **kwargs):
    reward = 0.

    # def count_substring_and_calc_reward(substring, completion):
    #     count = completion.count(substring)
    #     if count == 0:
    #         return 0.
    #     elif count  == 1:
    #         return 1.
    #     return max((10 - count) * 0.1, -1.)

    # keywords = ["<think>", "</think>", "<answer>", "</answer>", end_of_turn_token]
    # for keyword in keywords:
    #     reward += count_substring_and_calc_reward(keyword, completion)

    if completion.startswith("<think>"):
        reward += 1.

    if completion.endswith(end_of_turn_token):
        reward += 1.

    pattern = r"^<think>(.*?)</think>\n<answer>(.*?)</answer>" + end_of_turn_token + r"$"
    if re.match(pattern, completion, re.DOTALL):
        reward += 3.0

    # possible max value is 10
    # scale = 1./10
    scale = 1./5
    reward = reward * scale
    return reward

