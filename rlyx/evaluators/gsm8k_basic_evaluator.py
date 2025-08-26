"""
Basic evaluator for model validation
"""
import io
import numpy as np
import soundfile as sf
import wandb
from rlyx.utils.math_utils import extract_answer, extract_numbers, compare_numbers
from rlyx.rewards.transcribe_reward import transcribe_reward_func
from rlyx.registries import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register("gsm8k_basic_evaluator")
def evaluate_model(handle, valid_dataloader, rollout_max_tokens, stop_tokens, 
                  num_infer_workers, accelerator, is_wandb_logging, is_tb_logging, 
                  tb_writer, global_i, epoch, **kwargs):
    """
    Basic evaluation with gsm8k metrics
    
    Args:
        handle: Ray deployment handle
        valid_dataloader: Validation dataloader
        rollout_max_tokens: Maximum tokens for rollout
        stop_tokens: List of stop tokens for generation
        num_infer_workers: Number of inference workers
        accelerator: Accelerate instance
        is_wandb_logging: Whether to log to wandb
        is_tb_logging: Whether to log to tensorboard
        tb_writer: Tensorboard writer
        global_i: Global step
        epoch: Current epoch
        **kwargs: Additional arguments
    """
    if not accelerator.is_main_process:
        return
    
    completion_list = []
    gold_text_list = []
    pred_list = []
    batch_result_list = []
    
    for batch in valid_dataloader:
        # inference
        gold_text_list.extend(batch["gold_text"])
        
        sample_params = {
            "temperature": 0.01,
            "max_tokens": rollout_max_tokens,
            "n": 1,
            "include_stop_str_in_output": True,
            "skip_special_tokens": False,
            "stop": stop_tokens
        }
        future_policy_rollout_batch = handle.generate_text.remote(
            batch["user_input_text"],
            sample_params=sample_params
        )
        batch_result_list.append(future_policy_rollout_batch)
        
        if len(batch_result_list) <= num_infer_workers:
            continue
        
        for future_policy_rollout_batch in batch_result_list:
            policy_rollout_batch = future_policy_rollout_batch.result()
            for preds in policy_rollout_batch["text"]:
                completion_list.append(preds[0])
        
        batch_result_list = []
    
    if batch_result_list:
        for future_policy_rollout_batch in batch_result_list:
            policy_rollout_batch = future_policy_rollout_batch.result()
            for preds in policy_rollout_batch["text"]:
                completion_list.append(preds[0])
    

    for pred_raw in completion_list:
        # extract answer from <answer> </answer> tag
        answer_block = extract_answer(pred_raw)
        answer_number = extract_numbers(answer_block)
        pred = answer_number[0] if answer_number else None
        pred_list.append(pred)

    n_exact_correct = 0
    n_within_tolerance_correct = 0
    n_total = len(pred_list)
    
    for pred, gold in zip(pred_list, gold_text_list):
        result = compare_numbers(pred, gold)
        if result["exact_match"]:
            n_exact_correct += 1
        if result["within_tolerance"]:
            n_within_tolerance_correct += 1

    # Calc Accuracy
    exact_accuracy = n_exact_correct / n_total
    within_tolerance_accuracy = n_within_tolerance_correct / n_total

    metrics = {
        "gsm8k_accuracy_exact": exact_accuracy,
        "gsm8k_accuracy_within_tolerance": within_tolerance_accuracy,
    }

    if is_wandb_logging:
        wandb_metrics = {f"valid/{k}": v for k, v in metrics.items()}
        wandb.log(wandb_metrics)

    if is_tb_logging:
        for k, v in metrics.items():
            tb_writer.add_scalar(f"valid/{k}", v, global_i)
    
    accelerator.print(
        f"global_step: {global_i}, epoch: {epoch}, {metrics}"
    )
