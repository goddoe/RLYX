"""
Simple evaluator without audio processing
"""
import numpy as np
import wandb
from rlyx.rewards import format_reward_func
from rlyx.registries import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register("simple_evaluator")
def evaluate_model(handle, valid_dataloader, rollout_max_tokens, stop_tokens, 
                  num_infer_workers, accelerator, is_wandb_logging, is_tb_logging, 
                  tb_writer, global_i, epoch, **kwargs):
    """
    Simple evaluation focusing on format correctness only
    
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
    batch_result_list = []
    
    for batch in valid_dataloader:
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
    
    # Calculate format rewards
    format_rewards = []
    valid_completions = 0
    
    wb_table = wandb.Table(columns=[
        "global_step", "epoch", "gold_text", "completion", "format_reward"
    ])
    
    for eval_i, (completion, gold_text) in enumerate(zip(completion_list, gold_text_list)):
        format_reward = format_reward_func(completion, format_type="format_type_1")
        format_rewards.append(format_reward)
        
        if format_reward == 1.0:
            valid_completions += 1
        
        print(f"eval_i: {eval_i}, gold_text: {gold_text}, format_reward: {format_reward}")
        
        if is_wandb_logging and eval_i <= 10:
            wb_table.add_data(
                global_i, epoch, gold_text, completion, format_reward
            )
    
    metrics = {
        "format_reward_mean": np.mean(format_rewards),
        "valid_completion_rate": valid_completions / len(completion_list) if completion_list else 0.0
    }
    
    if is_wandb_logging:
        wandb_metrics = {f"valid/{k}": v for k, v in metrics.items()}
        wandb_metrics["valid/eval_simple_table"] = wb_table
        wandb.log(wandb_metrics)
    
    if is_tb_logging:
        for k, v in metrics.items():
            tb_writer.add_scalar(f"valid/{k}", v, global_i)
    
    accelerator.print(
        f"global_step: {global_i}, epoch: {epoch}, {metrics}"
    )