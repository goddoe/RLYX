"""
Detailed evaluator with comprehensive metrics and visualizations
"""
import io
import numpy as np
import soundfile as sf
import wandb
from rlyx.utils.transcribe_utils import batch_cvt_audio_and_transcribe_completion
from rlyx.rewards import transcribe_reward_func, format_reward_func
from rlyx.registries import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register("detailed_evaluator")
def evaluate_model(handle, valid_dataloader, rollout_max_tokens, stop_tokens, 
                  num_infer_workers, accelerator, is_wandb_logging, is_tb_logging, 
                  tb_writer, global_i, epoch, **kwargs):
    """
    Detailed evaluation with comprehensive metrics
    
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
    
    # Collect completions
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
    
    # Get audio transcriptions
    pred_text_and_audio_bin_list = batch_cvt_audio_and_transcribe_completion(
        completion_list,
        max_workers=8
    )
    
    # Initialize metric aggregators
    report_agg = {}
    format_rewards = []
    completion_lengths = []
    
    # Create detailed wandb table
    wb_table = wandb.Table(columns=[
        "global_step", "epoch", "gold_text", "pred_text_transcribed",
        'CER', 'WER', "tts_reward", "format_reward", "completion_length",
        "completion", "audio"
    ])
    
    # Process each sample
    for eval_i, ((pred_text, audio_bin), gold_text, completion) in enumerate(
        zip(pred_text_and_audio_bin_list, gold_text_list, completion_list)
    ):
        # Transcription metrics
        report = transcribe_reward_func(
            pred_text=pred_text,
            gold_text=gold_text,
            rich=True
        )
        
        # Format reward
        format_reward = format_reward_func(completion, format_type="format_type_1")
        format_rewards.append(format_reward)
        
        # Completion length
        completion_length = len(completion)
        completion_lengths.append(completion_length)
        
        # Aggregate metrics
        for k, v in report.items():
            if k not in report_agg:
                report_agg[k] = []
            report_agg[k].append(v)
        
        print(f"eval_i: {eval_i}, gold_text: {gold_text}, pred_text: {pred_text}")
        print(f"  - CER: {report['cer_value']:.3f}, WER: {report['wer_value']:.3f}")
        print(f"  - Format reward: {format_reward}, Completion length: {completion_length}")
        
        # Log samples to wandb
        if is_wandb_logging and eval_i <= 15:  # Log more samples
            audio = None
            if audio_bin:
                audio, sr = sf.read(io.BytesIO(audio_bin))
                audio = wandb.Audio(audio, sr, caption=f"gold_text: {gold_text}\npred_text: {pred_text}")
            
            wb_table.add_data(
                global_i, epoch, gold_text, pred_text,
                report['cer_value'], report['wer_value'], report['tts_reward'],
                format_reward, completion_length,
                completion, audio
            )
    
    # Calculate aggregate metrics
    metrics = {k: np.mean(v) for k, v in report_agg.items() if len(v) > 0}
    
    # Add additional metrics
    metrics.update({
        "format_reward_mean": np.mean(format_rewards),
        "completion_length_mean": np.mean(completion_lengths),
        "completion_length_std": np.std(completion_lengths),
        "valid_format_rate": sum(1 for r in format_rewards if r == 1.0) / len(format_rewards) if format_rewards else 0.0,
        "total_eval_samples": len(completion_list)
    })
    
    # Log to wandb
    if is_wandb_logging:
        wandb_metrics = {f"valid/{k}": v for k, v in metrics.items()}
        wandb_metrics["valid/eval_detailed_table"] = wb_table
        wandb_metrics["valid/global_step"] = global_i
        wandb_metrics["valid/epoch"] = epoch
        wandb.log(wandb_metrics)
        
        # Create histogram
        if completion_lengths:
            wandb.log({
                "valid/completion_length_histogram": wandb.Histogram(completion_lengths),
                "valid/format_reward_histogram": wandb.Histogram(format_rewards)
            })
    
    # Log to tensorboard
    if is_tb_logging:
        for k, v in metrics.items():
            tb_writer.add_scalar(f"valid/{k}", v, global_i)
        
        # Add histogram to tensorboard
        if completion_lengths:
            tb_writer.add_histogram("valid/completion_lengths", np.array(completion_lengths), global_i)
            tb_writer.add_histogram("valid/format_rewards", np.array(format_rewards), global_i)
    
    # Print summary
    accelerator.print("\n" + "="*50)
    accelerator.print(f"Evaluation Summary - Step {global_i}, Epoch {epoch}")
    accelerator.print("="*50)
    for k, v in sorted(metrics.items()):
        accelerator.print(f"{k:30s}: {v:.4f}")
    accelerator.print("="*50 + "\n")