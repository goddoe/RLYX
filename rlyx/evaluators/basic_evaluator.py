"""
Basic evaluator for model validation
"""
import io
import numpy as np
import soundfile as sf
import wandb
from rlyx.utils.transcribe_utils import batch_cvt_audio_and_transcribe_completion
from rlyx.rewards.transcribe_reward import transcribe_reward_func
from rlyx.registries import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register("basic_evaluator")
def evaluate_model(handle, valid_dataloader, rollout_max_tokens, stop_tokens, 
                  num_infer_workers, accelerator, is_wandb_logging, is_tb_logging, 
                  tb_writer, global_i, epoch, **kwargs):
    """
    Basic evaluation with TTS metrics
    
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
    
    pred_text_and_audio_bin_list = batch_cvt_audio_and_transcribe_completion(
        completion_list,
        max_workers=8
    )
    
    report_agg = {}
    
    wb_table = wandb.Table(columns=[
        "global_step", "epoch", "gold_text", "pred_text_transcribed",
        'CER', 'WER', "tts_reward", "completion", "audio"
    ])
    
    for eval_i, ((pred_text, audio_bin), gold_text, completion) in enumerate(
        zip(pred_text_and_audio_bin_list, gold_text_list, completion_list)
    ):
        report = transcribe_reward_func(
            pred_text=pred_text,
            gold_text=gold_text,
            rich=True
        )
        for k, v in report.items():
            if k not in report_agg:
                report_agg[k] = []
            report_agg[k].append(v)
        
        print(f"eval_i: {eval_i}, gold_text: {gold_text}, pred_text: {pred_text}, {report}")
        
        if is_wandb_logging and eval_i <= 10:
            audio = None
            if audio_bin:
                audio, sr = sf.read(io.BytesIO(audio_bin))
                audio = wandb.Audio(audio, sr, caption=f"gold_text: {gold_text}\npred_text: {pred_text}")
            
            wb_table.add_data(
                global_i, epoch, gold_text, pred_text,
                report['cer_value'], report['wer_value'], report['tts_reward'],
                completion, audio
            )
    
    metrics = {k: np.mean(v) for k, v in report_agg.items() if len(v) > 0}
    
    if is_wandb_logging:
        wandb_metrics = {f"valid/{k}": v for k, v in metrics.items()}
        wandb_metrics["valid/eval_sample_table"] = wb_table
        wandb.log(wandb_metrics)
    
    if is_tb_logging:
        for k, v in metrics.items():
            tb_writer.add_scalar(f"valid/{k}", v, global_i)
    
    accelerator.print(
        f"global_step: {global_i}, epoch: {epoch}, {metrics}"
    )
