import os
import time
import math
from argparse import ArgumentParser
from contextlib import nullcontext
from textwrap import dedent
from rlyx.arguments.base_arguments import BaseArgs
import io

import ray
import wandb
import numpy as np
import torch
import torch.distributed as dist
from ray import serve
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# Module registries
from rlyx.registries import (
    CHAT_TEMPLATE_REGISTRY,
    DATASET_LOADER_REGISTRY,
    EVALUATOR_REGISTRY,
    REWARD_REGISTRY,
    TOKENIZER_REGISTRY
)

# Utilities
from rlyx.utils import \
    prepare_deepspeed, get_all_inference_actors, \
    call_func_using_actor_handle, stateless_init_process_group, \
    get_per_token_logps

load_dotenv()


def setup_logging(accelerator, logging_methods, wandb_project, wandb_entity, exp_config, exp_name):
    """Setup logging for wandb and tensorboard"""
    tb_writer = None
    is_wandb_logging = "wandb" in logging_methods
    is_tb_logging = "tensorboard" in logging_methods
    
    if accelerator.is_main_process:
        if is_wandb_logging:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config=exp_config,
                name=exp_name
            )
        
        if is_tb_logging:
            tb_writer = SummaryWriter(f"tb/{exp_name}")
    
    return tb_writer, is_wandb_logging, is_tb_logging


def setup_tokenizer(model_name_or_path, chat_template_name=None):
    """Setup tokenizer with proper configurations"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load chat template dynamically only if specified
    if chat_template_name:
        get_chat_template = CHAT_TEMPLATE_REGISTRY.get(chat_template_name)
        chat_template = get_chat_template()
        tokenizer.chat_template = chat_template
    
    return tokenizer


def prepare_dataset(accelerator, dataset_name_or_path, dataset_loader_name, tokenized_dataset_path, 
                   overwrite_preprocess, batch_size_for_preproc, tokenizer, max_length, 
                   tokenizer_function_name, train_size_limit, valid_size_limit):
    """Prepare and tokenize datasets"""
    # Load tokenize function dynamically
    create_tokenize_function = TOKENIZER_REGISTRY.get(tokenizer_function_name)
    tokenize_function = create_tokenize_function(tokenizer, max_length)
    
    is_cache_exist = os.path.exists(tokenized_dataset_path)
    if accelerator.is_main_process and (not is_cache_exist or overwrite_preprocess):
        # Load dataset loader dynamically
        load_data = DATASET_LOADER_REGISTRY.get(dataset_loader_name)
        dataset = load_data(dataset_name_or_path, name='main')
        
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size_for_preproc,
            num_proc=8
        )
        tokenized_datasets.save_to_disk(tokenized_dataset_path)
    accelerator.wait_for_everyone()
    
    tokenized_datasets = load_from_disk(tokenized_dataset_path)
    
    # Apply size limits
    if train_size_limit > 0:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(min(train_size_limit, len(tokenized_datasets["train"]))))
    if valid_size_limit > 0:
        tokenized_datasets["test"] = tokenized_datasets["test"].select(range(min(valid_size_limit, len(tokenized_datasets["test"]))))
    
    return tokenized_datasets


def create_dataloaders(tokenized_datasets, tokenizer, train_batch_size_per_proc, eval_batch_size_per_proc):
    """Create train and validation dataloaders"""
    def collate_fn_all(batch):
        keys = [key for key in batch[0].keys()]
        data = {key: [] for key in keys}
        for item in batch:
            for key in keys:
                data[key].append(item[key])
        if "user_input_ids" in data:
            user_input = tokenizer.pad({"input_ids": data["user_input_ids"]},
                                       return_tensors="pt",
                                       padding=True,
                                       padding_side="left")
            data["user_input_ids"] = user_input.input_ids
        return data
    
    train_dataloader = DataLoader(tokenized_datasets["train"],
                                  batch_size=train_batch_size_per_proc,
                                  collate_fn=collate_fn_all,
                                  shuffle=True,
                                  drop_last=True)
    valid_dataloader = DataLoader(tokenized_datasets["test"],
                                  batch_size=eval_batch_size_per_proc,
                                  shuffle=False,
                                  collate_fn=collate_fn_all)
    
    return train_dataloader, valid_dataloader


def initialize_models(model_name_or_path, kl_coef, accelerator):
    """Initialize main model and reference model"""
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    
    ref_model = None
    if kl_coef > 0.:
        ref_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        ref_model.eval()
    
    return model, ref_model


def setup_optimizer_and_scheduler(model, accelerator, learning_rate, lr_scheduler_type, num_warmup_steps, num_training_steps):
    """Setup optimizer and learning rate scheduler"""
    optimizer_cls = (
        AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)
    
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=num_training_steps, warmup_num_steps=num_warmup_steps
        )
    
    return optimizer, lr_scheduler


def setup_ray_and_inference_workers(accelerator):
    """Setup Ray and get inference worker handles"""
    ray_master_address = os.environ["RAY_MASTER_ADDRESS"]
    ray_client_server_port = int(os.environ["RAY_CLIENT_SERVER_PORT"])
    ray_master_pg_port = int(os.environ["RAY_MASTER_PG_PORT"])
    
    if accelerator.is_main_process:
        ray.init(address="auto")
    else:
        ray.init(address=f"ray://{ray_master_address}:{ray_client_server_port}")
    
    handle = serve.get_deployment_handle("InferenceWorker", app_name="InferenceService")
    
    accelerator.print("init weight update group")
    
    num_infer_workers = -1
    model_update_group = None
    
    if accelerator.is_main_process:
        actor_handle_list = get_all_inference_actors()
        num_infer_workers = len(actor_handle_list)
        accelerator.print(actor_handle_list)
        
        worker_weight_init_handle_list = []
        for i, actor_handle in enumerate(actor_handle_list):
            worker_weight_init_handle = call_func_using_actor_handle(
                actor_handle,
                "init_weight_update_group",
                master_address=ray_master_address,
                master_port=ray_master_pg_port,
                rank=i+1,
                world_size=num_infer_workers + 1
            )
            worker_weight_init_handle_list.append(worker_weight_init_handle)
        
        model_update_group = stateless_init_process_group(
            ray_master_address,
            ray_master_pg_port,
            rank=0,
            world_size=num_infer_workers + 1,
            device=torch.device("cuda:0")
        )
        
        ray.get(worker_weight_init_handle_list)
    
    accelerator.wait_for_everyone()
    
    return handle, model_update_group, num_infer_workers


def perform_rollout(handle, batch, rollout_temperature, rollout_max_tokens, rollout_per_sample, stop_tokens):
    """Perform policy rollout with inference workers"""
    sample_params = {
        "temperature": rollout_temperature,
        "max_tokens": rollout_max_tokens,
        "n": rollout_per_sample,
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
        "stop": stop_tokens
    }
    
    future_policy_rollout_batch = handle.generate_text.remote(
        batch["user_input_text"],
        sample_params=sample_params
    )
    
    return future_policy_rollout_batch.result()


def calculate_rewards(text_compl_sample_list_batch, gold_text_list, reward_func_list, accelerator, stop_tokens):
    """Calculate rewards for rollout completions"""
    reward_list = []  # [batch_size, rollout_per_sample, num_reward_func]
    
    for j, (text_compl_sample_list, gold_text) in enumerate(zip(text_compl_sample_list_batch, gold_text_list)):
        curr_compl_reward_list = []  # [rollout_per_sample, num_reward_func]
        
        for k, text_compl_sample in enumerate(text_compl_sample_list):  # [rollout_per_sample]
            curr_sample_reward_list = []
            for l, reward_func in enumerate(reward_func_list):
                reward = reward_func(
                    completion=text_compl_sample,
                    gold_text=gold_text,
                    end_of_turn_token=stop_tokens[0] if stop_tokens else "",
                )
                
                log = (f"Gold Text: {gold_text}\n"
                       f"Pred Text: {text_compl_sample}\n"
                       f"{reward_func.__name__} {reward:.4f}")
                
                print("----------------------------\n"
                      f"{accelerator.process_index=}\n"
                      f"{log}\n")
                
                curr_sample_reward_list.append(reward)
            curr_compl_reward_list.append(curr_sample_reward_list)
        reward_list.append(curr_compl_reward_list)
    
    return torch.tensor(reward_list).float()


def calculate_advantages(rewards):
    """Calculate advantages from rewards"""
    total_reward_by_each_compl = torch.sum(rewards, dim=2)  # [batch_size, rollout_per_sample]
    reward_mean = torch.mean(total_reward_by_each_compl, dim=1)  # [batch_size]
    reward_std = torch.std(total_reward_by_each_compl, dim=1)  # [batch_size]
    
    # [batch_size, rollout_per_sample]
    advantages = (total_reward_by_each_compl - reward_mean.unsqueeze(1)) / (reward_std.unsqueeze(1) + 1e-4)
    
    return advantages, total_reward_by_each_compl


def prepare_completion_ids(raw_completion_ids_batch, tokenizer, model_device):
    """Prepare completion IDs for loss calculation"""
    batch_size = len(raw_completion_ids_batch)
    rollout_per_sample = len(raw_completion_ids_batch[0])
    
    # [batch_size * rollout_per_sample, length]
    completion_ids_list = []
    for raw_completion_ids in raw_completion_ids_batch:
        completion_ids_list.extend(raw_completion_ids)
    
    # [batch_size * rollout_per_sample, max_length]
    completion_padded = tokenizer.pad(
        {"input_ids": completion_ids_list},
        return_tensors="pt",
        padding=True,
        padding_side="right"
    )
    completion_ids = completion_padded.input_ids
    
    # [batch_size, rollout_per_sample, max_length]
    completion_ids = completion_ids.view(batch_size, rollout_per_sample, -1)
    completion_ids = completion_ids.to(model_device)
    
    return completion_ids, batch_size, rollout_per_sample


def calculate_policy_loss(model, ref_model, user_input_ids, completion_ids, advantages, kl_coef, pad_token_id, accelerator):
    """Calculate GRPO policy loss"""
    batch_size, rollout_per_sample, comp_length = completion_ids.shape
    
    # [batch_size, rollout_per_sample, max_length]
    user_input_ids_expanded = user_input_ids.unsqueeze(1).expand(-1, rollout_per_sample, -1)
    
    # [batch_size, rollout_per_sample, max_length]
    prompt_completion_ids = torch.cat([user_input_ids_expanded, completion_ids], dim=-1)
    
    logits_to_keep = comp_length
    
    # [batch_size * rollout_per_sample, max_length]
    flatten_prompt_completion_ids = prompt_completion_ids.view(batch_size * rollout_per_sample, -1)
    
    # [batch_size * rollout_per_sample, max_length]
    flatten_prompt_completion_attention_mask = (flatten_prompt_completion_ids != pad_token_id).long()
    
    policy_per_token_logps = get_per_token_logps(
        model,
        flatten_prompt_completion_ids,
        flatten_prompt_completion_attention_mask,
        logits_to_keep
    )
    
    per_token_kl = 0.
    if kl_coef > 0.:
        # Calc KLD
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(
                ref_model,
                flatten_prompt_completion_ids,
                flatten_prompt_completion_attention_mask,
                logits_to_keep
            )
        
        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - policy_per_token_logps) - (ref_per_token_logps - policy_per_token_logps) - 1
    
    # x - x.detach() allows for preserving gradients from x
    # [batch_size * rollout_per_sample, max_length]
    per_token_loss = torch.exp(policy_per_token_logps - policy_per_token_logps.detach()) * advantages.view(-1, 1)
    
    per_token_loss = -(per_token_loss - kl_coef * per_token_kl)
    
    if accelerator.is_main_process:
        print("*" * 30)
        print("advantages")
        print(advantages[:4, :4])
    
    # [batch_size * rollout_per_sample, max_length]
    completion_attention_mask = (completion_ids != pad_token_id).view(batch_size * rollout_per_sample, -1).long()
    train_loss = ((per_token_loss * completion_attention_mask).sum(dim=1) / completion_attention_mask.sum(dim=1)).mean()
    
    return train_loss


def update_model_weights(accelerator, model, model_update_group):
    """Update inference worker weights with trained model"""
    if not accelerator.is_main_process:
        return
    
    actor_handle_list = get_all_inference_actors(class_name="InferenceWorker", state="ALIVE")
    
    # Use get_state_dict to handle Stage 3 parameter gathering automatically
    state_dict = accelerator.get_state_dict(model)
    
    start_time = time.time()
    for name, param in state_dict.items():
        if not param.is_cuda:
            param = param.cuda()

        worker_update_weight_handle_list = []
        for i, actor_handle in enumerate(actor_handle_list):
            worker_update_weight_handle = call_func_using_actor_handle(
                actor_handle,
                "update_weight",
                name=name,
                dtype=param.dtype,
                shape=param.shape  # Now this will be the full shape even in Stage 3
            )
            worker_update_weight_handle_list.append(worker_update_weight_handle)
        model_update_group.broadcast(param, src=0, stream=torch.cuda.current_stream())
        ray.get(worker_update_weight_handle_list)
    
    print(f"Time for weight update: {time.time() - start_time}")


def log_metrics(global_i, epoch, train_loss, rewards, lr_scheduler, accelerator, is_wandb_logging, is_tb_logging, tb_writer, reward_func_list, text_compl_sample_list_batch):
    """Log training metrics to wandb and tensorboard"""
    # Collect metrics
    global_train_loss = accelerator.reduce(train_loss.detach().float(), reduction="mean").item()
    
    # [batch_size * world_size,  rollout_per_sample, num_reward_func]
    global_rewards = accelerator.gather(rewards.to(accelerator.device).float()).detach()
    global_reward_mean = torch.mean(global_rewards).item()
    
    if accelerator.is_main_process:
        length_list = []
        for text_compl_sample_list in text_compl_sample_list_batch:
            length_list.append([len(text_compl_sample) for text_compl_sample in text_compl_sample_list])
        
        length_mean = np.mean(length_list)
        length_std = np.std(length_list)
        
        reward_func_to_reward_map = {}
        for i, reward_func in enumerate(reward_func_list):
            reward_func_name = reward_func.__name__
            all_rewards = global_rewards[:, :, i]
            curr_reward_mean = torch.sum(all_rewards) / torch.numel(all_rewards)
            reward_func_to_reward_map[reward_func_name] = curr_reward_mean.item()
        
        metrics = {
            "epoch": epoch,
            "global_step": global_i,
            "reward_mean": global_reward_mean,
            "train_loss": global_train_loss,
            "lr": lr_scheduler.get_last_lr()[0],
            "length_mean": length_mean,
            "length_std": length_std,
            **reward_func_to_reward_map
        }
        
        print(metrics)
        
        if is_wandb_logging:
            wandb.log(metrics)
        
        if is_tb_logging:
            for k, v in metrics.items():
                tb_writer.add_scalar(f"train/{k}", v, global_i)



def save_checkpoint(accelerator, model, tokenizer, save_dir, epoch, global_i):
    """Save model checkpoint"""
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{save_dir}/ckpt_{global_i}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model)
        )
        
        tokenizer.save_pretrained(f"{save_dir}/ckpt_{global_i}")
        torch.save(
            {"epoch": epoch, "global_step": global_i},
            f"{save_dir}/ckpt_{global_i}/training_state.pt"
        )
    
    accelerator.wait_for_everyone()


def main():
    parser = ArgumentParser(description="Train for R1")
    parser.add_argument(
        "--exp-config-path",
        type=str, default="./exps/exp_debug/exp_config.yaml",
        help="Path to the experiment config file"
    )
    args = parser.parse_args()
    exp_args = BaseArgs.from_yaml(args.exp_config_path)

    accelerator = Accelerator(gradient_accumulation_steps=exp_args.gradient_accumulation_steps)
    is_deepspeed = accelerator.state.deepspeed_plugin is not None
    accelerator.print("Using DeepSpeed:", is_deepspeed)

    # Setup Logging
    tb_writer, is_wandb_logging, is_tb_logging = setup_logging(
        accelerator, 
        exp_args.logging_methods,
        exp_args.wandb_project,
        exp_args.wandb_entity,
        exp_args.to_dict(),
        exp_args.exp_name
    )

    # Prepare Tokenizer
    tokenizer = setup_tokenizer(exp_args.model_name_or_path, exp_args.chat_template_name)
    pad_token_id = tokenizer.pad_token_id
    stop_tokens = exp_args.stop_tokens

    # Prepare Dataset
    tokenized_datasets = prepare_dataset(
        accelerator,
        exp_args.dataset_name_or_path,
        exp_args.dataset_loader_name,
        exp_args.tokenized_dataset_path,
        exp_args.overwrite_preprocess,
        exp_args.batch_size_for_preproc,
        tokenizer,
        exp_args.max_length,
        exp_args.tokenizer_function_name,
        exp_args.train_size_limit,
        exp_args.valid_size_limit
    )
    train_dataloader, valid_dataloader = create_dataloaders(
        tokenized_datasets,
        tokenizer,
        exp_args.train_batch_size_per_proc,
        exp_args.eval_batch_size_per_proc
    )

    # Prepare Model
    model, ref_model = initialize_models(
        exp_args.model_name_or_path,
        exp_args.kl_coef,
        accelerator
    )

    # Prepare Optimizer and Scheduler
    num_processes = accelerator.num_processes
    accelerator.print("Number of processes (GPUs):", num_processes)
    
    num_update_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    num_training_steps = num_update_per_epoch * exp_args.num_train_epochs

    accelerator.print("Number of training steps:", num_training_steps)
    accelerator.print("Number of num_update_per_epoch:", num_update_per_epoch)
    accelerator.print("Number of num_train_epochs", exp_args.num_train_epochs)

    
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(
        model,
        accelerator,
        exp_args.learning_rate,
        exp_args.lr_scheduler_type,
        exp_args.num_warmup_steps,
        num_training_steps
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )


    if exp_args.kl_coef > 0.:
        if accelerator.state.deepspeed_plugin is None:
            ref_model = accelerator.prepare_model(ref_model, evaluation_mode=True)
        else:
            ref_model = prepare_deepspeed(ref_model, accelerator)

    # Prepare Reward functions from module registry
    reward_func_list = []
    for reward_name in exp_args.reward_function_names:
        reward_func = REWARD_REGISTRY.get(reward_name)
        reward_func_list.append(reward_func)

    # Load evaluator module
    evaluate_model_func = EVALUATOR_REGISTRY.get(exp_args.evaluator_name)

    # Prepare Inference Workers
    handle, model_update_group, num_infer_workers = setup_ray_and_inference_workers(accelerator)

    global_i = 0
    os.makedirs(exp_args.save_dir, exist_ok=True)
    model.train()
    pbar = tqdm(range(num_training_steps), total=num_training_steps)

    accelerator.print("Start training")
    for epoch in range(exp_args.num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                # Rollout
                policy_rollout_batch = perform_rollout(
                    handle,
                    batch,
                    exp_args.rollout_temperature,
                    exp_args.rollout_max_tokens,
                    exp_args.rollout_per_sample,
                    stop_tokens
                )
                text_compl_sample_list_batch = policy_rollout_batch["text"]  # [batch_size, rollout_per_sample]
                
                # Calculate Rewards
                rewards = calculate_rewards(
                    text_compl_sample_list_batch,
                    batch["gold_text"],
                    reward_func_list,
                    accelerator,
                    stop_tokens
                )
                
                # Calculate Advantages
                advantages, total_reward_by_each_compl = calculate_advantages(rewards)
                advantages = advantages.to(model.device)
                
                # Prepare completion IDs
                raw_completion_ids_batch = policy_rollout_batch["token_ids"]
                completion_ids, batch_size, rollout_per_sample = prepare_completion_ids(
                    raw_completion_ids_batch,
                    tokenizer,
                    model.device
                )
                
                # Calculate policy loss
                train_loss = calculate_policy_loss(
                    model,
                    ref_model,
                    batch["user_input_ids"],
                    completion_ids,
                    advantages,
                    exp_args.kl_coef,
                    pad_token_id,
                    accelerator
                )

                accelerator.backward(train_loss)
                if not is_deepspeed and accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), exp_args.max_grad_value)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Update Policy model
                update_model_weights(accelerator, model, model_update_group)
                accelerator.wait_for_everyone()
                print("-------------------------------\n"
                      f"{accelerator.process_index=}\n"
                      f"{rewards=}\n"
                      f"{advantages=}\n"
                      f"{train_loss=}\n"
                      f"{train_loss.item()=}\n")

            # Logging
            if global_i % exp_args.log_interval == 0:
                log_metrics(
                    global_i, epoch, train_loss, rewards, lr_scheduler,
                    accelerator, is_wandb_logging, is_tb_logging, tb_writer,
                    reward_func_list, text_compl_sample_list_batch
                )
            
            # Evaluation
            if accelerator.is_main_process and global_i % exp_args.eval_interval == 0:
                evaluate_model_func(
                    handle, valid_dataloader, exp_args.rollout_max_tokens, stop_tokens,
                    num_infer_workers, accelerator, is_wandb_logging,
                    is_tb_logging, tb_writer, global_i, epoch,
                    exp_args=exp_args  # Pass exp_args as kwargs for any additional params evaluator might need
                )
            
            accelerator.wait_for_everyone()
            
            # Save model
            if global_i % exp_args.save_interval == 0:
                save_checkpoint(accelerator, model, tokenizer, exp_args.save_dir, epoch, global_i)

            pbar.update(1)
            global_i += 1

    pbar.close()


if __name__ == "__main__":
    main()
