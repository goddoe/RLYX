import os
import time
import math
from argparse import ArgumentParser
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
import ray
from ray import serve
from textwrap import dedent
from arguments import BaseArgs
from torch.utils.tensorboard import SummaryWriter
import wandb
from dotenv import load_dotenv
from rewards import format_reward_func, math_reward_func
from utils import prepare_deepspeed, get_all_inference_actors, call_func_using_actor_handle, stateless_init_process_group, get_per_token_logps, extract_answer, extract_numbers, compare_numbers

load_dotenv()


def main():
    parser = ArgumentParser(description="Train for R1")
    parser.add_argument("--exp-config-path", type=str, default="./exps/exp_debug/exp_config.yaml", help="Path to the experiment config file")
    args = parser.parse_args()
    exp_args = BaseArgs.from_yaml(args.exp_config_path)

    accelerator = Accelerator(gradient_accumulation_steps=exp_args.gradient_accumulation_steps)

    is_deepspeed = accelerator.state.deepspeed_plugin is not None
    accelerator.print("Using DeepSpeed:", is_deepspeed)

    is_wandb_logging = "wandb" in exp_args.logging_methods
    is_tb_logging = "tensorboard" in exp_args.logging_methods
    if accelerator.is_main_process:
        if is_wandb_logging:
            wandb.init(
                project=exp_args.wandb_project,
                entity=exp_args.wandb_entity,
                config=exp_args.to_dict(),
                name=exp_args.exp_name
            )
        if is_tb_logging:
            tb_writer = SummaryWriter(f"tb/{exp_args.exp_name}")

    ###############################################################
    # Prepare Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(exp_args.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    stop_token = "<im_end>"
    
    chat_template = dedent("""
    {{- eos_token }}
    {%- for message in messages %}
        {{- '<im_start>' + message['role'] + '\n' + message['content'] + '<im_end>' + '\n' }}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<im_start>assistant\n' }}
    {%- endif %}""").strip()
    tokenizer.chat_template = chat_template

    def tokenize_function(examples):
        system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>\n<answer> answer here </answer>"
        fewshot_question_1 = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
        fewshot_answer_1 = dedent("""
        <think>
        Natalia sold 48/2 = 24 clips in May.
        Natalia sold 48+24 = 72 clips altogether in April and May.
        </think>
        <answer>
        72
        </answer>""").strip()
        
        gold_answer_list = []
        
        new_messages_list = []
        for q, a in zip(examples["question"], examples["answer"]):
            new_messages = [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": fewshot_question_1},
                            {"role": "assistant", "content": fewshot_answer_1},
                            {"role": "user", "content": q}
                            ]
            new_messages_list.append(new_messages)
            gold_answer_list.append(a.split("####")[-1].strip())

        batch = {}
        batch['gold_answer'] = gold_answer_list
        batch['solution'] = gold_answer_list

        batch['user_input_ids'] = tokenizer.apply_chat_template(new_messages_list,
                                                                add_generation_prompt=True,
                                                                return_tensors="pt",
                                                                padding=True,
                                                                truncation=True,
                                                                max_length=exp_args.max_length).tolist()
        batch['user_input_text'] = tokenizer.apply_chat_template(new_messages_list,
                                                                 tokenize=False,
                                                                 add_generation_prompt=True)

        return batch


    ###############################################################
    # Prepare Dataset
    is_cache_exist = os.path.exists(exp_args.tokenized_dataset_path) 
    if accelerator.is_main_process and (not is_cache_exist or exp_args.overwrite_preprocess):
        dataset = load_dataset(exp_args.dataset_name_or_path, 'main')
        tokenized_datasets = dataset.map(tokenize_function,
                                         batched=True,
                                         batch_size=exp_args.batch_size_for_preproc,
                                         num_proc=8
                                         )
        tokenized_datasets.save_to_disk(exp_args.tokenized_dataset_path)
    accelerator.wait_for_everyone()

    tokenized_datasets = load_from_disk(exp_args.tokenized_dataset_path)

    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(tokenized_datasets["train"], shuffle=True)
    else:
        train_sampler = RandomSampler(tokenized_datasets["train"])
    valid_sampler = SequentialSampler(tokenized_datasets["test"])

    def collate_fn_all(batch):
        keys = [key for key in batch[0].keys()]
        data = {key: [] for key in keys}
        for item in batch:
            for key in keys:
                data[key].append(item[key])
        if 'user_input_ids' in data:
            user_input = tokenizer.pad({"input_ids": data['user_input_ids']},
                                       return_tensors="pt",
                                       padding=True,
                                       padding_side='left')
            data['user_input_ids'] = user_input.input_ids
        return data

    train_dataloader = DataLoader(tokenized_datasets["train"],
                                  sampler=train_sampler,
                                  batch_size=exp_args.train_batch_size_per_proc,
                                  collate_fn=collate_fn_all,
                                  drop_last=True)
    valid_dataloader = DataLoader(tokenized_datasets["test"],
                                  sampler=valid_sampler,
                                  batch_size=exp_args.eval_batch_size_per_proc,
                                  collate_fn=collate_fn_all)

    ###############################################################
    # Prepare Model
    model = AutoModelForCausalLM.from_pretrained(exp_args.model_name_or_path)
    ref_model = AutoModelForCausalLM.from_pretrained(exp_args.model_name_or_path)
    ref_model.eval()


    ###############################################################
    # Prepare Optimizer and Scheduler
    optimizer_cls = (
        AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=exp_args.learning_rate)

    num_processes = accelerator.num_processes
    accelerator.print("Number of processes (GPUs):", num_processes)

    num_training_steps = math.ceil(len(tokenized_datasets['train']) / (exp_args.train_batch_size_per_proc * num_processes)) * exp_args.num_train_epochs
    accelerator.print("Number of training steps:", num_training_steps)

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=exp_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=exp_args.num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=num_training_steps, warmup_num_steps=exp_args.num_warmup_steps
        )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.state.deepspeed_plugin is None:
        ref_model = accelerator.prepare_model(ref_model, evaluation_mode=True)
    else:
        ref_model = prepare_deepspeed(ref_model, accelerator)

    ###############################################################
    # Prepare Reward function
    reward_func_list = [format_reward_func, math_reward_func]

    ###############################################################
    # Prepare Inference Workers
    ray_master_address = os.environ["RAY_MASTER_ADDRESS"]
    ray_client_server_port = int(os.environ["RAY_CLIENT_SERVER_PORT"])
    ray_master_pg_port = int(os.environ["RAY_MASTER_PG_PORT"])

    if accelerator.is_main_process:
        ray.init(address="auto")
    else:
        ray.init(address=f"ray://{ray_master_address}:{ray_client_server_port}")

    handle = serve.get_deployment_handle("InferenceWorker",
                                         app_name="default")

    print(f"result: {handle.generate_text.remote(['hello']).result()}")

    # init weight update group
    if accelerator.is_main_process:
        actor_handle_list = get_all_inference_actors()
        num_infer_workers = len(actor_handle_list)
        accelerator.print(actor_handle_list)

        worker_weight_init_handle_list = []
        for i, actor_handle in enumerate(actor_handle_list):
            worker_weight_init_handle = call_func_using_actor_handle(actor_handle,
                                         "init_weight_update_group",
                                         master_address=ray_master_address,
                                         master_port=ray_master_pg_port,
                                         rank=i+1,
                                         world_size=num_infer_workers+1)
            worker_weight_init_handle_list.append(worker_weight_init_handle)

        model_update_group = stateless_init_process_group(ray_master_address,
                                                          ray_master_pg_port,
                                                          rank=0,
                                                          world_size=num_infer_workers+1,
                                                          device=torch.device("cuda:0"))

        ray.get(worker_weight_init_handle_list)
    accelerator.wait_for_everyone()

    global_i = 0
    best_valid_loss = 9e9
    global_valid_loss = 9e9
    global_train_loss = 9e9

    os.makedirs(exp_args.save_dir, exist_ok=True)
    model.train()
    pbar = tqdm(range(num_training_steps), total=num_training_steps)

    accelerator.print("Start training")
    for epoch in range(exp_args.num_train_epochs):
        for batch in train_dataloader:
            # import pudb; pudb.set_trace()
            # context = accelerator.accumulate(model) if is_deepspeed else nullcontext()
            # context = accelerator.accumulate(model)
            context = nullcontext()
            with context:
                ###############################################################
                # Rollout
                sample_params = {'temperature': exp_args.rollout_temperature,
                                 'max_tokens': exp_args.rollout_max_tokens,
                                 'n': exp_args.rollout_per_sample,
                                 'include_stop_str_in_output': True,
                                 'stop': [stop_token]}

                future_policy_rollout_batch = handle.generate_text.remote(batch['user_input_text'],
                                                                          sample_params=sample_params)

                policy_rollout_batch = future_policy_rollout_batch.result()

                text_compl_sample_list_batch = policy_rollout_batch['text'] # [batch_size, rollout_per_sample]
                reward_list = [] # [batch_size, rollout_per_sample, num_reward_func]

                ###############################################################
                # Calc Reward
                for j, (text_compl_sample_list, solution) in enumerate(zip(text_compl_sample_list_batch, batch['solution'])):
                    curr_compl_reward_list = []  # [rollout_per_sample, num_reward_func]

                    for k, text_compl_sample in enumerate(text_compl_sample_list): # [rollout_per_sample]
                        curr_sample_reward_list = []
                        for l, reward_func in enumerate(reward_func_list):
                            reward = reward_func(text_compl_sample, solution=solution)
                            if reward > 0:
                                print("*"*30)
                                print(f"reward_func: {reward_func.__name__}, reward: {reward}")
                                print(f"text_compl_sample: {text_compl_sample}")
                                print("*"*30)
                            curr_sample_reward_list.append(reward)
                        curr_compl_reward_list.append(curr_sample_reward_list)
                    reward_list.append(curr_compl_reward_list)

                rewards = torch.tensor(reward_list)
                total_reward_by_each_compl = torch.sum(rewards, dim=2) # [batch_size, rollout_per_sample]
                reward_mean = torch.mean(total_reward_by_each_compl, dim=1) # [batch_size]
                reward_std = torch.std(total_reward_by_each_compl, dim=1) # [batch_size]

                ###############################################################
                # Calc Advantages
                # [batch_size, rollout_per_sample]
                advantages = (total_reward_by_each_compl - reward_mean.unsqueeze(1)) / (reward_std.unsqueeze(1) + 1e-4) 
                advantages = advantages.to(model.device)
                
                # [batch_size, rollout_per_sample, not fixed length ] 
                raw_completion_ids_batch = policy_rollout_batch['token_ids'] 
                
                ###############################################################
                # Calc KL divergence

                batch_size = len(raw_completion_ids_batch)
                rollout_per_sample = len(raw_completion_ids_batch[0])

                # [batch_size * rollout_per_sample, length]
                completion_ids_list = []
                for raw_completion_ids in raw_completion_ids_batch:
                    completion_ids_list.extend(raw_completion_ids)

                # [batch_size * rollout_per_sample, max_length]
                completion_padded= tokenizer.pad({"input_ids": completion_ids_list},
                                                  return_tensors="pt",
                                                  padding=True,
                                                  padding_side='right')
                completion_ids = completion_padded.input_ids

                # [batch_size, rollout_per_sample, max_length]
                completion_ids = completion_ids.view(batch_size, rollout_per_sample, -1)
                completion_ids = completion_ids.to(model.device)

                # [batch_size, max_length]
                user_input_ids = batch['user_input_ids']

                # [batch_size, rollout_per_sample, max_length]
                user_input_ids_expanded = user_input_ids.unsqueeze(1).expand(-1, rollout_per_sample, -1)
                
                # [batch_size, rollout_per_sample, max_length]
                prompt_completion_ids = torch.cat([user_input_ids_expanded,
                                                   completion_ids], dim=-1)

                logits_to_keep = completion_ids[0].size(1)

                print("*"*60)
                for item_list in prompt_completion_ids:
                    for item in item_list:
                        t = tokenizer.decode(item.cpu().tolist(), skip_special_tokens=True)
                        print("-"*30)
                        print(t)

                print("="*60)
                for item_list in completion_ids:
                    for item in item_list:
                        t = tokenizer.decode(item.cpu().tolist(), skip_special_tokens=True)
                        print("-"*30)
                        print(t)

                # [batch_size * rollout_per_sample, max_length]
                flatten_prompt_completion_ids = prompt_completion_ids.view(batch_size * rollout_per_sample, -1)

                # [batch_size, rollout_per_sample, max_length]
                flatten_prompt_completion_attention_mask = (flatten_prompt_completion_ids != pad_token_id).view(batch_size* rollout_per_sample, -1).long()

                # Calc KLD
                with torch.no_grad():
                    ref_per_token_logps = get_per_token_logps(
                        ref_model,
                        flatten_prompt_completion_ids,
                        flatten_prompt_completion_attention_mask,
                        logits_to_keep
                    )

                policy_per_token_logps = get_per_token_logps(
                    model,
                    flatten_prompt_completion_ids,
                    flatten_prompt_completion_attention_mask,
                    logits_to_keep
                )

                # Compute the KL divergence between the model and the reference model
                per_token_kl = torch.exp(ref_per_token_logps - policy_per_token_logps) - (ref_per_token_logps - policy_per_token_logps) - 1

                # x - x.detach() allows for preserving gradients from x
                # It is equivalent to updating the old policy model at every step.
                # [batch_size * rollout_per_sample, max_length]
                per_token_loss = torch.exp(policy_per_token_logps - policy_per_token_logps.detach()) * advantages.view(-1, 1)

                if accelerator.is_main_process:
                    print("*"*30)
                    print("per_token_kl")
                    print(per_token_kl[:4, :4])
                    print("*"*30)
                    print("policy_per_token_logps")
                    print(policy_per_token_logps[:4, :4])
                    print("*"*30)
                    print("per_token_loss")
                    print(per_token_loss[:4, :4])
                    print("*"*30)

                # Working version... However, I have no idea why it works
                # I think I need to multiply -1. to per_token_loss. Weird... 
                per_token_loss = per_token_loss + exp_args.kl_coef * per_token_kl
                # per_token_loss = -(per_token_loss - exp_args.kl_coef * per_token_kl)

                # [batch_size * rollout_per_sample, max_length]
                completion_attention_mask = (completion_ids != pad_token_id).view(batch_size* rollout_per_sample, -1).long() 
                train_loss = ((per_token_loss * completion_attention_mask).sum(dim=1) / completion_attention_mask.sum(dim=1)).mean()
                                                        
                accelerator.backward(train_loss)
                if not is_deepspeed and accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), exp_args.max_grad_value)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                ###############################################################
                # Update Policy model
                actor_handle_list = get_all_inference_actors(class_name='InferenceWorker', state='ALIVE')

                unwrapped_model = accelerator.unwrap_model(model)
                if accelerator.is_main_process:
                    start_time = time.time()
                    for name, p in unwrapped_model.named_parameters():
                        worker_update_weight_handle_list = []
                        for i, actor_handle in enumerate(actor_handle_list):
                            worker_update_weight_handle = call_func_using_actor_handle(actor_handle,
                                                         "update_weight",
                                                         name=name,
                                                         dtype=p.dtype,
                                                         shape=p.shape)
                            worker_update_weight_handle_list.append(worker_update_weight_handle)
                        model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
                        ray.get(worker_update_weight_handle_list)
                    
                    print(f"Time for weight update: {time.time() - start_time}")
                accelerator.wait_for_everyone()
                print(f"{accelerator.process_index} Train loss:", train_loss.item())

            ###############################################################
            # Logging
            if global_i % exp_args.log_interval == 0:
                # Collect metrics
                global_train_loss = accelerator.reduce(train_loss.detach(), reduction='mean').item()

                # [batch_size * world_size,  rollout_per_sample, num_reward_func] 
                global_rewards = accelerator.gather_for_metrics(rewards.to(model.device)).detach()
                global_reward_mean = torch.mean(global_rewards).item()
                
                if accelerator.is_main_process:
                    length_list = []
                    for text_compl_sample_list in text_compl_sample_list_batch:
                        length_list.append([len(text_compl_sample)
                                            for text_compl_sample in text_compl_sample_list])

                    length_mean = np.mean(length_list)
                    length_std = np.std(length_list)

                    reward_func_to_reward_map = {} 
                    for i, reward_func in enumerate(reward_func_list):
                        reward_func_name = reward_func.__name__
                        all_rewards = global_rewards[:, :, i]
                        curr_reward_mean = torch.sum(all_rewards) / torch.numel(all_rewards)
                        reward_func_to_reward_map[reward_func_name] = curr_reward_mean.item()

                    metrics = {"global_step": global_i,
                               "reward_mean": global_reward_mean,
                               "train_loss": global_train_loss,
                               "lr": lr_scheduler.get_last_lr()[0],
                               "length_mean": length_mean,
                               "length_std": length_std,
                                **reward_func_to_reward_map
                               }

                    if is_wandb_logging:
                        wandb.log(metrics)

                    if is_tb_logging:
                        for k, v in metrics.items():
                            tb_writer.add_scalar(f"train/{k}", v, global_i)

                desc = f"global_step: {global_i}, epoch: {epoch}, reward_mean: {global_reward_mean:0.4f}, train_loss: {global_train_loss:0.4f}, best_valid_loss: {global_valid_loss:0.4f}, recent_valid_loss: {best_valid_loss:0.4f}"
                accelerator.print(desc)


            if accelerator.is_main_process and global_i % exp_args.eval_interval == 0:
                pred_raw_list = []
                pred_list = []
                gold_list = []

                batch_result_list = []
                
                eval_sample = 30
                for batch in valid_dataloader:
                    if len(gold_list) > eval_sample:
                        break
                    # inference
                    gold_list.extend(batch['gold_answer'])

                    sample_params = {'temperature': 0.1,
                                     'max_tokens': exp_args.rollout_max_tokens,
                                     'n': 1,
                                     'include_stop_str_in_output': True,
                                     'stop': [stop_token]}
        
                    future_policy_rollout_batch = handle.generate_text.remote(batch['user_input_text'],
                                                                              sample_params=sample_params)
                    batch_result_list.append(future_policy_rollout_batch)

                    if len(batch_result_list) >= num_infer_workers:
                        continue

                    for future_policy_rollout_batch in batch_result_list:
                        policy_rollout_batch = future_policy_rollout_batch.result()
                        for preds in policy_rollout_batch['text']:
                            pred_raw_list.append(preds[0])

                    batch_result_list = []

                if batch_result_list:
                    for future_policy_rollout_batch in batch_result_list:
                        policy_rollout_batch = future_policy_rollout_batch.result()
                        for preds in policy_rollout_batch['text']:
                            pred_raw_list.append(preds[0])

                gold_list = gold_list[:eval_sample]
                pred_raw_list = pred_raw_list[:eval_sample]

                for pred_raw in pred_raw_list:
                    # extract answer from <answer> </answer> tag
                    answer_block = extract_answer(pred_raw)
                    answer_number = extract_numbers(answer_block)
                    pred = answer_number[0] if answer_number else None
                    pred_list.append(pred)

                n_exact_correct = 0
                n_within_tolerance_correct = 0
                n_total = len(pred_list)
                
                for pred, gold in zip(pred_list, gold_list):
                    result = compare_numbers(pred, gold)
                    if result['exact_match']:
                        n_exact_correct += 1
                    if result['within_tolerance']:
                        n_within_tolerance_correct += 1

                # Calc Accuracy
                exact_accuracy = n_exact_correct / n_total
                within_tolerance_accuracy = n_within_tolerance_correct / n_total

                metrics = {"gsm8k_accuracy_exact_trunc": exact_accuracy,
                           "gsm8k_accuracy_within_tolerance_trunc": within_tolerance_accuracy,

                          }
                if is_wandb_logging:
                    wandb.log(metrics)

                if is_tb_logging:
                    for k, v in metrics.items():
                        tb_writer.add_scalar(f"valid/{k}", v, global_i)

                accelerator.print(f"global_step: {global_i}, epoch: {epoch}, gsm8k_accuracy_exact: {exact_accuracy:0.4f}, gsm8k_accuracy_within_tolerance: {within_tolerance_accuracy:0.4f}")

            accelerator.wait_for_everyone()

            if global_i % exp_args.save_interval == 0:
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(f'{exp_args.save_dir}/ckpt_{global_i}',
                                                    is_main_process=accelerator.is_main_process,
                                                    save_function=accelerator.save,
                                                    state_dict=accelerator.get_state_dict(model))
                    
                    tokenizer.save_pretrained(f'{exp_args.save_dir}/ckpt_{global_i}')
                    torch.save({'epoch': epoch, 'global_step': global_i}, f'{exp_args.save_dir}/ckpt_{global_i}/training_state.pt')

                accelerator.wait_for_everyone()

            pbar.update(1)
            global_i += 1

    pbar.close()


if __name__ == "__main__":
    main()
