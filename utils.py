import re
import pickle
from copy import deepcopy

import ray
import yaml
import torch
from ray.util.state import list_actors
from ray.serve._private.common import RequestMetadata


def read_config(config_path):
    with open(config_path, 'rt') as f:
        config = yaml.safe_load(f)
    return config


def stateless_init_process_group(master_address, master_port,
                                 rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes) 
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


def get_all_inference_actors(class_name='InferenceWorker', state='ALIVE'
                             ) -> list[ray.actor.ActorHandle]:
    actor_state_list = []

    for actor in list_actors():
        if class_name in actor.class_name and actor.state == state:
            actor_state_list.append(actor)

    actor_handle_list = []
    for actor_state in actor_state_list:
        actor_handle = ray.get_actor(name=actor_state['name'], namespace=actor_state['ray_namespace'])
        actor_handle_list.append(actor_handle)

    return actor_handle_list


def call_func_using_actor_handle(actor_handle: ray.actor.ActorHandle,
                                 method_name: str,
                                 *method_args, **method_kwargs) -> ray.ObjectRef:
    request_metadata = RequestMetadata(
        request_id="dummy",
        internal_request_id="dummy",
        call_method=method_name
    )
    serialized_metadata = pickle.dumps(request_metadata)
    result = actor_handle.handle_request.remote(serialized_metadata, *method_args, **method_kwargs)
    return result



def prepare_deepspeed(model, accelerator):
    # Copy From: https://github.com/huggingface/trl/blob/af4ad47035529164799be10f3fe558ee642a9880/trl/models/utils.py#L199-L230

    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    import deepspeed
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    stage = config_kwargs["zero_optimization"]["stage"]


    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes)
            if getattr(model.config, "hidden_sizes", None)
            else getattr(model.config, "hidden_size", None)
        )
        if hidden_size is not None and stage == 3:
            # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
            # @ step 0: expected module 1, but got module 0`
            # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                }
            )


    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO
    # disabled (stage 0)
    if stage != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):

    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask
    ).logits  # (B, L, V)

    # [batch_size, max_length, vocab_size]
    logits = logits[:, -logits_to_keep:, :]  
    input_ids = input_ids[:, -logits_to_keep:]

    # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
    # per_token_logps = []
    # for logits_row, input_ids_row in zip(logits, input_ids[:, -logits_to_keep:]):
    #     log_probs = logits_row.log_softmax(dim=-1)
    #     token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
    #     per_token_logps.append(token_log_prob)

    # return torch.stack(per_token_logps)

    token_logits = logits.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1)
                                    for lg in logits])  # loop to reduce memory peak
    token_log_probs = token_logits - logsumexp_values  # log_softmax = logits - log(sum(exp(logits)))
    return token_log_probs


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1,
                                       index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = (
                row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1))
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def create_keyword_mask_from_offsets(tokenizer, input_texts, keywords):
    tokenized_inputs = tokenizer(input_texts, padding=True, truncation=True,
                                 return_tensors="pt", return_offsets_mapping=True)
    token_ids = tokenized_inputs["input_ids"]
    offset_mappings = tokenized_inputs["offset_mapping"]

    batch_size, seq_len = token_ids.shape

    mask = torch.zeros_like(token_ids, dtype=torch.float32)

    keyword_positions = []
    for keyword in keywords:
        for text in input_texts:
            start_idx = text.find(keyword)
            if start_idx != -1:
                keyword_positions.append((text, start_idx, start_idx + len(keyword)))

    for b in range(batch_size):
        text = input_texts[b]
        for _, start_pos, end_pos in keyword_positions:
            if text != _:
                continue
            for i in range(seq_len):
                token_start, token_end = offset_mappings[b, i]
                if token_start >= start_pos and token_end <= end_pos:
                    mask[b, i] = 1

    return mask


def extract_numbers(text):
    if text is None:
        return []
    
    text = text.replace(",", "")
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)

    return [float(num) for num in numbers] if numbers else []


def compare_numbers(pred, gold, tolerance=1e-5):
    if not pred or not gold:
        return {
            "exact_match": False,
            "within_tolerance": False,
            "pred": pred,
            "gold":gold
        }

    if isinstance(gold, str):
        gold = gold.replace(",", "")
    if isinstance(pred, str):
        pred = pred.replace(",", "")

    pred = float(pred)
    gold = float(gold)

    exact_match = pred == gold 
    within_tolerance = abs(pred - gold) <= tolerance

    return {
        "exact_match": exact_match,
        "within_tolerance": within_tolerance,
        "pred": pred,
        "gold":gold
    }


def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1) if match else None
