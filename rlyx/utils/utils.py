__all__ = ["read_config",
           "stateless_init_process_group",
           "get_all_inference_actors",
           "call_func_using_actor_handle",
           "prepare_deepspeed",
           "get_per_token_logps",
           "create_keyword_mask_from_offsets",
           "create_tool_response_mask",
           ]

import re
import pickle
from copy import deepcopy

import ray
import yaml
import torch
import torch.nn.functional as F
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
    """Calculate per-token log probabilities"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift logits and labels for next token prediction
    logits = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()

    # Get the last logits_to_keep tokens
    if logits_to_keep > 0:
        logits = logits[:, -logits_to_keep:, :]
        labels = labels[:, -logits_to_keep:]

    # Calculate log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probs for actual tokens
    labels_expanded = labels.unsqueeze(-1)
    per_token_logps = log_probs.gather(dim=-1, index=labels_expanded).squeeze(-1)

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


def create_tool_response_mask(tokenizer, completion_texts, completion_ids, pad_token_id):
    """
    Create mask for tool response blocks during training.
    Tool responses (<|im_start|>tool ... <|im_end|>) will be masked (0).
    Assistant responses will be kept (1).
    
    NOTE: This mask is for LABELS which are shifted by 1 position from input_ids.
    When position i in input predicts position i+1, we mask position i+1 in labels.
    
    Args:
        tokenizer: Tokenizer instance
        completion_texts: List of completion text strings or single string
        completion_ids: Tensor of token IDs [batch_size, seq_len] or list of lists
        pad_token_id: Padding token ID
        
    Returns:
        mask: Tensor with same shape as completion_ids where 1 = train, 0 = mask
              This mask should be applied to SHIFTED labels (labels = input_ids[:, 1:])
    """
    import torch
    
    # Handle single string input
    if isinstance(completion_texts, str):
        completion_texts = [completion_texts]
    
    # Convert completion_ids to tensor if needed
    if isinstance(completion_ids, list):
        # Pad to same length if list of lists
        if all(isinstance(item, list) for item in completion_ids):
            max_len = max(len(ids) for ids in completion_ids)
            padded_ids = []
            for ids in completion_ids:
                padded = ids + [pad_token_id] * (max_len - len(ids))
                padded_ids.append(padded)
            completion_ids = torch.tensor(padded_ids)
        else:
            completion_ids = torch.tensor(completion_ids).unsqueeze(0)
    
    batch_size, seq_len = completion_ids.shape
    
    # Initialize mask with 1s (keep all by default)
    mask = torch.ones_like(completion_ids, dtype=torch.float32)
    
    # Tool response pattern
    tool_start_pattern = r'<\|im_start\|>tool'
    tool_end_pattern = r'<\|im_end\|>'
    
    # Process each completion text
    for batch_idx, text in enumerate(completion_texts):
        if not text:
            continue
            
        # Find all tool response blocks
        current_pos = 0
        while True:
            # Find start of tool response
            start_match = re.search(tool_start_pattern, text[current_pos:])
            if not start_match:
                break
            
            tool_start = current_pos + start_match.start()
            
            # Find corresponding end
            end_match = re.search(tool_end_pattern, text[tool_start:])
            if not end_match:
                break
            
            tool_end = tool_start + end_match.end()
            
            # Tokenize the prefix to find token positions
            # This is approximate but should work for most cases
            prefix_text = text[:tool_start]
            tool_text = text[tool_start:tool_end]
            
            # Tokenize to get approximate token positions
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
            tool_tokens = tokenizer.encode(tool_text, add_special_tokens=False)
            
            # Calculate token range for tool response
            start_token_idx = len(prefix_tokens)
            end_token_idx = start_token_idx + len(tool_tokens)
            
            # IMPORTANT: Shift mask by 1 for label alignment
            # When input[i] predicts input[i+1], we mask label[i] which corresponds to input[i+1]
            # So if tool response is at input[40:63], we mask labels[39:62]
            # But since labels are already shifted (labels = input[1:]), 
            # we need to mask positions [39:62] in the label mask which means [40-1:63-1]
            if batch_idx < batch_size:
                # Shift indices by -1 for label alignment
                label_start = max(0, start_token_idx - 1)
                label_end = min(seq_len, end_token_idx - 1)
                if label_end > label_start:
                    mask[batch_idx, label_start:label_end] = 0
            
            # Move to next potential tool response
            current_pos = tool_end
    
    # Also mask padding tokens
    padding_mask = (completion_ids == pad_token_id)
    mask[padding_mask] = 0
    
    return mask

