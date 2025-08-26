"""
Few-shot tokenizer function with example pairs
"""
from rlyx.registries import TOKENIZER_REGISTRY


@TOKENIZER_REGISTRY.register("fewshot_tokenizer")
def create_tokenize_function(tokenizer, max_length):
    """
    Create tokenization function with few-shot examples
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length
    
    Returns:
        Tokenization function with few-shot examples
    """
    # Define few-shot examples
    few_shot_examples = [
        {
            "user": "안녕하세요, 반갑습니다.",
            "assistant": "<|unit1|><|unit2|><|unit3|><|unit4|><|unit5|>"
        },
        {
            "user": "오늘 날씨가 좋네요.",
            "assistant": "<|unit10|><|unit20|><|unit30|><|unit40|>"
        }
    ]
    
    def tokenize_function(examples):
        gold_answer_list = []
        new_messages_list = []
        
        for text in examples["text"]:
            # Build messages with few-shot examples
            messages = []
            
            # Add few-shot examples
            for example in few_shot_examples:
                messages.append({"role": "user-text", "content": example["user"]})
                messages.append({"role": "assistant-units", "content": example["assistant"]})
            
            # Add current example
            messages.append({"role": "user-text", "content": text})
            
            new_messages_list.append(messages)
            gold_answer_list.append(text)

        batch = {}
        batch["gold_text"] = gold_answer_list

        batch["user_input_ids"] = tokenizer.apply_chat_template(
            new_messages_list,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).tolist()
        batch["user_input_text"] = tokenizer.apply_chat_template(
            new_messages_list,
            tokenize=False,
            add_generation_prompt=True
        )

        return batch
    
    return tokenize_function