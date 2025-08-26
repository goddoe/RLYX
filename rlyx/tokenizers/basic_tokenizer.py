"""
Basic tokenizer function for dataset preprocessing
"""
from rlyx.registries import TOKENIZER_REGISTRY


@TOKENIZER_REGISTRY.register("basic_tokenizer")
def create_tokenize_function(tokenizer, max_length):
    """
    Create basic tokenization function for dataset preprocessing
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length
    
    Returns:
        Tokenization function
    """
    def tokenize_function(examples):
        gold_answer_list = []
        
        new_messages_list = []
        for text in examples["text"]:
            new_messages = [
                {"role": "user-text", "content": text}
            ]
            new_messages_list.append(new_messages)
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