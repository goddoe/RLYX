"""
Basic tokenizer function for dataset preprocessing
"""
from textwrap import dedent
from rlyx.registries import TOKENIZER_REGISTRY


@TOKENIZER_REGISTRY.register("gsm8k_default_chatml_tokenizer")
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
        system_prompt = dedent("""
            user와 assistant 간의 대화입니다.
            - assistant는 <think></think> tag속에서 사용자의 질문에 대해 정리하고 생각의 과정을 거쳐 생각한 다음 사용자에게 답변을 제공합니다.
            - 생각 과정과 답변은 각각 <think> </think> 태그와 <answer> </answer> 태그로 감싸집니다.
            
            **예시**
            ```
            <|im_start|>user
            Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
            <|im_end|>
            <|im_start|>assistant
            <think>
            나탈리아는 5월에 48 ÷ 2 = 24개의 클립을 팔았습니다.
            따라서 4월과 5월에 총 48 + 24 = 72개의 클립을 팔았습니다.
            </think>
            <answer>
            72
            </answer><|im_end|>
            ```
        """).strip()


        gold_answer_list = []
        
        new_messages_list = []
        for q, a in zip(examples["question"], examples["answer"]):
            new_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q}
            ]
            new_messages_list.append(new_messages)
            gold_answer_list.append(a.split("####")[-1].strip())

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
