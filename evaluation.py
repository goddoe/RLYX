import os
import json
import random
from textwrap import dedent
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import tqdm
from utils import  extract_answer, extract_numbers, compare_numbers


def prepare_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    chat_template = dedent("""
    {{- eos_token }}
    {%- for message in messages %}
        {{- '<im_start>' + message['role'] + '\n' + message['content'] + '<im_end>' + '\n' }}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<im_start>assistant\n' }}
    {%- endif %}""").strip()
    tokenizer.chat_template = chat_template
    return tokenizer

def build_gsm8k_input_and_output(tokenizer, question, answer):
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

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": fewshot_question_1},
                {"role": "assistant", "content": fewshot_answer_1},
                {"role": "user", "content": question}
                ]
    
    input_text = tokenizer.apply_chat_template(messages,
                                               tokenize=False,
                                               add_generation_prompt=True)

    gold = answer.split("####")[-1].strip()

    return input_text, gold


def load_gsm8k(split='test'):
    dataset = load_dataset("gsm8k", "main", split=split)
    return dataset

def generate_answer(llm, input_text):
    sampling_params = SamplingParams(max_tokens=1024, # 384
                                     temperature=0.0,
                                     stop=["<im_end>"]
                                     )
    outputs = llm.generate([input_text], sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()
    return generated_text

def evaluate_gsm8k(model_name_or_path, output_path):
    llm = LLM(model=model_name_or_path)
    dataset = load_gsm8k()
    tokenizer = prepare_tokenizer(model_name_or_path=model_name_or_path)

    n_total = 0
    n_exact_correct = 0
    n_within_tolerance_correct = 0
    
    pbar = tqdm.tqdm(dataset, total=len(dataset))
    for data in pbar:
        n_total += 1
        input_text, gold = build_gsm8k_input_and_output(tokenizer, data["question"], data["answer"])

        pred_raw = generate_answer(llm, input_text)

        print(pred_raw)

        pred_answer_block = extract_answer(pred_raw)
        pred_answer_number = extract_numbers(pred_answer_block)
        pred = pred_answer_number[0] if pred_answer_number else None

        result = compare_numbers(pred, gold)
        if result['exact_match']:
            n_exact_correct += 1
        if result['within_tolerance']:
            n_within_tolerance_correct += 1

        pbar.set_description(f"Exact: {n_exact_correct/n_total:.2f}, Tolerance: {n_within_tolerance_correct/n_total:.2f}")

    exact_accuracy = n_exact_correct / n_total
    within_tolerance_accuracy = n_within_tolerance_correct / n_total

    metrics = {"gsm8k_accuracy_exact": exact_accuracy,
               "gsm8k_accuracy_within_tolerance": within_tolerance_accuracy,
               "output_path": output_path,
               "model_name_or_path": model_name_or_path,
              }
    print(metrics)

    # make dir for ojutput_path file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "at") as f:
        metrics_jsonline = json.dumps(metrics, ensure_ascii=False)
        f.write(metrics_jsonline)
        f.write("\n")

    return metrics

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--output_path", type=str, default="eval_outs/evaluation.jsonl")
    parser.add_argument("--task", type=str, default="gsm8k")
    args = parser.parse_args()

    if args.task == "gsm8k":
        evaluate_gsm8k(args.model_name_or_path, args.output_path)

