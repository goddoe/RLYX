#!/bin/bash

BASE_DIR="./ckpts/exp-qwen0.5b-r1-zero-example"

NUM_GPUS=4

SAVE_PATH="./eval_outs/evaluation.jsonl"

COMMANDS=()

for ckpt in $(find "$BASE_DIR" -maxdepth 1 -type d -name "ckpt_*" | sort); do
    COMMANDS+=("python evaluation.py --model_name_or_path \"$ckpt\" --output_path $SAVE_PATH")
done

TOTAL_JOBS=${#COMMANDS[@]}

gpu_idx=0

for (( i=0; i<TOTAL_JOBS; i++ )); do
    CMD=${COMMANDS[$i]}
    
    CUDA_VISIBLE_DEVICES=$gpu_idx bash -c "$CMD" &

    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

    if (( (i + 1) % NUM_GPUS == 0 )); then
        wait
    fi
done

wait

echo "All evaluations completed. Results saved in $SAVE_PATH."
