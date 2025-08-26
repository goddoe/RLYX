#!/usr/bin/env bash

python -m rlyx.evaluation \
  --model_name_or_path "./ckpts/exp-qwen0.5b-r1-zero-example/ckpt_0" \
  --output_path "./eval_outs/evaluation.jsonl"

