#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT_DIR=$SCRIPT_DIR/../..
cd $REPO_ROOT_DIR

YAML_FILE=$(find "$SCRIPT_DIR" -maxdepth 1 -name "*.yaml" | head -n 1)

if [ -z "$YAML_FILE" ]; then
    echo "No YAML file found in the current directory: $(pwd)"
    exit 1
fi

ABS_YAML_PATH=$(realpath "$YAML_FILE")
echo "Using YAML file: $ABS_YAML_PATH"

# offline
# export WANDB_MODE=offline
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

# cache
# export HF_HOME="${REPO_ROOT_DIR}/cache"
# export HF_DATASETS_CACHE="${REPO_ROOT_DIR}/cache/datasets"

# Train Workers
NUM_PROCESSES=8
NUM_MACHINES=1
MAIN_PROCESS_IP=$(hostname -I | awk '{print $1}')
MAIN_PROCESS_PORT=23457
MACHINE_RANK=0

# Inference Workers
export RAY_MASTER_ADDRESS=$MAIN_PROCESS_IP
export RAY_CLIENT_SERVER_PORT=5000
export RAY_MASTER_PG_PORT=5001

accelerate launch \
  --num_processes $NUM_PROCESSES \
  --num_machines $NUM_MACHINES \
  --main_process_ip $MAIN_PROCESS_IP \
  --main_process_port $MAIN_PROCESS_PORT \
  --machine_rank $MACHINE_RANK \
  train.py \
    --exp-config-path $ABS_YAML_PATH


