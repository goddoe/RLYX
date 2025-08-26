#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT_DIR=$SCRIPT_DIR/../..
SERVE_CONFIG_PATH=$SCRIPT_DIR/serve_config.yaml
cd $REPO_ROOT_DIR

serve run $SERVE_CONFIG_PATH


