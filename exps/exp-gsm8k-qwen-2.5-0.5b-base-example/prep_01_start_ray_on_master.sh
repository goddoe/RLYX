export RLYX_PORT1=5001
export RLYX_PORT2=5002

ray stop && RAY_HEAD_NODE_ONLY=1 ray start --head \
    --port=$RLYX_PORT1 \
    --num-gpus=0 \
    --num-cpus=0 \
    --min-worker-port=13000 \
    --max-worker-port=14000 \
    --dashboard-port=1234 \
    --ray-client-server-port=$RLYX_PORT2 \
    --disable-usage-stats \
    --resources '{"head": 1}'
