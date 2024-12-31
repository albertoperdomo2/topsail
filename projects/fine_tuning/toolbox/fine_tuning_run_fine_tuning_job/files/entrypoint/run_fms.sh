# !/bin/bash

set -o pipefail
set -o errexit
set -o nounset
set -o errtrace
set -x

export SFT_TRAINER_CONFIG_JSON_PATH=$CONFIG_JSON_PATH

if [[ $WORLD_SIZE == 1 ]]; then
    echo "Running on a single machine."

    if [[ -z "${NUM_GPUS:-1}" || "${NUM_GPUS:-1}" == 1 ]]; then
        echo "Running with a single GPU"
    else
        echo "Running with a $NUM_GPUS GPUs"
    fi
    time python -m torch.distributed.run \
        --node_rank "$RANK" \
        --nnodes "$WORLD_SIZE" \
        --nproc_per_node "$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l) " \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        launch_training.py
    exit 0
fi
echo "Running on $WORLD_SIZE machines with $NUM_GPUS GPUs each."

time python -m torch.distributed.run \
     --node_rank "$RANK" \
     --nnodes "$WORLD_SIZE" \
     --nproc_per_node "$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l) " \
     --master_addr "$MASTER_ADDR" \
     --master_port "$MASTER_PORT" \
     launch_training.py

# --mixed_precision no --> disabled by default in fsdp
# --dynamo_backend no --> torch.compile disabled by default in fsdp
