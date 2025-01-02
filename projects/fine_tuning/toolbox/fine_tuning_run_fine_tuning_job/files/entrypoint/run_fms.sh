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
    python -m accelerate.commands.launch \
     --num_processes=8 \
     --num_machines=1 \
     --mixed_precision=no \
     --use_fsdp \
     --fsdp_auto_wrap_policy="HYBRID_SHARD" \
     --fsdp_forward_prefetch="BACKWARD_POST" \
     --fsdp_offload_params="false" \
     --fsdp_state_dict_type="SHARDED_STATE_DICT" \
     --fsdp_sync_module_states="true" \
     --fsdp_use_orig_params="false" \
     --rdzv_backend="static" \
     --same_network \
     --machine_rank=0 \
     --dynamo_backend=no \
     --module \
     tuning.sft_trainer
    exit 0
fi
