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
     --fsdp_auto_wrap_policy="TRANSFORMER_BASED_WRAP" \
     --fsdp_backward_prefetch="BACKWARD_PRE" \
     --fsdp_forward_prefetch="false" \
     --fsdp_offload_params="false" \
     --fsdp_sharding_strategy=1 \
     --fsdp_state_dict_type="FULL_STATE_DICT" \
     --fsdp_cpu_ram_efficient_loading="true" \
     --fsdp_sync_module_states="true" \
     --rdzv_backend="static" \
     --same_network \
     --machine_rank=0 \
     --dynamo_backend=no \
     --module \
     tuning.sft_trainer
    exit 0
fi
