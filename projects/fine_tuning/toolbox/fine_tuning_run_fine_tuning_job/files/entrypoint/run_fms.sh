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
    python -m torch.distributed.run \
     --nproc_per_node=8 \
     --nnodes=1 \
     --node_rank=0 \
     --master_addr=localhost \
     --master_port=29500 \
     --module \
     tuning.sft_trainer \
     --fsdp_auto_wrap_policy="HYBRID_SHARD" \
     --fsdp_backward_prefetch="BACKWARD_POST" \
     --fsdp_forward_prefetch=True \
     --fsdp_offload_params=False \
     --fsdp_state_dict_type="SHARDED_STATE_DICT" \
     --fsdp_sync_module_states=True \
     --fsdp_use_orig_params=False
    exit 0
fi
