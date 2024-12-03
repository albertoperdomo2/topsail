# !/bin/bash

set -o pipefail
set -o errexit
set -o nounset
set -o errtrace
set -x

cd /mnt/app

echo "# configuration:"
cat "$CONFIG_JSON_PATH"

if python3 ./test_network_overhead.py; then
    echo "SCRIPT SUCCEEDED"
else
    echo "SCRIPT FAILED"
    # don't exit with a return code != 0, otherwise the RayJob->Job retries 3 times ...
fi

# need to figure our how to parse config hyperparams into this
if python3 ./test_torch_benchmark.py run --num-runs 3 --num-epochs 20 --num-workers 4 --cpus-per-worker 8; then
    echo "TORCH SCRIPT SUCCEEDED"
else
    echo "TORCH SCRIPT FAILED"
fi

set +x
echo "*********"
echo "*********"
echo "*********"
echo "*********"
echo "********* Bye"
