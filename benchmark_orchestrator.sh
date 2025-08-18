#!/bin/bash
# benchmark the performance of multiple branches/forks against the baseline repo of BNB
# each benchmark will be run separately in a docker container
set -e 

# download Meta-Llama-3.1-8B-Instruct at the start
read -rsp "Enter your Hugging Face token: " HF_TOKEN
echo

mkdir -p models

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct',
    local_dir='models/Meta-Llama-3.1-8B-Instruct',
    token='${HF_TOKEN}'
)
"
unset HF_TOKEN

DOCKER_IMAGE="mhmdhisham/pytorch-2.8.0-cuda12.9-cudnn9-devel-ncu:testing"
FORK_URL="https://github.com/Mhmd-Hisham/bitsandbytes.git"
BASELINE_URL="https://github.com/bitsandbytes-foundation/bitsandbytes.git"
BASELINE_BRANCH="main"

# pull the docker image
docker pull $DOCKER_IMAGE

# branch list to benchmark
FORK_BRANCHES=(
    "cuda-branchless-quantization-float32"
    "cuda-branchless-quantization-float16"
    "cuda-branchless-quantization-float32-lut"
    "cuda-branchless-quantization-float16-lut"
    "cuda-branchless-quantization-float32-lut-bitwise"
    "cuda-branchless-quantization-float16-lut-bitwise"
    "cuda-branchless-dequantization-float32-lut"
)

mkdir -p benchmark_results
nvidia-smi -pm 1                       # enable persistence mode, stop gpu from powering down when idle
nvidia-smi --auto-boost-default=0      # disable auto boost aka automatic frequency scaling mechanism

run_benchmark_in_container() {
    local repo_url="$1"
    local branch="$2"
    local docker_image="$3"

    echo ">>> Running benchmark: ($branch from $repo_url)"

    docker run --user root --rm --gpus all \
        -v "$(pwd)/models:/workspace/models:ro" \
        -v "$(pwd)/benchmark_results:/workspace/benchmark_results" \
        -v "$(pwd)/stress_test.py:/workspace/stress_test.py" \
        -v "$(pwd)/normal_config.json:/workspace/normal_config.json" \
        -v "$(pwd)/ncu_config.json:/workspace/ncu_config.json" \
        -v "$(pwd)/benchmark_container.sh:/workspace/benchmark_container.sh" \
        "$docker_image" \
        bash /workspace/benchmark_container.sh "$repo_url" "$branch"
}

# benchmark the baseline repo in the container
run_benchmark_in_container $BASELINE_URL $BASELINE_BRANCH $DOCKER_IMAGE

# loop through each path
for BRANCH in "${FORK_BRANCHES[@]}"; do
    run_benchmark_in_container $FORK_URL $BRANCH $DOCKER_IMAGE
done

nvcc --version > benchmark_results/nvcc.txt
nvidia-smi > benchmark_results/nvidia-smi.txt

# install zip in case it is not installed
sudo apt update && sudo apt install zip

# zip the results to download with scp
zip -r h100_nebius_benchmark_results.zip benchmark_results