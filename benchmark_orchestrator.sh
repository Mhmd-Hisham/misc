#!/bin/bash
# benchmark the performance of multiple branches/forks against the baseline repo of BNB
# each benchmark will be run separately in a docker container
set -e 

OUTPUT_FILE="$1"
COMPUTE_CAPABILITY="$2"

# download the model at the start
MODEL_NAME="Llama-3.2-1B"               # 1B Model
MODEL_NAME="Meta-Llama-3.1-8B-Instruct" # 8B Model
MODEL_NAME="Llama-3.1-70B"              # 70B Model
MODEL_DIR="models/${MODEL_NAME}"
if [ -d "$MODEL_DIR" ]; then
    echo "Model already exists at $MODEL_DIR"
else
    read -rsp "Enter your Hugging Face token: " HF_TOKEN
    echo
    mkdir -p models
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/${MODEL_NAME}',
    local_dir='${MODEL_DIR}',
    token='${HF_TOKEN}'
)
"
    unset HF_TOKEN
fi

DOCKER_IMAGE="mhmdhisham/pytorch-2.8.0-cuda12.9-cudnn9-devel-ncu:testing"
FORK_URL="https://github.com/Mhmd-Hisham/bitsandbytes.git"
BASELINE_URL="https://github.com/bitsandbytes-foundation/bitsandbytes.git"
BASELINE_BRANCH="main"

# pull the docker image
docker pull $DOCKER_IMAGE

# branch list to benchmark
FORK_BRANCHES=(
    "cuda-branchless-dequantization-float32-lut"
)
    # "cuda-slow-dequantization"
    # "cuda-branchless-quantization-float16-lut-bitwise"
    # "cuda-branchless-quantization-float16-lut-bitwise-dequantization-float32-lut"
    # "cuda-branchless-quantization-float32"
    # "cuda-branchless-quantization-float16"
    # "cuda-branchless-quantization-float32-lut"
    # "cuda-branchless-quantization-float16-lut"
    # "cuda-branchless-quantization-float32-lut-bitwise"

mkdir -p benchmark_results
nvidia-smi -pm 1                       # enable persistence mode, stop gpu from powering down when idle
nvidia-smi --auto-boost-default=0      # disable auto boost aka automatic frequency scaling mechanism

# lock gpu clocks and power max values
MAX_GRAPHICS=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | tr -d ' ')
MAX_MEMORY=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | tr -d ' ')
MAX_POWER=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits | tr -d ' ')
echo "Locking graphics clock to ${MAX_GRAPHICS} MHz and memory clock to ${MAX_MEMORY} MHz"
nvidia-smi -lgc ${MAX_GRAPHICS},${MAX_GRAPHICS}
nvidia-smi -lmc ${MAX_MEMORY},${MAX_MEMORY}
echo "Setting power limit to ${MAX_POWER}W"
nvidia-smi -pl ${MAX_POWER}

# disable cpu frequency scaling if possible
if command -v cpupower &> /dev/null; then
    echo "Setting CPU governor to performance mode..."
    sudo cpupower frequency-set -g performance 2>/dev/null || echo "Warning: Could not set CPU governor"
fi

run_benchmark_in_container() {
    local repo_url="$1"
    local branch="$2"
    local docker_image="$3"
    local model_name="$4"
    local run_id="$5"

    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ">>> Running benchmark: ($branch from $repo_url)"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    
    # clear GPU memory and reset state before each benchmark
    echo "Clearing GPU memory..."
    nvidia-smi --gpu-reset || echo "GPU reset not supported, continuing..."
    sleep 5
    
    docker run --user root --rm --gpus all \
        --memory="128g" \
        --memory-swap="128g" \
        --shm-size="32g" \
        -v "$(pwd)/models:/workspace/models:ro" \
        -v "$(pwd)/benchmark_results:/workspace/benchmark_results" \
        -v "$(pwd)/stress_test.py:/workspace/stress_test.py" \
        -v "$(pwd)/normal_config.json:/workspace/normal_config.json" \
        -v "$(pwd)/ncu_config.json:/workspace/ncu_config.json" \
        -v "$(pwd)/inference_benchmark.py:/workspace/inference_benchmark.py" \
        -v "$(pwd)/benchmark_container.sh:/workspace/benchmark_container.sh" \
        -v "$(pwd)/functional.py:/workspace/functional.py" \
        "$docker_image" \
        bash /workspace/benchmark_container.sh "$repo_url" "$branch" "$model_name" "$run_id" "$COMPUTE_CAPABILITY"

    sleep 10
}

for RUN_ID in 1 2 3 4 5; do
    # loop through each branch
    for BRANCH in "${FORK_BRANCHES[@]}"; do
        # benchmark the baseline repo in the container
        run_benchmark_in_container $BASELINE_URL $BASELINE_BRANCH $DOCKER_IMAGE $MODEL_NAME "run_$RUN_ID"

        # benchmark the branch
        run_benchmark_in_container $FORK_URL $BRANCH $DOCKER_IMAGE $MODEL_NAME $RUN_ID
    done
done

nvidia-smi > benchmark_results/nvidia-smi.txt

# zip the results to download with scp
zip -r "$OUTPUT_FILE" "benchmark_results"