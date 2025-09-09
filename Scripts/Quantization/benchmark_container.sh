#!/bin/bash
# benchmark the performance of one branch from bitsandbytes

# set the testing device to be cuda only
export BNB_TEST_DEVICE="cuda"

set -e

# get repo url and branch name
REPO_URL=$1
BRANCH=$2
MODEL_NAME=$3
RUN_ID=$4
COMPUTE_CAPABILITY=$5
COMMIT=$6

OUTPUT_DIR="/workspace/benchmark_results/${RUN_ID}/${BRANCH}"

# clone repo
rm -rf bitsandbytes
git clone "$REPO_URL" bitsandbytes
cd bitsandbytes
git checkout "$BRANCH"
if [ -n "$COMMIT" ]; then
    git checkout "$COMMIT"
fi

# improve reproducibility
# https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# move test files
cp /workspace/model_load_benchmark.py .

# build for cuda and install
rm -rf build_cuda
cmake -B build_cuda -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=$COMPUTE_CAPABILITY .
cmake --build build_cuda --config Release
python -m pip install -e .
cd ..
mv bitsandbytes bitsandbytes_code
python -m pip uninstall bitsandbytes -y
cd bitsandbytes_code    
python -m pip install -e .
cd ..

nvidia-smi

python model_load_benchmark.py "meta-llama/${MODEL_NAME}" \
    --config nf4 \
    --iterations 200 \
    --warmup-runs 10 \
    --device "cuda:0" \
    --tmp-dir "/dev/shm" \
    --out-dir "${OUTPUT_DIR}/${MODEL_NAME}"