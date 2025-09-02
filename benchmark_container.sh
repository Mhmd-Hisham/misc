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

OUTPUT_DIR="/workspace/benchmark_results/${RUN_ID}/${BRANCH}"

# clone repo
rm -rf bitsandbytes
git clone "$REPO_URL" bitsandbytes
cd bitsandbytes
git checkout "$BRANCH"

# improve reproducibility
# https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# move test files
cp /workspace/stress_test.py .
cp /workspace/normal_config.json .
cp /workspace/ncu_config.json .
cp /workspace/inference_benchmark.py .
cp /workspace/functional.py ./bitsandbytes/functional.py # override functional.py to force set the blocksize 

# build for cuda and install
rm -rf build_cuda
cmake -B build_cuda -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=$COMPUTE_CAPABILITY .
cmake --build build_cuda --config Release
python -m pip install -e .

# run official bnb benchmark
cp -f ../inference_benchmark.py ./benchmarking/inference_benchmark.py
python ./benchmarking/inference_benchmark.py \
    "/workspace/models/${MODEL_NAME}" \
    --configs nf4 \
    --batches 1 4 8 \
    --nf4-blocksize 64 \
    --input-length 2048 \
    --output-length 128 \
    --iterations 35 \
    --warmup-runs 10 \
    --out-dir "${OUTPUT_DIR}/${MODEL_NAME}"

# profile the stress test with ncu
# only benchmark kQuantizeBlockwise and kDequantizeBlockwise kernels
mkdir -p "$OUTPUT_DIR"
ncu -f \
    --set full \
    --target-processes all \
    --kernel-name "regex:k(Quantize|Dequantize)Blockwise" \
    --export "${OUTPUT_DIR}/full.ncu-rep" \
    python stress_test.py ncu_config.json "${OUTPUT_DIR}/stress_test_ncu_run.csv" "${OUTPUT_DIR}/stress_test_ncu_metadata.csv"

# export ncu report as csv
ncu --import "${OUTPUT_DIR}/full.ncu-rep" --csv --page raw > "${OUTPUT_DIR}/ncu_rep.csv"

# benchmark with my custom stress test
python stress_test.py normal_config.json "${OUTPUT_DIR}/stress_test_run.csv" "${OUTPUT_DIR}/stress_test_metadata.csv"