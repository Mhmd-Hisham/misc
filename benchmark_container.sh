#!/bin/bash
# benchmark the performance of one branch from bitsandbytes

# set the testing device to be cuda only
export BNB_TEST_DEVICE="cuda"

set -e
sudo su

# get repo url and branch name
REPO_URL=$1
BRANCH=$2

OUTPUT_DIR="/workspace/benchmark_results/${BRANCH}_results"

# clone repo
rm -rf bitsandbytes
git clone "$REPO_URL" bitsandbytes
cd bitsandbytes
git checkout "$BRANCH"

# build for cuda and install
# get compute capability pytorch to avoid compiling for a different compute capability
capability=$(python -c "import torch; print('{}.{}'.format(*torch.cuda.get_device_capability()))")
rm -rf build_cuda
cmake -B build_cuda -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=$capability .
cmake --build build_cuda --config Release
python -m pip install -e .

# run official bnb benchmark
python ./benchmarking/inference_benchmark.py \
    "/workspace/models/Meta-Llama-3.1-8B-Instruct" \
    --configs int8 nf4 nf4-dq \
    --out-dir "${OUTPUT_DIR}/Llama-3.1-8B-Instruct"

# profile the stress test with ncu
# only benchmark kQuantizeBlockwise and kDequantizeBlockwise kernels
ncu -f \
    --set full \
    --target-processes all \
    --kernel-name "regex:k(Quantize|Dequantize)Blockwise" \
    --export "${OUTPUT_DIR}/${BRANCH}.ncu-rep" \
    python stress_test.py ncu_config.json "${OUTPUT_DIR}/stress_test_ncu_run.csv" "${OUTPUT_DIR}/stress_test_ncu_metadata.csv"

# export ncu report as csv
ncu --import "${OUTPUT_DIR}/${BRANCH}.ncu-rep" --csv --page raw > "${OUTPUT_DIR}/ncu_rep.csv"

# benchmark with my custom stress test
python stress_test.py normal_config.json "${OUTPUT_DIR}/stress_test_run.csv" "${OUTPUT_DIR}/stress_test_metadata.csv"