#!/bin/bash
# this script is used to benchmark the performance from a branch in my fork against the baseline repo of BNB
rm -rf bnb-benchmark
mkdir bnb-benchmark
cd bnb-benchmark

# set the testing device to be cuda only
export BNB_TEST_DEVICE="cuda"
export CUDA_LAUNCH_BLOCKING=1
mkdir benchmark_results

nvidia-smi -pm 1                       # enable persistence mode, stop gpu from powering down when idle
nvidia-smi --auto-boost-default=0      # disable auto boost aka automatic frequency scaling mechanism
nvidia-smi -c EXCLUSIVE_PROCESS        # restrict to only one process can create a cuda context on the GPU at any given time

# run nvidia-smi -q -d SUPPORTED_CLOCKS to know the possible ranges for your gpu
# nvidia-smi -lgc 2100,2100              # set min and max graphics freq in MHz
# nvidia-smi -lmc 5001                   # set memory freq in MHz

# get compute capability from nvidia-smi
# capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
capability="90"
benchmark_repo() {
    # take test from function call first arg
    # baseline or improved or branch name
    local test_type="$1"
    local output_dir="../benchmark_results/${test_type}_results"
    local test_name="${test_type}_bnb_kernel"

    # build for cuda
    rm -rf build_cuda
    cmake -B build_cuda -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=$capability .
    cmake --build build_cuda --config Release

    python -m pip install -e .

    # run official bnb benchmark
    python ./benchmarking/inference_benchmark.py "meta-llama/Meta-Llama-3.1-8B-Instruct" \
            --configs int8 nf4 nf4-dq \
            --out-dir "${output_dir}/Llamma-3.1-8B-Instruct"

    # profile the stress test with ncu
    # ncu -f \
    #     --set full \
    #     --target-processes all \
    #     --kernel-name "regex:k(Quantize|Dequantize)Blockwise" \
    #     --export "${output_dir}/${test_name}.ncu-rep" \
    #     python stress_test.py ncu_config.json "${output_dir}/${test_type}_ncu_run.csv" "${output_dir}/${test_type}_ncu_metadata.csv"

    # # export the csv
    # ncu --import "${output_dir}/${test_name}.ncu-rep" --csv --page raw > "${output_dir}/${test_name}.csv"

    # # benchmark with my custom stress test
    # python stress_test.py normal_config.json "${output_dir}/${test_type}_run.csv" "${output_dir}/${test_type}_metadata.csv"
    python -m pip uninstall bitsandbytes -y
}

# clone bnb original repo
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git

# copy the stress test files to the baseline repo
cp ../normal_config.json ../ncu_config.json ../stress_test.py bitsandbytes/

# benchmark
cd bitsandbytes
benchmark_repo "baseline"

# get back and rename the baseline repo
cd ..
mv bitsandbytes bitsandbytes_baseline

# clone my fork
git clone https://github.com/Mhmd-Hisham/bitsandbytes.git

# branch list to benchmark
benchmark_branches=(
    "cuda-branchless-quantization-float32"
    "cuda-branchless-quantization-float16"
    "cuda-branchless-quantization-float32-lut"
    "cuda-branchless-quantization-float16-lut"
    "cuda-branchless-dequantization-float32-lut"
)

cd bitsandbytes

# loop through each path
for branch in "${benchmark_branches[@]}"; do
    echo "Processing: $branch"

    # checkout the branch
    git checkout "$branch"
    echo "$(pwd)"
    # copy the stress test files to the branch
    cp ../../normal_config.json ../../ncu_config.json ../../stress_test.py .

    benchmark_repo "$branch"

    # reset so we can switch to a new branch
    git reset --hard
    git clean -fdx

done

# move back from the fork
cd ..

nvcc --version > benchmark_results/nvcc.txt
nvidia-smi > benchmark_results/nvidia-smi.txt

# zip the results to download with scp
zip -r h100_nebius_benchmark_results.zip benchmark_results