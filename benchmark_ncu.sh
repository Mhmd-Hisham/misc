#!/bin/bash
# benchmark with ncu

# Metrics explained:
#     gpu__time_duration_measured_user,`                         #  context-aware time duration in nanoseconds
#     sm__throughput.avg.pct_of_peak_sustained_elapsed,`         # avg streaming multiprocessor throughput as a percentage of peak sustained
#     dram__throughput.avg.pct_of_peak_sustained_elapsed,`       # avg memory throughput as a percentage of peak sustained
#     smsp__sass_branch_instructions_executed.sum,`              # total number of branch instructions executed.
#     smsp__sass_branch_targets_threads_divergent.sum,`          # number of threads that took divergent branches
#     sm__achieved_occupancy.avg.pct_of_peak_sustained_active,`  # actual occupancy achieved
#     inst_executed.sum,`                                        # total instructions executed
#     sm__warps_eligible_per_cycle.avg, `                        # warps eligible per cycle
run_ncu() {
    # define the metrics to include in the ncu report
    local metrics="gpu__time_duration_measured_user,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_branch_instructions_executed.sum,smsp__sass_branch_targets_threads_divergent.sum,sm__achieved_occupancy.avg.pct_of_peak_sustained_active,inst_executed.sum,sm__warps_eligible_per_cycle.avg"

    # take test from function call first arg
    # baseline or improved
    local test_type="$1"
    local test_name="${test_type}_bnb_kernel_benchmark"
    local kernels_regex="regex:k(Quantize|Dequantize)Blockwise"

    # profile the stress test with ncu
    ncu -f \
        --target-processes all \
        --kernel-name "${kernels_regex}" \
        --metrics "${metrics}" \
        --export "${test_name}.ncu-rep" \
        python stress_test.py "../benchmark_results/${test_type}_ncu_run.csv"

    # export the csv
    ncu --import "${test_name}.ncu-rep" --csv --page raw > "${test_name}.csv"
}


rm -rf bnb-benchmark
mkdir bnb-benchmark
cd bnb-benchmark

# the branch to benchmark
benchmark_branch="cuda-branchless-binary-search"

# set the testing device to be cuda only
export BNB_TEST_DEVICE="cuda"
export CUDA_LAUNCH_BLOCKING=1

mkdir benchmark_results

# clone my fork
git clone https://github.com/Mhmd-Hisham/bitsandbytes.git
cd bitsandbytes
git checkout "${benchmark_branch}"

# move stress test to the fork
cp ../../stress_test.py .

# build for cuda
rm -rf build_cuda
cmake -B build_cuda -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=90 .
cmake --build build_cuda --config Release

nvidia-smi -pm 1                       # enable persistence mode, stop gpu from powering down when idle
nvidia-smi --auto-boost-default=0      # disable auto boost aka automatic frequency scaling mechanism
nvidia-smi -c EXCLUSIVE_PROCESS        # restrict to only one process can create a cuda context on the GPU at any given time

# run nvidia-smi -q -d SUPPORTED_CLOCKS to know the possible ranges for your gpu
# nvidia-smi -lgc 2100,2100              # set min and max graphics freq in MHz
# nvidia-smi -lmc 5001                   # set memory freq in MHz

run_ncu "improved"
# copy ncu-rep and extracted .csv
cp "improved_bnb_kernel_benchmark.ncu-rep" "../benchmark_results/"
cp "improved_bnb_kernel_benchmark.csv" "../benchmark_results/"

# chdir and rename the fork
cd ..
mv bitsandbytes bitsandbytes_fork

# clone bnb original repo
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes

# copy the stress test to the baseline repo
cp ../../stress_test.py .

# build for cuda
rm -rf build_cuda
cmake -B build_cuda -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=90 .
cmake --build build_cuda --config Release

# benchmark the baseline repo
run_ncu "baseline"
cp "baseline_bnb_kernel_benchmark.ncu-rep" "../benchmark_results/"
cp "baseline_bnb_kernel_benchmark.csv" "../benchmark_results/"

mv stress_test_metadata.csv "../benchmark_results/"
mv stress_test.py "../benchmark_results/"

cd ..

# zip the results to download with scp
zip -r benchmark_results.zip benchmark_results
