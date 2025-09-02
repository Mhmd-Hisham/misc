# Huge thanks to Lawrence Atkins & David MacLeod
# https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch

import json
import random
import sys
from typing import Callable

import numpy as np
import pandas as pd
import torch
from bitsandbytes import functional as F

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# check if bfloat16 is supported
BFLOAT16_SUPPORT = torch.cuda.is_bf16_supported()

def clear_l2_cache(device=0):
    # get cache size from device
    cache_size = torch.cuda.get_device_properties(device).L2_cache_size
    dummy_data = torch.empty(cache_size, dtype=torch.int8, device=f"cuda:{device}")
    dummy_data.zero_()
    torch.cuda.synchronize()
    del dummy_data

def write_to_logger(params, n_times):
    for __ in range(n_times):
        LOGGER.append(params)

def benchmark_cuda_kernel(
    iterations: int, warmup_iterations: int, params_to_log: dict, n_logs: int, kernel: Callable, *args, **kwargs
):
    # warmup iterations
    params_to_log["is_warmup"] = True
    for _ in range(warmup_iterations):
        kernel(*args, **kwargs)
        write_to_logger(params_to_log, 1)
    torch.cuda.synchronize()

    # init cuda events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    clear_l2_cache()
    params_to_log["is_warmup"] = False
    for i in range(iterations):
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        kernel(*args, **kwargs)
        end_events[i].record()
        write_to_logger(params_to_log, 2)
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times


def quantize_dequantize_kernel(A1, blocksize=0, quant_type=0):
    (
        qa,
        SA,
    ) = F.quantize_4bit(A1, blocksize=blocksize, quant_type=quant_type)
    F.dequantize_4bit(qa, SA, blocksize=blocksize, quant_type=quant_type)


def save_metadata(path_to_json):
    df = pd.DataFrame(LOGGER)
    df.to_csv(path_to_json)


def get_stats(prefix, times):
    times = np.array(times)
    return {
        f"{prefix}_total": float(np.sum(times)),
        f"{prefix}_mean": float(np.mean(times)),
        f"{prefix}_median": float(np.median(times)),
        f"{prefix}_std": float(np.std(times)),
        f"{prefix}_min": float(np.min(times)),
        f"{prefix}_max": float(np.max(times)),
        f"{prefix}_p95": float(np.percentile(times, 95)),
        f"{prefix}_count": times.shape[0],
    }


def main():
    results = []
    for device in DEVICE:
        for dtype in DTYPE:
            for quant_type in QUANT_TYPE:
                for blocksize in BLOCKSIZE:
                    for tensor_shape in TENSOR_SHAPE:
                        A1 = torch.randn(tensor_shape, tensor_shape, device=device, dtype=dtype)
                        qa, SA = F.quantize_4bit(A1, blocksize=blocksize, quant_type=quant_type)
                        torch.cuda.synchronize()

                        # for NCU only
                        params_to_log = {
                            "tensor_shape": A1.shape[0],
                            "blocksize": blocksize,
                            "quant_type": quant_type,
                            "dtype": dtype,
                        }

                        print(f"[{device}-{tensor_shape}-{dtype}-{quant_type}-{blocksize}]: ", flush=True, end="")

                        # benchmark "F.quantize_4bit"
                        times = benchmark_cuda_kernel(
                            ITERATIONS,
                            WARMUP_ITER,
                            params_to_log,
                            1,
                            F.quantize_4bit,
                            A1,
                            blocksize=blocksize,
                            quant_type=quant_type,
                        )
                        quantization_stats = get_stats("quantize_latency", times)

                        # benchmark "F.dequantize_4bit"
                        times = benchmark_cuda_kernel(
                            ITERATIONS,
                            WARMUP_ITER,
                            params_to_log,
                            1,
                            F.dequantize_4bit,
                            qa,
                            SA,
                            blocksize=blocksize,
                            quant_type=quant_type,
                        )
                        dequantization_stats = get_stats("dequantize_latency", times)

                        # benchmark "F.quantize_4bit followed by F.dequantize_4bit"
                        # log the parameters two times, one for quantization and one for the dequantization
                        times = benchmark_cuda_kernel(
                            ITERATIONS,
                            WARMUP_ITER,
                            params_to_log,
                            2,
                            quantize_dequantize_kernel,
                            A1,
                            blocksize=blocksize,
                            quant_type=quant_type,
                        )
                        quantization_dequantization_stats = get_stats("quantize_dequantize_latency", times)

                        results.append(
                            {
                                "dtype": dtype,
                                "quant_type": quant_type,
                                "blocksize": blocksize,
                                "tensor_shape": tensor_shape,
                                **quantization_dequantization_stats,
                                **quantization_stats,
                                **dequantization_stats,
                            }
                        )
                        print(
                            {
                                "quant_dequant_latency": quantization_dequantization_stats[
                                    "quantize_dequantize_latency_total"
                                ],
                                "quant_latency": quantization_stats["quantize_latency_total"],
                                "dequant_latency": dequantization_stats["dequantize_latency_total"],
                            }
                        )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    save_metadata(METADATA_CSV)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python stress_test.py <config.json> <output.csv> <metadata.csv>")
        sys.exit()
    torch.cuda.empty_cache()

    CONFIG_FILE = sys.argv[1]
    OUTPUT_CSV = sys.argv[2]
    METADATA_CSV = sys.argv[3]

    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    ITERATIONS = config["ITERATIONS"]
    WARMUP_ITER = config["WARMUP_ITER"]
    DEVICE = config["DEVICE"]
    DTYPE = []
    for dtype in config["DTYPE"]:
        if dtype == "bfloat16" and not BFLOAT16_SUPPORT:
            continue
        DTYPE.append(getattr(torch, dtype))
    QUANT_TYPE = config["QUANT_TYPE"]
    BLOCKSIZE = config["BLOCKSIZE"]
    TENSOR_SHAPE = config["TENSOR_SHAPE"]

    # metadata logger, useful when benchmarking with ncu
    LOGGER = []

    main()
