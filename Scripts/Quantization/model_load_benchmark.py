import sys
import gc
import os
import random
import shutil
import argparse
import time
import tempfile
from typing import Callable
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import snapshot_download

BFLOAT16_SUPPORT = torch.cuda.get_device_capability()[0] >= 8

WEIGHTS_CONFIGS = {
    "fp16": {"torch_dtype": "float16", "quantization_config": {}},
    "bf16": {"torch_dtype": "bfloat16", "quantization_config": {}},
    "nf4": {
        "torch_dtype": "bfloat16" if BFLOAT16_SUPPORT else "float16",
        "quantization_config": {
            "quant_method": "bnb",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_compute_dtype": torch.bfloat16 if BFLOAT16_SUPPORT else "float16",
        },
    },
    "nf4-dq": {
        "torch_dtype": "bfloat16" if BFLOAT16_SUPPORT else "float16",
        "quantization_config": {
            "quant_method": "bnb",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.bfloat16 if BFLOAT16_SUPPORT else "float16",
        },
    },
    "int8-decomp": {
        "torch_dtype": "float16",
        "quantization_config": {
            "quant_method": "bnb",
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
        },
    },
    "int8": {
        "torch_dtype": "float16",
        "quantization_config": {
            "quant_method": "bnb",
            "load_in_8bit": True,
            "llm_int8_threshold": 0.0,
        },
    },
}

def clear_memory(device):
    # get cache size from device
    torch.cuda.empty_cache()
    free_mem = torch.cuda.mem_get_info(device)[0]
    # use 70% of memory, 4 bytes per float32
    cache_flush_size = int(free_mem * 0.7 // 4)
    if cache_flush_size <= 0:
        return

    try:
        dummy_data = torch.empty(cache_flush_size, dtype=torch.float32, device=device)
        dummy_data.zero_()
        torch.cuda.synchronize()
        del dummy_data
    except Exception as e:
        print(f"Warning: cache flush failed: {e}")
    gc.collect()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TRANSFORMERS_SEED'] = str(seed)
    
    # deterministic settings
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def benchmark_function(
    device: str, iterations: int, warmup_iterations: int, function: Callable, *args, **kwargs
):
    print(f"Warming up for {warmup_iterations} iterations...")
    # warmup iterations
    for _ in range(warmup_iterations):
        function(*args, **kwargs)
    torch.cuda.synchronize()

    # init cuda events and perf counters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    perf_start_events = []
    perf_end_events = []
    print(f"Running benchmark for {iterations} iterations...")
    for i in range(iterations):
        if (i+1) % 5 == 0 or i == 0:
            print(f"Iteration {i+1}/{iterations}", flush=True)
        set_seed()
        clear_memory(device)
        torch.cuda.synchronize()
        start_events[i].record()
        perf_start_events.append(time.perf_counter())

        data = function(*args, **kwargs)

        end_events[i].record()
        perf_end_events.append(time.perf_counter())
        torch.cuda.synchronize()
        del data
    print("Benchmarking complete.")
    times = [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    perf_times = [e-s for s, e in zip(perf_start_events, perf_end_events)]
    return times, perf_times

def get_stats(prefix: str, times: list):
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
        f"{prefix}_values": times.tolist(),
    }

def load_and_quantize_model(model_path, config, device):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        local_files_only=True,
        **WEIGHTS_CONFIGS[config]
    )

def parse_args():
    parser = argparse.ArgumentParser(description="bitsandbytes end-to-end model load benchmark tool")

    parser.add_argument("model_id", type=str, help="The model checkpoint to use.")

    parser.add_argument(
        "--config",
        choices=["bf16", "fp16", "nf4", "nf4-dq", "int8", "int8-decomp"],
        default="nf4",
    )

    parser.add_argument("--out-dir", type=str, default="reports")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations for each benchmark run")
    parser.add_argument("--warmup-runs", type=int, default=10, help="Number of warmup runs to discard before measurement")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for benchmarking")
    parser.add_argument("--download-path", type=str, default="models", help="Default path to download models to")
    parser.add_argument("--tmp-dir", type=str, default="/dev/shm", help="Temporary directory to use for RAM disk")

    return parser.parse_args()


def main(args):
    # download the model at first
    os.makedirs(args.download_path, exist_ok=True)
    if not os.path.exists(os.path.join(args.download_path, args.model_id.split("/")[-1])):
        cache_dir = snapshot_download(
            repo_id=args.model_id,
            local_dir=args.download_path,
            token=os.getenv("HF_HUB_TOKEN", None),
        )
    else:
        cache_dir = os.path.join(args.download_path, args.model_id.split("/")[-1])
        print(f"Model already exists at {cache_dir}, reusing...")

    # load the entire model into memory
    # you MUST choose a model that can fit into memory with full precision
    temp_dir = tempfile.mkdtemp(dir=args.tmp_dir)
    print("Loading full precision model into memory...")
    ramdisk_model_path = os.path.join(temp_dir, "model")
    if not os.path.exists(ramdisk_model_path):
        shutil.copytree(cache_dir, ramdisk_model_path)
        print(f"Model copied to {ramdisk_model_path}")
    else:
        print("Model already in RAM, reusing...")

    print("Starting benchmark...")
    # benchmark quantization and in-memory loading
    times, perf_times = benchmark_function(
        args.device,
        args.iterations,
        args.warmup_runs,
        load_and_quantize_model,
        ramdisk_model_path,
        args.config,
        args.device
    )

    cuda_stats = get_stats("cuda_event", times)
    perf_stats = get_stats("perf_counter", perf_times)

    results = {
        "model_name": args.model_id,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup_runs,
        "quantization_config": str(WEIGHTS_CONFIGS[args.config]),
        "cuda_event_times": cuda_stats,
        "perf_counter_times": perf_stats,
    }

    OUTPUT_DIR = args.out_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_JSON = f"{OUTPUT_DIR}/load_model_benchmark.json"

    with open(OUTPUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    
    print(f"Benchmark results saved to {OUTPUT_JSON}")
    print(f"Average load time (perf_counter): {perf_stats['perf_counter_mean']:.2f}s")
    print(f"Average load time (cuda_events): {cuda_stats['cuda_event_mean']:.2f}s")
    print("Cleaning up RAM copy...")
    shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    if sys.platform != "linux":
        print("This benchmark script only works on Linux.")
        sys.exit(1)

    args = parse_args()
    main(args)