"""
Inference benchmarking tool.

Requirements:
    transformers
    accelerate
    bitsandbytes
    optimum-benchmark

Usage: python inference_benchmark.py model_id

options:
    -h, --help            show this help message and exit
    --configs {bf16,fp16,nf4,nf4-dq,int8,int8-decomp} [{bf16,fp16,nf4,nf4-dq,int8,int8-decomp} ...]
    --bf16
    --fp16
    --nf4
    --nf4-dq
    --int8
    --int8-decomp
    --batches BATCHES [BATCHES ...]
    --input-length INPUT_LENGTH
    --out-dir OUT_DIR
    --seed SEED
    --iterations ITERATIONS
    --warmup-runs WARMUP_RUNS
    --nf4-blocksize [NF4_QUANTIZATION_BLOCKSIZE ..]
"""
import gc
import random
import os
import numpy as np
import argparse
from pathlib import Path

# disable "per_token" logs from pytorch backend, its slow with large number of tokens
from optimum_benchmark.scenarios.inference.scenario import PER_TOKEN_BACKENDS
if "pytorch" in PER_TOKEN_BACKENDS:
    PER_TOKEN_BACKENDS.remove("pytorch") 

from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, ProcessConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def clear_memory(device=0):
    # get cache size from device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        cache_size = torch.cuda.get_device_properties(device).L2_cache_size
        dummy_data = torch.empty(cache_size, dtype=torch.int8, device=f"cuda:{device}")
        dummy_data.zero_()
        torch.cuda.synchronize()
        del dummy_data
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
    

BFLOAT16_SUPPORT = torch.cuda.get_device_capability()[0] >= 8

WEIGHTS_CONFIGS = {
    "fp16": {"torch_dtype": "float16", "quantization_scheme": None, "quantization_config": {}},
    "bf16": {"torch_dtype": "bfloat16", "quantization_scheme": None, "quantization_config": {}},
    "nf4": {
        "torch_dtype": "bfloat16" if BFLOAT16_SUPPORT else "float16",
        "quantization_scheme": "bnb",
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_compute_dtype": torch.bfloat16 if BFLOAT16_SUPPORT else "float16",
        },
    },
    "nf4-dq": {
        "torch_dtype": "bfloat16" if BFLOAT16_SUPPORT else "float16",
        "quantization_scheme": "bnb",
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.bfloat16 if BFLOAT16_SUPPORT else "float16",
        },
    },
    "int8-decomp": {
        "torch_dtype": "float16",
        "quantization_scheme": "bnb",
        "quantization_config": {
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
        },
    },
    "int8": {
        "torch_dtype": "float16",
        "quantization_scheme": "bnb",
        "quantization_config": {
            "load_in_8bit": True,
            "llm_int8_threshold": 0.0,
        },
    },
}

def run_benchmark(args, config, batch_size, nf4_blocksize=None):
    print(f"[config={config}, batch_size={batch_size}, nf4_blocksize={nf4_blocksize}]")
    set_seed(args.seed)
    clear_memory()

    if nf4_blocksize:
        os.environ["BNB_BLOCKSIZE"] = str(nf4_blocksize)

    launcher_config = ProcessConfig(device_isolation=True, device_isolation_action="kill", start_method="spawn")
    scenario_config = InferenceConfig(
        latency=True,
        memory=False,
        input_shapes={"batch_size": batch_size, "sequence_length": args.input_length},
        iterations=args.iterations,
        warmup_runs=args.warmup_runs,
        duration=0,
    )
    backend_config = PyTorchConfig(
        device="cuda",
        device_ids="0",
        device_map="auto",
        no_weights=False,
        model=args.model_id,
        **WEIGHTS_CONFIGS[config],
    )
    benchmark_config = BenchmarkConfig(
        name=f"benchmark-{config}-bsz{batch_size}",
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend_config,
    )

    out_path = out_dir / f"benchmark_{config}_bsz{batch_size}.json"

    benchmark_report = Benchmark.launch(benchmark_config)
    benchmark_report.save_json(out_path)

if __name__ == "__main__":
    setup_logging(level="INFO")

    parser = argparse.ArgumentParser(description="bitsandbytes inference benchmark tool")

    parser.add_argument("model_id", type=str, help="The model checkpoint to use.")

    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["bf16", "fp16", "nf4", "nf4-dq", "int8", "int8-decomp"],
        default=["nf4", "int8", "int8-decomp"],
    )
    parser.add_argument("--bf16", dest="configs", action="append_const", const="bf16")
    parser.add_argument("--fp16", dest="configs", action="append_const", const="fp16")
    parser.add_argument("--nf4", dest="configs", action="append_const", const="nf4")
    parser.add_argument("--nf4-dq", dest="configs", action="append_const", const="nf4-dq")
    parser.add_argument("--int8", dest="configs", action="append_const", const="int8")
    parser.add_argument("--int8-decomp", dest="configs", action="append_const", const="int8-decomp")

    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 16, 32])
    parser.add_argument("--input-length", type=int, default=64)

    parser.add_argument("--out-dir", type=str, default="reports")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for each benchmark run")
    parser.add_argument("--warmup-runs", type=int, default=10, help="Number of warmup runs to discard before measurement")
    parser.add_argument("--nf4-blocksize", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048], help="NF4 quantization block size")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for batch_size in args.batches:
        print(f"Benchmarking batch size: {batch_size}")
        for config in args.configs:
            if "nf4" in config:
                for blocksize in args.nf4_blocksize:
                    run_benchmark(args, config, batch_size, nf4_blocksize=blocksize)
            else:
                run_benchmark(args, config, batch_size)
