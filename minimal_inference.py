# pip install -U sentencepiece transformers protobuf bitsandbytes accelerate
import sys
import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.cuda.nvtx as nvtx

MODEL_NAME = "arnir0/Tiny-LLM"
MODEL_NAME = "meta-llama/Llama-3.2-1B"

compute_dtype = torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
nvtx.range_push("MODEL_LOAD")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=False,
)
nvtx.range_pop()
print("Model loaded..")

@torch.inference_mode()
def prefill(prompt: str, kwargs: Dict[str, Any]) -> Dict:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    assert kwargs.get("max_new_tokens") == kwargs.get("min_new_tokens") == 1, (
        "For prefilling, max_new_tokens and min_new_tokens must be equal to 1"
    )
    nvtx.range_push("PREFILL")
    outputs = model.generate(**inputs, **kwargs)
    nvtx.range_pop()
    return {"outputs": outputs, "inputs": inputs}

@torch.inference_mode()
def generate(prefilled: Dict[str, Any], kwargs: Dict[str, Any]) -> list[str]:
    inputs = prefilled["inputs"]
    nvtx.range_push("GENERATE")
    outputs = model.generate(**inputs, **kwargs)
    nvtx.range_pop()
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def add_nvtx_label(label):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            nvtx.range_push(label)
            try:
                return fn(*args, **kwargs)
            finally:
                nvtx.range_pop()
        return wrapper
    return decorator

def main(batch_size):
    # prefill call
    prompts = ["According to all known laws of aviation, there is no way a bee should be able to fly."] * batch_size
    prefill_kwargs = {"max_new_tokens": 1, "min_new_tokens": 1}
    prefilled = prefill(prompts, prefill_kwargs)
    print("Prefill completed...")
    # decode call
    gen_kwargs = {"max_new_tokens": 1}
    responses = generate(prefilled, gen_kwargs)
    print(">>>>>>>>>>>>>>")
    print("Output:")
    print('\n=======\n'.join(responses))
    print(">>>>>>>>>>>>>>")
    print("Decode completed...")

if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    main(batch_size)
