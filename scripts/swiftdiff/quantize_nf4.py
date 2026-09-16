"""One-time: load base Qwen3.8-27B BF16 with bitsandbytes NF4 and save the 4-bit checkpoint.

Later forward passes and LoRA training load this copy instead of re-reading and re-quantising 52 GB.
Resource limits (lowres.limit): background priority, 6 CPU threads, GPU memory capped. The unquantised
input embedding and lm_head (~5 GB BF16) stay in system RAM so the GPU holds only the 4-bit decoder.

    python scripts/swiftdiff/quantize_nf4.py --src A:/models/Qwen3.8-27B-base-bf16 --dst A:/models/Qwen3.8-27B-base-nf4
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowres import limit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="A:/models/Qwen3.8-27B-base-bf16")
ap.add_argument("--dst", default="A:/models/Qwen3.8-27B-base-nf4")
ap.add_argument("--gpu-frac", type=float, default=0.6)
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()
limit(a.gpu_frac, a.threads)
t0 = time.time()
device_map = {"model.embed_tokens": "cpu", "lm_head": "cpu", "model.layers": 0, "model.norm": 0, "model.rotary_emb": 0}
model = AutoModelForCausalLM.from_pretrained(
    a.src, device_map=device_map, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                           llm_int8_enable_fp32_cpu_offload=True,
                                           llm_int8_skip_modules=["lm_head", "embed_tokens"]))
print(f"quantised in {time.time() - t0:.0f}s | GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB "
      f"(peak {torch.cuda.max_memory_allocated() / 2**30:.1f})", flush=True)
model.save_pretrained(a.dst, max_shard_size="4GB")
AutoTokenizer.from_pretrained(a.src).save_pretrained(a.dst)
print(f"saved to {a.dst} in {time.time() - t0:.0f}s", flush=True)
