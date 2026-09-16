"""Export UkisAI Swift's weight change as clean rank-32 PEFT LoRA adapters (full and subsets).

Source: rank-64 SVD factors of delta = W_swift - W_base from diff_swift.py. The singular spectrum drops ~10x
after index 32 (merge noise floor beyond), so the top 32 components are the original adapter.
LoRA convention: delta = B @ A * (alpha / r); with alpha = r, B = U sqrt(S), A = sqrt(S) V^T.

Variants (name -> tensor filter):
  full, mlp (gate/up/down), attn (q/k/v/o), early (layers 0-31), late (layers 32-63)

    python scripts/swiftdiff/export_swift_lora.py --out runs/swiftdiff/adapters
"""
import argparse
import glob
import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

VARIANTS = {
    "full": lambda mod, layer: True,
    "mlp": lambda mod, layer: mod.startswith("mlp."),
    "attn": lambda mod, layer: mod.startswith("self_attn."),
    "early": lambda mod, layer: layer < 32,
    "late": lambda mod, layer: layer >= 32,
}

ap = argparse.ArgumentParser()
ap.add_argument("--factors", default="A:/swift/runs/swiftdiff/factors")
ap.add_argument("--out", default="A:/swift/runs/swiftdiff/adapters")
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--base", default="A:/models/Qwen3.8-27B-base-bf16")
a = ap.parse_args()

files = sorted(glob.glob(f"{a.factors}/*.pt"))
loaded = []
for f in files:
    name = Path(f).name[:-3]
    m = re.search(r"layers\.(\d+)\.(.+)\.weight$", name)
    if not m or not name.startswith("model.language_model."):
        continue
    fac = torch.load(f)
    U, S, V = fac["U"].float(), fac["S"].float(), fac["V"].float()
    r = a.rank
    sq = S[:r].sqrt()
    B = (U[:, :r] * sq).contiguous()
    A = (sq[:, None] * V[:, :r].T).contiguous()
    loaded.append((name, int(m.group(1)), m.group(2), A, B))
print(f"{len(loaded)} adapted matrices")

for variant, keep in VARIANTS.items():
    tensors, targets = {}, set()
    for name, layer, mod, A, B in loaded:
        if not keep(mod, layer):
            continue
        key = "base_model.model." + name[: -len(".weight")]
        tensors[f"{key}.lora_A.weight"] = A.to(torch.bfloat16)
        tensors[f"{key}.lora_B.weight"] = B.to(torch.bfloat16)
        targets.add(mod.split(".")[-1])
    d = Path(a.out) / variant
    d.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(d / "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": a.base, "r": a.rank,
               "lora_alpha": a.rank, "lora_dropout": 0.0, "bias": "none", "target_modules": sorted(targets),
               "fan_in_fan_out": False, "inference_mode": True},
              open(d / "adapter_config.json", "w"), indent=1)
    print(f"{variant}: {len(tensors) // 2} matrices, targets {sorted(targets)}")
