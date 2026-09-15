"""Task-vector transfer ("adapter chunks"): move a behaviour delta from a donor fine-tune.

   delta = theta_donor - theta_donor_base          (e.g. ThinkingCap-Qwen3.6-27B minus Qwen3.6-27B)
   delta = trim_topk(delta, keep)                  (TIES-style: keep the largest |delta| entries per tensor)
   theta_target += alpha * delta                   (only for tensors whose shape matches, optional name filter)

Same-architecture transfer (3.6 -> 3.8 within a family) is what Swift's authors describe
as using "ThinkingCap adapter chunks".  Always re-run the paired eval afterwards; a task
vector can move accuracy either way.
"""
from __future__ import annotations
import json, re, shutil
from pathlib import Path


def trim_topk(delta, keep: float):
    import torch
    if keep >= 1.0:
        return delta
    k = max(1, int(delta.numel() * keep))
    thr = delta.abs().flatten().kthvalue(delta.numel() - k + 1).values
    return delta * (delta.abs() >= thr)


def transfer(target_dir: str, donor_dir: str, donor_base_dir: str, out_dir: str, alpha: float = 0.5, keep: float = 0.2,
             only: str | None = None, exclude: str = r"(embed_tokens|lm_head)") -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    target_dir, out_dir = Path(target_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    donor = _index(donor_dir); base = _index(donor_base_dir)
    only_re = re.compile(only) if only else None
    excl_re = re.compile(exclude) if exclude else None
    applied, skipped = [], []
    for shard in sorted(target_dir.glob("*.safetensors")):
        tensors = load_file(str(shard))
        for name, t in tensors.items():
            if name not in donor or name not in base or (only_re and not only_re.search(name)) or (excl_re and excl_re.search(name)):
                continue
            with safe_open(donor[name], "pt") as fd, safe_open(base[name], "pt") as fb:
                d = fd.get_tensor(name); b = fb.get_tensor(name)
            if d.shape != t.shape or b.shape != t.shape:
                skipped.append(name); continue
            delta = trim_topk(d.float() - b.float(), keep)
            tensors[name] = (t.float() + alpha * delta).to(t.dtype)
            applied.append(name)
        save_file(tensors, str(out_dir / shard.name), metadata={"format": "pt"})
    for f in target_dir.iterdir():
        if f.suffix in (".json", ".txt", ".model", ".jinja", ".py"):
            shutil.copy(f, out_dir / f.name)
    (out_dir / "swiftlab_transfer.json").write_text(json.dumps({"alpha": alpha, "keep": keep, "only": only, "applied": len(applied), "skipped": skipped[:50]}, indent=2))
    return {"applied": applied, "skipped": skipped}


def _index(model_dir: str) -> dict[str, str]:
    p = Path(model_dir)
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return {k: str(p / v) for k, v in wm.items()}
    from safetensors import safe_open
    out = {}
    for shard in p.glob("*.safetensors"):
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():
                out[k] = str(shard)
    return out
