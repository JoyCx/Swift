"""Decompose UkisAI Swift's weight change: delta = W_swift - W_base for every tensor of Qwen3.8-27B.

CPU only, below-normal priority. Reads one tensor pair at a time from the safetensors shards (mmap).
Per tensor: |delta|_F, |W_base|_F, relative change, max abs, exact-equality flag.
Per 2-D matrix with a non-zero delta: randomized SVD (rank q) -> top singular values, the share of
|delta|_F^2 captured by the top 1/4/8/16/32/64/128 directions (a merged LoRA of rank r puts ~100% in top r),
and the rank-64 factors saved as float16 for later partial merges.

    python scripts/swiftdiff/diff_swift.py --out runs/swiftdiff
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open

BASE = "A:/models/Qwen3.8-27B-base-bf16"
SWIFT = "A:/models/Qwen3.8-27B"


def lower_priority():
    if os.name == "nt":
        BELOW_NORMAL = 0x00004000
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(h, BELOW_NORMAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="A:/swift/runs/swiftdiff")
    ap.add_argument("--q", type=int, default=160, help="randomized SVD oversampled rank")
    ap.add_argument("--save-rank", type=int, default=64)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--skip-visual", action="store_true", default=True)
    a = ap.parse_args()
    lower_priority()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    out = Path(a.out); (out / "factors").mkdir(parents=True, exist_ok=True)
    wmap = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
    names = sorted(wmap, key=lambda n: (wmap[n], n))
    done = set()
    stats_path = out / "tensor_stats.jsonl"
    if stats_path.exists():
        done = {json.loads(l)["name"] for l in open(stats_path, encoding="utf-8")}
    fo = open(stats_path, "a", encoding="utf-8")
    t0 = time.time()
    handles = {}

    def tensor(root, name):
        key = (root, wmap[name])
        if key not in handles:
            for k in [k for k in handles if k[0] == root]:
                del handles[k]
            handles[key] = safe_open(f"{root}/{wmap[name]}", framework="pt", device="cpu")
        return handles[key].get_tensor(name)

    for i, name in enumerate(names):
        if name in done or (a.skip_visual and name.startswith("model.visual.")):
            continue
        wb = tensor(BASE, name).float()
        ws = tensor(SWIFT, name).float()
        d = ws - wb
        dn = d.norm().item()
        rec = dict(name=name, shape=list(wb.shape), base_norm=wb.norm().item(), delta_norm=dn,
                   rel=dn / max(wb.norm().item(), 1e-12), max_abs=d.abs().max().item(), identical=dn == 0.0)
        m = re.search(r"layers\.(\d+)\.", name)
        rec["layer"] = int(m.group(1)) if m else None
        rec["module"] = re.sub(r"^.*layers\.\d+\.", "", name) if m else name
        if d.dim() == 2 and dn > 0 and min(d.shape) > 8:
            q = min(a.q, min(d.shape))
            U, S, V = torch.svd_lowrank(d, q=q, niter=3)
            e = (S ** 2).cumsum(0) / (dn ** 2)
            rec["sv_top"] = [round(x, 6) for x in S[:16].tolist()]
            rec["energy_top"] = {str(r): round(e[min(r, len(e)) - 1].item(), 5) for r in (1, 4, 8, 16, 32, 64, 128) if r <= len(e)}
            r = min(a.save_rank, len(S))
            torch.save({"U": U[:, :r].half(), "S": S[:r].half(), "V": V[:, :r].half()},
                       out / "factors" / f"{name}.pt")
        fo.write(json.dumps(rec) + "\n"); fo.flush()
        if i % 25 == 0:
            print(f"{i}/{len(names)} {name} rel={rec['rel']:.2e} {time.time() - t0:.0f}s", flush=True)
        del wb, ws, d
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
