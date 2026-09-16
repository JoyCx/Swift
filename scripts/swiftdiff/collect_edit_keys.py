"""Collect linear-layer input keys for a null-space (AlphaEdit-style) loop edit feasibility check.

For selected modules (MLP down_proj, GDN linear_attn.out_proj, full-attention o_proj) record the INPUT vector
to the linear map at:
  pos  - the token before the first ENTRY token of a loop iteration (decision state)
  neg  - the token before a NORMAL paragraph start (matched boundary control)
and accumulate, over a random sample of NORMAL thinking tokens, the uncentred key covariance
C0 = sum k k^T (the knowledge to protect; its low-eigenvalue subspace is the edit's allowed null space).

Same resource limits and model load as extract_loop_direction.py (NF4 on load, GPU cap, background priority).
Output: <out>/<module>.pt with pos/neg keys (float16), window ids, C0 (float32, CPU) and token count.

    python scripts/swiftdiff/collect_edit_keys.py --windows 400
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))
from extract_loop_direction import positions  # noqa: E402
from lowres import limit  # noqa: E402

NORMAL = 1

MODULES = {  # name -> (layer, attribute path)
    **{f"L{l}.mlp.down_proj": (l, "mlp.down_proj") for l in (12, 24, 36, 48)},
    **{f"L{l}.linear_attn.out_proj": (l, "linear_attn.out_proj") for l in (8, 20, 32, 44, 56)},
    **{f"L{l}.self_attn.o_proj": (l, "self_attn.o_proj") for l in (19, 43)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="A:/models/Qwen3.8-27B-base-bf16")
    ap.add_argument("--data", default="A:/swift/runs/poc/labels_maxdevv")
    ap.add_argument("--out", default="A:/swift/runs/swiftdiff/edit_keys")
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--max-pos-per-window", type=int, default=6)
    ap.add_argument("--normal-per-window", type=int, default=96)
    ap.add_argument("--gpu-frac", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    limit(a.gpu_frac, a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    nrng = np.random.default_rng(a.seed + 1)

    from train_loop_lora import crop, load_windows
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(a.model)
    para_ids = {i for t, i in tok.get_vocab().items() if t.replace("\u010a", "\n").endswith("\n\n")}
    items = load_windows(a.data)
    rng.shuffle(items)
    items = items[: a.windows]

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True,
                                               llm_int8_skip_modules=["lm_head", "embed_tokens"]),
        device_map={"model.embed_tokens": "cpu", "lm_head": "cpu", "model.layers": 0, "model.norm": 0, "model.rotary_emb": 0})
    model.eval()
    print(f"loaded {time.time() - t0:.0f}s, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    store = {}
    cur = {}

    def get_module(layer, path):
        m = model.model.layers[layer]
        for p in path.split("."):
            m = getattr(m, p)
        return m

    def make_hook(name):
        def f(_m, inputs):
            x = inputs[0][0]
            cur[name] = (x[cur["sel"]].float().cpu(), x[cur["rnd"]].float().cpu())
        return f

    handles = []
    for name, (layer, path) in MODULES.items():
        mod = get_module(layer, path)
        handles.append(mod.register_forward_pre_hook(make_hook(name)))
        dim = mod.in_features
        store[name] = dict(pos=[], neg=[], win_pos=[], win_neg=[], C0=torch.zeros(dim, dim, dtype=torch.float32), n0=0, dim=dim)
    print({k: v["dim"] for k, v in store.items()}, flush=True)

    for w, (ids_np, lab_np, _m) in enumerate(items):
        ids_np, lab_np = crop(ids_np, lab_np, a.max_len)
        pi, ni = positions(ids_np, lab_np, para_ids, a.max_pos_per_window, rng)
        if not pi:
            continue
        normal = np.flatnonzero(lab_np[1:] == NORMAL)  # decision positions whose next token is NORMAL
        excl = set(pi) | set(ni)
        normal = np.array([k for k in normal if k not in excl])
        rnd = nrng.choice(normal, size=min(a.normal_per_window, len(normal)), replace=False) if len(normal) else np.array([], dtype=int)
        cur.clear()
        cur["sel"] = torch.tensor(pi + ni, device="cuda")
        cur["rnd"] = torch.tensor(np.sort(rnd), device="cuda", dtype=torch.long)
        with torch.no_grad():
            emb = model.model.embed_tokens(torch.from_numpy(ids_np.astype(np.int64))[None]).to("cuda")
            model.model(inputs_embeds=emb, use_cache=False)
        for name in MODULES:
            sel, r = cur[name]
            s = store[name]
            s["pos"].append(sel[: len(pi)].half()); s["neg"].append(sel[len(pi):].half())
            s["win_pos"] += [w] * len(pi); s["win_neg"] += [w] * len(ni)
            if len(r):
                s["C0"].addmm_(r.T, r)
                s["n0"] += len(r)
        if w % 20 == 0:
            print(f"window {w}/{len(items)} pos {len(store[next(iter(store))]['win_pos'])} "
                  f"normal keys {store[next(iter(store))]['n0']} {time.time() - t0:.0f}s "
                  f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    for h in handles:
        h.remove()
    for name, s in store.items():
        torch.save(dict(pos=torch.cat(s["pos"]), neg=torch.cat(s["neg"]), win_pos=s["win_pos"], win_neg=s["win_neg"],
                        C0=s["C0"], n0=s["n0"], dim=s["dim"]), out / f"{name}.pt")
    json.dump({"windows": a.windows, "modules": list(MODULES)}, open(out / "meta.json", "w"))
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
