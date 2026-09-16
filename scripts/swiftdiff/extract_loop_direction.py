"""Extract a per-layer "about to start a redundant loop iteration" direction from Qwen3.8-27B activations.

Data: loop-labelled windows from scripts/poc/label_loops.py. Inside each window, a POSITIVE position is the
token immediately before the first ENTRY token of a loop iteration (the residual state that decides the
next token). A NEGATIVE position is the token immediately before a NORMAL paragraph start ("\n\n" boundary)
that is not a loop entry, in the same window, so content/format/position confounds are matched.

For every decoder layer the residual stream output at those positions is collected (forward only, no grad),
then: diff-of-means direction on a train split, and held-out AUC of the 1-D projection (how linearly the loop
state is encoded at that layer). Also records the projection of Swift-removed marker positions for context.

Resources: NF4-quantises the BF16 base while loading (~1 min, ~12 GB GPU; embeddings/lm_head stay in RAM),
GPU memory cap, background priority, 6 CPU threads.

    python scripts/swiftdiff/extract_loop_direction.py --windows 400
"""
from __future__ import annotations

import argparse
import json
import sys
import random
import time
from pathlib import Path

import numpy as np
import torch

IGNORE, NORMAL, WASTE, ENTRY = 0, 1, 2, 3


def positions(ids, lab, para_ids, max_pos, rng):
    """Return (positive, negative) indices of the token BEFORE the decision."""
    n = len(ids)
    pos, neg = [], []
    for k in range(1, n):
        if lab[k] == ENTRY and lab[k - 1] != ENTRY:
            pos.append(k - 1)
        elif lab[k] == NORMAL and lab[k - 1] == NORMAL and ids[k - 1] in para_ids:
            neg.append(k - 1)
    rng.shuffle(pos); rng.shuffle(neg)
    m = min(len(pos), len(neg), max_pos)
    return sorted(pos[:m]), sorted(neg[:m])


def auc(p, n):
    s = np.concatenate([p, n]); y = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    return (r[y == 1].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="A:/models/Qwen3.8-27B-base-bf16")
    ap.add_argument("--data", default="A:/swift/runs/poc/labels_maxdevv")
    ap.add_argument("--out", default="A:/swift/runs/swiftdiff/loop_direction")
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--max-pos-per-window", type=int, default=6)
    ap.add_argument("--gpu-frac", type=float, default=0.7, help="cap on the process share of GPU memory")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lowres import limit
    limit(a.gpu_frac, a.threads)
    rng = random.Random(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))
    from train_loop_lora import load_windows, crop  # noqa: E402
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    para_ids = {i for t, i in tok.get_vocab().items() if t.replace("\u010a", "\n").endswith("\n\n")}
    items = load_windows(a.data)
    rng.shuffle(items)
    items = items[: a.windows]

    t0 = time.time()
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                 quantization_config=BitsAndBytesConfig(
                                                     load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                     bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True,
                                                     llm_int8_skip_modules=["lm_head", "embed_tokens"]),
                                                 device_map={"model.embed_tokens": "cpu", "lm_head": "cpu", "model.layers": 0,
                                                             "model.norm": 0, "model.rotary_emb": 0})
    model.eval()
    print(f"loaded {time.time() - t0:.0f}s, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    layers = model.model.layers
    L = len(layers)
    captured = {}

    def hook(i):
        def f(_m, _inp, outp):
            h = outp[0] if isinstance(outp, tuple) else outp
            captured[i] = h[0, captured["idx"]].float().cpu()
        return f

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    P = [[] for _ in range(L)]; N = [[] for _ in range(L)]; meta = []
    for w, (ids_np, lab_np, m) in enumerate(items):
        ids_np, lab_np = crop(ids_np, lab_np, a.max_len)
        pi, ni = positions(ids_np, lab_np, para_ids, a.max_pos_per_window, rng)
        if not pi:
            continue
        captured.clear()
        captured["idx"] = torch.tensor(pi + ni, device="cuda")
        with torch.no_grad():
            emb = model.model.embed_tokens(torch.from_numpy(ids_np.astype(np.int64))[None]).to("cuda")
            model.model(inputs_embeds=emb, use_cache=False)
        for i in range(L):
            P[i].append(captured[i][: len(pi)]); N[i].append(captured[i][len(pi):])
        meta.append(dict(window=w, id=m["id"], n=len(pi)))
        if w % 20 == 0:
            print(f"window {w}/{len(items)} pairs {sum(x['n'] for x in meta)} {time.time() - t0:.0f}s "
                  f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    for h in handles:
        h.remove()

    n_win = len(meta); split = int(n_win * 0.7)
    res, dirs = [], np.zeros((L, P[0][0].shape[-1]), dtype=np.float32)
    for i in range(L):
        ptr, ntr = torch.cat(P[i][:split]), torch.cat(N[i][:split])
        pte, nte = torch.cat(P[i][split:]), torch.cat(N[i][split:])
        d = ptr.mean(0) - ntr.mean(0)
        r = d / d.norm()
        dirs[i] = r.numpy()
        scale = (ptr.norm(dim=1).mean() + ntr.norm(dim=1).mean()).item() / 2
        res.append(dict(layer=i, auc_heldout=float(auc((pte @ r).numpy(), (nte @ r).numpy())),
                        auc_train=float(auc((ptr @ r).numpy(), (ntr @ r).numpy())),
                        diff_norm=d.norm().item(), resid_norm=scale, rel=d.norm().item() / scale,
                        n_train=len(ptr), n_test=len(pte)))
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res[-1].items()}), flush=True)
    np.save(out / "directions.npy", dirs)
    json.dump(res, open(out / "layer_stats.json", "w"), indent=1)
    json.dump(meta, open(out / "windows_used.json", "w"))
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
