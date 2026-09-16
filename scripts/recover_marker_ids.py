"""Recover UkisAI's 49-id penalised token pool from their Terminal-Bench 2.1 per-trial stats.

tb21_per_trial.csv gives, for each of 890 trials, reasoning_tokens and marker_hits (hits of the
49-id pool). Tokenising each trial's reasoning with the Qwen3.8 tokenizer gives a trial x vocab
count matrix C; the pool S satisfies marker_hits = C[:, S].sum(1). Solve by non-negative least
squares on the candidate columns, then verify the 0/1 set reproduces every trial.
"""
import csv, json, glob, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from tokenizers import Tokenizer

ROOT = Path("A:/swift/data/external/ukisai-evals/08-terminal-bench-2.1")
OUT = Path("A:/swift/runs/markers")
tok = Tokenizer.from_file("A:/models/Qwen3.8-27B/tokenizer.json")

rows = list(csv.DictReader(open(ROOT / "swift/analysis/tb21_per_trial.csv", encoding="utf-8")))
print("csv columns:", list(rows[0].keys()))
print(rows[0])

def trial_texts(arm, trial):
    d = ROOT / arm / trial
    tj = json.load(open(d / "agent/trajectory.json", encoding="utf-8"))
    out = []
    for s in tj["steps"]:
        if s.get("source") == "agent" and (s.get("metrics") or {}).get("completion_tokens", 0) > 0:
            if s.get("reasoning_content"):
                out.append(s["reasoning_content"])
    return out

armdir = {"base_qwen38_bf16": "base", "swift_bf16_adapter": "swift"}
counts, meta = [], []
for i, r in enumerate(rows):
    arm = armdir[r["arm"]]
    trial = r["trial"]
    if not (ROOT / arm / trial).exists():
        cands = glob.glob(str(ROOT / arm / f"{trial}*"))
        trial = Path(cands[0]).name if cands else trial
    try:
        texts = trial_texts(arm, trial)
    except FileNotFoundError:
        texts = []
    c = Counter()
    ntok = 0
    for enc in tok.encode_batch(texts, add_special_tokens=False) if texts else []:
        c.update(enc.ids); ntok += len(enc.ids)
    counts.append(c)
    meta.append((r["arm"], trial, int(float(r["reasoning_tokens"] or 0)), int(float(r["marker_hits"] or 0)), ntok))
    if i % 100 == 0:
        print(i, meta[-1], flush=True)

m = np.array([[x[2], x[3], x[4]] for x in meta], dtype=np.float64)
ratio = m[:, 2].sum() / m[:, 0].sum()
print(f"tokenised/their reasoning tokens: {ratio:.4f}; per-trial corr {np.corrcoef(m[:,0], m[:,2])[0,1]:.5f}")

vocab_total = Counter()
for c in counts: vocab_total.update(c)
cand = [t for t, n in vocab_total.items() if n >= 50]
print("candidate ids:", len(cand))
col = {t: j for j, t in enumerate(cand)}
C = np.zeros((len(counts), len(cand)), dtype=np.float64)
for i, c in enumerate(counts):
    for t, n in c.items():
        j = col.get(t)
        if j is not None: C[i, j] = n
y = m[:, 1]
# scale columns for conditioning, NNLS, then look at weights near 1
w, res = nnls(C, y, maxiter=20000)
order = np.argsort(-w)
print("NNLS residual", res)
top = [(cand[j], float(w[j]), tok.decode([cand[j]]), int(C[:, j].sum())) for j in order[:120] if w[j] > 0.05]
for t, wt, s, n in top: print(f"{t:7d} w={wt:.3f} {s!r:20} total={n}")
json.dump({"meta": meta, "nnls_top": top}, open(OUT / "tb_nnls.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
np.save(OUT / "tb_C.npy", C); json.dump(cand, open(OUT / "tb_cand.json", "w"))
