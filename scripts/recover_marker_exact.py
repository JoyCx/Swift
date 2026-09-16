"""Exact 0/1 recovery of the 49-id pool: greedy forward selection on integer residuals, then swap refinement."""
import json, numpy as np
from tokenizers import Tokenizer
tok = Tokenizer.from_file("A:/models/Qwen3.8-27B/tokenizer.json")
C = np.load("runs/markers/tb_C.npy"); cand = json.load(open("runs/markers/tb_cand.json"))
meta = json.load(open("runs/markers/tb_nnls.json", encoding="utf-8"))["meta"]
y = np.array([x[3] for x in meta], dtype=np.float64)
col = {t: j for j, t in enumerate(cand)}

def err(S):
    r = y - C[:, S].sum(1) if S else y.copy()
    return float(np.abs(r).sum()), r

S = []
while len(S) < 60:
    _, r = err(S)
    # gain of adding column j with weight 1: sum|r| - sum|r - c_j|
    gain = np.abs(r).sum() - np.abs(r[:, None] - C).sum(0)
    gain[S] = -np.inf
    j = int(np.argmax(gain))
    if gain[j] <= 0: break
    S.append(j)
    e, _ = err(S)
    print(f"{len(S):2d} +{cand[j]:7d} {tok.decode([cand[j]])!r:18} L1={e:.0f}  exact_trials={(np.abs(y - C[:, S].sum(1)) == 0).sum()}/890")
# swap refinement
improved = True
while improved:
    improved = False
    e0, _ = err(S)
    for k in range(len(S)):
        rest = S[:k] + S[k+1:]
        _, r = err(rest)
        l = np.abs(r[:, None] - C).sum(0); l[rest] = np.inf
        j = int(np.argmin(l))
        if l[j] < e0 and j != S[k]:
            print(f"swap {tok.decode([cand[S[k]]])!r} -> {tok.decode([cand[j]])!r}: {e0:.0f} -> {l[j]:.0f}")
            S[k] = j; improved = True; break
e, r = err(S)
ex = int((r == 0).sum())
print(f"\nFINAL |S|={len(S)} L1={e:.0f} exact trials={ex}/890 base_hits={C[:445][:, S].sum():.0f} swift_hits={C[445:][:, S].sum():.0f}")
ids = sorted(cand[j] for j in S)
json.dump([{"id": t, "text": tok.decode([t])} for t in ids], open("runs/markers/ukisai_pool_recovered.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print([tok.decode([t]) for t in ids])
bad = np.argsort(-np.abs(r))[:5]; print("worst trials", [(meta[i][1], int(r[i])) for i in bad])
