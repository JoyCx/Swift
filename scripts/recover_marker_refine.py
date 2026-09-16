import json, numpy as np
from scipy.optimize import nnls
from tokenizers import Tokenizer
tok = Tokenizer.from_file("A:/models/Qwen3.8-27B/tokenizer.json")
z = np.load("runs/markers/tb_shortlist.npz"); R, y, sid = z["R"], z["y"], list(z["ids"])
Cf = np.load("runs/markers/tb_C.npy"); cand = json.load(open("runs/markers/tb_cand.json"))
# union matrix: shortlist columns (exact reasoning counts) + full-vocab columns not in shortlist
extra = [j for j, t in enumerate(cand) if t not in set(sid)]
X = np.hstack([R, Cf[:, extra]]); ids = sid + [cand[j] for j in extra]
w, _ = nnls(R, y)
S = set(j for j in range(len(sid)) if 0.8 < w[j] < 1.3)
def l1(S): return float(np.abs(y - X[:, sorted(S)].sum(1)).sum())
e = l1(S); print("start", len(S), e)
while True:
    r = y - X[:, sorted(S)].sum(1)
    add = np.abs(r).sum() - np.abs(r[:, None] - X).sum(0); add[list(S)] = -np.inf
    rem = {j: np.abs(r).sum() - np.abs(r + X[:, j]).sum() for j in S}
    ja = int(np.argmax(add)); jr = max(rem, key=rem.get)
    if max(add[ja], rem[jr]) <= 0: break
    if add[ja] >= rem[jr]:
        S.add(ja); print(f"+ {tok.decode([ids[ja]])!r:16} gain {add[ja]:.0f}")
    else:
        S.discard(jr); print(f"- {tok.decode([ids[jr]])!r:16} gain {rem[jr]:.0f}")
r = y - X[:, sorted(S)].sum(1)
print(f"\n|S|={len(S)} L1={np.abs(r).sum():.0f} exact={(r==0).sum()}/890 base={X[:445][:, sorted(S)].sum():.0f} (227151) swift={X[445:][:, sorted(S)].sum():.0f} (74049)")
pool = sorted(ids[j] for j in S)
print([tok.decode([t]) for t in pool])
nz = np.nonzero(r)[0]; print("nonzero residual trials:", len(nz), "values:", sorted(set(r[nz].astype(int).tolist()))[:20])
json.dump([{"id": int(t), "text": tok.decode([t])} for t in pool], open("runs/markers/ukisai_pool_recovered.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
