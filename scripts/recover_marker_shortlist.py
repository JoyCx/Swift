import csv, json, glob, numpy as np
from pathlib import Path
from scipy.optimize import nnls
from tokenizers import Tokenizer
ROOT = Path("A:/swift/data/external/ukisai-evals/08-terminal-bench-2.1")
tok = Tokenizer.from_file("A:/models/Qwen3.8-27B/tokenizer.json")
words = """but maybe might could or wait actually hmm still however perhaps another alternatively instead rather rethink revisit
reconsider regardless hold either anyway though whether possibly unless yet retry backtrack wrong mistake incorrect error different
otherwise though although hmmm oops let double check verify recheck re-check again confirm hold on sure seems probably likely unclear
suppose assume what if no not doubt uncertain careful think thinking consider alternative options option second nevertheless nonetheless
also then so okay ok hm ah oh well""".split()
ids = {}
for w in set(words):
    for v in {w, w.capitalize(), w.upper()}:
        for form in (v, " " + v):
            e = tok.encode(form, add_special_tokens=False).ids
            if len(e) == 1: ids[e[0]] = form
ids = dict(sorted(ids.items()))
print(len(ids), "shortlist ids")
idx = {t: j for j, t in enumerate(ids)}
rows = list(csv.DictReader(open(ROOT / "swift/analysis/tb21_per_trial.csv", encoding="utf-8")))
armdir = {"base_qwen38_bf16": "base", "swift_bf16_adapter": "swift"}
R = np.zeros((len(rows), len(ids))); K = np.zeros_like(R); M = np.zeros_like(R)
y = np.array([float(r["marker_hits"]) for r in rows])
for i, r in enumerate(rows):
    tj = json.load(open(ROOT / armdir[r["arm"]] / r["trial"] / "agent/trajectory.json", encoding="utf-8"))
    rs, cs, ms = [], [], []
    for s in tj["steps"]:
        if s.get("source") == "agent" and (s.get("metrics") or {}).get("completion_tokens", 0) > 0:
            rs.append(s.get("reasoning_content") or "")
            ms.append(s.get("message") or "")
            cs.append(json.dumps([tc.get("arguments") for tc in s.get("tool_calls") or []]))
    for mat, texts in ((R, rs), (M, ms), (K, cs)):
        for e in tok.encode_batch([t for t in texts if t], add_special_tokens=False):
            for t in e.ids:
                j = idx.get(t)
                if j is not None: mat[i, j] += 1
names = list(ids.values()); tids = list(ids)
for label, X in (("reasoning", R), ("reasoning+message", R + M), ("reasoning+message+toolargs", R + M + K)):
    w, res = nnls(X, y)
    S = [j for j in range(len(tids)) if w[j] > 0.5]
    r = y - X[:, S].sum(1)
    print(f"\n== {label}: nnls resid {res:.1f}; 0/1 set |S|={len(S)} L1={np.abs(r).sum():.0f} exact={(r==0).sum()}/890 base={X[:445][:,S].sum():.0f} swift={X[445:][:,S].sum():.0f}")
    print(sorted(((names[j], round(w[j], 2)) for j in range(len(tids)) if w[j] > 0.05), key=lambda x: -x[1]))
np.savez("runs/markers/tb_shortlist.npz", R=R, M=M, K=K, y=y, ids=np.array(tids))
json.dump(ids, open("runs/markers/tb_shortlist_ids.json", "w"), ensure_ascii=False)
