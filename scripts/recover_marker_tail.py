import csv, json, itertools, numpy as np
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
ROOT = Path("A:/swift/data/external/ukisai-evals/08-terminal-bench-2.1")
tok = Tokenizer.from_file("A:/models/Qwen3.8-27B/tokenizer.json")
rows = list(csv.DictReader(open(ROOT / "swift/analysis/tb21_per_trial.csv", encoding="utf-8")))
armdir = {"base_qwen38_bf16": "base", "swift_bf16_adapter": "swift"}
pool = [p["id"] for p in json.load(open("runs/markers/ukisai_pool_recovered.json", encoding="utf-8"))]
full = []
for r in rows:
    tj = json.load(open(ROOT / armdir[r["arm"]] / r["trial"] / "agent/trajectory.json", encoding="utf-8"))
    c = Counter()
    texts = [s["reasoning_content"] for s in tj["steps"] if s.get("source") == "agent" and (s.get("metrics") or {}).get("completion_tokens", 0) > 0 and s.get("reasoning_content")]
    for e in tok.encode_batch(texts, add_special_tokens=False): c.update(e.ids)
    full.append(c)
y = np.array([float(r["marker_hits"]) for r in rows])
fit = np.array([sum(c[t] for t in pool) for c in full])
res = y - fit
nz = set(np.nonzero(res)[0].tolist())
print("residual trials", {i: int(res[i]) for i in nz})
tot = Counter(); [tot.update(c) for c in full]
# candidate: never exceeds residual anywhere
cands = [t for t in tot if t not in pool and tot[t] <= res[res > 0].sum() and all(full[i][t] <= max(res[i], 0) for i in range(len(full)) if full[i][t])]
print(len(cands), "candidates:", [(tok.decode([t]), tot[t]) for t in cands][:60])
best = None
for k in (1, 2, 3, 4):
    for combo in itertools.combinations(cands, k):
        v = np.array([sum(c[t] for t in combo) for c in full])
        if np.array_equal(v, res):
            print("EXACT with", [(t, tok.decode([t])) for t in combo]); best = combo; break
    if best: break
if best:
    final = sorted(pool + list(best))
    json.dump([{"id": int(t), "text": tok.decode([t])} for t in final], open("runs/markers/ukisai_pool_recovered.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("final pool size", len(final))
