"""Which penalised token did Swift need for math? AIME 2026 + HMMT Nov 2025, base vs swift, 30 problems x 5 seeds each."""
import glob, json, re, math, collections
import numpy as np, zstandard
from pathlib import Path
from tokenizers import Tokenizer
tok = Tokenizer.from_file("A:/models/Qwen3.8-27B/tokenizer.json")
pool = {p["id"]: p["text"] for p in json.load(open("runs/markers/ukisai_pool_recovered.json", encoding="utf-8"))}
E = "data/external/ukisai-evals/"
def load(bench, comp):
    rows = []
    for f in glob.glob(f"{E}{bench}/**/matharena_outputs/*/{comp}/*_s*/*.json.zst", recursive=True) + glob.glob(f"{E}{bench}/matharena_outputs/*/{comp}/*_s*/*.json.zst"):
        arm_seed = Path(f).parent.name; arm, s = arm_seed.rsplit("_s", 1)
        d = json.loads(zstandard.ZstdDecompressor().stream_reader(open(f, "rb")).read())
        cot = [m["content"] for m in d["messages"][0] if m.get("type") == "cot"]
        rows.append(dict(bench=comp, arm=arm, seed=int(s), idx=d["idx"], correct=bool(d["correct"][0]), cot=cot[0] if cot else ""))
    uniq = {(r["arm"], r["seed"], r["idx"]): r for r in rows}
    return list(uniq.values())
rows = load("05-aime-2026", "aime_2026") + load("06-hmmt-nov-2025", "hmmt_nov_2025")
print(collections.Counter((r["bench"], r["arm"]) for r in rows))
encs = tok.encode_batch([r["cot"] for r in rows], add_special_tokens=False)
for r, e in zip(rows, encs):
    r["n"] = len(e.ids); r["pc"] = collections.Counter(t for t in e.ids if t in pool); r["ids"] = e.ids
for b in ("aime_2026", "hmmt_nov_2025"):
    for a in ("base", "swift"):
        R = [r for r in rows if r["bench"] == b and r["arm"] == a]
        print(f"{b:14} {a:5} acc={np.mean([r['correct'] for r in R]):.3f} think_mean={np.mean([r['n'] for r in R]):.0f} marker/1k={1000*sum(sum(r['pc'].values()) for r in R)/sum(r['n'] for r in R):.2f}")

# 1) per-token suppression in math vs Terminal-Bench/GPQA style: rate ratio swift/base in math
def rate(R, t): return sum(r["pc"][t] for r in R) / max(1, sum(r["n"] for r in R)) * 1000
B = [r for r in rows if r["arm"] == "base"]; S = [r for r in rows if r["arm"] == "swift"]
# 2) problem-level: problems where swift lost accuracy vs base (same problem, 5 seeds each)
key = lambda r: (r["bench"], r["idx"])
acc = collections.defaultdict(lambda: {"base": [], "swift": []})
for r in rows: acc[key(r)][r["arm"]].append(r)
lost = [k for k, v in acc.items() if np.mean([x["correct"] for x in v["swift"]]) < np.mean([x["correct"] for x in v["base"]]) - 0.001]
gained = [k for k, v in acc.items() if np.mean([x["correct"] for x in v["swift"]]) > np.mean([x["correct"] for x in v["base"]]) + 0.001]
print("\nproblems swift lost:", [(k, [x['correct'] for x in acc[k]['base']].count(True), [x['correct'] for x in acc[k]['swift']].count(True)) for k in lost])
print("problems swift gained:", len(gained))
# 3) within base math traces: rate of each pool token in CORRECT vs INCORRECT traces, and in problems swift later lost
print(f"\n{'token':16} base/1k swift/1k ratio | base: correct-trace/1k wrong-trace/1k | lost-problems base/1k swift/1k ratio")
Bc = [r for r in B if r["correct"]]; Bw = [r for r in B if not r["correct"]]
Bl = [r for k in lost for r in acc[k]["base"]]; Sl = [r for k in lost for r in acc[k]["swift"]]
out = []
for t, s in pool.items():
    rb, rs = rate(B, t), rate(S, t)
    out.append((s, rb, rs, rs / rb if rb else float("nan"), rate(Bc, t), rate(Bw, t), rate(Bl, t), rate(Sl, t)))
for s, rb, rs, q, c, w, bl, sl in sorted(out, key=lambda x: -x[1]):
    if rb > 0.05: print(f"{s!r:16} {rb:7.2f} {rs:7.2f} {q:5.2f} | {c:7.2f} {w:7.2f} | {bl:7.2f} {sl:7.2f} {sl/bl if bl else float('nan'):5.2f}")
# 4) math-context usage: fraction of each token's base occurrences next to math (digits/LaTeX) within +-3 tokens
mathy = re.compile(r"[0-9=\$^_{}+\-*/<>]")
ctx = collections.defaultdict(lambda: [0, 0])
for r in B:
    ids = r["ids"]
    for i, t in enumerate(ids):
        if t in pool:
            win = tok.decode(ids[max(0, i-3):i]) + tok.decode(ids[i+1:i+4])
            ctx[t][0] += 1; ctx[t][1] += bool(len(mathy.findall(win)) >= 2)
print("\nmath-context share of base occurrences (>=2 math chars within +-3 tokens):")
print(sorted(((pool[t], v[0], round(v[1]/v[0], 2)) for t, v in ctx.items() if v[0] >= 100), key=lambda x: -x[2])[:15])
json.dump({"lost": lost, "rows": [(s, rb, rs, q, c, w, bl, sl) for s, rb, rs, q, c, w, bl, sl in out]}, open("runs/markers/math/math_bug.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
