"""Score a paired logit-bias test: same prompts, arms = harvest JSONL files.

    python scripts/score_bias_test.py runs/flashnext_bias/base.jsonl runs/flashnext_bias/bias_m1.jsonl
"""
import json, re, statistics, sys, unicodedata
from pathlib import Path
from tokenizers import Tokenizer

tok = Tokenizer.from_file("A:/models/Qwen3.8-27B/tokenizer.json")
pool = {p["id"] for p in json.load(open("A:/swift/runs/markers/ukisai_pool_recovered.json", encoding="utf-8"))}


def norm(s):
    s = unicodedata.normalize("NFKD", s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def correct(r):
    v, out = r.get("verify") or {}, r.get("content") or ""
    if v.get("kind") == "alias":
        m = re.findall(r"Answer:\s*(.+)", out)
        pred = norm(m[-1]) if m else ""
        return bool(pred) and any(norm(a) == pred or (norm(a) and norm(a) in pred) for a in v["answers"])
    if v.get("kind") == "integer":
        m = re.findall(r"\\boxed\{([^{}]*)\}", out)
        return bool(m) and re.sub(r"[^\d-]", "", m[-1]) == v["answer"]
    if v.get("kind") == "exact_boxed":
        pred = last_boxed(out)
        return pred is not None and canon(pred) == canon(v["answer"])
    return None


def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    start = j
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[start:j - 1] if depth == 0 else None


def canon(s):
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\[,;!]|\\ |\s|\$", "", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\(([^()+\-*/]*)\)", r"\1", s)
    return s.rstrip(".")


def load(p):
    rows = {}
    for l in Path(p).read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r.get("error"):
            continue
        ids = tok.encode(r.get("reasoning") or "", add_special_tokens=False).ids
        r["think"] = len(ids)
        r["markers"] = sum(1 for t in ids if t in pool)
        r["ok"] = correct(r)
        rows[(r["id"], r["sample"])] = r
    return rows


arms = {Path(p).stem: load(p) for p in sys.argv[1:]}
keys = set.intersection(*(set(a) for a in arms.values()))
print(f"paired items: {len(keys)}")
for dom in [None] + sorted({arms[next(iter(arms))][k]["domain"] for k in keys}):
    ks = [k for k in keys if dom is None or arms[next(iter(arms))][k]["domain"] == dom]
    if not ks:
        continue
    print(f"\n== {dom or 'ALL'} (n={len(ks)})")
    print(f"{'arm':12} {'acc':>6} {'think_mean':>10} {'think_p50':>9} {'markers/1k':>10} {'truncated':>9}")
    for name, a in arms.items():
        R = [a[k] for k in ks]
        th = [r["think"] for r in R]
        print(f"{name:12} {sum(bool(r['ok']) for r in R) / len(R):6.3f} {statistics.mean(th):10.0f} {statistics.median(th):9.0f} "
              f"{1000 * sum(r['markers'] for r in R) / max(1, sum(th)):10.2f} {sum(bool(r.get('truncated')) for r in R):9d}")
names = list(arms)
if len(names) == 2:
    a, b = arms[names[0]], arms[names[1]]
    ratio = [b[k]["think"] / a[k]["think"] for k in keys if a[k]["think"]]
    print(f"\npaired think ratio {names[1]}/{names[0]}: median {statistics.median(ratio):.2f}")
    print("fixed:", sum(1 for k in keys if not a[k]["ok"] and b[k]["ok"]), "broken:", sum(1 for k in keys if a[k]["ok"] and not b[k]["ok"]))
