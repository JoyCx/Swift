"""Paired evaluation: same task, same seed, same context, every arm.

Coverage metrics answer "how much of the thinking we measured as waste did the edit
actually remove, and did it touch thinking that was needed?"
  overspend_removed   = sum_i min(cut_i, overspent_i) / sum_i overspent_i        (higher is better)
  needed_cut          = sum_i max(0, cut_i - overspent_i) / sum_i settle_tokens_i  (lower is better)
where cut_i = think_A - think_B for the matched item and overspent_i / settle_tokens_i
come from settle probing of arm A (the base).
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

from .backends import Backend
from .rollout import run_rollouts
from .stats import paired_bootstrap, mcnemar_exact, median
from .tasks import Task


def run_arms(arms: dict[str, Backend], tasks: list[Task], seeds: int, out_dir: str | Path, effort: str | None = None,
             concurrency: int = 4, markers=None, logit_bias_by_arm: dict | None = None) -> dict[str, list[dict]]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, be in arms.items():
        lb = (logit_bias_by_arm or {}).get(name)
        rows[name] = run_rollouts(be, tasks, seeds, out_dir / f"eval_{name}.jsonl", concurrency=concurrency, effort=effort,
                                  logit_bias=lb, markers=markers, arm=name)
    return rows


def _key(r: dict):
    return (r["task_id"], r["seed"])


def summarize_arm(rows: list[dict]) -> dict:
    n = len(rows) or 1
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)
    dom = {d: {"n": len(rs), "accuracy": sum(r["correct"] for r in rs) / len(rs), "think_mean": sum(r["think_tokens"] for r in rs) / len(rs),
               "think_median": median([r["think_tokens"] for r in rs])} for d, rs in by_dom.items()}
    return {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / n, "think_mean": sum(r["think_tokens"] for r in rows) / n,
            "think_median": median([r["think_tokens"] for r in rows]), "latency_mean": sum(r["latency"] for r in rows) / n,
            "truncated": sum(1 for r in rows if r.get("finish_reason") == "length"), "domains": dom}


def compare(base: list[dict], other: list[dict], n_boot: int = 2000, seed: int = 0, settled_base: list[dict] | None = None) -> dict:
    A = {_key(r): r for r in base}; B = {_key(r): r for r in other}
    keys = sorted(set(A) & set(B))
    if not keys:
        raise ValueError("no matched (task, seed) pairs between arms")
    acc_a = [float(A[k]["correct"]) for k in keys]; acc_b = [float(B[k]["correct"]) for k in keys]
    th_a = [A[k]["think_tokens"] for k in keys]; th_b = [B[k]["think_tokens"] for k in keys]
    lat_a = [A[k]["latency"] for k in keys]; lat_b = [B[k]["latency"] for k in keys]
    fixed = sum(1 for k in keys if B[k]["correct"] and not A[k]["correct"])
    broken = sum(1 for k in keys if A[k]["correct"] and not B[k]["correct"])
    ratio = [b / a for a, b in zip(th_a, th_b) if a > 0]
    res = {"n_pairs": len(keys), "accuracy_base": sum(acc_a) / len(keys), "accuracy_other": sum(acc_b) / len(keys),
           "accuracy_delta": paired_bootstrap(acc_a, acc_b, n_boot, seed), "mcnemar_p": mcnemar_exact(fixed, broken),
           "fixed": fixed, "broken": broken, "think_mean_base": sum(th_a) / len(keys), "think_mean_other": sum(th_b) / len(keys),
           "think_median_base": median(th_a), "think_median_other": median(th_b),
           "think_reduction_mean": 1 - sum(th_b) / max(1, sum(th_a)), "think_reduction_median": 1 - median(th_b) / max(1, median(th_a)),
           "think_reduction_ci": paired_bootstrap(th_a, th_b, n_boot, seed, stat=lambda xs: sum(xs) / len(xs)),
           "per_item_ratio_median": median(ratio), "share_items_shorter": sum(1 for r in ratio if r < 1) / max(1, len(ratio)),
           "share_items_shorter_and_correct": sum(1 for k in keys if B[k]["think_tokens"] < A[k]["think_tokens"] and B[k]["correct"]) / len(keys),
           "speedup": (sum(lat_a) / max(1e-9, sum(lat_b))) if sum(lat_b) > 0 else None,
           "domains": {}}
    doms = sorted({A[k]["domain"] for k in keys})
    for d in doms:
        ks = [k for k in keys if A[k]["domain"] == d]
        ta = sum(A[k]["think_tokens"] for k in ks); tb = sum(B[k]["think_tokens"] for k in ks)
        res["domains"][d] = {"n": len(ks), "accuracy_base": sum(A[k]["correct"] for k in ks) / len(ks), "accuracy_other": sum(B[k]["correct"] for k in ks) / len(ks),
                             "think_reduction": 1 - tb / max(1, ta), "fixed": sum(1 for k in ks if B[k]["correct"] and not A[k]["correct"]),
                             "broken": sum(1 for k in ks if A[k]["correct"] and not B[k]["correct"])}
    if settled_base:
        S = {_key(r): r["settle"] for r in settled_base if "settle" in r}
        over_tot = removed = needed_cut = settle_tot = 0.0
        derailed_fixed = derailed_n = 0
        for k in keys:
            s = S.get(k)
            if not s or s["category"] == "short":
                continue
            cut = A[k]["think_tokens"] - B[k]["think_tokens"]
            over = s["overspent_tokens"]; st = s["settle_tokens"] or 0
            over_tot += over; settle_tot += st
            removed += max(0.0, min(cut, over)); needed_cut += max(0.0, cut - over) if st else 0.0
            if s["category"] == "derailed":
                derailed_n += 1; derailed_fixed += int(B[k]["correct"])
        res["coverage"] = {"overspend_removed": removed / over_tot if over_tot else None, "needed_cut": needed_cut / settle_tot if settle_tot else None,
                           "derailed_in_base": derailed_n, "derailed_fixed": derailed_fixed}
    return res


def compare_all(rows: dict[str, list[dict]], base: str = "base", settled_base: list[dict] | None = None, n_boot: int = 2000) -> dict:
    out = {"arms": {n: summarize_arm(r) for n, r in rows.items()}, "pairs": {}}
    for n, r in rows.items():
        if n != base:
            out["pairs"][f"{base}->{n}"] = compare(rows[base], r, n_boot=n_boot, settled_base=settled_base)
    return out
