"""Dataset-size scaling: how many long traces do you need before the waste patterns saturate?

For each n in the schedule we subsample n probed traces (stratified by domain, seeded) and
measure (a) how much removable thinking we have *observed* (overspent + wrong tokens),
(b) how many distinct enriched markers were *discovered*, (c) the Chao1 estimate of how
many markers exist in total, hence coverage = discovered / estimated.  A flat tail means
more traces of the same kind add nothing; a rising tail means keep sampling.
"""
from __future__ import annotations
import random
from collections import Counter

from .mining import mine_markers
from .trace import segment


def _stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["domain"], []).append(r)
    out = []
    for d in sorted(by):
        pool = list(by[d]); rng.shuffle(pool)
        out += pool[: max(1, round(n * len(pool) / len(rows)))]
    rng.shuffle(out)
    return out[:n]


def chao1(counts: Counter) -> float:
    f1 = sum(1 for c in counts.values() if c == 1)
    f2 = sum(1 for c in counts.values() if c == 2)
    s = len(counts)
    return s + (f1 * f1 / (2.0 * f2) if f2 > 0 else f1 * (f1 - 1) / 2.0)


def loop_signatures(rows: list[dict], markers=None) -> Counter:
    c = Counter()
    for r in rows:
        for s in segment(r["thinking"], markers):
            if s.kind in ("loop", "reverify"):
                sig = " ".join(s.text.lower().split()[:6])
                c[sig] += 1
    return c


def scaling_curve(rows: list[dict], sizes: list[int] | None = None, seed: int = 0, z_threshold: float = 3.0, markers=None) -> dict:
    N = len(rows)
    sizes = sizes or [s for s in [25, 50, 100, 200, 400, 800, 1600, 3200] if s < N] + [N]
    points = []
    for n in sizes:
        sub = _stratified(rows, n, seed)
        mined = mine_markers(sub, z_threshold=z_threshold, markers=markers)
        sigs = loop_signatures(sub, markers)
        think = sum(r["think_tokens"] for r in sub) or 1
        over = sum(r["settle"]["overspent_tokens"] for r in sub if "settle" in r)
        wrong = sum(r["think_tokens"] for r in sub if r.get("settle", {}).get("category") == "wrong")
        derailed = sum(1 for r in sub if r.get("settle", {}).get("category") == "derailed")
        est = chao1(sigs)
        points.append({"n": n, "think_tokens": think, "overspent_share": over / think, "wrong_share": wrong / think,
                       "removable_share": (over + wrong) / think, "n_derailed": derailed, "n_wrong": sum(1 for r in sub if not r["correct"]),
                       "markers_discovered": len(mined["phrases"]), "loop_signatures": len(sigs), "loop_signatures_chao1": est,
                       "signature_coverage": (len(sigs) / est) if est > 0 else 1.0})
    rec = None
    for a, b in zip(points, points[1:]):
        gain = (b["markers_discovered"] - a["markers_discovered"]) / max(1, a["markers_discovered"])
        if gain < 0.05 and b["signature_coverage"] > 0.9:
            rec = a["n"]; break
    return {"points": points, "recommended_n": rec or sizes[-1], "note": "recommended_n = first size after which marker discovery grows <5% and loop-signature coverage >90%"}
