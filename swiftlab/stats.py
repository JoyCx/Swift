"""Paired statistics: bootstrap CIs on paired deltas and an exact McNemar test."""
from __future__ import annotations
import math, random


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0, stat=None) -> dict:
    """CI for stat(b) - stat(a) over matched items (default: mean)."""
    stat = stat or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    n = len(a)
    if n == 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    obs = stat(b) - stat(a)
    ds = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ds.append(stat([b[i] for i in idx]) - stat([a[i] for i in idx]))
    ds.sort()
    return {"delta": obs, "lo": ds[int(0.025 * n_boot)], "hi": ds[int(0.975 * n_boot) - 1], "n": n}


def mcnemar_exact(b_only: int, a_only: int) -> float:
    """Two-sided exact p-value for discordant pairs (b right & a wrong vs a right & b wrong)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
