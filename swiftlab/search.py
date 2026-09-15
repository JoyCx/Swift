"""Hyper-parameter search for the edit: co-minimise thinking tokens and drift, keep accuracy.

score(params) = think_ratio + w_kl * KL(base || edited)  + big penalty if acc_delta < -tol
Search space: which layers (center, width), gamma, ridge lambda, matrices.
Backend-agnostic through an `evaluate(edit) -> metrics` callable; uses Optuna TPE when
installed, otherwise seeded random search + local refinement around the incumbent.
"""
from __future__ import annotations
import math, random
from typing import Callable

import numpy as np

from .edit import make_edit


def score(metrics: dict, w_kl: float = 2.0, acc_tol: float = 0.01, acc_penalty: float = 10.0) -> float:
    s = metrics["think_ratio"] + w_kl * metrics.get("kl", 0.0)
    short = -acc_tol - metrics["acc_delta"]
    if short > 0:
        s += acc_penalty * short * 100      # 1 point per pp below tolerance * 10
    return s


def _sample(rng: random.Random, layers: list[int], space: dict) -> dict:
    lo, hi = min(layers), max(layers)
    return {"center": rng.uniform(lo, hi), "width": rng.uniform(*space.get("width", (1.0, 8.0))),
            "gamma": rng.uniform(*space.get("gamma", (0.2, 1.5))), "kernel": rng.choice(space.get("kernel", ["gaussian", "flat"])),
            "matrices": rng.choice(space.get("matrices", [["self_attn.o_proj", "mlp.down_proj"], ["mlp.down_proj"], ["self_attn.o_proj"]]))}


def _perturb(rng: random.Random, p: dict, layers: list[int], space: dict, scale: float = 0.3) -> dict:
    q = dict(p)
    q["center"] = min(max(p["center"] + rng.gauss(0, scale * (max(layers) - min(layers)) / 4), min(layers)), max(layers))
    q["width"] = max(0.5, p["width"] * math.exp(rng.gauss(0, scale)))
    q["gamma"] = min(max(p["gamma"] * math.exp(rng.gauss(0, scale)), 0.05), 2.0)
    return q


def run_search(bundle: dict, evaluate: Callable[[dict], dict], n_trials: int = 30, seed: int = 0, w_kl: float = 2.0,
               acc_tol: float = 0.01, space: dict | None = None, layers: list[int] | None = None) -> dict:
    space = space or {}
    layers = layers or sorted(bundle["layers"])
    rng = random.Random(seed)
    trials = []

    def run(p: dict) -> float:
        edit = make_edit(bundle, layers=layers, gamma=p["gamma"], kernel=p["kernel"], center=p["center"], width=p["width"], matrices=p["matrices"])
        m = evaluate(edit)
        sc = score(m, w_kl, acc_tol)
        trials.append({"params": p, "metrics": m, "score": sc})
        return sc

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(t):
            p = {"center": t.suggest_float("center", min(layers), max(layers)), "width": t.suggest_float("width", *space.get("width", (1.0, 8.0))),
                 "gamma": t.suggest_float("gamma", *space.get("gamma", (0.2, 1.5))), "kernel": t.suggest_categorical("kernel", space.get("kernel", ["gaussian", "flat"])),
                 "matrices": list(t.suggest_categorical("matrices", [",".join(m) for m in space.get("matrices", [["self_attn.o_proj", "mlp.down_proj"], ["mlp.down_proj"], ["self_attn.o_proj"]])]).split(","))}
            return run(p)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)
        method = "optuna-tpe"
    except ImportError:
        method = "random+local"
        best = None
        for i in range(n_trials):
            p = _sample(rng, layers, space) if (best is None or i < n_trials // 2 or rng.random() < 0.3) else _perturb(rng, best["params"], layers, space)
            sc = run(p)
            if best is None or sc < best["score"]:
                best = trials[-1]
    trials.sort(key=lambda t: t["score"])
    best = trials[0]
    return {"method": method, "best": best, "trials": trials, "layers": layers,
            "best_edit": make_edit(bundle, layers=layers, gamma=best["params"]["gamma"], kernel=best["params"]["kernel"],
                                   center=best["params"]["center"], width=best["params"]["width"], matrices=best["params"]["matrices"])}


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    p = p / p.sum(-1, keepdims=True); q = q / q.sum(-1, keepdims=True)
    return float((p * (np.log(p) - np.log(q))).sum(-1).mean())
