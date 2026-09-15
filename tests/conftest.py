"""Shared fixtures: rollout+settle rows as plain data (no model stand-in anywhere).

These dicts have exactly the shape `run_rollouts` + `probe_all` produce, so the analysis
stages (mining, scaling, evaluate, sftdata) are exercised on realistic data without a model.
"""
import random
import pytest
from swiftlab.trace import marker_stats

NEED = "Step 1: compute the intermediate value. Step 2: combine the parts. Step 3: the result follows."
LOOP = " Wait, let me double-check that. Hmm, actually maybe I should reconsider. Same value again."


def _row(arm, tid, dom, seed, correct, think, settle_cat, settle_char, settle_tok, overspent, answer="Final answer: 42"):
    thinking = NEED + (LOOP * 3 if overspent else "")
    return {"arm": arm, "task_id": tid, "domain": dom, "difficulty": 0.5, "seed": seed, "k": seed,
            "thinking": thinking, "answer": answer, "think_tokens": think, "answer_tokens": 8, "latency": think / 400.0,
            "finish_reason": "stop", "model": arm, "correct": correct, "verdict": "", "verify_elapsed": 0.0,
            "markers": marker_stats(thinking), "extra": {},
            "settle": {"category": settle_cat, "settle_char": settle_char, "settle_tokens": settle_tok,
                       "overspent_tokens": overspent, "probes": []}}


@pytest.fixture
def settled_rows():
    doms = ["coding", "knowledge", "math"]
    rows = []
    rng = random.Random(0)
    for i in range(60):
        dom = doms[i % 3]
        cat = ["overspent", "tight", "wrong", "derailed"][i % 4]
        correct = cat in ("overspent", "tight")
        think = 120 if cat != "wrong" else 90
        sc = len(NEED) if cat in ("overspent", "derailed") else None
        stok = 40 if sc else (think if cat == "tight" else None)
        over = think - 40 if cat == "overspent" else 0
        rows.append(_row("base", f"t{i}", dom, 0, correct, think, cat, sc, stok, over))
    return rows


@pytest.fixture
def paired_rows(settled_rows):
    """A second arm that is shorter on every item and fixes some derailments."""
    other = []
    for r in settled_rows:
        rr = dict(r); rr["arm"] = "edited"; rr["model"] = "edited"
        rr["think_tokens"] = int(r["think_tokens"] * 0.7)
        rr["latency"] = rr["think_tokens"] / 400.0
        if r["settle"]["category"] == "derailed":
            rr["correct"] = True
        other.append(rr)
    return {"base": settled_rows, "edited": other}
