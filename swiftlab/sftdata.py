"""Build the penalised-SFT dataset the way Swift's authors described it.

Input: settled rollouts of the BASE model on the mining split (diverse, decontaminated).
Rows kept: correct traces (tight + overspent); overspent traces are truncated at the settle
point so the training target is the needed part of the trace. Derailed traces are kept with
their needed prefix and the *forced* answer is regenerated from that prefix. Wrong-throughout
traces are dropped. Output: jsonl {prompt, context, thinking, answer, task_id, domain, category}.
"""
from __future__ import annotations
import random
from pathlib import Path

from .backends import Backend, GenRequest
from .tasks import Task, verify
from .util import write_jsonl


def build_sft_rows(rows: list[dict], tasks_by_id: dict[str, Task], backend: Backend | None = None, keep_derailed: bool = True,
                   max_per_task: int = 2, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out, per_task = [], {}
    rows = list(rows); rng.shuffle(rows)
    for r in rows:
        s = r.get("settle", {}); cat = s.get("category", "short")
        t = tasks_by_id[r["task_id"]]
        if per_task.get(t.id, 0) >= max_per_task:
            continue
        if cat in ("tight", "short") and r["correct"]:
            th, ans = r["thinking"], r["answer"]
        elif cat == "overspent":
            th, ans = r["thinking"][: s["settle_char"]].rstrip(), r["answer"]
        elif cat == "derailed" and keep_derailed and backend is not None and s.get("settle_char"):
            th = r["thinking"][: s["settle_char"]].rstrip()
            ans = backend.complete_prefix(GenRequest(t.id, t.prompt, r["seed"], context=t.context, meta={"difficulty": t.difficulty}), th, 512)
            if not verify(t, ans).correct:
                continue
        else:
            continue
        per_task[t.id] = per_task.get(t.id, 0) + 1
        out.append({"task_id": t.id, "domain": t.domain, "prompt": t.prompt, "context": t.context, "thinking": th, "answer": ans,
                    "category": cat, "orig_think_tokens": r["think_tokens"]})
    return out


def write_sft(rows: list[dict], path: str | Path) -> dict:
    n = write_jsonl(path, rows)
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    return {"n": n, "categories": cats, "out": str(path)}
