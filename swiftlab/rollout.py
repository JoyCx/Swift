"""Rollouts: generate thinking traces for a task set with model-independent seeds.

Every (task, sample k) pair gets the seed derive_seed(task_id, k) so the base model,
an edited model and a quantized model all decode from the same seed with the same
context.  Rows are appended as they finish (resumable).
"""
from __future__ import annotations
from pathlib import Path

from .backends import Backend, GenRequest
from .tasks import Task, verify
from .trace import marker_stats
from .util import derive_seed, append_jsonl, read_jsonl


def make_requests(tasks: list[Task], samples_per_task: int, effort: str | None = None, logit_bias=None) -> list[GenRequest]:
    reqs = []
    for t in tasks:
        for k in range(samples_per_task):
            reqs.append(GenRequest(t.id, t.prompt, derive_seed(t.id, k), context=t.context, reasoning_effort=effort,
                                   logit_bias=logit_bias, meta={"difficulty": t.difficulty, "domain": t.domain, "k": k}))
    return reqs


def run_rollouts(backend: Backend, tasks: list[Task], samples_per_task: int, out: str | Path, concurrency: int = 4,
                 effort: str | None = None, logit_bias=None, markers: list[str] | None = None, arm: str = "base",
                 resume: bool = True) -> list[dict]:
    out = Path(out)
    done = set()
    rows = []
    if resume and out.exists():
        rows = [r for r in read_jsonl(out) if r.get("arm") == arm]
        done = {(r["task_id"], r["seed"]) for r in rows}
    by_id = {t.id: t for t in tasks}
    reqs = [r for r in make_requests(tasks, samples_per_task, effort, logit_bias) if (r.task_id, r.seed) not in done]
    batch = max(1, concurrency * 4)
    for i in range(0, len(reqs), batch):
        gens = backend.generate_many(reqs[i:i + batch], concurrency=concurrency)
        for req, g in zip(reqs[i:i + batch], gens):
            t = by_id[req.task_id]
            v = verify(t, g.answer)
            row = {"arm": arm, "task_id": t.id, "domain": t.domain, "difficulty": t.difficulty, "seed": g.seed, "k": req.meta["k"],
                   "thinking": g.thinking, "answer": g.answer, "think_tokens": g.think_tokens, "answer_tokens": g.answer_tokens,
                   "latency": g.latency, "finish_reason": g.finish_reason, "model": g.model, "correct": v.correct,
                   "verdict": v.detail[:300], "verify_elapsed": v.elapsed, "markers": marker_stats(g.thinking, markers), "extra": g.extra}
            append_jsonl(out, row)
            rows.append(row)
    return rows
