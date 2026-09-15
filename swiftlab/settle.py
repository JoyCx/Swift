"""Answer-settle probing: where in the trace did the answer stop changing?

For a trace T with final answer A, truncate T at checkpoint c (a segment boundary),
force-close the think block and let the model answer greedily.  The *settle point* is
the earliest checkpoint whose forced answer is already correct.  Tokens after the
settle point are *overspent*.  A trace whose forced answer was correct at some
checkpoint but whose real final answer is wrong was *derailed* by its own thinking —
that is the "overthinking error" Swift's authors target.

Categories: tight | overspent | derailed | wrong | short (not probed).
Cost: binary search uses ~log2(checkpoints) short generations per trace; `scan` probes all.
"""
from __future__ import annotations
from .backends import Backend, GenRequest
from .tasks import Task, verify
from .trace import segment


def checkpoints_for(thinking: str, n: int, markers=None) -> list[int]:
    segs = segment(thinking, markers)
    if len(segs) <= 1:
        return [len(thinking)]
    ends = [s.end for s in segs]
    if len(ends) <= n:
        return ends
    step = len(ends) / n
    return sorted({ends[min(len(ends) - 1, int((i + 1) * step) - 1)] for i in range(n)})


def probe_settle(backend: Backend, task: Task, row: dict, checkpoints: int = 8, answer_max_tokens: int = 256,
                 min_think_tokens: int = 200, mode: str = "bisect", markers=None) -> dict:
    thinking = row["thinking"]
    res = {"category": "short", "settle_char": None, "settle_tokens": None, "overspent_tokens": 0, "probes": []}
    if row["think_tokens"] < min_think_tokens or not thinking.strip():
        return res
    req = GenRequest(task.id, task.prompt, row["seed"], context=task.context, meta={"difficulty": task.difficulty})
    cps = checkpoints_for(thinking, checkpoints, markers)
    cache: dict[int, bool] = {}

    def ok_at(c: int) -> bool:
        if c not in cache:
            ans = backend.complete_prefix(req, thinking[:c], max_tokens=answer_max_tokens)
            cache[c] = verify(task, ans).correct
            res["probes"].append({"char": c, "correct": cache[c]})
        return cache[c]

    if mode == "scan":
        firsts = [c for c in cps if ok_at(c)]
        first_ok = firsts[0] if firsts else None
    else:  # bisect under the (approximate) monotone assumption; fall back to scanning left if violated
        lo, hi, first_ok = 0, len(cps) - 1, None
        if ok_at(cps[hi]):
            first_ok = cps[hi]
            while lo < hi:
                mid = (lo + hi) // 2
                if ok_at(cps[mid]):
                    first_ok = cps[mid]; hi = mid
                else:
                    lo = mid + 1
        else:
            for c in cps[:-1]:                 # derailment check: was it ever right?
                if ok_at(c):
                    first_ok = c; break
    final_ok = bool(row["correct"])
    tok = backend.count_tokens
    if first_ok is None:
        res["category"] = "wrong" if not final_ok else "tight"
        return res
    settle_tokens = tok(thinking[:first_ok])
    res.update({"settle_char": first_ok, "settle_tokens": settle_tokens})
    if not final_ok:
        res["category"] = "derailed"
        res["overspent_tokens"] = max(0, row["think_tokens"] - settle_tokens)
    else:
        over = max(0, row["think_tokens"] - settle_tokens)
        res["overspent_tokens"] = over
        res["category"] = "overspent" if over > 0.1 * row["think_tokens"] else "tight"
    return res


def probe_all(backend: Backend, tasks_by_id: dict[str, Task], rows: list[dict], **kw) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        r["settle"] = probe_settle(backend, tasks_by_id[r["task_id"]], r, **kw)
        out.append(r)
    return out


def waste_summary(rows: list[dict]) -> dict:
    n = len(rows) or 1
    cats = {}
    for r in rows:
        cats[r["settle"]["category"]] = cats.get(r["settle"]["category"], 0) + 1
    total = sum(r["think_tokens"] for r in rows) or 1
    over = sum(r["settle"]["overspent_tokens"] for r in rows)
    wrong_tok = sum(r["think_tokens"] for r in rows if r["settle"]["category"] == "wrong")
    return {"n": n, "categories": cats, "think_tokens": total, "overspent_tokens": over, "overspent_share": over / total,
            "wrong_trace_tokens": wrong_tok, "removable_share_upper_bound": (over + wrong_tok) / total,
            "derailed_share": cats.get("derailed", 0) / n, "wrong_share": cats.get("wrong", 0) / n}
