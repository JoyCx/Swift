"""Math tasks: numeric answer compared with tolerance (boxed or 'Final answer:' line)."""
from __future__ import annotations
import re
from .base import Task, Verdict

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def extract_number(answer: str) -> float | None:
    cands = BOXED.findall(answer)
    if not cands:
        m = re.findall(r"(?:final answer|answer)\s*[:：]\s*([^\n]+)", answer, re.I)
        cands = m
    if not cands:
        nums = NUM.findall(re.sub(r"(?<=\d),(?=\d{3}\b)", "", answer))
        cands = nums[-1:]
    for c in reversed(cands):
        m = NUM.search(c.replace(",", ""))
        if m:
            s = m.group(0)
            try:
                if "/" in s:
                    a, b = s.split("/"); return float(a) / float(b)
                return float(s)
            except (ValueError, ZeroDivisionError):
                continue
    return None


def verify_numeric(task: Task, answer: str) -> Verdict:
    got = extract_number(answer)
    gold = float(task.verify["gold"])
    tol = float(task.verify.get("tol", 1e-6))
    if got is None:
        return Verdict(False, "no number found")
    ok = abs(got - gold) <= tol * max(1.0, abs(gold))
    return Verdict(ok, f"got={got} gold={gold}")


def synthetic_math_tasks(n: int, seed: int = 0) -> list[Task]:
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            a, b, c = rng.randrange(2, 60), rng.randrange(2, 60), rng.randrange(1, 30)
            q, g = f"Compute {a} * {b} + {c}.", a * b + c
        elif kind == 1:
            a, d, k = rng.randrange(1, 20), rng.randrange(1, 9), rng.randrange(5, 40)
            q, g = f"An arithmetic sequence starts at {a} with common difference {d}. What is the sum of its first {k} terms?", k * (2 * a + (k - 1) * d) // 2
        elif kind == 2:
            a, b = rng.randrange(10, 400), rng.randrange(10, 400)
            import math; q, g = f"What is the greatest common divisor of {a} and {b}?", math.gcd(a, b)
        else:
            m = rng.randrange(3, 12); r = rng.randrange(1, m); x0 = rng.randrange(20, 200)
            q, g = f"Find the smallest integer x >= {x0} such that x mod {m} = {r}.", next(x for x in range(x0, x0 + m) if x % m == r)
        out.append(Task(id=f"syn-math-{seed}-{i}", domain="math", prompt=q + " Put the final numeric answer in \\boxed{}.",
                        verify={"kind": "numeric", "gold": g}, difficulty=0.2 + 0.15 * kind, source="synthetic", meta={"kind": kind}))
    return out
