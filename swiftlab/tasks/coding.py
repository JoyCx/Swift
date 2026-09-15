"""Coding tasks: the answer must contain a python code block that passes hidden tests.

Verification runs in a subprocess with a wall-clock limit, no network, and a
scratch working directory.  Time-to-pass is recorded so that speed regressions
are visible as well as correctness.
"""
from __future__ import annotations
import json, os, re, subprocess, sys, tempfile, textwrap, time
from .base import Task, Verdict

CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

RUNNER = textwrap.dedent('''
import sys, json, traceback
sys.setrecursionlimit(10000)
ns = {}
try:
    exec(compile(open("solution.py").read(), "solution.py", "exec"), ns)
except Exception:
    print(json.dumps({"ok": False, "stage": "load", "err": traceback.format_exc()[-800:]})); sys.exit(0)
tests = json.load(open("tests.json"))
failed = []
for i, t in enumerate(tests):
    try:
        fn = ns[t["fn"]]
        got = json.loads(json.dumps(fn(*t["args"]), default=str))   # canonicalise tuples/sets like the stored expectation
        if got != t["expect"]:
            failed.append({"i": i, "got": repr(got)[:200], "expect": repr(t["expect"])[:200]})
    except Exception:
        failed.append({"i": i, "err": traceback.format_exc()[-400:]})
print(json.dumps({"ok": not failed, "stage": "tests", "failed": failed[:5], "n": len(tests)}))
''')


def extract_code(answer: str) -> str | None:
    blocks = CODE_FENCE.findall(answer)
    if blocks:
        return max(blocks, key=len)
    # fallback: raw text that looks like python
    if re.search(r"^\s*def\s+\w+\(", answer, re.M):
        return answer
    return None


def verify_code(task: Task, answer: str, timeout: float | None = None) -> Verdict:
    code = extract_code(answer)
    if code is None:
        return Verdict(False, "no code block found")
    spec = task.verify
    timeout = timeout or float(spec.get("timeout", 10.0))
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "solution.py"), "w").write(code)
        open(os.path.join(d, "tests.json"), "w").write(json.dumps(spec["tests"]))
        open(os.path.join(d, "runner.py"), "w").write(RUNNER)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, "-I", "runner.py"], cwd=d, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Verdict(False, f"timeout after {timeout}s", elapsed=time.perf_counter() - t0)
        elapsed = time.perf_counter() - t0
    out = p.stdout.strip().splitlines()
    if not out:
        return Verdict(False, f"runner produced no output: {p.stderr[-300:]}", elapsed=elapsed)
    try:
        res = json.loads(out[-1])
    except json.JSONDecodeError:
        return Verdict(False, f"bad runner output: {out[-1][:200]}", elapsed=elapsed)
    limit = spec.get("time_limit")
    if res.get("ok") and limit and elapsed > float(limit):
        return Verdict(False, f"correct but too slow: {elapsed:.2f}s > {limit}s", elapsed=elapsed, extra=res)
    return Verdict(bool(res.get("ok")), json.dumps(res)[:500], elapsed=elapsed, extra=res)


# --- synthetic generator: unlimited, decontaminated-by-construction coding tasks --------------------
_TEMPLATES = [
    ("sum_even_squares", "Write a Python function `sum_even_squares(n)` that returns the sum of squares of all even integers in [0, n).",
     lambda n: sum(i * i for i in range(n) if i % 2 == 0), [(0,), (1,), (10,), (57,), (200,)]),
    ("count_vowels", "Write a Python function `count_vowels(s)` returning the number of vowels (aeiou, case-insensitive) in the string s.",
     lambda s: sum(c in "aeiouAEIOU" for c in s), [("",), ("hello",), ("Swift Qwen",), ("AEIOU xyz",)]),
    ("kth_largest", "Write a Python function `kth_largest(xs, k)` returning the k-th largest distinct value in list xs (k is 1-based); return None if it does not exist.",
     lambda xs, k: (sorted(set(xs), reverse=True)[k - 1] if 0 < k <= len(set(xs)) else None), [([3, 1, 2], 1), ([5, 5, 4], 2), ([1], 2), ([9, 8, 7, 7, 6], 3)]),
    ("run_length", "Write a Python function `run_length(s)` that returns the run-length encoding of s as a list of (char, count) tuples.",
     lambda s: [(c, len(list(g))) for c, g in __import__("itertools").groupby(s)], [("",), ("aaabcc",), ("abc",), ("zzzzzz",)]),
    ("balanced", "Write a Python function `balanced(s)` returning True if brackets ()[]{} in s are balanced and properly nested, else False.",
     None, [("()",), ("([)]",), ("{[()]}",), ("((",), ("",)]),
    ("digit_sum_until", "Write a Python function `digit_sum_until(n)` that repeatedly replaces n by the sum of its decimal digits until a single digit remains and returns it. n is a non-negative int.",
     None, [(0,), (9,), (38,), (999999,), (123456789,)]),
    ("merge_intervals", "Write a Python function `merge_intervals(iv)` that takes a list of [start, end] integer intervals and returns the merged, sorted list of non-overlapping intervals (as lists).",
     None, [([[1, 3], [2, 6], [8, 10], [15, 18]],), ([[1, 4], [4, 5]],), ([],), ([[5, 6], [1, 2]],)]),
]


def _balanced(s):
    st = []; pairs = {")": "(", "]": "[", "}": "{"}
    for c in s:
        if c in "([{": st.append(c)
        elif c in pairs:
            if not st or st.pop() != pairs[c]: return False
    return not st


def _digit_sum_until(n):
    while n >= 10: n = sum(int(c) for c in str(n))
    return n


def _merge_intervals(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out


_REF = {"balanced": _balanced, "digit_sum_until": _digit_sum_until, "merge_intervals": _merge_intervals}

# reference sources (used as the mock model's "correct" answer and as documentation of the spec)
REF_SOURCE = {
    "sum_even_squares": "def sum_even_squares(n):\n    return sum(i * i for i in range(n) if i % 2 == 0)",
    "count_vowels": "def count_vowels(s):\n    return sum(c in 'aeiouAEIOU' for c in s)",
    "kth_largest": "def kth_largest(xs, k):\n    u = sorted(set(xs), reverse=True)\n    return u[k - 1] if 0 < k <= len(u) else None",
    "run_length": "from itertools import groupby\ndef run_length(s):\n    return [(c, len(list(g))) for c, g in groupby(s)]",
    "balanced": "def balanced(s):\n    st, pairs = [], {')': '(', ']': '[', '}': '{'}\n    for c in s:\n        if c in '([{':\n            st.append(c)\n        elif c in pairs:\n            if not st or st.pop() != pairs[c]:\n                return False\n    return not st",
    "digit_sum_until": "def digit_sum_until(n):\n    while n >= 10:\n        n = sum(int(c) for c in str(n))\n    return n",
    "merge_intervals": "def merge_intervals(iv):\n    out = []\n    for a, b in sorted(iv):\n        if out and a <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out",
}


def synthetic_coding_tasks(n: int, seed: int = 0) -> list[Task]:
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        name, prompt, ref, cases = _TEMPLATES[i % len(_TEMPLATES)]
        ref = ref or _REF[name]
        # perturb: add extra random test cases so each instance is unique and decontaminated-by-construction
        extra = []
        for _ in range(3):
            base = cases[rng.randrange(len(cases))]
            if isinstance(base[0], int):
                extra.append((rng.randrange(0, 5000),) + base[1:])
            elif isinstance(base[0], str):
                extra.append(("".join(rng.choice("abcxyz(){}[] ") for _ in range(rng.randrange(0, 12))),) + base[1:])
            elif isinstance(base[0], list) and base[0] and isinstance(base[0][0], int):
                extra.append(([rng.randrange(-20, 20) for _ in range(rng.randrange(1, 8))],) + base[1:])
            else:
                extra.append(base)
        tests = [{"fn": name, "args": list(a), "expect": json.loads(json.dumps(ref(*a), default=str))} for a in list(cases) + extra]
        ex = tests[-1]
        example = f"Example: {name}({', '.join(repr(a) for a in ex['args'])}) should return {ex['expect']!r}."
        out.append(Task(id=f"syn-code-{seed}-{i}", domain="coding", prompt=prompt + "\n" + example + "\nReturn only one ```python code block.",
                        verify={"kind": "code_tests", "tests": tests, "timeout": 10}, difficulty=0.3 + 0.1 * (i % 5),
                        source="synthetic", meta={"template": name, "reference_code": REF_SOURCE[name]}))
    return out
