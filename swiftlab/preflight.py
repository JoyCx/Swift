"""Preflight: validate the real environment before spending hours of GPU time.

Two phases, both on real code (no stand-in model anywhere):

* local  — build a tiny real task set (one per domain) and confirm every verifier works:
           the coding reference solution executes and passes, a wrong one fails, the
           knowledge grep matches, the math numeric check matches, and decontamination
           drops an overlapping prompt. This proves the measurable half of the harness.
* server — construct the configured backend (openai/hf), generate one real completion per
           domain, confirm the thinking block splits and is non-empty, confirm the
           forced-prefix continuation (`complete_prefix`, needed by settle probing) returns
           an answer, and confirm token counting works. Then verify the model's own answers.

Exit 0 only if every hard check passes. WARNs (e.g. the real model got a task wrong) do not
fail preflight; they are expected and just reported.
"""
from __future__ import annotations
import traceback

from .tasks.coding import synthetic_coding_tasks
from .tasks.knowledge import synthetic_knowledge_tasks
from .tasks.math_ import synthetic_math_tasks
from .tasks import verify
from .tasks.decontam import Decontaminator
from .trace import split_thinking


class _Report:
    def __init__(self):
        self.rows = []
        self.hard_fail = False

    def add(self, name, status, detail=""):
        self.rows.append((name, status, detail))
        if status == "FAIL":
            self.hard_fail = True

    def show(self):
        w = max(len(n) for n, _, _ in self.rows)
        print("\n== swiftlab preflight ==")
        for n, st, d in self.rows:
            mark = {"PASS": "  ok", "FAIL": "FAIL", "WARN": "warn"}[st]
            print(f"[{mark}] {n.ljust(w)}  {d}")
        print(f"\n{'PREFLIGHT FAILED' if self.hard_fail else 'preflight passed'} "
              f"({sum(s=='PASS' for _,s,_ in self.rows)} pass, {sum(s=='WARN' for _,s,_ in self.rows)} warn, {sum(s=='FAIL' for _,s,_ in self.rows)} fail)")


def _sample_tasks():
    return {"coding": synthetic_coding_tasks(1, seed=7)[0], "knowledge": synthetic_knowledge_tasks(1, seed=7)[0], "math": synthetic_math_tasks(1, seed=7)[0]}


def check_verifiers(rep: _Report, tasks: dict) -> None:
    c = tasks["coding"]
    ok = verify(c, "```python\n" + c.meta["reference_code"] + "\n```")
    rep.add("verifier: coding executes reference", "PASS" if ok.correct else "FAIL", ok.detail[:80])
    fn = c.verify["tests"][0]["fn"]
    bad = verify(c, f"```python\ndef {fn}(*a):\n    return None\n```")
    rep.add("verifier: coding rejects wrong", "PASS" if not bad.correct else "FAIL")
    k = tasks["knowledge"]
    ok = verify(k, "Final answer: " + k.meta["gold_text"])
    rep.add("verifier: knowledge grep", "PASS" if ok.correct else "FAIL", ok.detail[:80])
    m = tasks["math"]
    ok = verify(m, f"\\boxed{{{m.verify['gold']}}}")
    rep.add("verifier: math numeric", "PASS" if ok.correct else "FAIL", ok.detail[:80])
    d = Decontaminator([k.prompt], ngram=8)
    rep.add("decontamination: flags overlap", "PASS" if d.contaminated(k.prompt) and not d.contaminated("write a function that adds two numbers") else "FAIL")


def check_server(rep: _Report, cfg, tasks: dict, n: int) -> None:
    from .backends import make_backend, GenRequest
    if cfg.backend.kind not in ("openai", "hf"):
        rep.add("backend kind", "FAIL", f"{cfg.backend.kind!r} is not a real backend"); return
    try:
        be = make_backend(cfg.backend)
    except Exception as e:  # noqa: BLE001
        rep.add(f"backend init ({cfg.backend.kind})", "FAIL", f"{type(e).__name__}: {e}")
        traceback.print_exc(); return
    rep.add(f"backend init ({cfg.backend.kind})", "PASS", f"model={cfg.backend.model}")
    probe = dict(list(tasks.items())[: max(1, n)]) if n else tasks
    for dom, t in probe.items():
        try:
            g = be.generate(GenRequest(t.id, t.prompt, seed=1, context=t.context, max_tokens=cfg.backend.max_tokens, reasoning_effort=cfg.backend.reasoning_effort))
        except Exception as e:  # noqa: BLE001
            rep.add(f"generate [{dom}]", "FAIL", f"{type(e).__name__}: {e}"); continue
        rep.add(f"generate [{dom}]", "PASS", f"think={g.think_tokens} tok, answer={g.answer_tokens} tok")
        rep.add(f"thinking splits [{dom}]", "PASS" if g.think_tokens > 0 or g.thinking else "WARN",
                "empty thinking — check reasoning parser / effort" if not (g.think_tokens or g.thinking) else "")
        v = verify(t, g.answer)
        rep.add(f"model answer [{dom}]", "PASS" if v.correct else "WARN", ("correct" if v.correct else "model got it wrong (fine for preflight)"))
        try:
            ans = be.complete_prefix(GenRequest(t.id, t.prompt, 1, context=t.context), (g.thinking or "Let me solve this.")[:400], max_tokens=cfg.settle.answer_max_tokens if hasattr(cfg, "settle") else 256)
            rep.add(f"forced-prefix probe [{dom}]", "PASS" if ans.strip() else "FAIL", "settle probing needs this" if not ans.strip() else f"{len(ans)} chars")
        except Exception as e:  # noqa: BLE001
            rep.add(f"forced-prefix probe [{dom}]", "FAIL", f"{type(e).__name__}: {e} (see docs/RUNBOOK.md pitfalls)")
    try:
        nt = be.count_tokens("hello world, this is a token count check")
        rep.add("token counting", "PASS" if nt > 0 else "FAIL", f"{nt} tokens")
    except Exception as e:  # noqa: BLE001
        rep.add("token counting", "WARN", f"falls back to estimate: {e}")


def run_preflight(cfg, server: bool = True, n: int = 1) -> bool:
    rep = _Report()
    tasks = _sample_tasks()
    check_verifiers(rep, tasks)
    if server:
        check_server(rep, cfg, tasks, n)
    else:
        rep.add("server checks", "WARN", "skipped (--local-only)")
    rep.show()
    return not rep.hard_fail
