"""Deterministic mock model for offline development and tests.

It behaves like a reasoning model with a *planted* overthinking direction d* and a
*planted* capability direction c* in a 64-dim residual stream:

* traces contain productive steps plus "Wait, let me double-check..." loops whose
  frequency is `state['overthink_rate']`;
* a runaway loop can derail an otherwise-correct answer (an "overthinking error");
* `capture()` returns activations where overthinking spans carry d* **and** a bit of
  c* (entanglement), so a naive diff-of-means direction is "dirty";
* `apply_edit()` reduces loops in proportion to cos²(v, d*) and damages accuracy in
  proportion to cos²(v, c*) — so surgical cleaning measurably matters.

All randomness is keyed on (task_id, seed) so paired A/B comparisons are exact.
"""
from __future__ import annotations
import math, re, time
from dataclasses import dataclass

import numpy as np

from .base import Backend, GenRequest, Generation
from ..util import rng_for, derive_seed

MARK_PHRASES = ["Wait, let me double-check that.", "Hmm, actually maybe I should reconsider.", "But wait, is that right?",
                "Hold on, let me verify this one more time.", "Actually, let me re-examine the previous step."]
_OT_RE = re.compile(r"\b(wait|hmm|double-check|reconsider|verify|re-examine|hold on|actually)\b", re.I)
_CAP_RE = re.compile(r"(def |```|return |Step \d|therefore|compute)", re.I)
_FMT_RE = re.compile(r"(Final answer|\\boxed)", re.I)
_HARM_TAG = "[harmful]"

# tiny vocabulary so logit_bias tests can address marker tokens by id
MOCK_VOCAB = {w: i for i, w in enumerate(["wait", "hmm", "actually", "double-check", "reconsider", "verify", "step", "the", "return", "answer"])}


@dataclass
class Plan:
    steps: list[str]
    loops: dict[int, int]      # step index -> loop repetitions
    settle_idx: int
    correct: bool
    runaway: bool
    would_correct: bool = False     # correct before a runaway loop derailed it
    runaway_step: int = -1


class MockBackend(Backend):
    name = "mock"

    def __init__(self, cfg=None, bank=None, hidden: int = 64, n_layers: int = 12, seed: int = 1234, **kw):
        self.cfg = cfg
        self.hidden, self.layers = hidden, n_layers
        g = np.random.default_rng(seed)
        self.d_star = _unit(g.standard_normal(hidden))
        c = g.standard_normal(hidden); c -= c @ self.d_star * self.d_star
        self.c_star = _unit(c)
        f = g.standard_normal(hidden); f -= f @ self.d_star * self.d_star; f -= f @ self.c_star * self.c_star
        self.f_star = _unit(f)
        # refusal direction, deliberately correlated with the overthinking direction (cos ~0.4)
        rr = g.standard_normal(hidden); rr -= rr @ self.c_star * self.c_star; rr -= rr @ self.f_star * self.f_star
        rr = _unit(rr - (rr @ self.d_star) * self.d_star)
        self.r_star = _unit(0.4 * self.d_star + 0.917 * rr)
        self.state = {"overthink_rate": 0.6, "acc_delta": 0.0, "drift": 0.0, "refuse_rate": 0.9, "variant": "mock-base"}
        self.tasks = dict(getattr(bank, "by_id", {}) or {})
        self.calls = 0

    # ------------------------------------------------------------------ helpers
    def n_layers(self) -> int:
        return self.layers

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def _layer_w(self, l: int) -> float:
        mid = self.layers / 2
        return math.exp(-((l - mid) ** 2) / (2 * (self.layers / 5) ** 2))

    def _task(self, req: GenRequest):
        return self.tasks.get(req.task_id)

    def _plan(self, req: GenRequest, effective_rate: float) -> Plan:
        task = self._task(req)
        diff = float(req.meta.get("difficulty", task.difficulty if task else 0.5))
        rng = rng_for(req.task_id, req.seed, "plan")
        k = 3 + int(diff * 6)
        steps = [f"Step {i + 1}: work out part {i + 1} of the problem, compute intermediate result r{i + 1}." for i in range(k)]
        u_correct = rng.random()
        p_correct = 0.92 - 0.45 * diff + self.state["acc_delta"]
        loops, runaway, run_step = {}, False, -1
        for i in range(k):
            u = rng.random(); m = 1 + int(rng.random() * 3); u_run = rng.random()
            if u < effective_rate:
                loops[i] = m
                if u_run < 0.25 * effective_rate and i >= k - 2 and not runaway:
                    loops[i] = 8 + int(rng.random() * 6); runaway = True; run_step = i
        would = u_correct < p_correct
        return Plan(steps, loops, max(1, math.ceil(k * 0.6)), would and not runaway, runaway, would, run_step)

    def _render(self, plan: Plan) -> tuple[str, list[int]]:
        parts, ends = [], []
        for i, s in enumerate(plan.steps):
            parts.append(s)
            for j in range(plan.loops.get(i, 0)):
                parts.append(MARK_PHRASES[(i + j) % len(MARK_PHRASES)] + f" Re-deriving r{i + 1} again... same value.")
            ends.append(len("\n".join(parts)))
        return "\n".join(parts), ends

    def _answer(self, req: GenRequest, correct: bool) -> str:
        task = self._task(req)
        if task is None:
            return "Final answer: 42" if correct else "Final answer: unknown"
        v = task.verify; kind = v.get("kind")
        if kind == "numeric":
            g = v["gold"]; return f"So the result is \\boxed{{{g if correct else g + 1}}}."
        if kind == "code_tests":
            ref = task.meta.get("reference_code")
            if correct and ref:
                return "Here is the solution:\n```python\n" + ref + "\n```"
            fn = v["tests"][0]["fn"]
            return f"```python\ndef {fn}(*a):\n    return None\n```"
        if kind == "regex":
            gold = task.meta.get("gold_text") or v["patterns"][0]
            return f"Final answer: {gold if correct else 'unknown'}"
        if kind == "exact":
            return f"Final answer: {v['gold'][0] if correct else 'unknown'}"
        return "Final answer: ?"

    def _effective_rate(self, req: GenRequest) -> float:
        rate = self.state["overthink_rate"]
        if req.logit_bias:
            mean_bias = float(np.mean([abs(b) for b in req.logit_bias.values()]))
            rate *= max(0.0, 1.0 - min(1.0, mean_bias / 5.0))
        eff = (req.reasoning_effort or (self.cfg.reasoning_effort if self.cfg else "xhigh"))
        rate *= {"low": 0.6, "medium": 0.8}.get(eff, 1.0)
        return rate

    # ------------------------------------------------------------------ API
    def generate(self, req: GenRequest) -> Generation:
        self.calls += 1
        t0 = time.perf_counter()
        plan = self._plan(req, self._effective_rate(req))
        thinking, _ = self._render(plan)
        answer = self._answer(req, plan.correct)
        if _HARM_TAG in req.prompt and rng_for(req.task_id, req.seed, "refuse").random() < self.state["refuse_rate"]:
            thinking = "Step 1: this request looks sensitive. Wait, let me double-check the policy. I should decline."
            answer = "I'm sorry, but I can't help with that."
        tt, at = self.count_tokens(thinking), self.count_tokens(answer)
        lat = (tt + at) / 400.0          # pretend 400 tok/s
        return Generation(req.task_id, req.seed, thinking, answer, tt, at, lat, model=self.state["variant"],
                          extra={"mock_correct": plan.correct, "runaway": plan.runaway, "wall": time.perf_counter() - t0})

    def complete_prefix(self, req: GenRequest, think_prefix: str, max_tokens: int = 256) -> str:
        plan = self._plan(req, self._effective_rate(req))
        _, ends = self._render(plan)
        settled_at = ends[plan.settle_idx - 1]
        if len(think_prefix) < settled_at:
            return self._answer(req, False)          # before settling: answer is still wrong / partial
        if plan.runaway:
            # correct until the runaway loop has run for a while, then derailed
            derail_at = (ends[plan.runaway_step - 1] if plan.runaway_step > 0 else 0) + 0.5 * (ends[plan.runaway_step] - (ends[plan.runaway_step - 1] if plan.runaway_step > 0 else 0))
            return self._answer(req, plan.would_correct and len(think_prefix) < derail_at)
        return self._answer(req, plan.correct)

    def capture(self, texts, spans, layers):
        out = {l: [] for l in layers}
        for text, (a, b) in zip(texts, spans):
            piece = text[a:b]
            ot = 1.0 if _OT_RE.search(piece) else 0.0
            harm = 1.0 if _HARM_TAG in text else 0.0
            cap = 1.0 if _CAP_RE.search(piece) else 0.0
            fmt = 1.0 if _FMT_RE.search(piece) else 0.0
            g = np.random.default_rng(derive_seed(text, a, b))
            for l in layers:
                w = self._layer_w(l)
                noise = 0.7 * g.standard_normal(self.hidden)
                vec = noise + w * (2.0 * ot * self.d_star + 1.1 * ot * self.c_star + 0.5 * cap * self.c_star + 1.2 * fmt * self.f_star + 2.5 * harm * self.r_star)
                out[l].append(vec)
        return {l: np.stack(v) for l, v in out.items()}

    def next_token_dist(self, prompts):
        rows = []
        for p in prompts:
            g = np.random.default_rng(derive_seed("dist", p))
            base = np.exp(g.standard_normal(len(MOCK_VOCAB)) * 2); base /= base.sum()
            d = min(1.0, self.state["drift"])
            rows.append((1 - d) * base + d / len(base))
        return np.stack(rows)

    # ------------------------------------------------------------------ surgery on the mock
    def apply_edit(self, edit: dict, variant: str = "mock-edited") -> dict:
        """edit = {"layers": {l: {"v": unit vec, "gamma": g}}}. Returns the new state."""
        eff, dmg, ref = 0.0, 0.0, 0.0
        for l, spec in edit["layers"].items():
            v = _unit(np.asarray(spec["v"], dtype=float)); gam = float(spec["gamma"]); w = self._layer_w(int(l))
            eff += gam * w * float(v @ self.d_star) ** 2
            dmg += gam * w * float(v @ self.c_star) ** 2
            ref += gam * w * float(v @ self.r_star) ** 2
        n = max(1, len(edit["layers"]))
        eff /= n; dmg /= n; ref /= n
        self.state["refuse_rate"] = self.state["refuse_rate"] * max(0.0, 1.0 - 1.4 * ref)
        self.state["overthink_rate"] = self.state["overthink_rate"] * max(0.0, 1.0 - 1.4 * eff)
        self.state["acc_delta"] -= 3.0 * dmg
        self.state["drift"] += 0.02 * eff + 2.0 * dmg
        self.state["variant"] = variant
        return dict(self.state)

    def clone(self) -> "MockBackend":
        m = MockBackend.__new__(MockBackend)
        m.__dict__.update({k: v for k, v in self.__dict__.items()})
        m.state = dict(self.state)
        return m


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
