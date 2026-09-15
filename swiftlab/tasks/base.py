"""Task schema and verification dispatch.

A task is measurable by construction: coding tasks execute, knowledge tasks are
grepped against gold strings/regexes, math tasks are compared numerically.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Task:
    id: str
    domain: str                      # coding | knowledge | math | custom
    prompt: str
    verify: dict                     # {"kind": "code_tests"|"regex"|"numeric"|"exact", ...}
    difficulty: float = 0.5          # 0 easy .. 1 hard (used for stratified sampling / reporting)
    context: str = ""                # optional system/context string, kept identical across arms
    source: str = "seed"             # seed | synthetic | user
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Task":
        return Task(**{k: d[k] for k in Task.__dataclass_fields__ if k in d})


@dataclass
class Verdict:
    correct: bool
    detail: str = ""
    elapsed: float = 0.0             # seconds spent in verification (e.g. test execution)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def verify(task: Task, answer: str) -> Verdict:
    kind = task.verify.get("kind")
    if kind == "code_tests":
        from .coding import verify_code
        return verify_code(task, answer)
    if kind in ("regex", "exact"):
        from .knowledge import verify_knowledge
        return verify_knowledge(task, answer)
    if kind == "numeric":
        from .math_ import verify_numeric
        return verify_numeric(task, answer)
    raise ValueError(f"unknown verify kind {kind!r} for task {task.id}")
