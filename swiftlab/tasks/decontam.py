"""Decontamination: drop tasks whose prompt shares an n-gram window with any eval question.

Uses the GPT-3 / Llama style 13-gram rule (configurable) plus exact normalized-hash dedupe.
Eval material may be jsonl (fields: prompt|question|text) or plain text (one item per line).
"""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Iterable
from .base import Task

_WS = re.compile(r"\s+")


def _norm(s: str) -> list[str]:
    return _WS.sub(" ", re.sub(r"[^\w\s]", " ", s.lower())).split()


def _ngrams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
    if len(tokens) < n:
        if tokens:
            yield tuple(tokens)      # short text: whole-text window
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i + n])


def load_eval_texts(paths: list[str | Path]) -> list[str]:
    texts = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        if p.suffix == ".jsonl":
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                texts.append(str(d.get("prompt") or d.get("question") or d.get("text") or ""))
        else:
            texts.extend(l for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
    return texts


class Decontaminator:
    def __init__(self, eval_texts: list[str], ngram: int = 13):
        self.n = ngram
        self.index: set[tuple[str, ...]] = set()
        for t in eval_texts:
            self.index.update(_ngrams(_norm(t), ngram))
        self.seen_hash: set[str] = set()

    def contaminated(self, text: str) -> bool:
        toks = _norm(text)
        return any(g in self.index for g in _ngrams(toks, self.n))

    def filter(self, tasks: list[Task]) -> tuple[list[Task], list[dict]]:
        keep, dropped = [], []
        for t in tasks:
            h = " ".join(_norm(t.prompt))
            if h in self.seen_hash:
                dropped.append({"id": t.id, "reason": "duplicate"}); continue
            self.seen_hash.add(h)
            if self.contaminated(t.prompt):
                dropped.append({"id": t.id, "reason": f"{self.n}-gram overlap with eval set"}); continue
            keep.append(t)
        return keep, dropped
