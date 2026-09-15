"""TaskBank: seeds + synthetic tasks -> decontaminated, hash-split, stratified bank."""
from __future__ import annotations
import random
from pathlib import Path
from ..util import read_jsonl, write_jsonl, stable_hash
from .base import Task
from .decontam import Decontaminator, load_eval_texts
from .coding import synthetic_coding_tasks
from .knowledge import synthetic_knowledge_tasks
from .math_ import synthetic_math_tasks

_SYNTH = {"coding": synthetic_coding_tasks, "knowledge": synthetic_knowledge_tasks, "math": synthetic_math_tasks}


class TaskBank:
    def __init__(self, tasks: list[Task]):
        self.tasks = tasks
        self.by_id = {t.id: t for t in tasks}

    @classmethod
    def build(cls, seed_paths: list[str], synthetic_per_domain: int = 0, domains: list[str] | None = None,
              decontam_against: list[str] | None = None, ngram: int = 13, seed: int = 0) -> tuple["TaskBank", list[dict]]:
        tasks: list[Task] = []
        for p in seed_paths:
            if Path(p).exists():
                tasks += [Task.from_dict(d) for d in read_jsonl(p)]
        for d in (domains or list(_SYNTH)):
            if synthetic_per_domain and d in _SYNTH:
                tasks += _SYNTH[d](synthetic_per_domain, seed=seed)
        if domains:
            tasks = [t for t in tasks if t.domain in domains]
        dec = Decontaminator(load_eval_texts(decontam_against or []), ngram=ngram)
        tasks, dropped = dec.filter(tasks)
        return cls(tasks), dropped

    @classmethod
    def load(cls, path: str) -> "TaskBank":
        return cls([Task.from_dict(d) for d in read_jsonl(path)])

    def save(self, path: str) -> int:
        return write_jsonl(path, (t.to_dict() for t in self.tasks))

    def split(self, fractions: dict[str, float], salt: str = "split") -> dict[str, list[Task]]:
        """Deterministic hash split so that eval tasks never leak into mining/calibration."""
        names = list(fractions)
        cum, acc = [], 0.0
        for n in names:
            acc += fractions[n]; cum.append(acc)
        out = {n: [] for n in names}
        for t in self.tasks:
            u = int(stable_hash(salt, t.id, n=8), 16) / 0xFFFFFFFF
            for n, c in zip(names, cum):
                if u <= c + 1e-12:
                    out[n].append(t); break
        return out

    def sample(self, n: int, seed: int = 0, stratify: bool = True) -> list[Task]:
        if n <= 0 or n >= len(self.tasks):
            return list(self.tasks)
        rng = random.Random(seed)
        if not stratify:
            return rng.sample(self.tasks, n)
        by_dom: dict[str, list[Task]] = {}
        for t in self.tasks:
            by_dom.setdefault(t.domain, []).append(t)
        out, doms = [], sorted(by_dom)
        per = max(1, n // len(doms))
        for d in doms:
            pool = by_dom[d]; rng.shuffle(pool)
            out += pool[:per]
        rest = [t for t in self.tasks if t not in out]
        rng.shuffle(rest)
        return (out + rest)[:n]
