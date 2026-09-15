from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

import numpy as np


@dataclass
class GenRequest:
    task_id: str
    prompt: str
    seed: int
    context: str = ""
    max_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    logit_bias: dict[int, float] | None = None      # token id -> bias (inference-time penalizer)
    meta: dict = field(default_factory=dict)


@dataclass
class Generation:
    task_id: str
    seed: int
    thinking: str
    answer: str
    think_tokens: int
    answer_tokens: int
    latency: float
    finish_reason: str = "stop"
    model: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Backend:
    """Minimal interface every backend implements. Activation/logit methods are optional."""
    name = "base"

    def generate(self, req: GenRequest) -> Generation:
        raise NotImplementedError

    def generate_many(self, reqs: Sequence[GenRequest], concurrency: int = 4) -> list[Generation]:
        if concurrency <= 1 or len(reqs) <= 1:
            return [self.generate(r) for r in reqs]
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            return list(ex.map(self.generate, reqs))

    def complete_prefix(self, req: GenRequest, think_prefix: str, max_tokens: int = 256) -> str:
        """Force-close the thinking block after `think_prefix` and return the answer text."""
        raise NotImplementedError

    # ---- optional, needed for direction extraction / KL / weight editing -----------------
    def capture(self, texts: list[str], spans: list[tuple[int, int]], layers: list[int]) -> dict[int, np.ndarray]:
        """Mean residual-stream vector per (text, char-span) at each layer -> {layer: [n, hidden]}"""
        raise NotImplementedError(f"{self.name} cannot capture activations")

    def next_token_dist(self, prompts: list[str]) -> np.ndarray:
        """Next-token probability rows [n, vocab] used for KL drift measurement."""
        raise NotImplementedError(f"{self.name} cannot return logits")

    def n_layers(self) -> int:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        from ..util import approx_tokens
        return approx_tokens(text)


def make_backend(cfg, **kw) -> Backend:
    kind = cfg.kind
    if kind == "mock":
        from .mock import MockBackend
        return MockBackend(cfg, **kw)
    if kind == "openai":
        from .openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(cfg, **kw)
    if kind == "hf":
        from .hf import HFBackend
        return HFBackend(cfg, **kw)
    raise ValueError(f"unknown backend kind {kind!r}")
