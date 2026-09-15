from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import yaml


@dataclass
class BackendCfg:
    kind: str = "mock"                 # mock | openai | hf
    model: str = "mock-model"          # served model name or HF path
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    reasoning_effort: str = "xhigh"    # Qwen3.8 template accepts xhigh|medium|low
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 32768
    concurrency: int = 8
    think_open: str = "<think>"
    think_close: str = "</think>"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    extra: dict = field(default_factory=dict)


@dataclass
class BankCfg:
    seeds: list[str] = field(default_factory=lambda: ["data/tasks/coding_seed.jsonl", "data/tasks/knowledge_seed.jsonl", "data/tasks/math_seed.jsonl"])
    domains: list[str] = field(default_factory=lambda: ["coding", "knowledge", "math"])
    decontam_against: list[str] = field(default_factory=list)   # jsonl/txt files of eval questions
    ngram: int = 13
    split: dict = field(default_factory=lambda: {"mine": 0.6, "calib": 0.2, "eval": 0.2})


@dataclass
class RolloutCfg:
    samples_per_task: int = 2
    max_tasks: int = 0
    out: str = "runs/rollouts.jsonl"


@dataclass
class SettleCfg:
    checkpoints: int = 8              # truncation points along the trace
    answer_max_tokens: int = 256
    min_think_tokens: int = 200       # skip probing very short traces


@dataclass
class DirectionCfg:
    layers: list[int] = field(default_factory=list)   # empty -> auto sweep middle third
    matrices: list[str] = field(default_factory=lambda: ["self_attn.o_proj", "mlp.down_proj"])
    ridge_lambda: float = 1e-2
    gamma: float = 1.0
    gamma_kernel: str = "flat"       # flat | gaussian
    kernel_width: float = 4.0
    atoms: list[str] = field(default_factory=lambda: ["coding", "knowledge", "format", "language"])


@dataclass
class QuantCfg:
    calib_samples: int = 512
    calib_max_tokens: int = 2048
    gguf_types: list[str] = field(default_factory=lambda: ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"])
    llamacpp_dir: str = "~/llama.cpp"
    w4a16_scheme: str = "W4A16"


@dataclass
class EvalCfg:
    seeds: int = 5
    max_tasks: int = 0
    bootstrap: int = 2000
    accuracy_tolerance: float = 0.01


@dataclass
class Config:
    name: str = "swiftlab-run"
    run_dir: str = "runs/default"
    backend: BackendCfg = field(default_factory=BackendCfg)
    bank: BankCfg = field(default_factory=BankCfg)
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    settle: SettleCfg = field(default_factory=SettleCfg)
    direction: DirectionCfg = field(default_factory=DirectionCfg)
    quant: QuantCfg = field(default_factory=QuantCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(dc, data: dict | None):
    if not data:
        return dc
    for k, v in data.items():
        if not hasattr(dc, k):
            raise KeyError(f"unknown config key: {k}")
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__"):
            _merge(cur, v)
        else:
            setattr(dc, k, v)
    return dc


def load_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for k, v in (overrides or {}).items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], v)
    return cfg
