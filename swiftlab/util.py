from __future__ import annotations
import hashlib, json, os, random, time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def stable_hash(*parts: Any, n: int = 16) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:n]


def derive_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from (task_id, sample_idx, ...), same for every model arm."""
    return int(stable_hash(*parts, n=8), 16) & 0x7FFFFFFF


def rng_for(*parts: Any) -> random.Random:
    return random.Random(derive_seed(*parts))


@contextmanager
def timer() -> Iterator[dict]:
    d = {"start": time.perf_counter(), "seconds": None}
    try:
        yield d
    finally:
        d["seconds"] = time.perf_counter() - d["start"]


def approx_tokens(text: str) -> int:
    """Cheap token estimate used only when a backend does not report counts."""
    return max(1, int(len(text) / 3.6 + 0.5))


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.lower() in ("1", "true", "yes", "on")
