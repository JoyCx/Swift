"""Rank-one surgical weight edit (SRA eq. 6 / heretic-style orthogonalization).

For every selected layer l and every output-side matrix W (rows = residual dims):
    W' = (I - gamma_l * v_l v_l^T) W
which removes (gamma=1) or attenuates the component of the module's *output* that
lies along the overthinking direction.  Nothing else in the model changes, the edit
is a pure function of the weights, and it survives quantization because it is baked
into the tensors before calibration.
"""
from __future__ import annotations
import json, re, shutil
from pathlib import Path

import numpy as np

from .directions import gamma_weights

DEFAULT_MATRICES = ["self_attn.o_proj", "mlp.down_proj"]


def rank_one_project(W: np.ndarray, v: np.ndarray, gamma: float) -> np.ndarray:
    """W: [out, in] as stored by torch Linear; v: [out] unit vector."""
    v = v / (np.linalg.norm(v) + 1e-12)
    return W - gamma * np.outer(v, v @ W)


def make_edit(bundle: dict, layers: list[int] | None = None, gamma: float = 1.0, kernel: str = "flat",
              center: float | None = None, width: float = 4.0, matrices: list[str] | None = None) -> dict:
    """Turn a direction bundle into a concrete edit spec: {"layers": {l: {"v", "gamma"}}, "matrices": [...]}."""
    layers = layers or bundle.get("best_layers") or list(bundle["layers"])
    center = float(np.mean(layers)) if center is None else center
    gw = gamma_weights(layers, center, width, gamma, kernel)
    return {"layers": {int(l): {"v": bundle["layers"][int(l)]["v"], "gamma": gw[l]} for l in layers},
            "matrices": matrices or DEFAULT_MATRICES, "kernel": kernel, "gamma": gamma, "center": center, "width": width}


# --------------------------------------------------------------------- HF / safetensors
def _tensor_layer_and_module(name: str) -> tuple[int, str] | None:
    m = re.match(r"^(?:model\.|transformer\.)?(?:language_model\.)?layers\.(\d+)\.(.+)\.weight$", name)
    if not m:
        m = re.match(r"^model\.layers\.(\d+)\.(.+)\.weight$", name)
    return (int(m.group(1)), m.group(2)) if m else None


def apply_in_place(model, edit: dict) -> dict:
    """Edit a loaded transformers model in place; returns originals for `restore_in_place`."""
    import torch
    saved = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            lm = _tensor_layer_and_module(name)
            if not lm or lm[0] not in edit["layers"] or lm[1] not in edit["matrices"]:
                continue
            spec = edit["layers"][lm[0]]
            v = torch.tensor(spec["v"], dtype=torch.float32, device=p.device)
            v = v / v.norm()
            saved[name] = p.detach().clone()
            W = p.float()
            p.copy_((W - spec["gamma"] * torch.outer(v, v @ W)).to(p.dtype))
    return saved


def restore_in_place(model, saved: dict) -> None:
    import torch
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in saved:
                p.copy_(saved[name])


def apply_to_safetensors(model_dir: str | Path, out_dir: str | Path, edit: dict, dtype: str | None = None) -> dict:
    """Stream every shard, edit matching tensors, write a new model directory (configs/tokenizer copied)."""
    import torch
    from safetensors.torch import load_file, save_file
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    touched = []
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {model_dir}")
    for shard in shards:
        tensors = load_file(str(shard))
        for name, t in tensors.items():
            lm = _tensor_layer_and_module(name)
            if not lm or lm[0] not in edit["layers"] or lm[1] not in edit["matrices"]:
                continue
            spec = edit["layers"][lm[0]]
            v = torch.tensor(spec["v"], dtype=torch.float32); v = v / v.norm()
            W = t.float()
            tensors[name] = (W - spec["gamma"] * torch.outer(v, v @ W)).to(getattr(torch, dtype) if dtype else t.dtype)
            touched.append(name)
        save_file(tensors, str(out_dir / shard.name), metadata={"format": "pt"})
    for f in model_dir.iterdir():
        if f.suffix in (".json", ".txt", ".model", ".jinja", ".py") or f.name.endswith(".tiktoken"):
            shutil.copy(f, out_dir / f.name)
    (out_dir / "swiftlab_edit.json").write_text(json.dumps({"matrices": edit["matrices"], "gamma": edit.get("gamma"), "kernel": edit.get("kernel"),
                                                            "layers": {str(l): s["gamma"] for l, s in edit["layers"].items()}, "touched": touched}, indent=2))
    return {"touched": touched, "out_dir": str(out_dir)}
