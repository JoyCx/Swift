"""How much of Swift's weight change survives each NInfer storage format (round-to-nearest).

For a matrix pair (W_base, W_swift = W_base + D) and a quantizer Q:
  preserved = <Q(W_swift) - Q(W_base), D> / |D|^2   (1.0 = Swift's change fully kept, 0 = erased)
  cos       = cosine(Q(W_swift) - Q(W_base), D)     (direction fidelity of the kept change)
  noise/D   = |Q(W_swift) - W_swift| / |D|          (quantisation error size relative to Swift's change)
Also the same for the rank-32 clean delta applied to a quantized base as an unquantized side branch
(noise-free by construction, reported for reference).

Formats (symmetric, as defined for NInfer artifacts):
  nvfp4   E2M1 codes, group 16, FP8-E4M3 group scale under a per-tensor global scale (6*448/amax)
  fp8row  E4M3 per-row scale amax/448
  q4g64 / q5g64 / q6g64  int grouped (qmax 7/15/31), group 64, FP16 scale amax/qmax
  w8g32  int8 grouped (qmax 127), group 32

CPU only, background priority.
"""
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowres import limit  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fp8_round(x):
    return x.clamp(-448, 448).to(torch.float8_e4m3fn).float()


def q_nvfp4(w):
    rows, cols = w.shape
    pad = (-cols) % 16
    x = torch.nn.functional.pad(w, (0, pad)).reshape(rows, -1, 16)
    g = 6.0 * 448.0 / w.abs().max()
    s = fp8_round(x.abs().amax(-1, keepdim=True) / 6.0 * g).clamp(min=1e-8)
    v = x * g / s
    idx = (v.abs().unsqueeze(-1) - E2M1).abs().argmin(-1)
    q = E2M1[idx] * v.sign() * s / g
    return q.reshape(rows, -1)[:, :cols]


def q_fp8row(w):
    s = (w.abs().amax(1, keepdim=True) / 448.0).clamp(min=1e-12)
    return fp8_round(w / s) * s


def q_group(bits, group):
    qmax = 2 ** (bits - 1) - 1

    def f(w):
        rows, cols = w.shape
        pad = (-cols) % group
        x = torch.nn.functional.pad(w, (0, pad)).reshape(rows, -1, group)
        s = (x.abs().amax(-1, keepdim=True) / qmax).half().float().clamp(min=2.0 ** -24)
        q = torch.clamp(torch.round(x / s), -qmax - 1, qmax) * s
        return q.reshape(rows, -1)[:, :cols]
    return f


FORMATS = {"nvfp4": q_nvfp4, "fp8row": q_fp8row, "q4g64": q_group(4, 64), "q5g64": q_group(5, 64),
           "q6g64": q_group(6, 64), "w8g32": q_group(8, 32)}
BASE, SWIFT = "A:/models/Qwen3.8-27B-base-bf16", "A:/models/Qwen3.8-27B"
NAMES = [f"model.language_model.layers.{l}.{m}.weight" for l in (5, 20, 35, 50, 60)
         for m in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")] + \
        [f"model.language_model.layers.{l}.self_attn.{m}.weight" for l in (7, 31, 55) for m in ("q_proj", "o_proj")]


def main():
    limit(0.1, 6) if torch.cuda.is_available() and torch.cuda.device_count() else None
    torch.set_num_threads(6)
    wmap = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
    out = []
    for name in NAMES:
        get = lambda root: safe_open(f"{root}/{wmap[name]}", "pt").get_tensor(name).float()
        wb, ws = get(BASE), get(SWIFT)
        d = ws - wb
        dd = (d * d).sum()
        rec = dict(name=name.replace("model.language_model.layers.", "L"), delta_rel=(d.norm() / wb.norm()).item())
        for fmt, q in FORMATS.items():
            qs, qb = q(ws), q(wb)
            k = qs - qb
            rec[fmt] = dict(preserved=round(((k * d).sum() / dd).item(), 3),
                            cos=round(((k * d).sum() / (k.norm() * d.norm())).item(), 3),
                            noise_over_delta=round(((qs - ws).norm() / d.norm()).item(), 2))
        out.append(rec)
        print(json.dumps(rec), flush=True)
    Path("A:/swift/runs/swiftdiff/quant_survival.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
