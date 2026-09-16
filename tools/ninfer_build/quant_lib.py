"""NVFP4 + row-scaled FP8 weight quantizers matching unsloth/llm-compressor.

Uses compressed_tensors' own observer + compressor so the produced words decode
under the exact convention the ninfer engine already consumes for the stock
unsloth source. Only weight statistics are (re)computed; activation divisors
(input_global_scale / d_x) are preserved from the stock source by the caller.
"""
from __future__ import annotations

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils.helpers import (
    calculate_qparams,
    generate_gparam,
)
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor

F8_MAX = 448.0

_NVFP4_ARGS = QuantizationArgs(
    num_bits=4,
    type=QuantizationType.FLOAT,
    group_size=16,
    symmetric=True,
    strategy=QuantizationStrategy.TENSOR_GROUP,
    dynamic=False,
    scale_dtype=torch.float8_e4m3fn,
)
_NVFP4_SCHEME = QuantizationScheme(targets=["Linear"], weights=_NVFP4_ARGS)


def nvfp4_global_scale(amax: torch.Tensor) -> torch.Tensor:
    """d_w/d_x global scale = 448*6/amax as f32[1] (compressed-tensors generate_gparam)."""
    a = amax.reshape(1).float()
    return generate_gparam(a.neg(), a)


def quant_nvfp4(weight: torch.Tensor, d_w: torch.Tensor | None = None):
    """Return (weight_packed[u8 N,K/2], weight_scale[f8 N,K/16], d_w[f32 1]).

    If d_w is given (e.g. a gate/up-shared global scale) it is used verbatim so
    the fused parent's divisors stay bit-identical; otherwise it is derived from
    this tensor's own amax.
    """
    w = weight.detach().to(torch.bfloat16)
    n, k = w.shape
    assert k % 16 == 0, k
    wf = w.float()
    if d_w is None:
        d_w = nvfp4_global_scale(wf.abs().amax())
    else:
        d_w = d_w.reshape(1).float()
    groups = wf.reshape(n, k // 16, 16)
    gmin = groups.amin(dim=-1)
    gmax = groups.amax(dim=-1)
    scale, _ = calculate_qparams(gmin, gmax, _NVFP4_ARGS, global_scale=d_w)
    sd = {
        "weight": w,
        "weight_scale": scale,
        "weight_global_scale": d_w,
    }
    out = NVFP4PackedCompressor.compress(sd, _NVFP4_SCHEME)
    return out["weight_packed"], out["weight_scale"], d_w


def quant_fp8_row(weight: torch.Tensor):
    """Return (code[f8 N,K], scale[bf16 N,1]) row-scaled to amax/448."""
    wf = weight.detach().float()
    amax = wf.abs().amax(dim=1, keepdim=True)
    amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    scale = (amax / F8_MAX).to(torch.bfloat16)
    code = (wf / scale.float()).clamp_(-F8_MAX, F8_MAX).to(torch.float8_e4m3fn)
    return code, scale
