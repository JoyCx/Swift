"""Build a Swift-abliterated compressed-tensors NVFP4/FP8 source for the ninfer converter.

Weights come from the abliterated Swift BF16 (`swift/`). Weight global scales d_w are
recomputed from the Swift weights (gate/up share a joint d_w so their fused parent stays
bit-identical). Activation divisors d_x are reused from the stock unsloth source (a <2%
approximation; abliteration barely shifts activation magnitudes). Layout/field structure and
config mirror the unsloth source so the converter's preflight accepts it unchanged.
Embedding, vision, MTP and k/v_scale are NOT emitted: the converter reads those from --model
or excludes them.
"""
from __future__ import annotations

import json, shutil, time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import quant_lib as q

ROOT = Path("A:/models/Qwen3.8-27B")
SWIFT = ROOT / "swift"
UNS = ROOT / "NVFP4_unsloth"
OUT = Path("A:/models/swift-nvfp4-source")
DEV = "cuda"

SKIP = ("model.language_model.embed_tokens", "model.visual.")
SKIP_SUFFIX = (".k_scale", ".v_scale")


def swift_reader():
    wm = json.load(open(SWIFT / "model.safetensors.index.json"))["weight_map"]
    handles: dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        shard = wm[name]
        if shard not in handles:
            handles[shard] = safe_open(str(SWIFT / shard), "pt")
        return handles[shard].get_tensor(name)

    return get, wm


def main() -> None:
    started = time.perf_counter()
    get_swift, swift_map = swift_reader()
    fu = safe_open(str(UNS / "model.safetensors"), "pt")
    keys = list(fu.keys())

    # categorize unsloth bases
    nvfp4_bases, fp8_bases, bf16_keys = set(), set(), []
    for k in keys:
        if k.startswith(SKIP) or k.endswith(SKIP_SUFFIX):
            continue
        if k.endswith(".weight_packed"):
            nvfp4_bases.add(k[: -len(".weight_packed")])
        elif k.endswith((".weight_scale", ".weight_global_scale", ".input_global_scale")):
            continue  # handled with its base
        elif k.endswith(".weight") and fu.get_slice(k).get_dtype() == "F8_E4M3":
            fp8_bases.add(k[: -len(".weight")])
        else:
            bf16_keys.append(k)  # copy verbatim from swift

    out: dict[str, torch.Tensor] = {}

    # --- BF16 aux: copy verbatim from Swift ---
    for k in bf16_keys:
        if k not in swift_map:
            raise KeyError(f"swift lacks bf16 aux {k}")
        out[k] = get_swift(k).contiguous()
    print(f"bf16 aux copied: {len(bf16_keys)}", flush=True)

    # --- FP8 row-scaled: quantize Swift weight ---
    for i, b in enumerate(sorted(fp8_bases)):
        w = get_swift(b + ".weight").to(DEV)
        code, scale = q.quant_fp8_row(w)
        out[b + ".weight"] = code.cpu().contiguous()
        out[b + ".weight_scale"] = scale.cpu().contiguous()
        del w
    print(f"fp8 quantized: {len(fp8_bases)}", flush=True)

    # --- NVFP4: gate/up share a joint d_w; reuse unsloth d_x ---
    def dx(base):  # activation divisor reused from stock unsloth
        return fu.get_tensor(base + ".input_global_scale").float().reshape(1)

    layers = sorted({int(b.split(".layers.")[1].split(".")[0]) for b in nvfp4_bases})
    for L in layers:
        pfx = f"model.language_model.layers.{L}.mlp."
        g, u, d = pfx + "gate_proj", pfx + "up_proj", pfx + "down_proj"
        wg = get_swift(g + ".weight").to(DEV)
        wu = get_swift(u + ".weight").to(DEV)
        wd = get_swift(d + ".weight").to(DEV)
        dw_gu = q.nvfp4_global_scale(torch.maximum(wg.float().abs().amax(), wu.float().abs().amax()))
        dw_d = q.nvfp4_global_scale(wd.float().abs().amax())
        for base, w, dw in ((g, wg, dw_gu), (u, wu, dw_gu), (d, wd, dw_d)):
            pk, sc, dwr = q.quant_nvfp4(w, d_w=dw)
            out[base + ".weight_packed"] = pk.cpu().contiguous()
            out[base + ".weight_scale"] = sc.cpu().contiguous()
            out[base + ".weight_global_scale"] = dwr.cpu().reshape(1).contiguous()
            out[base + ".input_global_scale"] = dx(base).cpu().contiguous()
        del wg, wu, wd
    # enforce gate/up d_x identical (they are in unsloth, but be explicit)
    for L in layers:
        pfx = f"model.language_model.layers.{L}.mlp."
        out[pfx + "up_proj.input_global_scale"] = out[pfx + "gate_proj.input_global_scale"].clone()
    print(f"nvfp4 quantized: {len(nvfp4_bases)} across {len(layers)} layers", flush=True)

    import gc
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.safetensors"):
        f.unlink()
    # Shard the write to bound peak host memory; free each tensor as it lands.
    SHARD_MAX = 2 * 1024**3
    order = list(out.keys())
    weight_map: dict[str, str] = {}
    cur: list[str] = []
    cur_bytes = 0
    idx = 1
    total = len(order)

    def flush():
        nonlocal cur, cur_bytes, idx
        if not cur:
            return
        fn = f"model-{idx:05d}-of-XXXXX.safetensors"
        save_file({k: out[k] for k in cur}, str(OUT / fn), metadata={"format": "pt"})
        for k in cur:
            weight_map[k] = fn
            del out[k]
        gc.collect()
        print(f"  shard {idx}: {len(cur)} tensors, {cur_bytes/2**30:.2f} GiB", flush=True)
        idx += 1
        cur = []
        cur_bytes = 0

    for k in order:
        nb = out[k].numel() * out[k].element_size()
        if cur and cur_bytes + nb > SHARD_MAX:
            flush()
        cur.append(k)
        cur_bytes += nb
    flush()

    # rename shards to final -of-N and rewrite index
    nshards = idx - 1
    remap = {}
    for i in range(1, nshards + 1):
        old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
        new_name = f"model-{i:05d}-of-{nshards:05d}.safetensors"
        old.rename(OUT / new_name)
        remap[f"model-{i:05d}-of-XXXXX.safetensors"] = new_name
    weight_map = {k: remap[v] for k, v in weight_map.items()}
    index = {"metadata": {}, "weight_map": weight_map}
    (OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    shutil.copyfile(UNS / "config.json", OUT / "config.json")
    print(f"wrote {total} tensors in {nshards} shards to {OUT} in {time.perf_counter()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
