"""Remap the GPTQ output (text-only naming/config from AutoModelForCausalLM) into the
multimodal layout convert_nvfp4 expects: rename model.layers/norm/embed_tokens ->
model.language_model.*, and rebuild config.json = swift multimodal root + the GPTQ
quantization_config. No re-quant; tensor bytes are copied verbatim.
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("A:/models/swift-nvfp4-gptq")
DST = Path("A:/models/swift-nvfp4-gptq-ninfer")
SWIFT_CFG = Path("A:/models/Qwen3.8-27B/swift/config.json")

def remap(k: str) -> str:
    for p in ("model.layers.", "model.norm.", "model.embed_tokens."):
        if k.startswith(p):
            return "model.language_model." + k[len("model."):]
    return k

def main():
    if DST.exists(): shutil.rmtree(DST)
    DST.mkdir(parents=True)
    idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
    shards = sorted(set(idx.values()))
    new_map = {}
    for shard in shards:
        print(f"remapping {shard} ...", flush=True)
        out = {}
        with safe_open(str(SRC / shard), "pt") as f:
            for k in f.keys():
                nk = remap(k)
                out[nk] = f.get_tensor(k)
                new_map[nk] = shard
        save_file(out, str(DST / shard), metadata={"format": "pt"})
        del out
    (DST / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": new_map}, indent=2))

    # config = swift multimodal root + GPTQ quantization_config
    swift_cfg = json.load(open(SWIFT_CFG))
    gptq_cfg = json.load(open(SRC / "config.json"))
    swift_cfg["quantization_config"] = gptq_cfg["quantization_config"]
    (DST / "config.json").write_text(json.dumps(swift_cfg, indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        if (SRC / f).exists(): shutil.copyfile(SRC / f, DST / f)
    print(f"remapped {len(new_map)} tensors -> {DST}", flush=True)
    print("sample:", [k for k in list(new_map)[:3]], flush=True)
    print("has language_model:", any("language_model" in k for k in new_map),
          "| norm:", "model.language_model.norm.weight" in new_map,
          "| lm_head:", any(k.startswith("lm_head") for k in new_map))

if __name__ == "__main__":
    main()
