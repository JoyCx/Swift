"""GPTQ re-quant of the abliterated Swift BF16 -> compressed-tensors NVFP4/FP8, matching
unsloth's exact config_groups so tools.convert.qwen3_8_27b.convert_nvfp4 accepts it.

Run with the swift 3.12 venv python. Modes:
  --validate         build recipe + tokenize a tiny calib set, no model load
  --smoke N          real oneshot with N calibration samples (small, for a fast end-to-end check)
  (default)          full run, NCAL samples
"""
from __future__ import annotations
import argparse, glob, json, random, sys, time
from pathlib import Path

MODEL = "A:/models/Qwen3.8-27B/swift"              # abliterated BF16 (embeddings/vision/mtp source too)
UNS   = "A:/models/Qwen3.8-27B/NVFP4_unsloth"
OUT   = "A:/models/swift-nvfp4-gptq"
MAXLEN = 1024
NCAL   = 64


def load_calib_texts(tokenizer, n: int, max_len: int) -> list[str]:
    """Assemble a diverse general corpus from local text, sliced into max_len-token windows."""
    srcs: list[str] = []
    globs = [
        "A:/swift/docs/*.md", "A:/swift/README.md", "A:/swift/swiftlab/*.py",
        "A:/ninfer-spark/docs/**/*.md", "A:/ninfer-spark/docs/*.md",
        "A:/models/Qwen3.8-27B/README.md",
    ]
    for g in globs:
        for p in glob.glob(g, recursive=True):
            try:
                t = Path(p).read_text(encoding="utf-8", errors="ignore")
                if len(t) > 200:
                    srcs.append(t)
            except Exception:
                pass
    # multilingual prose/code/math from the swiftlab calibration corpus
    try:
        sys.path.insert(0, "A:/swift")
        from swiftlab.calibrate import CORPUS_DOCUMENTS
        srcs.extend(CORPUS_DOCUMENTS)
    except Exception as e:
        print("(swiftlab corpus unavailable:", e, ")")
    random.Random(0).shuffle(srcs)
    big = "\n\n".join(srcs)
    ids = tokenizer(big, add_special_tokens=False)["input_ids"]
    windows = [ids[i:i + max_len] for i in range(0, len(ids) - max_len, max_len)]
    print(f"calib pool: {len(srcs)} docs -> {len(ids)} tokens -> {len(windows)} windows of {max_len}")
    random.Random(1).shuffle(windows)
    if len(windows) < n:
        windows = (windows * (n // max(1, len(windows)) + 1))
    return [tokenizer.decode(w) for w in windows[:n]]


def build_recipe():
    from llmcompressor.modifiers.quantization import GPTQModifier
    uns = json.load(open(UNS + "/config.json"))["quantization_config"]
    groups = uns["config_groups"]
    ignore = list(uns.get("ignore") or [])
    print(f"recipe: {len(groups)} config_groups; ignore={len(ignore)} modules "
          f"(group_1 actorder={groups['group_1']['weights'].get('actorder')})")
    mod = GPTQModifier(config_groups=groups, ignore=ignore, dampening_frac=0.01)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    n = args.smoke if args.smoke else NCAL

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    recipe = build_recipe()
    texts = load_calib_texts(tok, n, MAXLEN)
    from datasets import Dataset
    ds = Dataset.from_list([{"text": t} for t in texts])
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAXLEN, add_special_tokens=False),
                remove_columns=["text"])
    print(f"calibration dataset: {len(ds)} samples")

    if args.validate:
        print("VALIDATE OK: recipe instantiated + calibration tokenized; skipping model load.")
        return

    import torch
    from transformers import AutoModelForCausalLM
    from llmcompressor import oneshot
    t0 = time.perf_counter()
    print("loading model (device_map=auto, offload)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")
    print(f"model loaded in {time.perf_counter()-t0:.0f}s; starting oneshot GPTQ", flush=True)
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=MAXLEN,
            num_calibration_samples=n, processor=tok)
    out_dir = OUT + ("-smoke" if args.smoke else "")
    print("quantization complete; saving (save_original_format=False to skip the offload revert bug)", flush=True)
    saved = False
    for attempt, kw in enumerate([
        dict(save_compressed=True, save_original_format=False),
        dict(save_compressed=True, save_original_format=False, max_shard_size="4GB"),
        dict(save_compressed=True, save_original_format=False, max_shard_size="2GB"),
    ]):
        try:
            model.save_pretrained(out_dir, **kw); saved = True
            print(f"SAVED {out_dir} (attempt {attempt+1}, {kw}) in {time.perf_counter()-t0:.0f}s", flush=True)
            break
        except Exception as e:
            print(f"save attempt {attempt+1} failed: {type(e).__name__}: {e}", flush=True)
    if not saved:
        # last resort: raw state_dict so the ~6.5h run is never lost
        import torch, os as _os
        _os.makedirs(out_dir, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, _os.path.join(out_dir, "state_dict_fallback.pt"))
        print(f"SAVED FALLBACK state_dict to {out_dir}/state_dict_fallback.pt", flush=True)
    tok.save_pretrained(out_dir)
    print(f"done in {time.perf_counter()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
