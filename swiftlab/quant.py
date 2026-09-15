"""High-quality quantization of the edited model.

Principles
* Calibrate on the *edited* model's own short-thinking traces (in-distribution: the quant
  should preserve the behaviour you just installed), mixed with plain text, all drawn from
  the calib split (never the eval split).
* GGUF: convert -> imatrix on that calibration text -> quantize with the imatrix -> measure
  KL(bf16 || quant) with llama-perplexity on held-out text, plus the paired task eval.
* W4A16 / NVFP4: llmcompressor GPTQ with the same calibration samples; ignore lm_head.
* Never quantize before editing: the rank-one edit must be in the tensors the calibration sees.
"""
from __future__ import annotations
import json, random, re
from pathlib import Path


def build_calibration(rows: list[dict], tasks_by_id: dict, n: int = 512, max_chars: int = 8000, seed: int = 0,
                      extra_texts: list[str] | None = None, think_open: str = "<think>", think_close: str = "</think>") -> list[dict]:
    """Full (prompt, thinking, answer) samples from edited-model rollouts, preferring correct + tight traces."""
    rng = random.Random(seed)
    good = [r for r in rows if r["correct"]]
    rng.shuffle(good)
    out = []
    for r in good[:n]:
        t = tasks_by_id[r["task_id"]]
        text = f"{t.prompt}\n{think_open}\n{r['thinking']}\n{think_close}\n\n{r['answer']}"
        out.append({"text": text[:max_chars], "task_id": r["task_id"], "domain": r["domain"]})
    for t in (extra_texts or []):
        out.append({"text": t[:max_chars], "task_id": "extra", "domain": "text"})
    rng.shuffle(out)
    return out[:n]


def write_calibration(samples: list[dict], out_dir: str | Path) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calib.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n", encoding="utf-8")
    (out_dir / "calib.txt").write_text("\n\n".join(s["text"] for s in samples) + "\n", encoding="utf-8")
    return {"jsonl": str(out_dir / "calib.jsonl"), "txt": str(out_dir / "calib.txt"), "n": len(samples)}


def gguf_commands(model_dir: str, out_dir: str, calib_txt: str, heldout_txt: str, types: list[str], llamacpp_dir: str = "~/llama.cpp",
                  ctx: int = 8192, threads: int = 16, ngl: int = 99) -> list[str]:
    name = Path(model_dir).name
    f16 = f"{out_dir}/{name}-bf16.gguf"
    cmds = [f"mkdir -p {out_dir}",
            f"python {llamacpp_dir}/convert_hf_to_gguf.py {model_dir} --outtype bf16 --outfile {f16}",
            f"{llamacpp_dir}/build/bin/llama-imatrix -m {f16} -f {calib_txt} -o {out_dir}/imatrix.gguf -c {ctx} -t {threads} -ngl {ngl} --chunks 400",
            f"{llamacpp_dir}/build/bin/llama-perplexity -m {f16} -f {heldout_txt} -c {ctx} -ngl {ngl} --kl-divergence-base {out_dir}/kld-base.bin"]
    for q in types:
        cmds.append(f"{llamacpp_dir}/build/bin/llama-quantize --imatrix {out_dir}/imatrix.gguf {f16} {out_dir}/{name}-{q}.gguf {q} {threads}")
        cmds.append(f"{llamacpp_dir}/build/bin/llama-perplexity -m {out_dir}/{name}-{q}.gguf --kl-divergence-base {out_dir}/kld-base.bin --kl-divergence -c {ctx} -ngl {ngl} | tee {out_dir}/kld-{q}.txt")
    return cmds


def parse_kl_output(text: str) -> dict:
    """Parse llama-perplexity --kl-divergence summary lines."""
    out = {}
    for key, pat in {"mean_kl": r"Mean\s+KLD:\s*([0-9.]+)", "p99_kl": r"99\.0%\s+KLD:\s*([0-9.]+)", "max_kl": r"Maximum\s+KLD:\s*([0-9.]+)",
                     "top1_agreement": r"Same top p:\s*([0-9.]+)", "mean_ppl": r"Mean PPL\(Q\)\s*:\s*([0-9.]+)"}.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def w4a16_script(model_dir: str, out_dir: str, calib_jsonl: str, scheme: str = "W4A16", n: int = 512, max_len: int = 4096) -> str:
    return f'''"""GPTQ {scheme} with llmcompressor, calibrated on the edited model's own traces."""
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL = {model_dir!r}; OUT = {out_dir!r}
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")
rows = [json.loads(l) for l in open({calib_jsonl!r})][:{n}]
ds = Dataset.from_list([{{"text": r["text"]}} for r in rows])
ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length={max_len}, add_special_tokens=False), remove_columns=["text"])
recipe = GPTQModifier(targets="Linear", scheme={scheme!r}, ignore=["lm_head"], dampening_frac=0.01,
                      sequential_targets=None)   # set to the decoder layer class name for very large models
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length={max_len}, num_calibration_samples={n})
model.save_pretrained(OUT, save_compressed=True); tok.save_pretrained(OUT)
print("saved", OUT)
'''


def size_table(n_params_b: float, types: list[str]) -> list[dict]:
    bpw = {"Q2_K": 3.35, "Q3_K_M": 3.91, "Q4_K_S": 4.58, "Q4_K_M": 4.85, "Q5_K_M": 5.69, "Q6_K": 6.57, "Q8_0": 8.5, "W4A16": 4.5, "NVFP4": 4.5, "FP8": 8.0, "BF16": 16.0}
    return [{"type": t, "bits_per_weight": bpw.get(t), "approx_gb": round(n_params_b * bpw.get(t, 0) / 8, 1)} for t in types]
