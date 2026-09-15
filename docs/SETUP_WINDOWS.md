# Setup on Windows with one RTX 5090 (32 GB)

## The honest constraint

Swift-Qwen3.8-27B in BF16 is ~54 GB of weights. Your 5090 has 32 GB. So on this machine:

| you want to… | fits on one 5090? | how |
|---|---|---|
| measure overthinking, mine markers, scaling, eval, inference-time penalizer, quantize | **yes** | serve a **quantized 27B GGUF** with llama-server, backend `openai` |
| direction / edit / search / abliterate / LoRA / OPD / GSPO on the **27B** | **no** (needs BF16 ~54 GB, or QLoRA which is fiddly on Windows) | rent a multi-GPU box, or use WSL2 + a 4-bit load for capture only |
| the **full** surgery + training pipeline locally | **yes, on an 8B** | `configs/qwen3-8b-5090.yaml`, prove the method, then scale the recipe up |

Recommended: do the **measurement path on the 27B quant now** (below), and run the **full
pipeline on Qwen3-8B** to exercise every stage end to end on hardware that fits.

## 0. Fix the two blockers in your `nvidia-smi` session

**Python 3.14 is too new** — torch/vLLM/transformers have no wheels for it yet. Install 3.12:

```powershell
winget install Python.Python.3.12
py -3.12 --version
```

**Blackwell (sm_120) needs CUDA 12.8 torch wheels.** The default `pip install torch` gives a
build that does not support the 5090 and you get "no kernel image is available".

```powershell
cd A:\swift
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

That last line must print `True` and `NVIDIA GeForce RTX 5090`. Then install swiftlab:

```powershell
pip install -e ".[hf,search,dev]"
pytest -q                                   # offline tests, no GPU/model needed
swiftlab preflight --config configs\qwen3.8-27b.yaml --local-only
```

Note: in PowerShell/cmd the null device is `2>nul`, not `2>/dev/null` (that is why your
`pip show ... 2>/dev/null` printed "cannot find the path"). Just drop the redirect.

## 1. Measurement path on the 27B quant (fits 32 GB, native Windows)

### 1a. Get llama.cpp with CUDA and a GGUF quant

Download a prebuilt CUDA build of llama.cpp from its GitHub releases
(`llama-*-bin-win-cuda-x64.zip`), unzip to e.g. `A:\llama.cpp`. Download one quant of Swift
(Q4_K_M ~16.5 GB fits with room for context; Q5_K_M ~19 GB is tighter):

```powershell
pip install huggingface_hub
huggingface-cli download bartowski/ukisai_Swift-Qwen3.8-27b-GGUF ukisai_Swift-Qwen3.8-27b-Q4_K_M.gguf --local-dir A:\models
```

### 1b. Serve it (OpenAI-compatible, thinking separated)

```powershell
A:\llama.cpp\llama-server.exe -m A:\models\ukisai_Swift-Qwen3.8-27b-Q4_K_M.gguf `
  --alias swift-27b --host 127.0.0.1 --port 8080 `
  -ngl 99 -c 16384 --jinja --reasoning-format deepseek
```

`-ngl 99` offloads all layers to the GPU, `--jinja` uses the model's own chat template,
`--reasoning-format deepseek` puts the `<think>` content in `reasoning_content` where the
`openai` backend reads it. Leave this window running.

### 1c. Validate, then run the analysis

In your activated venv:

```powershell
swiftlab preflight --config configs\llama-server-27b-5090.yaml
swiftlab bank    --config configs\llama-server-27b-5090.yaml --synthetic 120 --out runs\swift27b-5090\bank.jsonl
swiftlab rollout --config configs\llama-server-27b-5090.yaml --bank runs\swift27b-5090\bank.jsonl --split mine --max-tasks 40 --out runs\swift27b-5090\rollouts.jsonl
swiftlab settle  --config configs\llama-server-27b-5090.yaml --bank runs\swift27b-5090\bank.jsonl --rollouts runs\swift27b-5090\rollouts.jsonl --out runs\swift27b-5090\settled.jsonl
swiftlab mine    --settled runs\swift27b-5090\settled.jsonl --out runs\swift27b-5090\markers.json
swiftlab scale   --settled runs\swift27b-5090\settled.jsonl --out runs\swift27b-5090\scaling.json
```

Now you have the real waste breakdown, the mined markers, and the scaling curve for the
actual model — the "how much does it overthink and how many traces do I need" answer.

### 1d. Inference-time penalizer (no training, works through llama-server)

Turn the mined markers into a `logit_bias` and re-roll to see the effect. The mined phrases
in `markers.json` map to token ids and are sent as `logit_bias`; add an arm in `eval` that
carries it. This is the one behavioural change you can make on the 27B without the BF16
weights, and it is a real test of whether suppressing those tokens shortens thinking without
hurting accuracy. (Token-id mapping needs the tokenizer; `pip install transformers` and
point `penalty_vocab_to_logit_bias` at `AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")`.)

## 2. Full pipeline on Qwen3-8B (every stage, on hardware that fits)

```powershell
swiftlab preflight --config configs\qwen3-8b-5090.yaml
# rollouts through transformers are slower than vLLM; keep task counts modest to start
swiftlab bank      --config configs\qwen3-8b-5090.yaml --synthetic 120 --out runs\qwen3-8b\bank.jsonl
swiftlab rollout   --config configs\qwen3-8b-5090.yaml --bank runs\qwen3-8b\bank.jsonl --split mine --max-tasks 60 --out runs\qwen3-8b\rollouts.jsonl
swiftlab settle    --config configs\qwen3-8b-5090.yaml --bank runs\qwen3-8b\bank.jsonl --rollouts runs\qwen3-8b\rollouts.jsonl --out runs\qwen3-8b\settled.jsonl
swiftlab mine      --settled runs\qwen3-8b\settled.jsonl --out runs\qwen3-8b\markers.json
swiftlab direction --config configs\qwen3-8b-5090.yaml --bank runs\qwen3-8b\bank.jsonl --settled runs\qwen3-8b\settled.jsonl --out runs\qwen3-8b\bundle.json
swiftlab search    --config configs\qwen3-8b-5090.yaml --bank runs\qwen3-8b\bank.jsonl --bundle runs\qwen3-8b\bundle.json --trials 20 --max-tasks 30 --out runs\qwen3-8b\search.json
python -c "import json; s=json.load(open(r'runs\qwen3-8b\search.json')); json.dump(s['best_edit'], open(r'runs\qwen3-8b\edit.json','w'))"
swiftlab edit      --config configs\qwen3-8b-5090.yaml --bundle runs\qwen3-8b\bundle.json --edit-json runs\qwen3-8b\edit.json --model-dir <path-to-Qwen3-8B-snapshot> --out A:\models\Qwen3-8B-surgical
```

For LoRA / OPD / GSPO on the 8B, use `scripts/train_penalized_lora.py`, `scripts/opd_restore.py`,
`scripts/gspo_restore.py` as in `docs/SWIFT_RECIPE.md`; an 8B LoRA fits in 32 GB without QLoRA.
Everything you validate here transfers unchanged to the 27B on a bigger box.

## 3. When you want the 27B surgery/training for real

* **Rent** a 1–2× H100/A100 (80 GB) box or an 8×H100 node like the Swift authors used. Put the
  BF16 weights on it, run `scripts/run_swift_recipe.sh`. The 5090 stays your inference/eval box.
* **Or WSL2 + 4-bit capture** on the 5090: load the 27B with `load_in_4bit` for direction
  capture only (activations stay fp16, fits ~16 GB), then `swiftlab edit` writes the edit into
  the BF16 safetensors on disk (CPU streaming, no VRAM). Training the 27B still wants QLoRA and
  more patience than a single 32 GB card makes pleasant.

## Common Windows/Blackwell gotchas

* `torch.cuda.is_available()` is False → you installed the CPU or non-cu128 wheel; reinstall
  from the cu128 index.
* llama-server "no CUDA devices" → you grabbed the CPU zip; get `-bin-win-cuda-`.
* Out of memory serving the GGUF → lower `-c` (context) or use a smaller quant (Q4_K_M).
* Thinking not splitting (`think tokens 0` in preflight) → add `--reasoning-format deepseek`
  to llama-server, or check `--jinja` is set so the Qwen template is used.
* Paths: the CLI takes Windows paths fine; in configs use forward slashes to avoid YAML escaping.
