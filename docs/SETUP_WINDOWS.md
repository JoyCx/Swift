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

## 4. Using the community artifacts (their quants, not the base)

You linked two very different things. Know which job each does before downloading 54 GB.

| artifact | format / size | what it's for on a 5090 |
|---|---|---|
| `bartowski/ukisai_Swift-Qwen3.8-27b-GGUF` | GGUF quants (Q2..Q8) | **serve for inference.** This is the measurement / rollout / eval path (§1). Q4_K_M fits 32 GB. GGUF is inference-only — you cannot run direction/edit/transfer on it. |
| `d0xin/Swift-Qwen3.8-27B-Uncensored-BF16` | BF16 safetensors, ~54 GB | **the abliteration is already done.** Editable weights for the surgery stages; too big to serve in BF16 on 32 GB (quantize it first). |

### If your goal is "uncensored Swift on my 5090"

You do **not** need `swiftlab abliterate` — d0xin already removed refusal. Just quantize that
BF16 to GGUF yourself and serve it like §1 (no uncensored GGUF is published, so you make it):

```powershell
huggingface-cli download d0xin/Swift-Qwen3.8-27B-Uncensored-BF16 --local-dir A:\models\swift-unc-bf16
python A:\llama.cpp\convert_hf_to_gguf.py A:\models\swift-unc-bf16 --outtype bf16 --outfile A:\models\swift-unc-bf16.gguf
A:\llama.cpp\llama-quantize.exe A:\models\swift-unc-bf16.gguf A:\models\swift-unc-Q4_K_M.gguf Q4_K_M
A:\llama.cpp\llama-server.exe -m A:\models\swift-unc-Q4_K_M.gguf --alias swift-unc --host 127.0.0.1 --port 8080 -ngl 99 -c 16384 --jinja --reasoning-format deepseek
```

Then point `configs/llama-server-27b-5090.yaml` `backend.model: swift-unc` and run §1. For a
better quant, generate an imatrix from your own calibration text first (see `docs/QUANT.md`).

### If your goal is "cut Swift's thinking further" (weight surgery, no training)

This is doable on one 5090 without a rented box, because capture can run 4-bit and the edit
runs on CPU. Start from the BF16 safetensors (d0xin uncensored, or the original
`ukisai/Swift-Qwen3.8-27b`):

```powershell
pip install bitsandbytes            # for 4-bit capture; if it won't build on Blackwell, use the max_memory CPU-offload line in the config
# rollouts + settle through the served GGUF (fast), directions through the BF16 (4-bit load)
swiftlab rollout   --config configs\llama-server-27b-5090.yaml --bank runs\u\bank.jsonl --split mine --out runs\u\rollouts.jsonl
swiftlab settle    --config configs\llama-server-27b-5090.yaml --bank runs\u\bank.jsonl --rollouts runs\u\rollouts.jsonl --out runs\u\settled.jsonl
swiftlab direction --config configs\uncensored-27b-5090.yaml --bank runs\u\bank.jsonl --settled runs\u\settled.jsonl --out runs\u\bundle.json
swiftlab search    --config configs\uncensored-27b-5090.yaml --bank runs\u\bank.jsonl --bundle runs\u\bundle.json --trials 15 --max-tasks 25 --out runs\u\search.json
python -c "import json;s=json.load(open(r'runs\u\search.json'));json.dump(s['best_edit'],open(r'runs\u\edit.json','w'))"
swiftlab edit      --config configs\uncensored-27b-5090.yaml --bundle runs\u\bundle.json --edit-json runs\u\edit.json --model-dir A:\models\swift-unc-bf16 --out A:\models\swift-unc-surgical
# quantize the edited BF16 to GGUF and serve as a new eval arm (§1c/§1 quant)
```

Notes: 4-bit capture makes the *direction* slightly noisier than full BF16 but the edit is
still written into the true BF16 tensors, so quality of the shipped model is unaffected. The
`search` stage re-generates through the 4-bit model and is the slow part — keep `--max-tasks`
small. **Training** the 27B (penalised LoRA / OPD / GSPO) still wants QLoRA and more than one
32 GB card is comfortable with; prototype those on Qwen3-8B (§2) or rent a box (§3).

### Which base to start from

* Want uncensored + shorter thinking → start from **d0xin BF16**, run the surgery above, quantize.
* Want only shorter thinking, keep original alignment → start from **ukisai/Swift-Qwen3.8-27b** BF16.
* Just want to measure/serve Swift as-is → **bartowski GGUF**, §1, done.

## 5. The exact "cut Swift's thinking further" run (one 5090, no training)

`scripts/run_surgery_5090.ps1` is this whole loop. Edit the paths at the top, then run it
block by block (the two `llama-server` launches go in a second terminal, shown as comments).
The split is deliberate:

* **rollout / settle / eval** hit the **served GGUF** — fast, fits 32 GB.
* **direction / search / edit** use the **BF16 loaded 4-bit** for capture; the edit writes the
  true BF16 tensors on CPU (no VRAM). Stop llama-server before these so the 4-bit load has room.

### Two things that make it work on 32 GB

* **Paired eval without both models resident.** Two Q4_K_M servers (~33 GB) don't co-fit. So
  eval runs in two passes using resume: pass 7a serves the base and writes `eval_base.jsonl`;
  pass 7b serves the surgical model and, because the base rows are already complete, the base
  arm makes zero calls and only the surgical arm generates. Same quant type both sides, so the
  delta is honest.
* **Keep `search --max-tasks` and `--trials` small.** The search re-generates through the 4-bit
  27B, which is the slow step. 12 trials × 15 tasks is enough to pick gamma/layers; you are not
  training, just choosing an edit.

### A fidelity caveat, stated plainly

Traces are generated by the GGUF quant but activations are captured on the BF16-in-4-bit model
— two slightly different numeric versions of the same weights. That is fine for a first cut:
the direction is robust to it, and the edit lands in the true BF16 tensors. If you want maximal
fidelity, roll out through the `hf` backend too (set `configs/uncensored-27b-5090.yaml` and run
`rollout` there) — correct but much slower than the GGUF, so most people only do it for the
final calibration pass.

### What "it worked" looks like

Open `runs\swift-surgery\eval\report.html`. You want, on the base→surgical row:

* thinking reduction positive (even 10–20% on top of Swift is a real win — Swift is already lean)
* accuracy delta CI containing 0, McNemar p > 0.05, fixed ≥ broken
* `needed cut` near 0 (you cut waste, not signal), `overspend removed` clearly positive
* truncated count unchanged

If accuracy drops, lower `direction.gamma` in the config (or re-run `search` with a tighter
`--w-kl`) and re-edit. If thinking barely moves, raise gamma or widen the layer set. The edit
is cheap to redo — only steps 4–6 repeat, and only step 6 (quantize) is slow.

### Disk budget

BF16 base download ~54 GB + edited BF16 copy ~54 GB + GGUFs ~17 GB each. Keep ~150 GB free on A:.
