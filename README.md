# swiftlab — surgical overthinking ablation, quantization and paired evaluation for reasoning LLMs

A harness that reproduces and extends what UkisAI did for
[Swift-Qwen3.8-27B](https://huggingface.co/ukisai/Swift-Qwen3.8-27b) (−58% thinking tokens,
<1% accuracy loss): find the thinking that provably does not change the answer, cut it
surgically, quantize without undoing the cut, and prove the result with same-task /
same-seed / same-context comparisons on tasks that are measurable by construction.

Works on any HF-format model with a thinking block. No stand-in model anywhere: you run it
against a real server (vLLM/llama-server/SGLang) and the real weights. `swiftlab preflight`
validates that setup before any long run.

```
bank → rollout → settle → mine → scale → direction → search → edit → (transfer/LoRA/OPD) → quant → eval → report
```

## What you get

| stage | question it answers | output |
|---|---|---|
| `bank` | are my tasks measurable and decontaminated? | `bank.jsonl` (coding = executes, knowledge = grep, math = numeric) |
| `rollout` | what does the model do at `xhigh`? | traces with per-item seeds shared by every arm |
| `settle` | which tokens did not change the answer? | per-trace category: tight / overspent / derailed / wrong |
| `mine` | which tokens mark the waste? | penalty vocabulary with z-scores |
| `scale` | how many long traces do I need? | scaling table + Chao1 coverage + recommended n |
| `direction` | where in the residual stream is "re-verify", and is it entangled with capability? | per-layer dirty vs cleaned directions, AUC, atom cosines |
| `search` | how hard can I cut before drift/accuracy move? | best (layers, γ, matrices) by tokens + KL + accuracy constraint |
| `edit` | apply it | edited safetensors (rank-one projection on `o_proj`/`down_proj`) |
| `transfer` | borrow a donor's efficiency delta | TIES-trimmed task-vector merge |
| `quant` | keep the cut through quantization | calibration set from the edited model's own traces + GGUF/W4A16 scripts + KL check |
| `eval` | did it work, per task and per seed? | paired accuracy CI, McNemar, token reduction, **overspend removed**, **needed cut**, per-domain |
| `report` | show it | `report.md` / `report.html` |

## Quick start

```bash
pip install -e ".[hf,search]"      # numpy, pyyaml, torch, transformers, safetensors, optuna
# serve the model, then validate the whole setup (server + verifiers) before a long run:
swiftlab preflight --config configs/qwen3.8-27b.yaml
# validate just the task verifiers / decontamination without a server:
swiftlab preflight --config configs/qwen3.8-27b.yaml --local-only
```

Preflight builds one real task per domain, checks that the coding verifier executes, the
knowledge grep and math check work, decontamination flags overlap, then (server mode) runs
one real completion per domain and confirms the thinking block splits, the forced-prefix
probe that settle depends on works, and token counting works. It exits non-zero on any hard
failure, so you find a broken chat template or reasoning parser in seconds, not after hours
of rollouts.

## Full run (Swift-Qwen3.8-27B or any base)

```bash
pip install -e ".[hf,search]"      # torch, transformers, safetensors, optuna
vllm serve /models/Swift-Qwen3.8-27b --served-model-name ukisai/Swift-Qwen3.8-27b \
     --reasoning-parser qwen3 --max-model-len 262144 --enable-prefix-caching
CFG=configs/qwen3.8-27b.yaml HFCFG=configs/hf-capture.yaml MODEL_DIR=/models/Swift-Qwen3.8-27b \
     bash scripts/run_pipeline.sh
```

`scripts/run_pipeline.sh` is the full sequence with comments; each `swiftlab` sub-command
also runs alone. Compute budget for a 27B: rollouts dominate (≈ 2× the number of
mining tasks long generations at xhigh); settle probing adds ~3 short generations per
trace; capture/edit take minutes on one 80 GB GPU; the search re-generates the calib
split once per trial.

## Adding your own goal-specific tasks

Append JSONL rows to `data/tasks/*.jsonl` (schema in `swiftlab/tasks/base.py`):

```json
{"id":"my-1","domain":"coding","prompt":"Write `f(x)` that …","verify":{"kind":"code_tests","tests":[{"fn":"f","args":[3],"expect":9}],"timeout":10,"time_limit":2}}
{"id":"my-2","domain":"knowledge","prompt":"What port does Redis use? End with 'Final answer: <port>'","verify":{"kind":"regex","patterns":["\\b6379\\b"]}}
```

For knowledge at scale, `knowledge_tasks_from_table(rows, "What is the {field} of {entity}?", "value")`
turns any fact table you own into grep-verified tasks. List your benchmark dumps under
`bank.decontam_against` before mining.

## The Swift recipe itself (training-based) + abliteration

`scripts/run_swift_recipe.sh` chains the authors' actual pipeline, from the plain base:

1. diverse decontaminated rollouts → settle → mine (`bank`, `rollout`, `settle`, `mine`, `scale`)
2. `swiftlab sftdata` + `scripts/train_penalized_lora.py` — LoRA SFT with CE + β·P(mined tokens | think)
3. restore accuracy: `scripts/opd_restore.py` (on-policy distillation) and/or `scripts/gspo_restore.py` (GSPO RL with verifiable rewards + settle-keyed brevity); optional `swiftlab transfer` for ThinkingCap adapter chunks
4. `swiftlab abliterate` — extra: surgical refusal ablation with the overthinking direction as a protected atom, searched on refusal + KL with thinking/accuracy constraints
5. `swiftlab quant` → `swiftlab eval` — base vs swift vs abliterated vs quant, same seeds

Step-by-step mapping of their write-up to commands: [docs/SWIFT_RECIPE.md](docs/SWIFT_RECIPE.md).
The activation-only edit (`direction` → `search` → `edit`) remains available as a cheaper first cut.

## Docs

* [docs/ALGORITHM.md](docs/ALGORITHM.md) — every stage, the math, and why it is built this way
* [docs/QUANT.md](docs/QUANT.md) — the quantization recipe and acceptance thresholds
* [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md) — Windows + single RTX 5090 (32 GB): what fits, Blackwell/Python fixes, llama-server measurement path, 8B full-pipeline path, and how to use the community GGUF/uncensored-BF16 quants
* [docs/SWIFT_RECIPE.md](docs/SWIFT_RECIPE.md) — the authors' pipeline mapped to commands, with abliteration as an extra stage
* [docs/RUNBOOK.md](docs/RUNBOOK.md) — step-by-step commands, hardware, time, and first-contact pitfalls for a real model
* [docs/PLAN_CODER_5090.md](docs/PLAN_CODER_5090.md) — plan for one RTX 5090: honest coding eval, NInfer speed (prefill, profile, speculation), quant quality, and QLoRA + rejection-sampling training to make Swift a better coder

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```

Tests cover the verifiers (real subprocess execution), trace parsing, the SRA/edit/ablation
linear algebra, the GSPO reward and advantage math, and the analysis stages (mining,
scaling, paired comparison, SFT-data building) on data fixtures. No model is mocked; the
parts that need a model are validated live by `swiftlab preflight`.

## Sources

* r/LocalLLaMA thread by the Swift authors (method description, benchmark tables)
* [Surgical Refusal Ablation](https://arxiv.org/abs/2601.08489) — concept atoms, ridge residualisation, rank-one update
* [heretic](https://github.com/p-e-w/heretic) — directional ablation with Optuna co-minimising behaviour and KL
* [ThinkingCap-Qwen3.6-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B) — donor for the transfer step
* Manifold steering / ThinkEdit / Halt Vector — overthinking as a low-dimensional activation direction
