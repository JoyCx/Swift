# Runbook: doing it to the real model

Target: `ukisai/Swift-Qwen3.8-27b` (or plain `Qwen/Qwen3.8-27B` if you want to reproduce
Swift from the base). Everything below also works for any other HF reasoning model.

## 0. Hardware and time

| step | needs | rough time (27B) |
|---|---|---|
| rollouts + settle (vLLM) | 1× 80 GB GPU (BF16, `--max-model-len 65536`) or 2× 80 GB for 262k ctx; a Q8 GGUF on 2× 24 GB works for rollouts only | 200 tasks × 2 samples at xhigh ≈ 2–4 h; settle ≈ 3 short gens per trace ≈ 1 h |
| direction capture (transformers) | 1× 80 GB, or 24 GB + CPU offload (slow but only a few hundred short forward passes) | 10–30 min |
| search | same as rollouts; each trial re-generates the calib split | 30 trials × ~5 min |
| edit (safetensors) | CPU + RAM ≥ largest shard (~5 GB); no GPU needed | 5 min |
| quant | llama.cpp: CPU ok, imatrix faster with GPU; llmcompressor W4A16: 1× 80 GB | 1–3 h |
| eval | one vLLM per arm (or serve sequentially and reuse the same `--out`) | eval split × 5 seeds × arms |

Disk: BF16 weights 54 GB, edited copy 54 GB, quants 16–29 GB each.

## 1. Install

```bash
git clone https://github.com/JoyCx/Swift && cd Swift
python -m venv .venv && source .venv/bin/activate
pip install -e ".[hf,search,dev]"          # torch, transformers, safetensors, optuna, pytest
pip install vllm                            # for serving; pin to the version you serve with
huggingface-cli download ukisai/Swift-Qwen3.8-27b --local-dir /models/Swift-Qwen3.8-27b
pytest -q                                   # offline tests, ~30 s
```

## 2. Serve the base for rollouts

```bash
vllm serve /models/Swift-Qwen3.8-27b --served-model-name ukisai/Swift-Qwen3.8-27b \
  --reasoning-parser qwen3 --max-model-len 65536 --enable-prefix-caching --seed 0 --port 8000
```

Smoke test the backend before spending hours:

```bash
python - <<'PY'
from swiftlab.config import load_config; from swiftlab.backends import make_backend, GenRequest
cfg = load_config("configs/qwen3.8-27b.yaml"); be = make_backend(cfg.backend)
g = be.generate(GenRequest("t", "What is 17*23? Put the answer in \\boxed{}.", seed=1, max_tokens=4096))
print("think tokens:", g.think_tokens, "| answer:", g.answer[:200])
print("prefix probe:", be.complete_prefix(GenRequest("t", "What is 17*23? Put the answer in \\boxed{}.", seed=1), "17*23 = 391.", 64)[:200])
PY
```

Both lines must print sensible text. Or just run the bundled check, which does this plus the
verifier and decontamination checks and exits non-zero on failure:

```bash
swiftlab preflight --config configs/qwen3.8-27b.yaml
```

If `think tokens` is 0, the reasoning parser is not splitting; if the prefix probe errors,
see pitfalls below.

## 3. Bank, rollouts, settle

```bash
export CFG=configs/qwen3.8-27b.yaml RUN=runs/swift27b; mkdir -p $RUN
# put GPQA/LCB/AIME question dumps in bank.decontam_against first if you will benchmark on them
swiftlab bank    --config $CFG --synthetic 200 --out $RUN/bank.jsonl
swiftlab rollout --config $CFG --bank $RUN/bank.jsonl --split mine --max-tasks 40 --out $RUN/rollouts.jsonl   # pilot
swiftlab settle  --config $CFG --bank $RUN/bank.jsonl --rollouts $RUN/rollouts.jsonl --out $RUN/settled.jsonl
```

Look at `waste_summary` printed by `settle`. If `overspent_share` is under 5% on the pilot,
raise `settle.checkpoints` to 12 and check the prefix probe: the settle point is only as
good as the forced-answer continuation. Then drop `--max-tasks` and run the full split;
`rollout` resumes, so you can stop and restart.

## 4. Mine and scale

```bash
swiftlab mine  --settled $RUN/settled.jsonl --out $RUN/markers.json
swiftlab scale --settled $RUN/settled.jsonl --sizes 25,50,100,200,400 --out $RUN/scaling.json
```

If the last two rows of the scaling table still add markers and the Chao1 coverage is
below 90%, generate more traces (raise `--synthetic`, add seeds, or add your own tasks)
and re-run settle. Prioritise hard tasks: derailments live in long traces.

## 5. Directions (needs the weights in-process)

Stop vLLM or use a second GPU. Edit `configs/hf-capture.yaml` `backend.model` to the local path.

```bash
swiftlab direction --config configs/hf-capture.yaml --bank $RUN/bank.jsonl --settled $RUN/settled.jsonl --out $RUN/bundle.json
```

Read the printed table: you want layers where `auc clean` is clearly above 0.5 (0.7+ is
typical in the middle of the stack) and `energy removed` between 5% and 40%. If a layer
loses most of its energy to cleaning, the "overthinking" signal there was mostly a
capability signal; do not edit it.

## 6. Search and edit

```bash
swiftlab search --config configs/hf-capture.yaml --bank $RUN/bank.jsonl --bundle $RUN/bundle.json --trials 30 --max-tasks 40 --out $RUN/search.json
python -c "import json; s=json.load(open('$RUN/search.json')); json.dump(s['best_edit'], open('$RUN/edit.json','w'))"
swiftlab edit --config configs/hf-capture.yaml --bundle $RUN/bundle.json --edit-json $RUN/edit.json \
              --model-dir /models/Swift-Qwen3.8-27b --out /models/Swift-Qwen3.8-27b-surgical
```

The search runs the edit in-process (`apply_in_place` / `restore_in_place`) so nothing is
written until `edit`. Start with `--trials 10` and `--max-tasks 20` to see one trial's cost.
Expect the best γ between 0.4 and 1.0; γ above 1 over-rotates.

## 7. Serve the edited model, calibrate, quantize

```bash
vllm serve /models/Swift-Qwen3.8-27b-surgical --served-model-name surgical --reasoning-parser qwen3 --max-model-len 65536 --port 8001
swiftlab rollout --config $CFG --set backend.base_url=http://localhost:8001/v1 --set backend.model=surgical \
                 --bank $RUN/bank.jsonl --split calib --arm edited --out $RUN/rollouts_edited.jsonl
swiftlab quant --config $CFG --bank $RUN/bank.jsonl --rollouts $RUN/rollouts_edited.jsonl \
               --model-dir /models/Swift-Qwen3.8-27b-surgical --out $RUN/quant
bash $RUN/quant/quant_gguf.sh            # needs llama.cpp built at quant.llamacpp_dir
python $RUN/quant/quant_w4a16.py         # needs llmcompressor + one 80 GB GPU
```

Check `$RUN/quant/gguf/kld-*.txt`: mean KLD under 0.02 for Q5/Q6, under 0.06 for Q4_K_M.

## 8. Paired evaluation

Serve each arm (base on :8000, surgical on :8001, a quant on :8002 via `llama-server` or vLLM), then:

```bash
swiftlab eval --config $CFG --bank $RUN/bank.jsonl --settled $RUN/settled.jsonl --seeds 5 \
  --arm base:openai:ukisai/Swift-Qwen3.8-27b@http://localhost:8000/v1 \
  --arm surgical:openai:surgical@http://localhost:8001/v1 \
  --arm surgical_q4:openai:surgical-q4@http://localhost:8002/v1 \
  --out $RUN/eval
```

`--settled` is optional but gives the `overspend removed` and `needed cut` columns, which
need settle data for the eval split's base rows (run `settle` on `eval_base.jsonl` first for
full coverage). Report: `$RUN/eval/report.html`.

What "it worked" looks like: accuracy delta CI contains 0, McNemar p > 0.05, fixed ≥
broken, thinking reduction 20%+ on top of Swift (or 40–60% if starting from plain Qwen),
`needed cut` under 5%, truncation count unchanged.

## 9. Optional: training-based second pass

```bash
python scripts/train_penalized_lora.py --model /models/Swift-Qwen3.8-27b-surgical --settled $RUN/settled.jsonl --markers $RUN/markers.json --out $RUN/lora-pen --beta 0.5
# merge: PeftModel.from_pretrained(base, "$RUN/lora-pen").merge_and_unload().save_pretrained(...)
python scripts/opd_restore.py --student /models/merged-pen --teacher /models/Swift-Qwen3.8-27b --bank $RUN/bank.jsonl --out $RUN/lora-opd --steps 300
```

Re-run step 8 with the merged model as another arm.

## Pitfalls to expect on first contact

* **reasoning_effort**: Qwen3.8's template accepts only `xhigh|medium|low` and raises on
  anything else. The backend sends it in `chat_template_kwargs` and top-level; if vLLM
  rejects the top-level key set `backend.extra.send_reasoning_effort_top_level: false`.
* **prefix probe**: `complete_prefix` uses vLLM's `continue_final_message`. If your server
  lacks it, it falls back to a raw `/completions` prompt built from
  `backend.extra.prompt_template`; copy the real template from `chat_template.jinja` into
  that field (system, user, assistant placeholders).
* **think tags**: the template may emit `<think>` itself. `HFBackend.complete_prefix`
  strips a trailing opener before appending the prefix; verify once by printing the rendered text.
* **token counts**: the OpenAI backend reads `completion_tokens_details.reasoning_tokens`
  when the server reports it, otherwise estimates by character share. `/tokenize` is used
  for settle lengths when available.
* **layer paths**: `edit._tensor_layer_and_module` matches `model.layers.N.<module>.weight`.
  For multimodal checkpoints with `model.language_model.layers.N` the regex already
  matches; anything else, extend it.
* **memory in search**: `apply_in_place` clones every touched matrix; with 2 matrices × 20
  layers that is a few GB extra on the GPU. Reduce layers or move `saved` to CPU if tight.
* **free API**: the ukisai endpoint (5 RPM) is fine for the smoke test, not for rollouts.
