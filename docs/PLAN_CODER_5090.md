# Plan: a better coder, faster on NInfer, on one RTX 5090 (2026-09-17)

Goal: turn Swift-Qwen3.8-27B into a measurably better coding model and serve it faster on
NInfer, using the local machine only (RTX 5090 32 GB, 128 GB RAM, Windows 11) plus optional
small API spend. Companion to `RESEARCH_REPORT_2026-09-16.md` and
`NINFER_IMPLEMENTATION_SUMMARY.md`; every number below that is not marked as an estimate comes
from those two documents.

Markers: **[measured]** = produced and checked here · **[estimate]** = arithmetic or expectation,
verify before relying on it · **[claimed]** = third-party claim · **[to write]** = a script or
change this plan needs that does not exist yet.

---

## 0. Where you stand, and what it implies

| Item | State | Consequence for this plan |
|---|---|---|
| Coding eval | 12 easy function tasks 12/12 on every artifact; 5 SWE-style proxies 5/5 (GPTQ 4/5) **[measured]** | Saturated. Nothing can be improved until the eval can tell two artifacts apart. Track A comes first. |
| Swift-based NInfer builds | only the two *abliterated* ones exist; `out/qwen3_8_27b_nvfp4.ninfer` is effectively base **[measured]** | Every base-vs-Swift number so far is confounded by abliteration and by quant. Build the plain Swift artifact first (A1). |
| Decode | ~75 tok/s plain; DFlash2 166–212 tok/s at 28–38 % accept; MTP3 190–218 tok/s at 76–88 % accept on short answers **[measured]** | Decode is already good. Gains come from acceptance on *code* output and from fewer thinking tokens, not from kernels. |
| Prefill on Windows | 1.07–1.74 k tok/s on short prompts vs 8,340 tok/s Linux reference on a 7.7 k prompt **[measured]** | The NVFP4 W4A4 TMA kernels are stubbed under MSVC. For agentic coding (20–40 k contexts, dozens of turns) this is the single largest latency term (B1). |
| Quant quality | AIME 2026 81.4 % on Swift-abliterated GPTQ vs 94.0 % BF16 Swift; one DeepSWE proxy regressed on GPTQ **[measured, small n]** | Quantization noise, not Swift, is the measured quality loss. Trace-calibrated GPTQ and sensitivity-chosen FP8 layers are quality wins with zero speed cost (C2). |
| Training on this box | BF16 27B = 54 GB, does not fit; NF4 load ~12 GB GPU, forward jobs peak ~20 GB; `scripts/poc/train_loop_lora.py` already does NF4 + LoRA + adapter-disabled anchor, its dry run never completed **[measured / partial]** | QLoRA on the 27B fits in 32 GB. The first training job is a 3-step dry run for tok/s and peak memory (D0). |
| RL as written | `opd_restore.py` / `gspo_restore.py` load two BF16 27Bs and sample through HF `generate` | Not runnable here. Replace with rejection sampling through `ninfer-serve` plus offline preference training (D1–D3). |

## 1. Constraints that shape everything

- **32 GB VRAM**: one 27B at a time. 4-bit for training, NVFP4/FP8 for serving. Never a server and a
  training job at once; the calendar in §6 serialises the GPU.
- **128 GB RAM**: LoRA merge (54 GB BF16) and RTN source builds run on CPU. CPU-offloaded GPTQ
  needed ~83 GB committed with heavy paging.
- **Thermal rule** from `tools/ninfer_build/eval/run_all.sh`: never sustain ≥ 70 °C for more than
  10 min; the last eval peaked at 75 °C. Set a power limit before any multi-day job and keep the
  `cool_gate` pattern in every long loop.
- **Windows**: run every Python GPU job through `scripts/swiftdiff/lowres.py` (hard CUDA cap,
  background priority); set the driver's "CUDA – Sysmem Fallback Policy" to "Prefer No Sysmem
  Fallback" for the venv `python.exe` and `ninfer-serve.exe`.
- **Disk**: keep ≥ 150 GB free. Each merged BF16 copy is 54 GB, each `.ninfer` artifact 23.7 GB.

---

## 2. Track A — measure coding for real (week 1)

### A1. Build the plain (non-abliterated) Swift NVFP4 artifact — day 1
Without it there is no honest base-vs-Swift-on-code number and no clean starting point for
training.

1. `tools/ninfer_build/build_swift_nvfp4_source.py` hard-codes `SWIFT = ROOT / "swift"`, which is
   the *abliterated* copy. Point it at the plain Swift shards (`A:\models\Qwen3.8-27B`, the
   hash-verified Swift release) and a new `OUT` (`A:\models\swift-plain-nvfp4-source`); better,
   give it `--swift` / `--out` flags **[to write, small]**. RTN source build: ~70 s.
2. Convert with the Swift staging dir (`A:\models\qwen38-staging-official`) as `--model` and the
   new source as `--quantized-model`; ~3–4 min. Preflight must show 1190 objects, 233 FP8 + 168
   NVFP4, 81 DFlash2 tensors.
3. Before serving, diff a few quantized matrices of the source against the intended BF16 (the
   `scripts/swiftdiff/diff_swift.py` approach) so the mix-up of §0 cannot recur silently. Make
   this a habit for every artifact.

### A2. A three-tier coding eval that is not saturated — days 1–3
Run everything through `swiftlab eval` (same task / same seed / same context per arm, bootstrap
CI, McNemar, thinking tokens, latency, truncation). Serve arms one at a time and use the resume
trick from `docs/SETUP_WINDOWS.md` §5 so two models never have to co-reside. Verify first with
`swiftlab preflight` against `ninfer-serve`: the OpenAI backend sends
`chat_template_kwargs.reasoning_effort`, which NInfer rejected in the harvester; if preflight
gets a 400, add a config switch to omit it **[to write, 3 lines]**.

| Tier | What | Verifier | Size / cost |
|---|---|---|---|
| 1. Function-level | EvalPlus HumanEval+ and MBPP+; a *hard* slice: LiveCodeBench problems dated after the model's cutoff; BigCodeBench-Hard for library use | existing `code_tests`; stdin/stdout problems need a `code_stdio` verify kind **[to write]** | ~600 tasks, 3 seeds; hours per arm |
| 2. Edit-with-context | Aider polyglot subset (multi-language edits with tests) | its own harness against `ninfer-serve` | ~60 exercises; ~1 h per arm |
| 3. Agentic | SWE-bench Verified fixed 50-task subset via mini-SWE-agent (Docker Desktop, WSL2 backend) at 2 lanes; Terminal-Bench 2.1 subset, for which UkisAI's 5-seed base/Swift logs are an external reference | resolved / pass | ~50 tasks × 2 seeds; overnight per arm |

Decontaminate before building any training bank: list every Tier-1/2/3 question file under
`bank.decontam_against` (13-gram) and keep a private held-out slice that never touches training.

### A3. The zero-training grid — days 2–4
Arms: {base NVFP4 (neroued), Swift plain RTN (A1)} × {effort `medium`, `xhigh`} × {stock system
prompt, a Sharp/Dirk-style "act, don't re-read" prompt}. Tier 1 full, Tier 3 subset. Before
spending GPU time read `data/external/deepswe11-qwen38-27b`: it already has the same SWE tasks at
low/medium/xhigh for the 27B, i.e. the effort effect for free. The Sharp/Dirk claim is
54.6 → 20.0 min per SWE-bench-Live task at equal solves **[claimed]**; a 2×2 here settles it.

Deliverable: one table saying which effort and prompt to serve for coding, and how far plain
Swift sits from base *on code* (UkisAI reports < 1 % overall loss; verify per tier).

---

## 3. Track B — faster on NInfer (parallel with A, no training)

### B1. Prefill: get the TMA W4A4 path back — biggest single win
`tools/ninfer_build/windows-port.patch` stubs `src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu` and
`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu` and gates five launch sites behind
`NINFER_DISABLE_TMA`; every prefill and batch-verify GEMM falls to the `tokens <= 64`-style
branches. Two ways out, try (a) first:

- **(a) WSL2 build.** NInfer targets Linux natively; CUDA on WSL2 supports sm_120. Build in an
  Ubuntu WSL2 with CUDA 13.x and Ninja, copy artifacts into the WSL2 filesystem (23.7 GB each;
  `/mnt/a` is slow), serve on `127.0.0.1` (WSL2 forwards localhost). Expect the Linux numbers minus
  WSL overhead **[estimate]**. Also removes the MSVC-only test failures. Cost: about a day.
- **(b) Native fix.** Unset `NINFER_DISABLE_TMA`, build only those two objects, collect the actual
  MSVC/nvcc errors and patch them (the `__int128` → `_umul128` fix in the patch is the same kind
  of work). Time-box to one day; fall back to (a).

Why it matters **[estimate, arithmetic]**: at a 30 k-token agent context, prefill per turn is
~18 s at 1.7 k tok/s vs ~4 s at 8 k tok/s; over 30 turns that is ~7 min vs ~2 min of pure prefill
per SWE task, before any decode.

Also test whether `ninfer-serve` reuses KV for a repeated prefix across turns (the docs here do
not say): send the same 20 k prefix twice and compare prefill time. If it does not, agent loops
pay full prefill every turn, which makes B1 matter even more.

### B2. Re-pick the weights profile on Windows — one converter run
`groupwise-int` (`qwen3_8_27b-v2`) is 3 GB smaller (16.67 vs 19.73 GiB) and decodes faster on
Linux (79.8 vs 71.2 tok/s); its prefill (3,275) loses to NVFP4 (8,340) only when the TMA path
works. On Windows today NVFP4 prefill is 1.7 k, so groupwise-int may win outright. Measure both
on Windows with the 7.7 k-token prompt (like-for-like with the Linux table). Revisit after B1.

### B3. Speculative decoding on *code* output
All acceptance numbers so far are prose probes. Code is more predictable, so acceptance and the
best draft length should be higher. Sweep `--spec mtp --draft-tokens {3,4,5}` and
`--spec dflash2 --draft-tokens {5,7,9}` at temperature 0 and 0.6 on Tier-1 tasks; pick per
workload. GPTQ raised DFlash2 acceptance from 31.7 % to 37.7 %, so C2 helps here too. After any
fine-tune (Track D) acceptance drops, because the MTP head and the DFlash2 drafter were trained
on base; realign them by self-distillation on the final model's outputs (NInfer open item 5).

### B4. The thinking budget is the largest latency knob
Swift's −58 % thinking is why decode "feels" fast. From A3, serve `medium` for coding wherever
accuracy holds, and the better system prompt. D2's loop-aware training compounds on top.

### B5. Context and lanes
Keep `--kv-dtype int8` (FP8/INT8 KV is near-lossless for long reasoning; avoid FP4 KV),
`--max-context 40960`, `--max-concurrency 2` with a long `--pending-timeout-ms` (8 lanes expired
in admission). Two lanes is the agent throughput ceiling; groupwise-int's 3 GB buys KV for a
third lane **[estimate]**. Measure `--preserve-thinking` on and off for multi-turn agents: it
changes both context length and behaviour.

### B6. Free quality that is also free speed
Sensitivity-chosen FP8 MLP layers (zero speed cost at 8 layers) and trace-calibrated GPTQ: see
C2. Skipping lm_head GPTQ saves ~70 min and the ~83 GB paging spike.

### B7. Housekeeping
Sysmem fallback policy; a power limit that holds < 70 °C; `lowres.py` on every Python GPU job;
start `ninfer-serve` with `--log-level debug` redirected to a file (one silent exit seen).

---

## 4. Track C — a better coder without training (weeks 1–2)

### C1. Serving config
Effort and prompt from A3; temperature 0.2–0.6 for code; `--default-max-tokens` sized from the
Tier-1 p99 thinking length so truncation stays at zero (the LiveCodeBench "+4.8 pp" in Swift's own
table was truncation).

### C2. Quantization done for coding
- **Calibration set = the model's own code-heavy traces**, 4–8 k tokens per sample (not 1024):
  resolved Open-SWE runs by Qwen3.8-27B (agentic, long context), correct Tier-1 solutions from
  the A2 rollouts (`swiftlab rollout --split calib`, never `eval`), plus plain code and docs.
  `swiftlab quant` builds `calib.jsonl` from rollouts.
- **GPTQ NVFP4**: act-order, `ignore` lm_head, MSE block-scale search if llmcompressor exposes it
  (global rotations hurt at group-16), `save_original_format=False`, 3.12 venv with
  `processor=tok`. ~5–6 h CPU-offloaded.
- **FP8 layer choice by sensitivity**: rank the 64 MLP layers by their KL contribution on the
  calibration set and give the top 8 FP8 instead of the fixed 56–63. Needs a new weights profile
  (converter, binder, workspace sizing, tests) on the NInfer side **[to write]**.
- **Gauge**: `ninfer-perplexity` NLL on held-out BF16 Swift *coding* traces for each artifact; NLL
  differences are KL differences on Swift's own distribution. This is the KL tool you do not
  otherwise have (serve rejects logprobs; the HF script fails on FP4).
- **Accept** when Tier-1 pass@1 is within CI of the RTN artifact, thinking tokens are not up, and
  MTP/DFlash2 acceptance is not down.

### C3. Task-vector transfer, only if a donor exists
If a same-architecture coding fine-tune of Qwen3.8-27B exists, `swiftlab transfer --only mlp
--alpha 0.3 --keep 0.2` is the cheapest capability gain there is; verify with the paired eval.
Never splice quantized sources of a different base (the mix-up lesson).

---

## 5. Track D — training the coder on the 5090 (weeks 2–6)

### D0. Feasibility dry run — first GPU day of the track
```
python scripts/poc/train_loop_lora.py --model A:\models\Qwen3.8-27B --data <any labelled windows> --out runs\dry --dry-run 3 --train-len 8192
```
under `lowres.limit()`. It logs `tok_per_s` and `max_mem_gib`; everything below is sized by those
two numbers. Expectation **[estimate]**: on the order of 100–300 tok/s, so 10 M training tokens
is roughly 10–30 h. Risk: fla/Triton backward on Windows + Blackwell (triton-windows 3.7.1 and
fla 0.5.2 are installed); if it fails, train in the same WSL2 as B1(a).

Adapter: rank 32–64, alpha 2r, targets = what Swift itself changed (q/k/v/o on the 16
full-attention layers, gate/up/down on all 64), optionally DeltaNet `in_proj_qkv` / `in_proj_z` /
`out_proj` as the PoC trainer already does. NF4 double-quant base, gradient checkpointing, 8 k
windows (agent transcripts cropped to the last 8 k like `crop()`), batch 1 × accumulate 8.

### D1. Data — three sources, two already on disk
1. **Agentic SWE** (on disk: `data/external/open-swe-traces`, 14 k Qwen3.8-27B runs with a
   `resolved` label). Keep resolved, dedupe by repo + issue, decontaminate against SWE-bench
   Verified/Lite/Live by repo + issue and 13-gram, then **trim the loops that predict failure**:
   drop cross-step re-read iterations (77 % of loop tokens; OR 0.89 per +10 points of loop share)
   using the `scripts/loops/detect.py` / `swe_analyze.py` labels, and keep edit-test-fail cycles
   (they go *with* resolution). Format as multi-turn mini-SWE-agent conversations with thinking,
   loss on assistant turns only **[to write: `scripts/coder/swe_traces_to_sft.py`]**. Add
   `data/external/ukisai-agent-sft` as-is and the kaitchup DeepSWE medium-effort runs as
   short-thinking exemplars.
2. **Verifiable function / contest tasks by rejection sampling** (the practical RL here). Build a
   hard bank with hidden tests: the seeds and synthetic templates are too easy
   **[to write: importer into `code_tests` / `code_stdio` rows + decontam]**. Sample N = 8 per
   task at T = 0.8, at both efforts, through `ninfer-serve` on the *current* artifact with
   speculation at 2 lanes. Keep correct samples, prefer the shortest thinking among them, and keep
   (correct-short, wrong-or-looping) pairs for D3 **[to write: `scripts/coder/sample_rft.py`]**.
   Budget **[estimate]**: 1 k tasks × 8 × ~3 k tokens ≈ 24 M tokens ≈ 1–1.5 days at 200–300 tok/s
   aggregate; start at 1 k tasks, not 5 k. This is where the 5090's inference speed does the work
   instead of training compute.
3. **A stronger sibling for the tasks the 27B cannot solve** (optional, paid). Flash-Next via
   OpenRouter ($0.15 / $0.47 per M tokens) shares the 27B tokenizer. Do **not** SFT on its
   thinking. Use its test-verified solution as a hint so the 27B produces its *own* correct trace,
   then train on the hint-free prompt. 5 k tasks × ~8 k tokens ≈ 40 M output tokens ≈ $20
   **[estimate]**.

On disk but not a coding target: MaxDevv 1M (the 27B's own long xhigh traces, many capped at
32 k). Use it only for loop labels.

### D2. Loss — one trainer generalising `train_loop_lora.py` and `train_penalized_lora.py`
**[to write: `scripts/coder/train_coder_qlora.py`]**
```
CE on kept assistant tokens
+ β · P(pool | think positions)        46-id corrected pool; drop ` or` (UkisAI's AIME regression)
+ unlikelihood on labelled loop-entry tokens
+ KL anchor to the adapter-disabled model on a sample of normal tokens   (forgetting guard, no second model)
```
β 0.2 → 0.5; watch `penalty_mass` fall with `ce` flat, as in `docs/SWIFT_RECIPE.md`.

### D3. Offline preference stage — replaces GSPO/OPD on this machine
DPO/ORPO-style on the pairs from D1.2, and on resolved-vs-unresolved SWE runs of the same
issue where both exist. Reference log-probs come from the same NF4 model with
`model.disable_adapter()`, so it fits. Use a sequence-level, length-normalised objective: the
GSPO argument in `swiftlab/rl.py` (per-token ratios are unstable on long traces) applies.
Reward shape from `efficiency_reward`: correctness first, brevity only as a bonus for correct
answers. **[to write, ~150 lines on D2's loader]**. `gspo_restore.py` as written is not runnable
here: HF `generate` at 4-bit is roughly 10–20 tok/s **[estimate]**, and G = 8 × 6 k tokens per
step is hours.

### D4. The round loop — 3–4 days per round **[estimate]**
```
train (D2 → D3)
→ scripts/merge_lora.py on CPU                       (BF16, 54 GB, needs the 128 GB RAM)
→ build_swift_nvfp4_source.py --swift <merged>       (~70 s; recomputes d_w)
→ convert_nvfp4                                       (3–4 min)
→ Track A: Tier 1 every round, Tier 3 subset every round, TB subset for release candidates
→ keep or revert
```
`d_x` reuse from the unsloth source is a < 2 % approximation for small deltas; after larger
fine-tunes recompute `d_x` from the merged model's own calibration activations
**[to write: `--recompute-dx`]**. GPTQ (5–6 h) only for a release candidate.

### D5. Gates per round (paired, same seeds)
- Tier-1 accuracy delta CI must not exclude 0 downward; Tier-3 resolved not down.
- Thinking tokens not up (mean and p90); truncation count unchanged; `needed_cut` ≈ 0 where
  settle data exists.
- Math and knowledge arms of the existing bank unchanged (catches an ` or`-type mistake early).
- Refusal rate unchanged if the abliterated base is used.

### D6. Release
GPTQ + FP8-layer selection (C2) on the final merged model → realign MTP and DFlash2 (B3) →
full Track A with 3–5 seeds → publish artifact + eval JSON, and record exactly which quantized
source built it.

---

## 6. Order and calendar (one GPU, so serialise)

| When | GPU | CPU / side work |
|---|---|---|
| Week 1 | A1 plain Swift artifact; A2 baseline runs; A3 grid; B2 profile measurement; B3 acceptance sweep | B1(a) WSL2 build; Tier-1/2/3 harness code; kaitchup effort table |
| Week 2 | C2 GPTQ overnight; D0 dry run; D1.2 sampling starts (server) | D1.1 SWE trace conversion; D2/D3 trainer |
| Weeks 3–5 | D rounds: train ~1 day, sample ~2 days, eval ~half day | inspect failures, curate data |
| Week 6 | D6: GPTQ, drafter realign, final eval | write-up |

Highest value per GPU-hour, in order: A1 + A2 (you cannot see anything without them) → B1
(prefill 4–5× for agent use) → A3/C1 (effort + prompt, near-free) → C2 (quant quality) → D.

## 7. What to skip on this machine
- OPD/GSPO as implemented (two BF16 27Bs, HF sampling).
- Full-parameter training; dense → MoE conversion (report §7).
- The null-space edit and loop-direction steering: research tracks, not on the coder critical
  path. Revisit after D shows where the remaining loops are.
- llama.cpp for serving (24 tok/s decode); keep it only for runtime-LoRA experiments.
- Abliteration work: it is weakened through FP4 anyway. Start the coder from plain Swift unless
  uncensored is a requirement, in which case use the d0xin BF16 as the base and keep D5's refusal gate.

## 8. Risks and the mitigation already chosen
| Risk | Mitigation |
|---|---|
| fla/Triton backward fails on Windows | WSL2 (shared with B1a) |
| Windows sysmem fallback stalls the PC | `lowres.py` cap + driver policy |
| Thermal cap on multi-day sampling | power limit + `cool_gate` in every loop |
| Eval contamination | `decontam_against` before any bank; private held-out slice |
| Regressing math/general (the ` or` lesson) | D5 gates every round |
| Building the wrong thing again | converter preflight + tensor diff before every artifact (A1.3) |
| Drafter acceptance collapses after fine-tune | B3 realignment on the final model |
| Sampling budget balloons | start at 1 k tasks; scale only after one round shows a gain |

## 9. Scripts and changes this plan needs **[to write]**
| Path | Purpose |
|---|---|
| `tools/ninfer_build/build_swift_nvfp4_source.py` | `--swift`, `--out`, `--recompute-dx` flags |
| `swiftlab/tasks/coding.py` | `code_stdio` verify kind (stdin/stdout, per-test time limit) |
| `swiftlab/backends/openai_compat.py` | switch to omit `chat_template_kwargs` for NInfer |
| `scripts/coder/import_code_bank.py` | EvalPlus / LiveCodeBench / BigCodeBench + hard train sets → bank rows, decontam |
| `scripts/coder/swe_traces_to_sft.py` | Open-SWE resolved runs → loop-trimmed multi-turn SFT |
| `scripts/coder/sample_rft.py` | N-sample rejection sampling through `ninfer-serve`, verification, pair extraction |
| `scripts/coder/train_coder_qlora.py` | D2 loss; `--dpo` mode for D3 |
| `scripts/coder/eval_swe_subset.ps1` | mini-SWE-agent on a fixed 50-task subset against `ninfer-serve` |
| `scripts/coder/sensitivity_fp8_layers.py` | per-layer KL contribution → FP8 layer list; NInfer weights profile |
