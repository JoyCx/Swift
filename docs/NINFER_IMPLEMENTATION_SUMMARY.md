# NInfer implementation summary — Swift-Qwen3.8-27B on Windows / RTX 5090 (as of 2026-09-17)

What was done to build, run and evaluate Qwen3.8-27B (base, Swift, Swift-abliterated) on the NInfer engine,
and everything needed to reproduce it. Companion to `RESEARCH_REPORT_2026-09-16.md`.

Status markers: **[working]** reproduced and in use · **[measured]** · **[caveat]** known limitation ·
**[not done]**.

---

## 1. Environment

| Item | Value |
|---|---|
| GPU | RTX 5090 32 GB, compute capability 12.0 (`sm_120a`) |
| OS | Windows 11 (NInfer officially targets Linux; runs via a local Windows port) |
| Compiler | MSVC 14.38 (VS 2022 Community), CUDA 13.3 `nvcc`, CMake + Ninja, `Release`, `CMAKE_CUDA_ARCHITECTURES=120a` |
| Media deps | FFmpeg shared (`A:\ffmpeg-master-latest-win64-gpl-shared`), vcpkg (`A:\vcpkg`) via `PKG_CONFIG_PATH` |
| Repo | `A:\ninfer` (master, `d4929686`) with worktree `A:\ninfer-spark` (branch `spark-x25-port`) |
| Binaries | `A:\ninfer-spark\build\apps\{ninfer,ninfer-serve,ninfer-perplexity}.exe` |
| Converter Python | NInfer tools env (Python 3.14, torch 2.11+cu128) |
| Quantization Python | `A:\swift\.venv` (Python 3.12): llmcompressor 0.13.0, compressed-tensors 0.18, transformers 5.14.1, triton-windows 3.7.1.post27, fla 0.5.2, bitsandbytes 0.50.2, peft 0.21 |

### 1.1 Windows port **[working]** (`build/windows-port.patch`, copy in `tools/ninfer_build/`)
20 files, uncommitted in the worktree:
- **Build flags** (`CMakeLists.txt`): MSVC `/Zc:preprocessor /utf-8`, `NOMINMAX`, `WIN32_LEAN_AND_MEAN`, CCCL
  preprocessor warning suppressed; same flags passed to CUDA via `-Xcompiler`; `UTF8PROC_STATIC`.
- **POSIX shims:** `src/compat/win_posix.h` included by the perplexity app, artifact reader (Windows file
  mapping), logging, startup log, request log; `getpid` → `_getpid`.
- **128-bit math:** `_umul128` in place of `__int128` in `runtime/contract/types.h` and
  `engine/materialization_planner.h`.
- **Move semantics:** explicit `SequencePlan` move constructor/assignment (`qwen3_6/impl/runtime/api_impl.h`).
- **NVFP4 W4A4 "TMA" kernels disabled on MSVC** (`NINFER_DISABLE_TMA=1`): the TMA launchers become stubs and the
  W4A4 routes fall back to the non-TMA path (`tokens <= 64` branches). **[caveat]** This likely explains why
  prefill on Windows measured ~1.7 k tok/s on a short prompt vs 8.3 k tok/s in the Linux performance report
  (7.7 k prompt; not a like-for-like comparison).
- POSIX-only tests (pretty_logging, context_cost, request_log, host_timing, gdn_replay*, …) do not build under
  MSVC; `ninfer_resource_manager_test` fails on Windows independently of the port.

Build:
```bat
A:\ninfer-spark\build\vsenv.cmd cmake -S A:\ninfer-spark -B A:\ninfer-spark\build -G Ninja -DCMAKE_BUILD_TYPE=Release
A:\ninfer-spark\build\vsenv.cmd cmake --build A:\ninfer-spark\build -j
```
`vsenv.cmd` loads `vcvars64.bat -vcvars_ver=14.38` and sets `PKG_CONFIG_PATH`, then runs its arguments.

---

## 2. How Qwen3.8-27B maps onto NInfer

- Qwen3.8-27B is architecturally identical to the registered `qwen3_6_27b` target; the registry routes it via
  `Qwen3_6_27B::qwen3_8_model_id`. Swift and abliterated variants are weight conversions, not engine ports.
- Two weight profiles exist for identity `qwen3.8-27b`:

| Role | `nvfp4` profile (`qwen3_8_27b_nvfp4-v2`) | `groupwise-int` profile (`qwen3_8_27b-v2`) |
|---|---|---|
| Attention q/k/gate/v (fused), attention out | FP8 row-scaled | Q4 / Q5 |
| DeltaNet q/k/v/z (fused), DeltaNet out | FP8 row-scaled | Q4 / Q5 |
| DeltaNet A/B, conv, norms | BF16 (A_log, dt_bias FP32) | BF16 |
| MLP gate_up / down | layers 0–55 NVFP4 + FP32 input divisor; 56–63 FP8 | Q4 / Q5 all layers |
| Embedding | FP8 re-encoded from official BF16 | W8G32 |
| lm_head | FP8 (from quantized source) | W8G32 |
| MTP / draft head (131,072 rows) | W8G32 / Q4G64 from official BF16 | same |
| Vision | groupwise Q4/Q5/Q6/W8 + BF16, from official BF16 | same |
| DFlash2 drafter | W8G32 matrices, BF16 norms/conv/selector | same |
| Weight arena | 19.73 GiB | 16.67 GiB |

- The layer split is hard-coded (`bindings.cpp:380`, `layer < 56`; converter regex `convert_nvfp4.py:49-55`).
  Changing which layers are FP8 requires a new weights profile (converter, binder, workspace sizing, tests).
- Runtime precision: NVFP4 MLP runs **weight-only at small token counts** (single-request decode) and W4A4 at
  prefill, batches and wide speculative verification.

---

## 3. Build pipeline (converter) **[working]**

The NVFP4 converter is a three-source repack; it does not quantize itself:
```text
python -m tools.convert.qwen3_8_27b.convert_nvfp4 \
  --model <official-layout BF16 dir>            # embeddings, MTP, draft head, vision, frontend resources
  --quantized-model <compressed-tensors dir>    # every FP8 / NVFP4 matrix, d_w, d_x
  --dflash2-model A:\models\Qwen3.8-27B\dflash2
  --out out\<name>.ninfer --device cuda
```
Preflight (`convert_nvfp4.preflight_conversion`) must be green: 1190 objects, 233 FP8 + 168 NVFP4 source
matrices, 81 DFlash2 tensors.

### 3.1 Gotchas and fixes
| Problem | Fix |
|---|---|
| Converter pins SHA-256 of 6 frontend files to `Qwen/Qwen3.8-27B@1d4bf0f2`; the Swift release's `generation_config.json` differs | `tools/ninfer_build/stage_official.py`: staging dir with hardlinked shards + canonical `generation_config.json` (sha `e70c136c…`). Staging dirs: `A:\models\qwen38-staging-official` (Swift), `A:\models\swift-abliterated-staging-official` |
| `A:\models\Qwen3.8-27B\NVFP4\` is a ModelOpt export | Rejected by converter; use compressed-tensors sources only |
| Quantized sources of a different base (unsloth base, OrcaRouter) | Never splice into a Swift build — bases differ per layer |
| `--quantized-model` supplies *all* quantized matrices | Using `NVFP4_unsloth` makes the artifact effectively **base**, whatever `--model` is (see 4.1) |
| gate/up must share identical `d_w` and `d_x` | Enforced in source builders (joint scale) |
| llmcompressor output uses text-only names / config | `tools/ninfer_build/remap_gptq_source.py`: rename `model.layers.*` → `model.language_model.*`, rebuild multimodal config + quantization_config |
| `save_pretrained` of an offloaded llmcompressor model crashes at the end ("could not revert weight conversions because of offloading") | Pass `save_original_format=False` (lost a 6.5 h run before this was found) |
| llmcompressor broken on Python 3.14 (pydantic) | Run in `A:\swift\.venv` (3.12); pass `processor=tok` to `oneshot` (else it builds the multimodal processor and needs torchvision) |

### 3.2 Source builders
- **RTN source for Swift-abliterated:** `tools/ninfer_build/build_swift_nvfp4_source.py` + `quant_lib.py`
  (compressed-tensors observers/compressor). 233 FP8 (bit-exact row scale amax/448) + 168 NVFP4 (d_w recomputed
  from Swift weights; **d_x reused from unsloth**, <2 % approximation) + 449 BF16 aux → `A:\models\swift-nvfp4-source`.
- **GPTQ source:** `tools/ninfer_build/gptq_nvfp4.py` (llmcompressor GPTQModifier; recipe = unsloth's
  `config_groups` + `ignore` verbatim; 64 samples × 1024 tokens of local docs/code/corpus text; act-order static)
  → `A:\models\swift-nvfp4-gptq` → remap → `A:\models\swift-nvfp4-gptq-ninfer`. Runtime ~6.5 h CPU-offloaded
  (~5 min/layer; lm_head alone ~70 min and ~83 GB committed memory, heavy paging).

---

## 4. Artifacts

### 4.1 Inventory
| Artifact | Built from | What it really is |
|---|---|---|
| `A:\models\Qwen3.8-27B-nvfp4-NInfer\qwen3_8_27b_nvfp4.ninfer` | neroued release | base NVFP4 ("regular") |
| `out\qwen3_8_27b_nvfp4.ninfer` | `--model` Swift staging + `--quantized-model NVFP4_unsloth` | **effectively base** (all quantized matrices from unsloth base). Do not use as "Swift". |
| `out\qwen3_8_27b_nvfp4_abliterated.ninfer` | Swift-abliterated staging + `swift-nvfp4-source` (RTN) | Swift + abliteration, RTN |
| `out\qwen3_8_27b_nvfp4_abliterated_gptq.ninfer` | Swift-abliterated staging + `swift-nvfp4-gptq-ninfer` | Swift + abliteration, GPTQ |
| (not built) non-abliterated Swift NVFP4 | would need a Swift RTN/GPTQ source | **[not done]** |

All artifacts: 23.7 GB file, 1190 objects, conversion ~3–4 min, vision and DFlash2 included.

### 4.2 Abliteration through FP4 **[caveat]**
The abliterated BF16 differs from Swift by a rank-1 edit on 131 tensors (~1.4 % norm). RTN NVFP4 likely weakens
it (direction cosine ~0.3 on `down_proj`; FP8 ~0.73). Refusal-rate evaluation was inconclusive (thinking consumed
the token budget). Later analysis of Swift's own delta showed low-rank edits survive quantization *in projection*
(98–100 %) even when cosine is low, so the effect is probably weakened rather than erased — not re-measured for
abliteration.

---

## 5. Running

### 5.1 Server
```bat
ninfer-serve.exe <artifact> --host 127.0.0.1 --port 8090 --model-id <id> ^
  --max-context 40960 --kv-capacity auto --max-concurrency 2 --kv-dtype int8 ^
  --spec mtp --draft-tokens 3 --lm-head-draft --default-max-tokens 32768 --pending-timeout-ms 36000000
```
- `--spec dflash2 --draft-tokens 7` for the DFlash2 drafter; `--vision` for images (trim context, e.g. 16384 →
  25.5 GB used); `--preserve-thinking` keeps earlier-turn reasoning.
- With 8 concurrent lanes and 32 k outputs, requests expired in admission (`request_queue_timeout`); use 2 lanes
  or a long `--pending-timeout-ms`.
- One server started from a background shell exited silently once; restart with `--log-level debug` and redirect
  output to a file.

### 5.2 Requests
- OpenAI chat completions; send `reasoning_effort` as a **top-level** field. NInfer returns HTTP 400
  `chat_template_option_not_supported` for `chat_template_kwargs.reasoning_effort`
  (`scripts/harvest_api.py --no-template-kwargs`).
- Thinking is returned in `message.reasoning_content`; usage reports `completion_tokens_details.reasoning_tokens`.
- `enable_thinking: false` via `chat_template_kwargs` for direct vision answers.
- Not supported: `logprobs` / `top_logprobs` (rejected), `/v1/completions`.

### 5.3 Quality tooling
- `ninfer-perplexity`: CausalScoring over windows (context 4096, stride 2048, 1024-token prefill chunks → W4A4
  path); `--corpus` manifest or `--text FILE`; writes `report.json`. No KL divergence or logits export.
- Planned gauge: score UkisAI's BF16 Swift reasoning traces on each artifact; NLL differences equal KL
  differences on Swift's own distribution **[not done]**.
- HF-level KL script `tools/ninfer_build/kld_measure.py` fails on compressed-tensors NVFP4 (no FP4 matmul in plain
  transformers).

---

## 6. Measurements

### 6.1 Speed (short prompt probe, Windows, 1 request) **[measured]** (`tools/ninfer_build/eval/speed_*.txt`)
| Artifact | Prefill tok/s | Decode tok/s (no spec) | DFlash2 decode | DFlash2 accept |
|---|---:|---:|---:|---:|
| regular (base) | 1.07 k | ~74–77 | 165.6 | 28 % |
| rtn_abl | 1.70 k | ~74–77 | 174 | 31.7 % |
| gptq_abl | 1.74 k | 76.6 | 211.7 | 37.7 % |

Other runs: `out\qwen3_8_27b_nvfp4.ninfer` 60.6 tok/s text decode, DFlash2 K=7 178.6 tok/s at 40 % accept;
MTP3 on Swift-abliterated GPTQ 190–218 tok/s decode at 76–88 % accept on short answers (server log).
Linux reference (repo docs, 7.7 k prompt): nvfp4 prefill 8,340 / decode 71.2; groupwise-int 3,275 / 79.8;
MTP3 C=1 143.8 tok/s at 48.9 %; DFlash2 192.5 tok/s at 37 %.

### 6.2 Quality **[measured, small samples]** (`tools/ninfer_build/eval/eval_*.json`)
| Check | regular | rtn_abl | gptq_abl |
|---|---|---|---|
| Median reasoning tokens (short set) | 96 | 78.5 | 74.5 |
| Coding pass@1 (12 easy tasks) | 12/12 | 12/12 | 12/12 |
| DeepSWE-proxy (5 tasks) | 5/5 | 5/5 | 4/5 (failed `fix_binary_search`) |
| Refusals | 0 | 0 | 0 (inconclusive: truncated answers) |

AIME 2026 at xhigh, 32 k cap, Swift-abliterated GPTQ via `ninfer-serve` (43 answers, 2 seeds partial): 81.4 %
correct, thinking mean 8,124 / median 3,975 tokens, 5 truncated. UkisAI's BF16 Swift reports 94.0 % on AIME 2026.
Base NVFP4 run on the same set stopped at 18/120 answers (unscored).

Thermals during the eval: max 75 °C, longest ≥ 70 °C window 1 m 31 s.

### 6.3 Net assessment
- GPTQ's clear win is speculative-decoding acceptance and speed; quality gain over RTN is unproven (KL not
  measurable yet) and one harder coding task regressed.
- The abliterated artifacts are the only Swift-based NInfer builds.

---

## 7. Resource handling on this PC
- Heavy jobs (52 GB BF16 reads + GPU over-subscription) lagged the whole PC; Windows' shared-GPU-memory fallback
  is the likely cause of the throttling.
- `A:\swift\scripts\swiftdiff\lowres.py`: background priority, 6 CPU threads, hard CUDA memory cap (OOM instead
  of spilling). With it, NF4-on-load of the 27B takes ~1 min at ~12 GB GPU and forward jobs peak ~20 GB with
  no shared-memory growth.
- Suggested driver setting: NVIDIA "CUDA – Sysmem Fallback Policy: Prefer No Sysmem Fallback" for the venv
  `python.exe` and `ninfer-serve.exe` **[not done — user setting]**.
- llama.cpp on this architecture is much slower (24 tok/s decode, 497 tok/s prefill for base Q4_K_M) than NInfer;
  use it only for runtime LoRA/steering experiments.

---

## 8. Open items for NInfer
1. Build a **non-abliterated Swift** NVFP4 artifact (RTN and/or GPTQ source from `A:\models\Qwen3.8-27B`).
2. Swift-trace-calibrated GPTQ (4–8 k-token sequences, MSE block scales, act-order; skip lm_head GPTQ).
3. Perplexity-based KL gauge on BF16 Swift traces; multi-seed long-reasoning eval with length/loop/truncation.
4. Sensitivity-chosen FP8 MLP layers (new weights profile; zero speed cost at 8 layers).
5. MTP / draft-head realignment to Swift if acceptance lags base.
6. Investigate enabling the TMA W4A4 kernels under MSVC for Windows prefill speed.

## 9. File index
| What | Path |
|---|---|
| Staging, source builders, GPTQ, remap, KL, refusal, vision demo | `A:\swift\tools\ninfer_build\*.py` |
| Conversion / GPTQ / bench logs | `A:\swift\tools\ninfer_build\logs\` |
| Eval harness and results | `A:\swift\tools\ninfer_build\eval\` |
| Windows port patch, env wrapper | `A:\swift\tools\ninfer_build\windows-port.patch`, `vsenv.cmd` (originals in `A:\ninfer-spark\build\`) |
| NInfer artifact docs | `A:\ninfer-spark\docs\maintainer\qwen3.8-27b-artifact.md`, `tensor-formats.md`, `docs\performance\qwen3.8-27b.md` |
| Research report | `A:\swift\docs\RESEARCH_REPORT_2026-09-16.md` |

Note: `kld_measure.py` and `refusal_bench.py` still write to their original temporary scratchpad path; update the
output paths before reuse.
