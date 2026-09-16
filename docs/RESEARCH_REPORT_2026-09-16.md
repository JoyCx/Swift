# Swift / Qwen3.8 efficiency research — full report (2026-09-16)

Scope: reverse-engineer UkisAI's Swift-Qwen3.8-27B (shorter reasoning at ~same accuracy), find better
methods to cut wasted thinking, and carry the results to fast local serving (NInfer on one RTX 5090) and to
Qwen3.8-Flash-Next. Hardware: RTX 5090 32 GB, 128 GB RAM, Windows 11.

Status markers: **[measured]** = produced and checked here · **[claimed]** = third-party claim, not
reproduced · **[in progress]** / **[not run]**.

---

## 1. Executive summary

1. **Swift's hidden recipe was recovered.** The 47 (of 49) penalised token IDs were reconstructed exactly
   from UkisAI's own published eval logs, and the base-vs-Swift weight diff shows Swift is a **rank-32 LoRA on
   attention q/k/v/o (16 full-attention layers) and MLP gate/up/down (all 64 layers)** — the 48 DeltaNet
   layers were never touched. **[measured]**
2. **Swift mostly suppresses hedge words, not loops.** Its marker words drop 55–67 %, but real repetition
   loops (code redrafts, re-derivation, re-reading the task) are kept or grow; loops explain only 4–17 % of
   Swift's length reduction. UkisAI's own math regression traces to penalising `' or'`. **[measured]**
3. **Loops that matter in real agentic work are cross-step re-reading.** On 14 k Qwen3.8-27B SWE-agent
   runs, more loop share predicts lower resolution after controlling for length (OR 0.89 per +10 points,
   p = 6e-6); retries of failing commands do not. **[measured]**
4. **Inference-time word penalties do not work** (the model rephrases). A linear "about to loop" direction
   exists but is weak (held-out AUC ≤ 0.69). **[measured]**
5. **Quantization does not erase Swift's change** (98–100 % of its projection survives every NInfer
   format); the remaining quality problem is ordinary quantization noise. **[measured]**
6. **A key artifact mix-up was found:** `out/qwen3_8_27b_nvfp4.ninfer` is effectively *base* Qwen3.8-27B,
   not Swift (all quantized matrices came from unsloth's base NVFP4). **[measured]**
7. Current work: a closed-form, curvature/null-space-constrained ("Riemannian") loop edit — feasibility
   data collection **[in progress]**.

---

## 2. Data acquired

| Source | Model | Content | Size | Path |
|---|---|---|---|---|
| UkisAI Swift-Qwen3.8-27B-evals (GitHub) | 27B base vs Swift, BF16, paired | 9 benchmarks × 5 seeds incl. Terminal-Bench 2.1 trajectories, raw thinking | 4.5 GB | `data/external/ukisai-evals` |
| nvidia/Open-SWE-Traces (qwen38_27b) | 27B base | 107 k mini-SWE-agent runs on real repos, `resolved` label, median ~20 k thinking tokens | 10 GB | `data/external/open-swe-traces` |
| MaxDevv/Qwen3.8-27B-Distill-1M | 27B (NVFP4 backend, xhigh, T=1.0) | 992 k conversations, 3.57 B reasoning tokens, many capped at 32 k | 5.5 GB | `data/external/maxdevv-qwen38-27b-distill-1m` |
| kaitchup DeepSWE 1.1 trajectories | 27B | same tasks at low/medium/xhigh + Claude Code / mini-SWE harnesses | 10 MB | `data/external/deepswe11-qwen38-27b` |
| Lottolabs Terminal-Bench 2.1 traces | 27B | low/medium effort agent traces | 80 MB | `data/external/lottolabs-tb21-qwen38-27b` |
| ukisai/Qwen3.8-27B-multi-turn-agent-sft | 27B | UkisAI's own agent SFT data | 150 MB | `data/external/ukisai-agent-sft` |
| aswinkumar99/qwen3.8-flash-next-expert-traces (token files only) | **Flash-Next** NVFP4, T=0.2 | only public Flash-Next text: 2,494 agentic/doc responses rebuilt from token IDs | 44 MB | `data/external/flashnext-expert-traces` → `runs/flashnext_traces/responses.jsonl` |

Checked and not useful: DaoCloud Drafter-SFT (repetition-filtered), RadixArk Regen (private), MathArena
(no Qwen3.8/Flash-Next runs), HF Flash-Next community endpoint (retired), UkisAI free API (502 all day;
rate-limited harvester `scripts/harvest_api.py` kept). OpenRouter serves Flash-Next only via Alibaba
(paid, $0.15/$0.47 per M tokens).

Flash-Next shares the 27B tokenizer: all 47 pool IDs identical; 248,077 / 248,320 tokens identical.

---

## 3. Reverse-engineering Swift

### 3.1 The penalised token pool **[measured]**
- Terminal-Bench `tb21_per_trial.csv` gives per-trial `marker_hits` for their 49-id pool. Tokenising each
  trial's reasoning with the Qwen3.8 tokenizer reproduces their reasoning-token counts exactly (ratio
  1.0000); a 0/1 subset search found **47 ids that reproduce marker hits on all 890 trials exactly**
  (totals 227,151 base / 74,049 Swift).
- Independent check: GPQA Swift marker rate **6.69 / 1k** matches UkisAI's published number (base 15.29).
- Pool: leading-space hedge/backtrack words — ` or, but, But, Or, could, might, maybe, perhaps, possibly,
  still, yet, either, whether, though, although, however, However, instead, Instead, rather, otherwise,
  unless, regardless, anyway, actually, wait, Wait, hmm, Hmm, hold, another, different, Alternatively,
  alternatively, error, wrong, incorrect, mistake, doubt, uncertain, unsure, confused, retry, reconsider,
  rethink, revisit, backtrack`. Last 2 ids never occur in TB reasoning.
- Files: `runs/markers/ukisai_pool_recovered.json`, `scripts/recover_marker_*.py`.

### 3.2 UkisAI's math regression (`' or'`) **[measured]**
- AIME 2026 base 98.7 % → Swift 94.0 %; HMMT 99.3 % → 96.0 % (their traces).
- On the 10 problems Swift lost, Swift's wrong traces use `' or'` at 0.68/1k vs 1.41/1k in its correct
  traces; other pool words are flat (0.8–0.9×). Base uses `' or'` mostly for case enumeration
  ("0 or 1", "P(A<2 or B<2)"). Small sample (15 wrong traces), but the only strong outlier.

### 3.3 Waste Swift still leaves **[measured]** (`runs/markers/residual/`)
- "Post-decision" thinking share (correct traces), base → Swift: MMLU-Pro 0.34 → 0.26, GPQA 0.55 → 0.42,
  C-Eval 0.27 → 0.22; Swift keeps ~45 % of base's post-decision tokens (detector precision ~93 %).
- Answer flip-flops, base → Swift: GPQA 40.9 % → 30.2 % of traces, MMLU-Pro 15.5 % → 9.0 %.
- Missed variants: capitalised / line-start forms (`Maybe`, ` Maybe`, ` Could`, `But`, ` Actually`, …)
  suppressed only 0.65–0.80× vs 0.35–0.42× for the penalised forms.
- Untouched categories: re-verification ("Double-check", "trap", "careful"), guessing the test-maker
  ("dataset", "distractor", "intended"), format re-planning (Swift does *more*: "Ensure", "concise" 1.4–2.1×).

### 3.4 Swift's weights **[measured]** (`runs/swiftdiff/`)
- Diff of `A:\models\Qwen3.8-27B` (Swift, hash-verified) vs `A:\models\Qwen3.8-27B-base-bf16` (official,
  hash-verified): 256 of 866 text tensors changed.
- Singular values drop ~10× exactly after index 32 (e.g. L20 down_proj S32 0.0302 → S33 0.0032) → **rank-32
  LoRA**; residual beyond is BF16 merge rounding (max ≤ 3e-4).
- Changed: `self_attn.{q,k,v,o}_proj` (16 layers), `mlp.{gate,up,down}_proj` (64 layers). Unchanged:
  all `linear_attn.*` (DeltaNet), norms, embeddings, lm_head, MTP, vision.
- Relative change 0.25–0.75 %; energy weighted early (layers 0–19 ~2.2 % each, 36–61 ~0.7 %, layer 63 4.0 %).
- Exported clean adapters (PEFT r=32) `adapters/{full,mlp,attn,early,late}` and GGUF LoRA
  `adapters_gguf/swift-*.gguf`; llama.cpp confirmed the adapter changes greedy output.

---

## 4. Loop analysis on long traces **[measured]** (`runs/loops/`)

Detectors (validated by reading examples): paragraph near-duplicates (shingle Jaccard ≥ 0.5 +
containment), topic revisits, degenerate repeats, answer flip-flops, Terminal-Bench action loops.

- **Swift vs base (UkisAI data):** loop tokens per sample fall (TB 1,690 → 1,250) because Swift writes fewer
  long traces, but loop *share* inside long traces is equal or higher. Loops explain only 4–17 % of the
  length gap.
- By category (loop tokens per 1k, Swift/base): hedge-led re-derivation **0.57×**, code redraft **1.24×**,
  re-derivation **1.36×**, verification 1.05×, format re-plan 1.19×.
- Flip-flops predict wrong answers (GPQA base: 43 % vs 83 % accuracy); plain loop share does not.
- Terminal-Bench action loops go with failure (base solved 32 % with vs 71 % without); JSON-format failure
  loops worst.
- **Open-SWE (14 k Qwen3.8-27B runs):** loop share unresolved 14.4 % vs resolved 12.6 %; controlled for
  length/steps/dataset/category OR 0.89 per +10 points (p = 6e-6); 77 % of loop tokens are cross-step
  re-reads of the PR/patch; retries and edit-test-fail cycles go *with* resolution.
- **Pool words are depleted inside loops** (e.g. ` but` z −47): loops start with neutral restarts
  ("Now, let's consider…", "Let's re-read the PR description once more", "Actually, wait.").
- Flash-Next's own style (rebuilt traces): short thinking (median 99, p99 6.3 k tokens), no text repetition
  loops; decision dithering instead ("Keep simple… Actually… Simpler… Just return error").

---

## 5. Experiments on cutting thinking

### 5.1 Inference-time logit penalty on Flash-Next **[measured, small]**
- MixedINT4 GGUF on llama.cpp, 46-id corrected pool at −1.0, MuSiQue multi-hop (20 paired):
  base acc 0.90, thinking mean 1,451 / median 296; biased acc 0.95, mean 1,008 / median 438; marker rate
  23.4 → 11.1 /1k; paired median length ratio 0.99. The model rephrases hedges ("?", "Need consider",
  "Could answer"). Questions were too short to contain loops; the long AIME/HMMT arm was stopped early.
- Speed note: the same GPU runs Flash-Next at ~21 tok/s total across 2 slots.

### 5.2 Loop-labelled training data **[measured]** (`runs/poc/labels_maxdevv/`)
- 1,200 long MaxDevv traces → 1,819 windows, 16.8 M tokens, 18,836 loop-entry tokens, 5.9 % waste tokens.
  Labels: IGNORE / NORMAL / WASTE (2nd+ prose iteration) / ENTRY (first tokens of an iteration); code and
  markup blocks excluded.

### 5.3 Loop-aware LoRA trainer **[written, not run]** (`scripts/poc/train_loop_lora.py`)
- Loss: KL anchor to frozen base on NORMAL tokens + unlikelihood on ENTRY tokens + small pool-mass penalty.
- Dry run aborted: loading 52 GB BF16 while a game used GPU/RAM lagged the PC. Resource limits added since.

### 5.4 Linear "about to loop" direction **[measured]** (`runs/swiftdiff/loop_direction/`)
- 905 matched pairs (loop start vs normal paragraph start, same trace), residual stream at all 64 layers.
- Held-out AUC 0.59–0.69 (best 0.687 at layer 61); gap 2–8 % of residual norm; direction stable across
  adjacent layers (cos 0.93–0.97). Real but weak (refusal directions typically ≥ 0.95).
- Control-vector file built (`runs/steer/loop_gap.gguf`, per-layer increments) — **not yet tested**.

### 5.5 Swift adapter / steering sweep on llama.cpp **[partial]** (`runs/steer/`)
- Base Q4_K_M + Swift adapters, 12 AIME/HMMT problems × 1 seed. Base arm 11/12 finished: 100 % correct,
  thinking mean 11,176 / median 9,354, marker rate 14.15/1k; session ended before the Swift arm.
- llama.cpp is slow for this architecture: 24 tok/s decode (18 with LoRA), 497 tok/s prefill, GPU at 95 %.
- Test power (from UkisAI 5-seed data): single-run log-length σ = 0.43; 12 problems × 1 seed detects only
  ≥ 39 % changes (Swift's true effect on this set: 0.60×). Adequate for verifying Swift, too weak for steering,
  and too easy for accuracy (base 60/60, Swift 58/60).

### 5.6 Null-space ("Riemannian") loop edit **[in progress]**
- Idea: the Fisher / activation-covariance metric defines which weight directions matter for protected
  behaviour. Edit `down_proj`, DeltaNet `out_proj`, attention `o_proj` only inside the low-curvature
  (null) space of normal reasoning keys, mapping loop-entry keys toward "move on" (AlphaEdit-style,
  closed form, no training).
- Collecting inputs to 11 modules (MLP L12/24/36/48, GDN out L8/20/32/44/56, attn o L19/43) at loop starts,
  matched boundaries and 38 k random normal tokens (`scripts/swiftdiff/collect_edit_keys.py`).
- Feasibility metrics next (`scripts/swiftdiff/nullspace_feasibility.py`): null-space dimension, normal
  leakage, share of loop-vs-normal difference kept, held-out response on loop vs boundary vs normal tokens.

---

## 6. Serving and quantization

### 6.1 Local model inventory (hash-verified)
| Path | Actually is |
|---|---|
| `A:\models\Qwen3.8-27B-base-bf16` | official base BF16 (downloaded 2026-09-16) |
| `A:\models\Qwen3.8-27B\*.safetensors` (+ `qwen38-staging-official` hardlinks) | **Swift** BF16 |
| `A:\models\swift-abliterated-staging-official` | Swift + abliteration BF16 |
| `A:\models\Qwen3.8-27B-nvfp4-NInfer\qwen3_8_27b_nvfp4.ninfer` | neroued base NVFP4 artifact |
| `out/qwen3_8_27b_nvfp4.ninfer` | **effectively base** (quantized matrices from unsloth base NVFP4) |
| `out/qwen3_8_27b_nvfp4_abliterated{,_gptq}.ninfer` | Swift-abliterated (RTN / GPTQ) |
| `A:\models\Qwen3.8-27B\Q4_K_M` | unsloth base Q4_K_M GGUF |
| `A:\models\Qwen3.8-Flash-Next-abliterated-MixedINT4.gguf` | only Flash-Next copy (no BF16/FP8) |

### 6.2 NInfer facts (from repo docs and code)
- nvfp4 profile: attention/DeltaNet/lm_head FP8-row; MLP 0–55 NVFP4, 56–63 FP8 (hard-coded,
  `bindings.cpp:380`); MTP W8, draft head Q4, vision groupwise. NVFP4 MLP is **weight-only at small
  decode widths** and W4A4 at prefill/batch/wide verify.
- Formats are a closed set; a different layer split = new weights profile (converter + binder + tests).
- Speeds (7.7 k prompt): nvfp4 prefill 8,340 / decode 71.2 tok/s; groupwise-int 3,275 / 79.8; MTP3 C=1
  143.8 tok/s at 48.9 % accept (Qwen3.6: 69 %); DFlash2 192.5 tok/s at 37 %.
- Quality tooling: `ninfer-perplexity` scores given text (log p); no KL tool; serve rejects logprobs.
- Swift-abliterated GPTQ artifact on 43 AIME 2026 answers (xhigh, 32 k cap): 81.4 % correct, thinking
  mean 8,124 / median 3,975, 5 truncated, marker rate 5.03/1k (UkisAI BF16 Swift: 94.0 %). Small sample,
  confounded by quant + abliteration.

### 6.3 Does quantization erase Swift? **[measured]** (`runs/swiftdiff/quant_survival.json`)
Round-to-nearest on base and Swift weights, 21 matrices:
| Format | Swift projection kept | Direction cosine | Quant error / Swift change |
|---|---|---|---|
| NVFP4 | 0.98–1.00 | 0.12–0.16 | 14–59× |
| FP8 row | 1.00 | 0.28–0.36 | 4–17× |
| Q5G64 | 1.00 | 0.13–0.22 | 7.5–33× |
| Q6G64 | 1.00 | 0.19–0.32 | 3.6–16× |
| W8G32 | 1.00 | 0.41–0.70 | 0.8–3.5× |
Swift's low-rank change survives in expectation; noise is uncorrelated and spread over all directions.
A separate BF16 LoRA branch would add decode cost for little gain.

### 6.4 Quantization research summary (external evidence)
- NVFP4: GPTQ helps; global rotations (Hadamard/QuaRot/SpinQuant) hurt with group-16 scales; MSE/"4-over-6"
  block-scale search is free at runtime (MR-GPTQ, Four Over Six, ScaleSweep).
- Reasoning models: 4-bit (esp. W4A4) can lengthen CoT; calibrate GPTQ on the model's own reasoning traces.
- QAD/QAT recovers near-BF16 (e.g. QUASAR Qwen3.8-27B NVFP4 GPQA-D 90.9 vs 91.4) but full QAD does not fit
  32 GB.
- Drafters trained on base lose acceptance on fine-tunes; self-distilling MTP/DFlash2 on target outputs is cheap.
- KV: FP8 near-lossless for long reasoning; avoid NVFP4 KV.

### 6.5 Recommended quant plan (not started)
1. Quality gauge without engine changes: `ninfer-perplexity` on UkisAI's BF16 Swift traces (NLL differences
   = KL differences) + multi-seed thinking-length/loop/truncation runs.
2. Re-run GPTQ on merged Swift weights with Swift-trace calibration (4–8 k tokens), MSE block scales, act-order;
   skip lm_head GPTQ.
3. Choose the 8 FP8 MLP layers by measured sensitivity instead of fixed 56–63 (zero speed cost).
4. Realign MTP head / draft-head ranking to Swift if acceptance is lower than base.
5. Keep FP8/INT8 KV.

### 6.6 Resource handling on this PC
- Earlier lag came from GPU over-subscription (Windows shared-memory fallback) plus heavy disk reads.
- `scripts/swiftdiff/lowres.py`: background priority, 6 CPU threads, hard CUDA memory cap. NF4-on-load of the
  27B takes ~1 min and ~12 GB GPU; forward-pass jobs peak ~20 GB with no shared-memory spill.
- Saving an offloaded NF4 checkpoint fails in transformers 5.14 → quantize on load instead.

---

## 7. Other questions examined

- **Dense 27B → MoE:** carving (MoEfication/contextual sparsity) is possible only at mild sparsity and conflicts
  with speculative decoding; a full MoE needs pretraining-scale retraining; even a perfect conversion leaves
  attention/DeltaNet always active (~2.5–3× ceiling). Use pretrained MoEs (Qwen3.6-35B-A3B, Flash-Next).
- **peculiar-ragdoll (Dirk / Nail / Tiel):** template + system prompt + `medium` effort default (Dirk is
  template-only on Qwen3.8-27B). Claims **[claimed]**: SWE-bench-Live median time 54.6 → 20.0 min at 16 → 15
  solves; "~70 % more solves" belongs to CyberTiel (abliteration + custom imatrix + template together).
  One run per problem on 25 tasks — time effect robust, solve gains within noise. Serving config stacks on
  top of any weight edit; worth a 2×2 test (Swift/base × stock/Sharp prompt).

## 8. Flash-Next quant plan (blocked)
- Source: official `Qwen/Qwen3.8-Flash-Next-FP8` (185.6 GB) → `--fp8-as-q8` GGUF → edits → requant with
  trace-based imatrix and sensitivity-driven bit allocation (current MixedINT4: KLD 0.122 vs Q8, top-1 90.4 %).
- Blocked on disk space (~400 GB peak on one drive) and user download approval.

## 9. Next steps
1. Finish null-space edit feasibility; build and evaluate the edit only if held-out selectivity is clear.
2. Quality gauge (6.5 step 1) and a multi-seed NInfer eval set with hard, loop-prone problems.
3. Swift-vs-base × prompt 2×2 on NInfer.
4. Loop-aware LoRA dry run with resource limits.
5. Flash-Next once source weights and disk are available.

## 10. Key files
| What | Path |
|---|---|
| Marker recovery | `scripts/recover_marker_*.py`, `runs/markers/` |
| Residual waste mining | `scripts/mine_residual_markers.py`, `runs/markers/residual/` |
| Loop detectors and results | `scripts/loops/`, `runs/loops/` (`gallery.md`) |
| Flash-Next trace rebuild | `scripts/flashnext/rebuild_expert_traces.py`, `runs/flashnext_traces/` |
| Bias tests | `scripts/score_bias_test.py`, `runs/flashnext_bias/` |
| Loop labels, LoRA trainer | `scripts/poc/label_loops.py`, `scripts/poc/train_loop_lora.py`, `runs/poc/` |
| Swift diff, adapters, directions, quant survival | `scripts/swiftdiff/`, `runs/swiftdiff/` |
| llama.cpp sweep | `runs/steer/` |
| API harvester | `scripts/harvest_api.py` |
