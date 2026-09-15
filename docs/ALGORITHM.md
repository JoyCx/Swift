# The algorithm: measure waste, cut it surgically, prove it with paired evals

This document is the "why" behind every stage of `swiftlab`. The Swift authors described
their recipe in the r/LocalLLaMA thread as: generate a large, diverse, decontaminated set
of traces → group the ones that overthink → find the common-denominator tokens → build a
loss that penalises those tokens during thinking (LoRA SFT) → restore accuracy with
on-policy distillation and ThinkingCap adapter chunks → verify with 5-seed benchmark
runs. `swiftlab` reproduces that recipe end to end and adds two things they did not
publish: an **activation-level surgical edit** (no training needed for the first cut) and a
**measurable task design** (code that executes, facts that grep) so every claim is a number.

## 0. Vocabulary

| term | meaning |
|---|---|
| trace | the `<think>…</think>` block of one rollout |
| settle point | earliest prefix of the trace from which a forced answer is already correct |
| overspent tokens | tokens after the settle point in a trace whose final answer is correct |
| derailed | trace that was correct at some checkpoint but wrong at the end (an "overthinking error") |
| waste | overspent + derailed-tail + wrong-throughout + loop segments |
| need | productive segments before the settle point |
| dirty direction | difference of means of residual activations, waste minus need |
| atom | difference-of-means direction for a *protected* concept (coding, knowledge, format, language) |
| clean direction | dirty direction with the atom-predictable part removed (ridge residualisation) |

## 0.5 Preflight

Before any long run, `swiftlab preflight` validates the real setup end to end: the verifiers
(coding executes, knowledge greps, math checks), decontamination, and — in server mode — one
real completion per domain with thinking-split and forced-prefix checks. Nothing in the
pipeline uses a stand-in model; preflight is how the model-dependent stages are smoke-tested.

## 1. Task bank: measurable by construction

`swiftlab bank` merges hand-written seeds (`data/tasks/*.jsonl`), unlimited synthetic
instances (unique prompts, generated from parametrised templates, so they cannot be in any
benchmark) and your own tasks. Every task carries a verifier:

* **coding** — `code_tests`: hidden test cases executed in an isolated subprocess with a
  timeout and optional `time_limit`. Correctness *and* speed are measurable.
* **knowledge** — `regex`/`exact`: gold strings from real data you own (tables, docs, wikis),
  matched by grep. No LLM judge anywhere in the loop.
* **math** — `numeric`: boxed / final-answer number with tolerance.

Decontamination: 13-gram overlap (GPT-3/Llama rule) against any eval dumps you list in
`bank.decontam_against`, plus exact-normalised de-duplication. The bank is then hash-split
into `mine` / `calib` / `eval` so evaluation tasks never touch mining or calibration.

## 2. Rollouts with model-independent seeds

Seed for (task, sample k) = `sha256(task_id, k)`. The base, the edited model and every
quant decode the same task with the same seed, the same context and the same sampling
settings. That is what makes the later comparison *paired* instead of two noisy averages.

## 3. Settle probing: where does the answer stop changing?

For each trace pick 8 segment boundaries as checkpoints. Truncate the thinking there,
force-close `</think>` and let the model answer greedily (vLLM `continue_final_message`,
or a raw prompt template). Binary-search the earliest checkpoint whose forced answer is
correct.

```
correct at end, first-ok early    -> overspent   (waste = tokens after first-ok)
correct at end, first-ok at end   -> tight
wrong at end,   ok somewhere      -> derailed    (the model talked itself out of the right answer)
wrong at end,   never ok          -> wrong
```

`waste_summary` reports the removable-share upper bound. The Swift authors' claim that
"these patterns do not contribute to answer quality" becomes testable here: the overspent
share is exactly the thinking that provably did not change the answer.

## 4. Marker mining: the common-denominator tokens

Contrast NEED text against WASTE text with the "fightin' words" statistic (log-odds with an
informative Dirichlet prior, reported as z). Phrases with z ≥ 3 form the penalty
vocabulary. On real models this recovers `wait`, `but wait`, `hmm`, `let me double-check`,
`actually`, plus model-specific tics you would not have guessed. Output is used by
(a) the inference-time `logit_bias` penalizer and (b) the penalised LoRA loss.

## 5. Scaling: how many long traces are enough?

`swiftlab scale` subsamples the probed traces at increasing n and reports, per size:
removable share, wrong / derailed counts, markers discovered, distinct loop signatures and
the Chao1 estimate of how many signatures exist in total. Coverage = discovered /
estimated. The recommended n is the first size after which marker discovery grows < 5%
and signature coverage > 90%. Long, hard traces are worth disproportionately more than
short ones: they are where derailments live, so the curve is computed on the
`mine` split with the highest reasoning effort.

## 6. Directions and surgical cleaning (SRA)

From [Surgical Refusal Ablation](https://arxiv.org/abs/2601.08489) (Cristofano 2026),
applied to overthinking instead of refusal:

```
r_dirty[l] = mean_l(WASTE spans) - mean_l(NEED spans)
a_k[l]     = mean_l(D_k+) - mean_l(D_k-)                   k in {coding, knowledge, format, language}
w_hat      = argmin_w ||r_dirty - A w||^2 + lambda ||w||^2   (A = [a_1 … a_K], ridge)
r_clean    = r_dirty - A w_hat
v[l]       = r_clean / ||r_clean||
```

The dirty vector is polysemantic: in a reasoning model the "re-verify" direction is
entangled with "I am doing maths/code right now". Removing that component is what keeps
the capability intact — at identical edit strength an uncleaned direction breaks more
items than a cleaned one, which the per-domain eval table makes visible. Per layer the
bundle records separation AUC on held-out
spans before/after cleaning, energy removed, and cosines with each atom; `best_layers` are
the top third by clean AUC.

## 7. The edit

Rank-one projection on the output-side matrices of the chosen layers
(`self_attn.o_proj`, `mlp.down_proj`):

```
W' = (I - gamma_l * v_l v_l^T) W
```

gamma_l follows a flat or Gaussian kernel over layers (heretic-style). `swiftlab search`
picks (center, width, gamma, matrices) by co-minimising
`think_ratio + w_kl * KL(base || edited)` on the calib split, with a hard penalty whenever
accuracy drops more than `eval.accuracy_tolerance`. Optuna TPE if installed, otherwise
seeded random + local search.

Why not just penalise length? Because, as the Swift authors put it, length is important
and should be optimised, not shortened by force. The direction is estimated from the
*overspent* region only, so the edit targets the loop behaviour, not the step count.

## 8. Training-based path (the Swift recipe proper)

When the pure weight edit is not enough, or for a second pass:

1. `scripts/train_penalized_lora.py` — LoRA SFT on the model's own correct traces truncated
   at the settle point, loss = CE + β · P(penalty tokens | think position). The model learns
   not to *want* the loop.
2. `scripts/opd_restore.py` — on-policy distillation: sample from the student, score the
   student's own tokens with the frozen base, minimise reverse KL. Restores token-level
   judgement without re-lengthening traces. Wrong traces get a higher weight.
3. `swiftlab transfer` — task-vector transfer from a donor efficient fine-tune of the same
   architecture (ThinkingCap-Qwen3.6-27B minus Qwen3.6-27B), TIES-trimmed and scaled, on a
   module subset (`--only mlp`). This is the "adapter chunks" trick.

Every path ends in the same paired eval, so you can compare edit-only vs edit+LoRA vs
edit+LoRA+OPD on identical seeds.

## 9. Quantization after the edit

See `docs/QUANT.md`. Short version: the edit is baked into the tensors first, calibration
uses the edited model's own short traces, GGUF gets an imatrix from that text and a
KL-vs-BF16 check, W4A16/NVFP4 go through llmcompressor GPTQ with the same samples.

## 10. Paired evaluation and coverage

`swiftlab eval` runs every arm on the `eval` split with the same seeds and reports:

* accuracy per arm, paired accuracy delta with bootstrap CI, exact McNemar p, fixed/broken
* thinking tokens mean/median and reduction, per-item ratio, share of items shorter
* **overspend removed** = Σ min(cut, overspent) / Σ overspent — how much of the *measured*
  waste the edit removed
* **needed cut** = Σ max(0, cut − overspent) / Σ settle tokens — how much *needed* thinking
  it took away (should be near zero)
* derailed-in-base vs derailed-fixed, speed-up, per-domain breakdown, truncation counts
  (the LiveCodeBench +4.8pp in the Swift table came from truncation, which is why this
  column exists).

## 11. Applying this to any model

Nothing above is Qwen-specific. Requirements: a chat template with a thinking block
(`backend.think_open/close`), an OpenAI-compatible server for rollouts, and the HF
weights for capture/edit. `HFBackend._find_layers` handles Llama/Qwen/Mistral/GPT-NeoX
layouts; add the attribute path for anything exotic. For non-reasoning models the same
harness measures verbosity instead of thinking (set `think_open` to an empty string and
the whole completion is treated as the trace).
