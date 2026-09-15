# Replicating the Swift pipeline, plus abliteration as an extra

What the Swift authors wrote in the r/LocalLLaMA thread, sentence by sentence, and the
command that does it here. Everything they kept private (exact token list, loss weights,
data mix, OPD schedule) is a labelled guess you tune with the paired eval.

| what they said | command | notes / what is a guess |
|---|---|---|
| "generated a large amount of different (out-of-distribution) domain traces" on 8×H100 | `swiftlab bank --synthetic 400` + your seeds; `swiftlab rollout --split mine --samples 2` at `xhigh` | Domains here: coding (executes), knowledge (grep), math (numeric). Add agentic / vision tasks as custom verifiers if you need their coverage. Decontaminate against the benchmarks you will report on. |
| "grouped the ones with overthinking" | `swiftlab settle` | Their grouping was by eye/heuristics; here a trace is overspent when a forced answer at an earlier checkpoint is already correct, derailed when it was right then went wrong. |
| "found common-denominator tokens and targeted the most prominent ones" | `swiftlab mine` → `markers.json` | Log-odds z-score contrast of waste vs needed text. They referenced a Meta PTQ paper's token list as a starting point and said their own mined list worked better; take the top 20–40 by z. |
| "built an inference-time penalizer … did not work at all" (as a data source for plain SFT) | `penalty_vocab_to_logit_bias` + `--logit-bias` on rollouts | Kept as a diagnostic: shows what the tokens do at inference, not used for training data. |
| "built a loss function using the tokens and ran LoRA SFT over the traces" | `swiftlab sftdata` → `scripts/train_penalized_lora.py --beta 0.5` | Loss = CE on the needed prefix + β·P(penalty tokens at think positions). β, rank 32, lr 1e-4, 1 epoch are guesses; watch `penalty_mass` fall and `ce` stay flat. |
| "reasoning was falling off … accuracy seemed to follow" | `swiftlab eval` with arm `swift` | Expect a few pp loss after this stage; that is what step 4 repairs. |
| "restored accuracy with RL (GSPO), on-policy distillation and ThinkingCap adapter chunks" | `scripts/opd_restore.py` (OPD) and/or `scripts/gspo_restore.py` (GSPO); optional `swiftlab transfer --only mlp --alpha 0.4 --keep 0.2` | Both restore paths are implemented. OPD = sample from the student, reverse-KL to the frozen base on the student's own tokens. GSPO = on-policy RL with verifiable rewards (see below). ThinkingCap adapter chunks = `swiftlab transfer`. |
| "one token relevant for math reasoning was penalised by mistake" (AIME −4.6pp) | per-domain table in the eval report | If math drops while others hold, look for a math-specific phrase in `markers.json` and remove it from the penalty list. |
| "ran each benchmark 5× on base + 5× with our adapter" | `swiftlab eval --seeds 5` | Same seeds per task across arms, bootstrap CI, McNemar, fixed/broken, truncation column (their LCB +4.8pp was truncation). |
| community asks for an abliterated version | `swiftlab abliterate --protect-bundle bundle.json` | Extra stage, see below. |
| GGUF / W4A16 / NVFP4 releases | `swiftlab quant` + generated scripts | Calibrate on the final model's own traces; check KL vs BF16. |

`scripts/run_swift_recipe.sh` runs the table top to bottom.

## Order matters

1. Thinking work first (penalised SFT → OPD → optional transfer). This produces `model-swift`.
2. Abliteration second, on `model-swift`, with the overthinking direction as a protected atom.
3. Quantization last, calibrated on the final model's own rollouts.

Abliterating before the thinking work would make the mined markers and the settle data stale;
quantizing before either would put rounding noise exactly on the rows the edits touch.

## Abliteration, surgically

The refusal direction is the harmful-minus-harmless difference of means at the end of the
prompt (Arditi et al.), cleaned the same way as the overthinking direction:

```
r_clean = r_dirty − A ŵ,   A = [coding, knowledge, format atoms, overthink direction v_l]
```

Including `v_l` from `bundle.json` matters: in reasoning models "I should be careful here"
is both a refusal cue and a re-check cue, so an unprotected refusal edit shifts the
thinking behaviour you just tuned. The search co-minimises refusal rate and KL drift
(heretic's objective) and adds penalties when thinking tokens grow or accuracy drops. The
report shows `think_ratio` for the chosen edit; it should sit at 1.00 ± 0.05.

Prompt sets are yours to supply (`data/abliterate/README.md`). Refusal is measured with a
regex over the first 400 characters of the answer, thinking on, 512-token cap.

## Budget for a 27B on one 8×H100 node (order of magnitude)

| stage | GPU-hours |
|---|---|
| 800 rollouts at xhigh + settle | 25–40 |
| penalised LoRA (800 rows, 1 epoch) | 2–4 |
| OPD 300 steps (student sampling dominates) | 10–20 |
| abliteration capture + 20-trial search | 2–4 |
| GGUF ladder + W4A16 | 4–8 |
| 5-seed eval, 4 arms, 300 eval tasks | 30–60 |

## Knobs, in the order to turn them

1. `--beta` in the penalised SFT (0.2 → 1.0): higher = shorter, riskier.
2. `--steps` and `--wrong-weight` in OPD: more steps recover accuracy, too many re-lengthen.
3. `--alpha`/`--keep` in transfer: start small (0.3 / 0.2), it moves accuracy both ways.
4. `gamma` range in abliteration (0.5–1.5): stop at the first γ where refusal ≈ 0.
5. Quant type: pick the smallest with mean KLD under the thresholds in `docs/QUANT.md`.
