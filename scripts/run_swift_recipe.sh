#!/usr/bin/env bash
# The Swift recipe, as the authors described it, starting from the PLAIN base (e.g. Qwen/Qwen3.8-27B),
# followed by abliteration as an extra, then quantization and the paired eval.
# Each numbered block maps to a sentence of their write-up; see docs/SWIFT_RECIPE.md.
set -euo pipefail
CFG=${CFG:-configs/qwen3.8-27b.yaml}
HFCFG=${HFCFG:-configs/hf-capture.yaml}
RUN=${RUN:-runs/swift-recipe}
BASE=${BASE:-/models/Qwen3.8-27B}                    # plain base to reproduce Swift; or Swift itself to push further
DONOR=${DONOR:-/models/ThinkingCap-Qwen3.6-27B}      # optional adapter-chunk donor
DONOR_BASE=${DONOR_BASE:-/models/Qwen3.6-27B}
HARM=${HARM:-data/abliterate/harmful.txt}
SAFE=${SAFE:-data/abliterate/harmless.txt}
mkdir -p "$RUN"

# ---- 1. "generated a large amount of different (out-of-distribution) domain traces" ---------------
#   serve BASE on :8000 first:  vllm serve $BASE --served-model-name base --reasoning-parser qwen3 --port 8000
swiftlab bank    --config "$CFG" --synthetic 400 --out "$RUN/bank.jsonl"           # + your own seeds, decontam list in config
swiftlab rollout --config "$CFG" --set backend.model=base --bank "$RUN/bank.jsonl" --split mine --samples 2 --out "$RUN/rollouts.jsonl"

# ---- 2. "grouped the ones with overthinking and found common-denominator tokens" -----------------
swiftlab settle  --config "$CFG" --set backend.model=base --bank "$RUN/bank.jsonl" --rollouts "$RUN/rollouts.jsonl" --out "$RUN/settled.jsonl"
swiftlab mine    --settled "$RUN/settled.jsonl" --out "$RUN/markers.json"
swiftlab scale   --settled "$RUN/settled.jsonl" --out "$RUN/scaling.json"      # tells you if step 1 needs more traces

# ---- 3. "built a loss function using the tokens we identified and ran LoRA SFT over the traces" --
swiftlab sftdata --config "$CFG" --set backend.model=base --bank "$RUN/bank.jsonl" --settled "$RUN/settled.jsonl" --out "$RUN/sft.jsonl"
python scripts/train_penalized_lora.py --model "$BASE" --sft "$RUN/sft.jsonl" --markers "$RUN/markers.json" \
       --out "$RUN/lora-pen" --beta 0.5 --epochs 1 --lr 1e-4 --rank 32
python scripts/merge_lora.py --base "$BASE" --lora "$RUN/lora-pen" --out "$RUN/model-pen"

# ---- 4. "restored accuracy: on-policy distillation (+ ThinkingCap adapter chunks)" ---------------
#   two restore options — run either or both (OPD then GSPO is a good order):
#   (a) on-policy distillation from the frozen base (cheap, dense token-level signal):
python scripts/opd_restore.py --student "$RUN/model-pen" --teacher "$BASE" --bank "$RUN/bank.jsonl" --out "$RUN/lora-opd" --steps 300
python scripts/merge_lora.py --base "$RUN/model-pen" --lora "$RUN/lora-opd" --out "$RUN/model-swift"
#   (b) GSPO RL with VERIFIABLE rewards (recovers accuracy AND holds thinking short via the
#       settle-keyed brevity term; more compute, directly optimises the eval metric):
python scripts/gspo_restore.py --model "$RUN/model-swift" --bank "$RUN/bank.jsonl" --settled "$RUN/settled.jsonl"        --out "$RUN/lora-gspo" --group-size 8 --steps 400 --clip 0.2 --brevity-coef 0.3
python scripts/merge_lora.py --base "$RUN/model-swift" --lora "$RUN/lora-gspo" --out "$RUN/model-swift-rl"
#   pick the model that wins the paired eval (step 7); default the rest of the pipeline to it:
ln -sfn "$(basename "$RUN/model-swift-rl")" "$RUN/model-swift" 2>/dev/null || true
# optional adapter-chunk transfer (same architecture family only; re-evaluate after):
# swiftlab transfer --target "$RUN/model-swift" --donor "$DONOR" --donor-base "$DONOR_BASE" --alpha 0.4 --keep 0.2 --only mlp --out "$RUN/model-swift-tv"

# ---- 5. extra: surgical abliteration, protected by the overthinking direction ---------------------
#   needs the weights in-process (HFCFG.backend.model = $RUN/model-swift) and the two prompt files
swiftlab direction  --config "$HFCFG" --set backend.model="$RUN/model-swift" --bank "$RUN/bank.jsonl" --settled "$RUN/settled.jsonl" --out "$RUN/bundle.json"
swiftlab abliterate --config "$HFCFG" --set backend.model="$RUN/model-swift" --bank "$RUN/bank.jsonl" --harmful "$HARM" --harmless "$SAFE" \
       --protect-bundle "$RUN/bundle.json" --trials 20 --out-bundle "$RUN/refusal_bundle.json" --out "$RUN/abliterate.json" \
       --model-dir "$RUN/model-swift" --apply-out "$RUN/model-swift-abliterated"

# ---- 6. quantize (calibrate on the final model's own traces) --------------------------------------
#   serve $RUN/model-swift-abliterated on :8001 as "final"
swiftlab rollout --config "$CFG" --set backend.base_url=http://localhost:8001/v1 --set backend.model=final --bank "$RUN/bank.jsonl" --split calib --arm final --out "$RUN/rollouts_final.jsonl"
swiftlab quant   --config "$CFG" --bank "$RUN/bank.jsonl" --rollouts "$RUN/rollouts_final.jsonl" --model-dir "$RUN/model-swift-abliterated" --out "$RUN/quant"
bash "$RUN/quant/quant_gguf.sh"

# ---- 7. paired eval: base vs penalised vs +OPD vs +abliterated vs quant, same tasks / seeds -------
swiftlab eval --config "$CFG" --bank "$RUN/bank.jsonl" --settled "$RUN/settled.jsonl" --seeds 5 \
  --arm base:openai:base@http://localhost:8000/v1 \
  --arm swift:openai:swift@http://localhost:8002/v1 \
  --arm final:openai:final@http://localhost:8001/v1 \
  --arm final_q4:openai:final-q4@http://localhost:8003/v1 \
  --out "$RUN/eval"
echo "report: $RUN/eval/report.html"
