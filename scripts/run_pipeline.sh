#!/usr/bin/env bash
# Full real-model pipeline. Requires: a vLLM server for rollouts (step 0) and a GPU box with the
# BF16 weights for direction capture / editing (steps 4-6). Edit CFG paths first.
set -euo pipefail
CFG=${CFG:-configs/qwen3.8-27b.yaml}
HFCFG=${HFCFG:-configs/hf-capture.yaml}
RUN=${RUN:-runs/swift27b}
MODEL_DIR=${MODEL_DIR:-/models/Swift-Qwen3.8-27b}
mkdir -p "$RUN"

# 0. serve the BF16 model (separate terminal):
#    vllm serve $MODEL_DIR --served-model-name ukisai/Swift-Qwen3.8-27b --max-model-len 262144 \
#         --reasoning-parser qwen3 --seed 0 --enable-prefix-caching

# 1. task bank (add your own jsonl seeds + eval dumps for decontamination in the config)
swiftlab bank --config "$CFG" --synthetic 200 --out "$RUN/bank.jsonl"

# 2. long-trace rollouts on the mining split, then settle probing
swiftlab rollout  --config "$CFG" --bank "$RUN/bank.jsonl" --split mine --out "$RUN/rollouts.jsonl"
swiftlab settle   --config "$CFG" --bank "$RUN/bank.jsonl" --rollouts "$RUN/rollouts.jsonl" --out "$RUN/settled.jsonl"

# 3. marker mining + dataset-size scaling curve (tells you whether to generate more traces)
swiftlab mine  --settled "$RUN/settled.jsonl" --out "$RUN/markers.json"
swiftlab scale --settled "$RUN/settled.jsonl" --out "$RUN/scaling.json"

# 4. directions (dirty + surgically cleaned) from residual activations
swiftlab direction --config "$HFCFG" --bank "$RUN/bank.jsonl" --settled "$RUN/settled.jsonl" --out "$RUN/bundle.json"

# 5. search gamma / layers on the calib split (co-minimise tokens + KL, accuracy within tolerance)
swiftlab search --config "$HFCFG" --bank "$RUN/bank.jsonl" --bundle "$RUN/bundle.json" --trials 30 --out "$RUN/search.json"
python - <<PY
import json; s=json.load(open("$RUN/search.json")); json.dump(s["best_edit"], open("$RUN/edit.json","w"))
PY

# 6. write the edited BF16 checkpoint
swiftlab edit --config "$HFCFG" --bundle "$RUN/bundle.json" --edit-json "$RUN/edit.json" --model-dir "$MODEL_DIR" --out "$MODEL_DIR-surgical"

# 7. (optional) task-vector transfer from a donor efficient fine-tune of the same architecture
# swiftlab transfer --target "$MODEL_DIR-surgical" --donor /models/ThinkingCap-Qwen3.6-27B --donor-base /models/Qwen3.6-27B \
#                   --alpha 0.4 --keep 0.2 --only 'mlp' --out "$MODEL_DIR-surgical-tv"

# 8. serve the edited model (second vLLM on :8001), roll out on calib split, build calibration + quant scripts
swiftlab rollout --config "$CFG" --set backend.base_url=http://localhost:8001/v1 --bank "$RUN/bank.jsonl" --split calib --arm edited --out "$RUN/rollouts_edited.jsonl"
swiftlab quant   --config "$CFG" --bank "$RUN/bank.jsonl" --rollouts "$RUN/rollouts_edited.jsonl" --model-dir "$MODEL_DIR-surgical" --out "$RUN/quant"
bash "$RUN/quant/quant_gguf.sh"          # GGUF + imatrix + KL vs bf16
# python "$RUN/quant/quant_w4a16.py"     # W4A16 via llmcompressor

# 9. paired evaluation: base vs edited vs quant, same tasks / seeds / context
swiftlab eval --config "$CFG" --bank "$RUN/bank.jsonl" --settled "$RUN/settled.jsonl" \
  --arm base:openai:ukisai/Swift-Qwen3.8-27b@http://localhost:8000/v1 \
  --arm surgical:openai:surgical@http://localhost:8001/v1 \
  --arm surgical_q4:openai:surgical-q4@http://localhost:8002/v1 \
  --out "$RUN/eval"
echo "report: $RUN/eval/report.html"
