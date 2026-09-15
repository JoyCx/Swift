# Cut Swift's thinking further on a single RTX 5090 (32 GB), no training, no rented box.
#   rollout + settle + eval  -> served GGUF (fast, fits 32 GB)
#   direction + search + edit -> BF16 loaded 4-bit for capture; edit streams on CPU (no VRAM)
# Run blocks one at a time; the llama-server commands are shown as comments to run in a 2nd window.
# Prereqs: docs/SETUP_WINDOWS.md sections 0 and 1 done (venv, cu128 torch, swiftlab, llama.cpp).

$ErrorActionPreference = "Stop"
$RUN   = "runs\swift-surgery"
$BF16  = "A:\models\swift-unc-bf16"                 # d0xin BF16 (uncensored) OR ukisai/Swift-Qwen3.8-27b snapshot
$GGUF  = "A:\models\swift-unc-Q4_K_M.gguf"          # a quant of $BF16 to serve (see docs 4)
$LCPP  = "A:\llama.cpp"
$MEAS  = "configs\llama-server-27b-5090.yaml"
$SURG  = "configs\uncensored-27b-5090.yaml"
New-Item -ItemType Directory -Force -Path $RUN | Out-Null

# 0. one-time: quantize your BF16 to a GGUF to serve (skip if you already have $GGUF)
#   python $LCPP\convert_hf_to_gguf.py $BF16 --outtype bf16 --outfile A:\models\swift-unc-bf16.gguf
#   & $LCPP\llama-quantize.exe A:\models\swift-unc-bf16.gguf $GGUF Q4_K_M

# 1. SERVE (window A): & $LCPP\llama-server.exe -m $GGUF --alias swift-27b --host 127.0.0.1 --port 8080 -ngl 99 -c 16384 --jinja --reasoning-format deepseek
swiftlab preflight --config $MEAS
swiftlab bank    --config $MEAS --synthetic 150 --out $RUN\bank.jsonl

# 2. rollouts + settle through the served GGUF (this is the fast part)
swiftlab rollout --config $MEAS --bank $RUN\bank.jsonl --split mine --out $RUN\rollouts.jsonl
swiftlab settle  --config $MEAS --bank $RUN\bank.jsonl --rollouts $RUN\rollouts.jsonl --out $RUN\settled.jsonl
swiftlab mine    --settled $RUN\settled.jsonl --out $RUN\markers.json
swiftlab scale   --settled $RUN\settled.jsonl --out $RUN\scaling.json

# 3. directions on the BF16 loaded 4-bit (~16 GB). Stop llama-server first to free VRAM.
#    (if bitsandbytes won't build, comment load_in_4bit and set max_memory in $SURG for CPU offload)
swiftlab direction --config $SURG --bank $RUN\bank.jsonl --settled $RUN\settled.jsonl --out $RUN\bundle.json

# 4. search gamma/layers on a SMALL calib slice (4-bit regen is slow -> keep these low)
swiftlab search --config $SURG --bank $RUN\bank.jsonl --bundle $RUN\bundle.json --trials 12 --max-tasks 15 --out $RUN\search.json
python -c "import json;s=json.load(open(r'$RUN\search.json'));json.dump(s['best_edit'],open(r'$RUN\edit.json','w'))"

# 5. write the edited BF16 (CPU streaming, no VRAM; needs ~54 GB free disk for the copy)
swiftlab edit --config $SURG --bundle $RUN\bundle.json --edit-json $RUN\edit.json --model-dir $BF16 --out A:\models\swift-surgical-bf16

# 6. quantize the edited model to the SAME quant type you serve the base with (paired!)
python $LCPP\convert_hf_to_gguf.py A:\models\swift-surgical-bf16 --outtype bf16 --outfile A:\models\swift-surgical-bf16.gguf
& $LCPP\llama-quantize.exe A:\models\swift-surgical-bf16.gguf A:\models\swift-surgical-Q4_K_M.gguf Q4_K_M

# 7. PAIRED EVAL without needing both models in VRAM at once (resume trick):
#    7a. SERVE base (window A): ...llama-server -m $GGUF --alias base --port 8080 ...
swiftlab eval --config $MEAS --bank $RUN\bank.jsonl --settled $RUN\settled.jsonl --seeds 5 `
  --arm base:openai:base@http://127.0.0.1:8080/v1 --out $RUN\eval
#    7b. stop base; SERVE surgical (window A): ...llama-server -m A:\models\swift-surgical-Q4_K_M.gguf --alias surgical --port 8080 ...
#        base arm resumes from eval_base.jsonl (already complete -> zero calls); only surgical generates:
swiftlab eval --config $MEAS --bank $RUN\bank.jsonl --settled $RUN\settled.jsonl --seeds 5 `
  --arm base:openai:base@http://127.0.0.1:8080/v1 --arm surgical:openai:surgical@http://127.0.0.1:8080/v1 --out $RUN\eval

Write-Host "report: $RUN\eval\report.html"
