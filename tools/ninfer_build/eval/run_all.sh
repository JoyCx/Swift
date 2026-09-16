#!/usr/bin/env bash
# Autonomous comparison eval, hard thermal cap (never sustain >=70C >10min).
# Compares: regular non-Swift quant (baseline) vs Swift RTN-abl vs Swift GPTQ-abl.
# Per model: precise CLI speed probe + reasoning-tokens + refusal + coding@1 + deepswe@1, outputs saved.
set -u
EVAL="C:/Users/ilaya/AppData/Local/Temp/claude/A--ninfer-spark/79ffab74-e953-4647-8f64-a06ac226f6f9/scratchpad/eval"
NINFER="/a/ninfer-spark"
PY314="python"
PY312="/a/swift/.venv/Scripts/python.exe"
SERVE="$NINFER/build/apps/ninfer-serve.exe"
CLI="$NINFER/build/apps/ninfer.exe"
TEMPCSV="$EVAL/temp_log.csv"; LOG="$EVAL/run_all.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== run_all start $(date) ==="

gpu_temp(){ nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -dc '0-9'; }
cool_gate(){ echo "[thermal] gate: ${1:-}$(gpu_temp)C, need <=58C"; while [ "$(gpu_temp)" -gt 58 ]; do sleep 10; done; echo "[thermal] open ($(gpu_temp)C)"; }
apath(){ case "$1" in /*) echo "$1";; *) echo "$NINFER/$1";; esac; }

( echo "epoch,tempC"; while true; do echo "$(date +%s),$(gpu_temp)"; sleep 10; done ) > "$TEMPCSV" &
TEMPPID=$!; trap 'kill $TEMPPID 2>/dev/null' EXIT

speed_probe(){  # artifact_abspath, label
  local art="$1" label="$2" f="$EVAL/speed_${2}.txt"
  echo "=== speed_probe $label ===" | tee "$f"
  cool_gate
  echo "-- no-spec --" >> "$f"
  "$CLI" "$art" --prompt "Write a short paragraph explaining what an inference engine does." \
    --greedy --max-new 160 --max-context 2048 --log-level warning 2>&1 | grep -E "prefill speed|decode speed|throughput|generated tokens" >> "$f" || echo "no-spec probe failed" >> "$f"
  cool_gate
  echo "-- dflash2 --" >> "$f"
  "$CLI" "$art" --prompt "Write a short paragraph explaining what an inference engine does." \
    --greedy --max-new 160 --max-context 2048 --spec dflash2 --draft-tokens 7 --log-level warning 2>&1 | grep -E "prefill speed|decode speed|acceptance rate|generated tokens" >> "$f" || echo "dflash2 probe failed" >> "$f"
  echo "[speed] $label:"; cat "$f"
}

serve_eval(){  # artifact(rel or abs), port, model_id, label
  local art; art=$(apath "$1"); local port="$2" mid="$3" label="$4"
  echo "=== $label ($art) ==="
  [ -f "$art" ] || { echo "MISSING $art -> skip $label"; return; }
  speed_probe "$art" "$label"
  cool_gate
  "$SERVE" "$art" --host 127.0.0.1 --port "$port" --model-id "$mid" --greedy \
    --max-concurrency 2 --default-max-tokens 3072 --max-context 4096 --log-level error &
  local spid=$!
  for i in $(seq 1 90); do curl -s -o /dev/null "http://127.0.0.1:$port/v1/models" && break; sleep 3; done
  echo "[serve] $label ready (pid $spid)"
  "$PY314" "$EVAL/eval_model.py" "$port" "$mid" "$label" || echo "eval_model FAILED for $label"
  kill $spid 2>/dev/null; sleep 5; echo "[serve] $label stopped"
}

# ---- Phase 1: per-artifact (regular non-swift baseline, swift rtn, swift gptq) ----
serve_eval "/a/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer" 8110 regular  regular
serve_eval "out/qwen3_8_27b_nvfp4_abliterated.ninfer"                    8111 rtn_abl  rtn_abl
serve_eval "out/qwen3_8_27b_nvfp4_abliterated_gptq.ninfer"               8112 gptq_abl gptq_abl

# ---- Phase 2: HF-level KLD vs BF16 (heavy) ----
HELD="$EVAL/../heldout.txt"
echo "=== KLD: BF16 || GPTQ ==="; cool_gate
"$PY312" "$EVAL/../kld_measure.py" "A:/models/Qwen3.8-27B/swift" "A:/models/swift-nvfp4-gptq" "$HELD" 512 \
  && cp "$EVAL/../kld_result.json" "$EVAL/kld_gptq.json" || echo "KLD gptq FAILED"
echo "=== KLD: BF16 || stock-unsloth-NVFP4 (RTN reference) ==="; cool_gate
"$PY312" "$EVAL/../kld_measure.py" "A:/models/Qwen3.8-27B/swift" "A:/models/Qwen3.8-27B/NVFP4_unsloth" "$HELD" 512 \
  && cp "$EVAL/../kld_result.json" "$EVAL/kld_rtn.json" || echo "KLD rtn FAILED"

# ---- Phase 3: thermal audit + summary ----
echo "=== thermal audit ==="
"$PY314" - "$TEMPCSV" <<'PYEOF'
import sys,csv
ts=[(int(r["epoch"]),int(r["tempC"])) for r in csv.DictReader(open(sys.argv[1])) if r.get("tempC","").isdigit()]
mx=max((t for _,t in ts),default=0); worst=0; start=None
for e,t in ts:
    if t>=70: start=start if start is not None else e; worst=max(worst,e-start)
    else: start=None
print(f"THERMAL max_temp={mx}C longest_>=70C_window={worst//60}m{worst%60}s (limit 10m) -> {'OK' if worst<600 else 'VIOLATION'}")
PYEOF
echo "=== SUMMARY ==="
"$PY314" - "$EVAL" <<'PYEOF'
import json,sys,glob,os
d=sys.argv[1]
print(f"{'label':10} {'mean_think':>10} {'refusal':>8} {'code@1':>7} {'deepswe@1':>10} {'tok/s':>7} {'reason_acc':>10}")
for f in sorted(glob.glob(os.path.join(d,"eval_*.json"))):
    r=json.load(open(f))
    print(f"{r['label']:10} {r['reasoning']['mean_think']:>10} {r['refusal']['refusal_rate']:>8} {r['coding']['pass_at_1']:>7} {r['deepswe']['pass_at_1']:>10} {r['speed']['mean_tok_s']:>7} {str(r['reasoning']['verifiable_acc']):>10}")
for f in ("kld_gptq.json","kld_rtn.json"):
    p=os.path.join(d,f)
    if os.path.exists(p):
        k=json.load(open(p)); print(f"{f}: mean_kl={k['mean_kl']:.4f} p99={k['p99_kl']:.4f} top1={k['top1_agreement']:.4f}")
PYEOF
echo "=== run_all done $(date) ==="
