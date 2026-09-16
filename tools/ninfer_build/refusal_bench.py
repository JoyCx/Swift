"""Streaming, resumable refusal bench against a served ninfer model.

Writes one JSONL line per prompt as it completes (refusal boolean + metadata only —
never the completion text), so a kill keeps finished results and a re-run resumes.
Prints ONLY aggregate progress/metrics.
"""
from __future__ import annotations
import json, sys, threading, types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "A:/swift")
from swiftlab.backends.openai_compat import OpenAICompatBackend
from swiftlab.backends.base import GenRequest
from swiftlab.abliterate import is_refusal, load_prompts

port = int(sys.argv[1]); model_id = sys.argv[2]; label = sys.argv[3]
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 1536
CONC = int(sys.argv[5]) if len(sys.argv) > 5 else 6
out_path = Path(f"C:/Users/ilaya/AppData/Local/Temp/claude/A--ninfer-spark/79ffab74-e953-4647-8f64-a06ac226f6f9/scratchpad/refusal_{label}.jsonl")

harmful = load_prompts("A:/swift/data/abliterate/harmful.txt")

done: dict[str, bool] = {}
if out_path.exists():
    for ln in out_path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln); done[r["task_id"]] = r["refused"]
print(f"{label}: {len(harmful)} prompts, {len(done)} already done, resuming rest", flush=True)

cfg = types.SimpleNamespace(
    base_url=f"http://127.0.0.1:{port}/v1", model=model_id, api_key="x",
    reasoning_effort="medium", temperature=0.0, top_p=1.0, max_tokens=MAXTOK,
    think_open="<think>", think_close="</think>", concurrency=CONC, extra={"timeout": 1200},
)
be = OpenAICompatBackend(cfg)
lock = threading.Lock()
fh = out_path.open("a", encoding="utf-8")
counter = {"n": len(done)}


def work(i: int, p: str):
    tid = f"h{i}"
    if tid in done:
        return
    g = be.generate(GenRequest(tid, p, seed=1, max_tokens=MAXTOK, reasoning_effort="medium", meta={"harmful": True}))
    ref = is_refusal(g.answer)
    rec = {"task_id": tid, "refused": bool(ref), "answer_len": len(g.answer),
           "think_tokens": g.think_tokens, "finish": g.finish_reason}
    with lock:
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        done[tid] = ref; counter["n"] += 1
        if counter["n"] % 10 == 0 or counter["n"] == len(harmful):
            refused = sum(done.values())
            print(f"  progress {counter['n']}/{len(harmful)}  refused so far {refused}", flush=True)


todo = [(i, p) for i, p in enumerate(harmful) if f"h{i}" not in done]
with ThreadPoolExecutor(max_workers=CONC) as ex:
    futs = [ex.submit(work, i, p) for i, p in todo]
    for fu in as_completed(futs):
        fu.result()
fh.close()

n = len(done)
refused = sum(done.values())
print(f"RESULT {label}: refusal_rate={refused/max(1,n)*100:.1f}%  ({refused}/{n} refused, {n-refused} complied)", flush=True)
