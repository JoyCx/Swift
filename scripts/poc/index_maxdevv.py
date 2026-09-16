"""Stream all MaxDevv Qwen3.8-27B distill shards; index every conversation by reasoning length and keep
full rows for long-thinking single-turn prompts (loop-prone candidates for the proof of concept)."""
import glob, io, json, re, sys, collections
import zstandard

SRC = sorted(glob.glob("A:/swift/data/external/maxdevv-qwen38-27b-distill-1m/data/*.jsonl.zst"))
OUT_IDX = open("A:/swift/runs/poc/maxdevv_index.jsonl", "w", encoding="utf-8")
OUT_LONG = open("A:/swift/runs/poc/maxdevv_long.jsonl", "w", encoding="utf-8")
stats = collections.Counter()
for path in SRC:
    with open(path, "rb") as fh:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8", errors="replace")
        for line in reader:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                stats["bad"] += 1; continue
            conv = r.get("conversations") or []
            users = [m for m in conv if m.get("role") == "user"]
            asst = [m for m in conv if m.get("role") == "assistant"]
            sysm = next((m.get("content", "") for m in conv if m.get("role") == "system"), "")
            reg = r.get("regeneration") or {}
            samp = reg.get("sampling") or {}
            effort = samp.get("reasoning_effort") or (r.get("regen_contract") or {}).get("reasoning_effort")
            if not effort:
                m = re.search(r"Reasoning effort is set to (\w+)", sysm); effort = m.group(1) if m else None
            rc = sum(len(m.get("reasoning_content") or "") for m in asst)
            finish = [m.get("finish_reason") for m in asst]
            rt = [u.get("reasoning_tokens") for u in (reg.get("usage") or []) if isinstance(u, dict)]
            rec = {"id": r.get("id"), "source": r.get("source"), "effort": effort, "turns": len(users),
                   "reasoning_chars": rc, "reasoning_tokens": sum(x for x in rt if x) if rt and all(rt) else None,
                   "finish": finish, "prompt_chars": sum(len(u.get("content") or "") for u in users)}
            OUT_IDX.write(json.dumps(rec) + "\n")
            stats["rows"] += 1
            if rc >= 40000 and len(users) == 1:
                OUT_LONG.write(line if line.endswith("\n") else line + "\n"); stats["long"] += 1
            if stats["rows"] % 100000 == 0:
                print(dict(stats), path.split("/")[-1], flush=True)
print("done", dict(stats), flush=True)
