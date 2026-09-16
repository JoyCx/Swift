"""Rebuild Qwen3.8-Flash-Next generated text from aswinkumar99/qwen3.8-flash-next-expert-traces compact token ids.

token_id[row] is the id the model consumed at that decode row = the previously sampled token, so a request's
rows ordered by req_pos give its generated tokens (the final sampled token of each request is not present)."""
import glob, json, collections
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT = Path("data/external/flashnext-expert-traces/compact")
tok = Tokenizer.from_file("A:/models/Qwen3.8-27B/tokenizer.json")
THINK_END = tok.token_to_id("</think>")
out = open("runs/flashnext_traces/responses.jsonl", "w", encoding="utf-8")
seen = set(); stats = collections.Counter()
for d in sorted(ROOT.iterdir()):
    if not d.is_dir() or d.name.endswith("_mtp"):
        continue  # *_mtp duplicates the same rows with extra MTP states
    L = {k: np.load(d / f"{k}.npy") for k in ("token_id", "req_index", "req_pos", "req_id_raw", "session_index")}
    names = json.load(open(d / "sessions.json"))["names_by_session_index"]
    order = np.lexsort((L["req_pos"], L["req_index"]))
    ri = L["req_index"][order]
    bounds = np.flatnonzero(np.diff(ri)) + 1
    for seg in np.split(order, bounds):
        ids = L["token_id"][seg].tolist()
        key = (d.name, int(L["req_id_raw"][seg[0]]))
        if key in seen: continue
        seen.add(key)
        si = int(L["session_index"][seg[0]]); sess = names[si] if 0 <= si < len(names) else None
        if THINK_END in ids:
            k = ids.index(THINK_END); think_ids, ans_ids = ids[:k], ids[k + 1:]
        else:
            think_ids, ans_ids = ids, []
        rec = {"batch": d.name, "req_id": key[1], "session": sess, "n_tokens": len(ids),
               "think_tokens": len(think_ids), "closed_think": THINK_END in ids,
               "reasoning": tok.decode(think_ids, skip_special_tokens=False), "content": tok.decode(ans_ids, skip_special_tokens=False)}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats["requests"] += 1; stats["tokens"] += len(ids); stats["think_tokens"] += len(think_ids)
        stats["unclosed"] += not rec["closed_think"]
    print(d.name, dict(stats), flush=True)
print("done", dict(stats))
