"""Turn base Qwen3.8-27B reasoning traces into loop-labelled training windows.

For every trace the full chat sequence is rendered exactly as served (reasoning-effort system prompt,
<think>, reasoning, </think>, answer, <|im_end|>) and tokenised with offsets. The validated loop detector
(scripts/loops/detect.py) runs on the reasoning text; its loop iterations are mapped to token positions.

Per-token labels (uint8):
  0 IGNORE  prompt / system tokens (never trained)
  1 NORMAL  assistant tokens outside detected waste (anchored to the frozen base with KL)
  2 WASTE   tokens inside a 2nd+ prose loop iteration (no anchor: free to change)
  3 ENTRY   first ENTRY_TOKENS non-whitespace tokens of a prose loop iteration (unlikelihood target)
Code-redraft iterations (detector code fraction >= 0.5) are left NORMAL: in the loop study they go with
correct answers on coding tasks.

Output: <out>/shard-XXX.npz with flat input_ids (int32) and labels (uint8) plus window offsets, and
<out>/windows.jsonl with per-window metadata.

    python scripts/poc/label_loops.py --src runs/poc/maxdevv_long.jsonl --out runs/poc/labels_maxdevv --limit 3000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "loops"))
import detect  # noqa: E402

MODEL_DIR = "A:/models/Qwen3.8-27B-base-bf16"
IGNORE, NORMAL, WASTE, ENTRY = 0, 1, 2, 3
MIN_ITER_TOKENS = 20
MARKUP_LINE = re.compile(r"^\s*(</?[A-Za-z][\w:-]*[\s>/]|[{}\[\]]|\||[.#]?[\w-]+\s*\{|\"[\w-]+\"\s*:|```|[-*]\s*\(|\(?-?\d+\s*,\s*-?\d+\)?)")
_tok = None


def structured(text):
    """True for markup/code/data blocks the prose-code heuristic misses (HTML, CSS, JSON, tables, coordinate lists)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True
    if sum(c.isalpha() for c in chars) / len(chars) < 0.65:
        return True
    lines = [l for l in text.split("\n") if l.strip()]
    return bool(lines) and sum(1 for l in lines if MARKUP_LINE.match(l)) / len(lines) >= 0.3
_hf = None


def tokenizers():
    global _tok, _hf
    if _tok is None:
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
        _tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
        _hf = AutoTokenizer.from_pretrained(MODEL_DIR)
    return _tok, _hf


def render(row):
    """Return (full_text, reasoning_char_span, effort) or None."""
    _, hf = tokenizers()
    conv = row["conversations"]
    effort = None
    msgs = []
    for m in conv:
        if m["role"] == "system" and (m.get("content") or "").startswith("Reasoning effort is set to"):
            effort = m["content"].split("Reasoning effort is set to", 1)[1].split(".", 1)[0].strip()
            continue
        msgs.append({k: v for k, v in m.items() if k in ("role", "content", "reasoning_content")})
    samp = ((row.get("regeneration") or {}).get("sampling") or {})
    effort = samp.get("reasoning_effort") or (row.get("regen_contract") or {}).get("reasoning_effort") or effort or "xhigh"
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    if len(asst) != 1 or asst[0] != len(msgs) - 1:
        return None
    reasoning = msgs[-1].get("reasoning_content") or ""
    if len(reasoning) < 2000:
        return None
    gen = hf.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False, reasoning_effort=effort)
    full = hf.apply_chat_template(msgs, tokenize=False, reasoning_effort=effort)
    if not full.startswith(gen):
        return None
    r0 = full.find(reasoning.strip()[:200], len(gen))
    if r0 < 0:
        return None
    r1 = r0 + len(reasoning.strip())
    if full[r0:r1] != reasoning.strip():
        return None
    return full, len(gen), (r0, r1), effort


def label_row(args):
    row, max_len, entry_tokens, max_windows, pre_entry_ctx = args
    tok, _ = tokenizers()
    try:
        r = render(row)
    except Exception:
        return None
    if r is None:
        return None
    full, prompt_end, (r0, r1), effort = r
    enc = tok.encode(full, add_special_tokens=False)
    ids = np.asarray(enc.ids, dtype=np.int32)
    starts = np.asarray([o[0] for o in enc.offsets], dtype=np.int64)
    n = len(ids)
    labels = np.full(n, NORMAL, dtype=np.uint8)
    p_tok = int(np.searchsorted(starts, prompt_end))
    labels[:p_tok] = IGNORE
    reasoning = full[r0:r1]
    renc = tok.encode(reasoning, add_special_tokens=False)
    t = dict(text=reasoning, ids=np.asarray(renc.ids), starts=np.asarray([o[0] for o in renc.offsets]),
             bench="maxdevv", arm="base", tid=row.get("id"), group=row.get("source"), correct=None, truncated=None)
    try:
        res, iters, _, _, _, _ = detect.analyse(t)
    except Exception:
        return None
    roff = [o[0] for o in renc.offsets]
    entries = []
    waste_tok = 0
    for it in iters:
        if it["code"] >= 0.5 or it["tok"] < MIN_ITER_TOKENS:
            continue
        ca = r0 + roff[it["ta"]] if it["ta"] < len(roff) else r1
        cb = r0 + roff[it["tb"]] if it["tb"] < len(roff) else r1
        if structured(full[ca:cb]):
            continue
        a = int(np.searchsorted(starts, ca)); b = int(np.searchsorted(starts, cb))
        if b <= a:
            continue
        labels[a:b] = WASTE
        waste_tok += b - a
        k, placed = a, 0
        while k < b and placed < entry_tokens:
            if full[enc.offsets[k][0]:enc.offsets[k][1]].strip():
                labels[k] = ENTRY; placed += 1
            k += 1
        entries.append(a)
    if not entries:
        return None
    windows = []
    body = max_len - p_tok
    if body < 2048:
        return None
    if n <= max_len:
        windows.append((0, n))
    else:
        for e in sorted(entries):
            end = min(n, e + (max_len - p_tok) - pre_entry_ctx)
            start = max(p_tok, end - body)
            if windows and start < windows[-1][1]:
                continue
            windows.append((start, end))
            if len(windows) >= max_windows:
                break
    out = []
    for s, e in windows:
        seq = np.concatenate([ids[:p_tok], ids[s:e]]) if s > 0 else ids[s:e]
        lab = np.concatenate([labels[:p_tok], labels[s:e]]) if s > 0 else labels[s:e]
        if (lab == ENTRY).sum() == 0:
            continue
        out.append(dict(ids=seq, labels=lab, meta=dict(id=row.get("id"), source=row.get("source"), effort=effort,
                    trace_tokens=n, window=[int(s), int(e)], len=int(len(seq)),
                    n_entry=int((lab == ENTRY).sum()), n_waste=int((lab == WASTE).sum()),
                    n_normal=int((lab == NORMAL).sum()), trace_loop_share=round(res["loop_share"], 4),
                    trace_prose_loop_share=round(res["loop_share_prose"], 4))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=3000, help="number of source traces to label")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--entry-tokens", type=int, default=3)
    ap.add_argument("--max-windows", type=int, default=2)
    ap.add_argument("--pre-entry-ctx", type=int, default=10240,
                    help="tokens of context kept before an entry inside a window (rest goes after it)")
    ap.add_argument("--min-prose-loop-share", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=500)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    # reservoir-sample source rows so every source/effort is represented without loading 8 GB
    reservoir, seen = [], 0
    with open(a.src, encoding="utf-8") as f:
        for line in f:
            seen += 1
            if len(reservoir) < a.limit * 3:
                reservoir.append(line)
            else:
                j = rng.randrange(seen)
                if j < len(reservoir):
                    reservoir[j] = line
    rows = [json.loads(l) for l in reservoir]
    print(f"sampled {len(rows)} of {seen} source rows", flush=True)
    meta_f = open(out / "windows.jsonl", "w", encoding="utf-8")
    buf_ids, buf_lab, buf_meta, shard, kept_traces, total = [], [], [], 0, 0, 0

    def flush():
        nonlocal buf_ids, buf_lab, buf_meta, shard
        if not buf_ids:
            return
        lens = np.array([len(x) for x in buf_ids], dtype=np.int64)
        np.savez(out / f"shard-{shard:03d}.npz", input_ids=np.concatenate(buf_ids), labels=np.concatenate(buf_lab),
                 offsets=np.concatenate([[0], np.cumsum(lens)]))
        for m in buf_meta:
            m["shard"] = shard
            meta_f.write(json.dumps(m) + "\n")
        buf_ids, buf_lab, buf_meta = [], [], []
        shard += 1

    jobs = ((r, a.max_len, a.entry_tokens, a.max_windows, a.pre_entry_ctx) for r in rows)
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(label_row, jobs, chunksize=4):
            total += 1
            if not res:
                continue
            res = [w for w in res if w["meta"]["trace_prose_loop_share"] >= a.min_prose_loop_share]
            if not res:
                continue
            kept_traces += 1
            for w in res:
                w["meta"]["index"] = len(buf_ids)
                buf_ids.append(w["ids"]); buf_lab.append(w["labels"]); buf_meta.append(w["meta"])
                if len(buf_ids) >= a.shard_size:
                    flush()
            if kept_traces >= a.limit:
                break
            if total % 200 == 0:
                print(f"processed {total} kept traces {kept_traces}", flush=True)
    flush()
    meta_f.close()
    print(f"done: processed {total}, kept traces {kept_traces}, shards {shard}", flush=True)


if __name__ == "__main__":
    main()
