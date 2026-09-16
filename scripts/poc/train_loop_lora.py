"""Loop-aware penalty LoRA for Qwen3.8-27B (proof of concept of the Swift-style recipe, generalised).

Loss per window (labels from label_loops.py; position t predicts token t+1):
  kl_w   * KL(p_base || p_lora)        on a subsample of NORMAL targets (anchor: keep behaviour elsewhere)
  ul_w   * -log(1 - p_lora(target))    on ENTRY targets (unlikelihood: stop starting redundant iterations)
  pool_w * sum_{v in pool} p_lora(v)   on thinking positions (corrected Swift marker pool)
WASTE targets get no term. The base distribution comes from the same 4-bit weights with the adapter disabled.

    python scripts/poc/train_loop_lora.py --data runs/poc/labels_maxdevv --out runs/poc/lora_v1 --dry-run 3
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE, NORMAL, WASTE, ENTRY = 0, 1, 2, 3
POST, SKIP = 4, 6  # internal kinds: anchor-only after </think>, and excluded


def load_windows(data_dir):
    meta = [json.loads(l) for l in open(Path(data_dir) / "windows.jsonl", encoding="utf-8")]
    shards = {}
    for f in sorted(glob.glob(str(Path(data_dir) / "shard-*.npz"))):
        z = np.load(f)
        shards[int(Path(f).stem.split("-")[1])] = (z["input_ids"], z["labels"], z["offsets"])
    items = []
    for m in meta:
        ids, lab, off = shards[m["shard"]]
        s, e = off[m["index"]], off[m["index"] + 1]
        items.append((ids[s:e], lab[s:e], m))
    return items


def crop(ids, lab, train_len):
    if len(ids) <= train_len:
        return ids, lab
    p = int(np.argmax(lab != IGNORE))  # prompt length
    keep = train_len - p
    return np.concatenate([ids[:p], ids[-keep:]]), np.concatenate([lab[:p], lab[-keep:]])


def pool_ids(path, drop_or=True):
    ids = {p["id"] for p in json.load(open(path, encoding="utf-8")) if not (drop_or and p["text"] == " or")}
    extra = [16018, 10380, 20734, 12525, 32645, 3850, 7834, 10560, 35542, 15896, 18200, 12908, 13634, 37488,
             10883, 33713, 14205, 6084, 32148, 88842, 13784, 27167, 76475, 50821, 4387, 59869, 32290]
    return sorted(ids | set(extra))


def chunk_terms(h_s, h_t, lm_head, tgt, kind, pool):
    """Per-chunk loss terms, recomputed in backward via checkpoint. Returns stacked sums."""
    logits_s = lm_head(h_s).float()
    logp_s = F.log_softmax(logits_s, dim=-1)
    with torch.no_grad():
        logp_t = F.log_softmax(lm_head(h_t).float(), dim=-1)
    zero = logp_s.new_zeros(())
    normal = (kind == NORMAL) | (kind == POST)
    entry = kind == ENTRY
    think = (kind >= NORMAL) & (kind <= ENTRY)
    kl = (logp_t[normal].exp() * (logp_t[normal] - logp_s[normal])).sum() if normal.any() else zero
    if entry.any():
        lp = logp_s[entry].gather(-1, tgt[entry][:, None]).squeeze(-1)
        ul = -torch.log1p(-lp.exp().clamp(max=1 - 1e-6)).sum()
        with torch.no_grad():
            lp_t = logp_t[entry].gather(-1, tgt[entry][:, None]).squeeze(-1)
            p_entry_s, p_entry_t = lp.exp().sum().detach(), lp_t.exp().sum()
    else:
        ul, p_entry_s, p_entry_t = zero, zero.detach(), zero.detach()
    if think.any():
        pm = torch.logsumexp(logp_s[think][:, pool], dim=-1).exp()
        pool_mass = pm.sum()
        with torch.no_grad():
            pool_mass_t = torch.logsumexp(logp_t[think][:, pool], dim=-1).exp().sum()
    else:
        pool_mass, pool_mass_t = zero, zero.detach()
    return torch.stack([kl, ul, pool_mass, p_entry_s, p_entry_t, pool_mass_t.detach()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="A:/models/Qwen3.8-27B-base-bf16")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="A:/swift/runs/markers/ukisai_pool_recovered.json")
    ap.add_argument("--train-len", type=int, default=8192)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--kl-w", type=float, default=1.0)
    ap.add_argument("--ul-w", type=float, default=1.0)
    ap.add_argument("--pool-w", type=float, default=0.2)
    ap.add_argument("--kl-positions", type=int, default=2048, help="NORMAL targets sampled per window for the anchor")
    ap.add_argument("--chunk", type=int, default=768)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--dry-run", type=int, default=0, help="run N optimizer micro-steps, report speed/memory, exit")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    torch.manual_seed(a.seed); random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = load_windows(a.data)
    random.shuffle(items)
    print(f"{len(items)} windows", flush=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
    )
    print(f"loaded in {time.time() - t0:.0f}s, mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    backbone = model.base_model.model.model
    lm_head = model.base_model.model.lm_head
    from transformers import AutoTokenizer
    think_end = AutoTokenizer.from_pretrained(a.model).convert_tokens_to_ids("</think>")
    print("</think> id", think_end, flush=True)
    pool = torch.tensor(pool_ids(a.pool), device="cuda")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    total_steps = max(1, math.ceil(len(items) * a.epochs / a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)

    step, micro, n_seen, tok_seen = 0, 0, 0, 0
    agg = torch.zeros(6, device="cuda"); cnt = torch.zeros(3, device="cuda")
    tstart = time.time()
    order = [items[i % len(items)] for i in range(int(len(items) * a.epochs))]
    for ids_np, lab_np, meta in order:
        ids_np, lab_np = crop(ids_np, lab_np, a.train_len)
        ids = torch.from_numpy(ids_np.astype(np.int64)).cuda()[None]
        lab = torch.from_numpy(lab_np.astype(np.int64)).cuda()
        T = ids.shape[1]
        # thinking region ends at </think>; after it only the anchor applies
        te = (ids[0] == think_end).nonzero()
        think_stop = int(te[0]) if len(te) else T
        tgt = ids[0, 1:]
        kind = lab[1:].clone()
        pos = torch.arange(T - 1, device="cuda")
        post = (pos + 1 >= think_stop) & (kind != IGNORE)
        kind[post] = POST
        anchor_idx = ((kind == NORMAL) | (kind == POST)).nonzero().squeeze(-1)
        if len(anchor_idx) > a.kl_positions:
            drop = anchor_idx[torch.randperm(len(anchor_idx), device="cuda")[a.kl_positions:]]
            kind[drop] = torch.where(kind[drop] == NORMAL, torch.full_like(kind[drop], WASTE), torch.full_like(kind[drop], SKIP))
        sel = ((kind != IGNORE) & (kind != SKIP)).nonzero().squeeze(-1)
        with torch.no_grad(), model.disable_adapter():
            h_t = backbone(input_ids=ids).last_hidden_state[0, :-1][sel]
        h_s = backbone(input_ids=ids).last_hidden_state[0, :-1][sel]
        k_sel, t_sel = kind[sel], tgt[sel]
        sums = torch.zeros(6, device="cuda")
        for c0 in range(0, len(sel), a.chunk):
            c1 = min(len(sel), c0 + a.chunk)
            sums = sums + checkpoint(chunk_terms, h_s[c0:c1], h_t[c0:c1], lm_head, t_sel[c0:c1], k_sel[c0:c1], pool,
                                     use_reentrant=False)
        n_norm = ((k_sel == NORMAL) | (k_sel == POST)).sum().clamp(min=1); n_ent = (k_sel == ENTRY).sum().clamp(min=1)
        n_think = ((k_sel >= NORMAL) & (k_sel <= ENTRY)).sum().clamp(min=1)
        loss = a.kl_w * sums[0] / n_norm + a.ul_w * sums[1] / n_ent + a.pool_w * sums[2] / n_think
        (loss / a.accum).backward()
        agg += sums.detach(); cnt += torch.stack([n_norm, n_ent, n_think]).float()
        micro += 1; n_seen += 1; tok_seen += T
        del h_s, h_t, sums, loss
        if micro % a.accum == 0 or (a.dry_run and micro >= a.dry_run):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            el = time.time() - tstart
            rec = dict(step=step, windows=n_seen, tokens=tok_seen, lr=sched.get_last_lr()[0],
                       kl=(agg[0] / cnt[0]).item(), ul=(agg[1] / cnt[1]).item(), pool_mass=(agg[2] / cnt[2]).item(),
                       p_entry_lora=(agg[3] / cnt[1]).item(), p_entry_base=(agg[4] / cnt[1]).item(),
                       pool_mass_base=(agg[5] / cnt[2]).item(), tok_per_s=tok_seen / el,
                       max_mem_gib=torch.cuda.max_memory_allocated() / 2**30, elapsed_s=el)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            agg.zero_(); cnt.zero_()
            if a.dry_run and micro >= a.dry_run:
                print("dry run done"); return
            if step % a.save_every == 0:
                model.save_pretrained(out / f"step-{step}")
    model.save_pretrained(out / "final")
    print("training done", flush=True)


if __name__ == "__main__":
    main()
