#!/usr/bin/env python
"""Penalized LoRA SFT (the Swift recipe, step 1): cross-entropy on the model's own traces plus a
penalty on the probability mass assigned to mined overthinking tokens at think positions.

Data: settled.jsonl rows (use only correct traces, truncated at the settle point so the
target trace is the *needed* part).  Penalty vocab: markers.json from `swiftlab mine`.

  python scripts/train_penalized_lora.py --model /models/Qwen3.8-27B --settled runs/settled.jsonl \
      --markers runs/markers.json --out runs/lora-penalized --beta 0.5 --epochs 1
"""
import argparse, json, random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from swiftlab.penalty import penalized_loss, think_position_mask
from swiftlab.mining import penalty_vocab_to_logit_bias

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--settled", required=True); ap.add_argument("--markers", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--beta", type=float, default=0.5); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--bs", type=int, default=2); ap.add_argument("--max-len", type=int, default=8192)
ap.add_argument("--rank", type=int, default=32); ap.add_argument("--truncate-at-settle", action="store_true", default=True)
ap.add_argument("--think-open", default="<think>"); ap.add_argument("--think-close", default="</think>")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

markers = json.load(open(args.markers))["phrases"]
penalty_ids = sorted(penalty_vocab_to_logit_bias(markers, tok, strength=0).keys())
print(f"penalty token ids: {len(penalty_ids)}")
open_id = tok.convert_tokens_to_ids(args.think_open); close_id = tok.convert_tokens_to_ids(args.think_close)

rows = [json.loads(l) for l in open(args.settled)]
rows = [r for r in rows if r["correct"] and r.get("settle", {}).get("category") in ("tight", "overspent")]
random.Random(0).shuffle(rows)


def build(r):
    th = r["thinking"]
    if args.truncate_at_settle and r["settle"].get("settle_char"):
        th = th[: r["settle"]["settle_char"]]
    msgs = [{"role": "user", "content": r.get("prompt", "")}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + f"{args.think_open}\n{th}\n{args.think_close}\n\n{r['answer']}" + tok.eos_token
    ids = tok(full, add_special_tokens=False, truncation=True, max_length=args.max_len)["input_ids"]
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    labels = [-100] * n_prompt + ids[n_prompt:]
    return {"input_ids": ids, "labels": labels}


def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id or 0); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"]); ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


# rollouts store task_id; join prompt text back from the bank if needed
bank = {}
bank_path = Path(args.settled).parent / "bank.jsonl"
if bank_path.exists():
    bank = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(bank_path)}
for r in rows:
    r.setdefault("prompt", bank.get(r["task_id"], ""))
data = [build(r) for r in rows if r["prompt"]]
dl = DataLoader(data, batch_size=args.bs, shuffle=True, collate_fn=collate)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
model.train()
for ep in range(args.epochs):
    for step, batch in enumerate(dl):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = out.logits[:, :-1]; labels = batch["labels"][:, 1:]
        tmask = think_position_mask(batch["input_ids"], open_id, close_id)[:, 1:]
        loss, info = penalized_loss(logits, labels, tmask, penalty_ids, beta=args.beta)
        loss.backward(); opt.step(); opt.zero_grad()
        if step % 10 == 0:
            print(f"ep{ep} step{step} loss={loss.item():.4f} ce={info['ce'].item():.4f} penalty_mass={info['penalty_mass'].item():.4f}")
model.save_pretrained(args.out); tok.save_pretrained(args.out)
print("saved LoRA to", args.out, "- merge with: model.merge_and_unload()")
