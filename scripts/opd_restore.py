#!/usr/bin/env python
"""On-policy distillation to restore accuracy after the penalized SFT / surgical edit (Swift step 2).

Student = edited/penalized model (LoRA trainable). Teacher = frozen base model.
Loop: sample a trace from the *student* on a calib task -> score every think+answer token with
teacher and student -> minimise per-token reverse KL(student || teacher) on those positions.
Because the trajectory is the student's own, the short reasoning style is kept while the
teacher's token-level judgement is re-absorbed.  Optionally weight tokens by correctness
(verified by the task) so wrong short traces are pushed harder toward the teacher.

  python scripts/opd_restore.py --student runs/merged-penalized --teacher /models/Qwen3.8-27B \
      --bank runs/bank.jsonl --out runs/lora-opd --steps 300
"""
import argparse, json, random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from swiftlab.penalty import reverse_kl_on_positions
from swiftlab.tasks import Task, verify

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True); ap.add_argument("--bank", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--max-new", type=int, default=6144); ap.add_argument("--wrong-weight", type=float, default=2.0); ap.add_argument("--topk", type=int, default=64)
ap.add_argument("--reasoning-effort", default="xhigh")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.student)
student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, device_map="auto")
student = get_peft_model(student, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16, device_map="auto").eval()
for p in teacher.parameters():
    p.requires_grad_(False)

tasks = [Task.from_dict(json.loads(l)) for l in open(args.bank)]
random.Random(0).shuffle(tasks)
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)

for step in range(args.steps):
    t = tasks[step % len(tasks)]
    msgs = ([{"role": "system", "content": t.context}] if t.context else []) + [{"role": "user", "content": t.prompt}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort=args.reasoning_effort)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(student.device)
    student.eval()
    with torch.no_grad():
        gen = student.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.eos_token_id)
    student.train()
    n_prompt = enc["input_ids"].shape[1]
    text = tok.decode(gen[0, n_prompt:], skip_special_tokens=True)
    answer = text.split("</think>")[-1]
    correct = verify(t, answer).correct
    w = 1.0 if correct else args.wrong_weight
    ids = gen[:, : n_prompt + args.max_new]
    out_s = student(input_ids=ids).logits[:, n_prompt - 1: -1]
    with torch.no_grad():
        out_t = teacher(input_ids=ids.to(teacher.device)).logits[:, n_prompt - 1: -1].to(out_s.device)
    mask = torch.ones(out_s.shape[:2], dtype=torch.bool, device=out_s.device)
    loss = w * reverse_kl_on_positions(out_s, out_t, mask, topk=args.topk)
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 10 == 0:
        print(f"step {step} task={t.id} correct={correct} gen_tokens={ids.shape[1] - n_prompt} rKL={loss.item() / w:.4f}")
student.save_pretrained(args.out); tok.save_pretrained(args.out)
print("saved", args.out)
