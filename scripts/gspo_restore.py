#!/usr/bin/env python
"""GSPO restore (Group Sequence Policy Optimization) — the RL accuracy-restore path the Swift
authors listed alongside on-policy distillation. Reward is VERIFIABLE (our task verifiers),
with an efficiency term keyed on the settle point so accuracy is recovered without
re-lengthening thinking.

  python scripts/gspo_restore.py --model runs/model-pen --bank runs/bank.jsonl \
      --settled runs/settled.jsonl --out runs/lora-gspo --group-size 8 --steps 400 --clip 0.2

Notes
* On-policy: each step samples G completions per prompt from the current policy, scores them,
  computes group-normalised advantages, and takes one clipped GSPO step (sequence-level ratio).
* LoRA policy; the "old" policy log-probs are taken from the sampling forward pass (single-step
  off-policyness), matching GSPO's practical setup.
* settle_tokens (per task, if --settled given) sets the brevity target; else the base rollout's
  own thinking length is the reference.
"""
import argparse, json, random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from swiftlab.tasks import Task, verify
from swiftlab.trace import split_thinking
from swiftlab.rl import efficiency_reward, group_advantages, reward_summary

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--bank", required=True); ap.add_argument("--settled", default=None)
ap.add_argument("--out", required=True); ap.add_argument("--group-size", type=int, default=8); ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--max-new", type=int, default=6144)
ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--brevity-coef", type=float, default=0.3)
ap.add_argument("--reasoning-effort", default="xhigh"); ap.add_argument("--think-open", default="<think>"); ap.add_argument("--think-close", default="</think>")
ap.add_argument("--prompts-per-step", type=int, default=2)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
policy = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
policy = get_peft_model(policy, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
policy.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

tasks = [Task.from_dict(json.loads(l)) for l in open(args.bank)]
settle_tok = {}
if args.settled:
    for l in open(args.settled):
        r = json.loads(l); s = r.get("settle", {})
        if s.get("settle_tokens"):
            settle_tok.setdefault(r["task_id"], []).append(s["settle_tokens"])
    settle_tok = {k: int(np.median(v)) for k, v in settle_tok.items()}
random.Random(0).shuffle(tasks)


def render(t: Task) -> str:
    msgs = ([{"role": "system", "content": t.context}] if t.context else []) + [{"role": "user", "content": t.prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort=args.reasoning_effort)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def seq_logprobs(input_ids, gen_len):
    """Sum/token log-probs of the generated tail under the current policy."""
    out = policy(input_ids=input_ids)
    logits = out.logits[:, -gen_len - 1:-1]
    logp = F.log_softmax(logits.float(), dim=-1)
    tgt = input_ids[:, -gen_len:]
    return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)          # [B, gen_len]


for step in range(args.steps):
    batch_tasks = [tasks[(step * args.prompts_per_step + j) % len(tasks)] for j in range(args.prompts_per_step)]
    seq_ratios, advs_all, losses = [], [], []
    step_rewards = []
    opt.zero_grad()
    for t in batch_tasks:
        prompt = render(t)
        enc = tok(prompt, return_tensors="pt").to(policy.device)
        n_prompt = enc["input_ids"].shape[1]
        policy.eval()
        with torch.no_grad():
            gen = policy.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=args.temperature,
                                  top_p=0.95, num_return_sequences=args.group_size, pad_token_id=tok.eos_token_id)
        policy.train()
        rewards, gen_lens, old_logps_list, seqs = [], [], [], []
        for g in range(gen.shape[0]):
            ids = gen[g]
            gen_ids = ids[n_prompt:]
            gen_len = int((gen_ids != tok.eos_token_id).sum().item()) or gen_ids.shape[0]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            thinking, answer = split_thinking(text, args.think_open, args.think_close)
            v = verify(t, answer)
            think_tokens = len(tok(thinking, add_special_tokens=False)["input_ids"])
            fmt_ok = bool(answer.strip())
            rewards.append(efficiency_reward(v.correct, think_tokens, settle_tok.get(t.id), brevity_coef=args.brevity_coef, format_ok=fmt_ok))
            gen_lens.append(gen_len); seqs.append(ids[: n_prompt + gen_len].unsqueeze(0))
        adv = group_advantages(np.array(rewards), args.group_size)
        step_rewards.extend(rewards)
        # old log-probs: recompute once under no_grad (single-step off-policy), then a grad pass for the surrogate
        for g in range(len(seqs)):
            if abs(adv[g]) < 1e-8:
                continue
            ids = seqs[g].to(policy.device); gl = gen_lens[g]
            with torch.no_grad():
                old_lp = seq_logprobs(ids, gl)
            new_lp = seq_logprobs(ids, gl)
            mask = torch.ones_like(new_lp)
            seq_log_ratio = ((new_lp - old_lp) * mask).sum() / mask.sum().clamp_min(1)   # sequence-level
            s = torch.exp(seq_log_ratio)
            a = torch.tensor(adv[g], device=s.device, dtype=s.dtype)
            loss = -torch.min(s * a, torch.clamp(s, 1 - args.clip, 1 + args.clip) * a) / (len(batch_tasks) * args.group_size)
            loss.backward()
            losses.append(float(loss))
    torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
    opt.step()
    if step % 5 == 0:
        summ = reward_summary(np.array(step_rewards), args.group_size)
        print(f"step {step} loss={np.sum(losses):.4f} mean_reward={summ['mean_reward']:.3f} solve_rate={summ['solve_rate']:.2f} dead_groups={summ['all_same_groups']}")
policy.save_pretrained(args.out); tok.save_pretrained(args.out)
print("saved GSPO LoRA to", args.out)
