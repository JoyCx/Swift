"""Token-penalized objectives (the Swift recipe) and the on-policy-distillation restore step.

Two ways to use a mined penalty vocabulary:

1. Inference-time penalizer: `logit_bias` on the mined token ids (no training).  Cheap,
   reversible, works through vLLM/llama.cpp, but it is a blunt filter.
2. Penalized SFT loss (LoRA): standard next-token cross-entropy on self-generated traces
   PLUS beta * (probability mass the model puts on penalty tokens at think positions).
   The model learns to not *want* the loop, instead of being blocked from emitting it.

After the penalized SFT, accuracy is restored with on-policy distillation: sample from the
student, score the student's own think tokens with the frozen base as teacher, and
minimise the per-token reverse KL (student || teacher) on those positions.  Because the
student chooses the trajectory, it keeps its short traces while re-absorbing the base's
token-level judgement.  See scripts/train_penalized_lora.py and scripts/opd_restore.py.
"""
from __future__ import annotations


def penalized_loss(logits, labels, think_mask, penalty_ids, beta: float = 0.5, ce_weight: float = 1.0):
    """logits [B,T,V] (positions predicting labels[:, t]), labels [B,T] (-100 = ignore),
    think_mask [B,T] bool (positions inside the thinking block), penalty_ids: list[int]."""
    import torch
    import torch.nn.functional as F
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V).float(), labels.reshape(-1), ignore_index=-100)
    probs = torch.softmax(logits.float(), dim=-1)
    pen_mass = probs[..., penalty_ids].sum(-1)                        # [B,T]
    mask = think_mask & (labels != -100)
    pen = (pen_mass * mask).sum() / mask.sum().clamp_min(1)
    return ce_weight * ce + beta * pen, {"ce": ce.detach(), "penalty_mass": pen.detach()}


def reverse_kl_on_positions(student_logits, teacher_logits, mask, topk: int | None = None):
    """Per-token reverse KL  KL(student || teacher) averaged over masked positions (on-policy distillation)."""
    import torch
    import torch.nn.functional as F
    s = F.log_softmax(student_logits.float(), -1)
    t = F.log_softmax(teacher_logits.float(), -1)
    if topk:
        idx = s.topk(topk, dim=-1).indices
        s = s.gather(-1, idx); t = t.gather(-1, idx)
    kl = (s.exp() * (s - t)).sum(-1)
    return (kl * mask).sum() / mask.sum().clamp_min(1)


def think_position_mask(input_ids, open_id: int, close_id: int):
    """Boolean mask of positions strictly inside <think> ... </think> for a batch of token ids."""
    import torch
    B, T = input_ids.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
    for b in range(B):
        inside = False
        for t in range(T):
            tok = int(input_ids[b, t])
            if tok == open_id:
                inside = True; continue
            if tok == close_id:
                inside = False; continue
            mask[b, t] = inside
    return mask
