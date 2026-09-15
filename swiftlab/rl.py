"""GSPO reward shaping and advantage math (pure numpy, torch-free so it is unit-tested).

GSPO (Group Sequence Policy Optimization, Qwen team, arXiv:2507.18071) differs from GRPO in
one place that matters for long reasoning traces: the importance ratio is defined at the
**sequence** level (the geometric mean of per-token ratios, i.e. exp of the mean log-ratio)
instead of per token. That removes the per-token variance that makes GRPO unstable on long
completions and MoE models — exactly the regime here. Advantages are group-normalised
rewards: sample G completions per prompt, subtract the group mean, divide by the group std.

Why GSPO helps *this* task (answering the user's push-back): the reward is a **verifiable**
signal from our own task verifiers (code executes, knowledge greps, math checks), so unlike
the penalised SFT it optimises accuracy directly rather than imitating truncated traces. And
because the reward includes an efficiency term keyed on the settle point, it can *keep*
short thinking while recovering the accuracy the SFT lost — the two objectives are in the
same reward, not fighting across two training stages. The cost is on-policy sampling every
step (like OPD but with a sparse reward), and a badly shaped reward can re-lengthen traces,
which is why brevity is a bonus for *correct* answers only, never a penalty that can be won
by giving up.
"""
from __future__ import annotations
import numpy as np


def efficiency_reward(correct: bool, think_tokens: int, settle_tokens: int | None, ref_think: int | None = None,
                      brevity_coef: float = 0.3, format_ok: bool = True, format_penalty: float = 0.2) -> float:
    """Verifiable reward for one completion.

    base:      +1 correct, 0 wrong  (never negative for being short — brevity must not beat correctness)
    brevity:   correct answers get up to +brevity_coef for using at most the settled/needed length;
               the bonus decays smoothly as thinking exceeds what the answer needed.
    format:    small penalty if the answer is malformed (no boxed / no final line), correct or not.
    """
    r = 1.0 if correct else 0.0
    if correct:
        target = settle_tokens or ref_think or think_tokens
        if target and think_tokens > 0:
            over = max(0, think_tokens - target) / max(1, target)      # 0 = at or under target
            r += brevity_coef * float(np.exp(-over))                    # 1.0 at target, decays as it overshoots
    if not format_ok:
        r -= format_penalty
    return r


def group_advantages(rewards: np.ndarray, group_size: int, eps: float = 1e-4) -> np.ndarray:
    """Group-normalised advantages. rewards laid out as [n_prompts * group_size] or [n_prompts, G]."""
    r = np.asarray(rewards, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, group_size)
    mean = r.mean(axis=1, keepdims=True)
    std = r.std(axis=1, keepdims=True)
    adv = (r - mean) / (std + eps)
    return adv.reshape(-1)


def sequence_log_ratio(new_logps: np.ndarray, old_logps: np.ndarray, mask: np.ndarray) -> float:
    """Sequence-level log importance ratio = mean over valid tokens of (new_logp - old_logp).

    exp() of this is GSPO's s_i(theta). Length-normalising by the token count is what makes it
    a per-token *geometric* mean, so long and short sequences are on the same scale.
    """
    m = np.asarray(mask, dtype=float)
    d = (np.asarray(new_logps) - np.asarray(old_logps)) * m
    n = m.sum()
    return float(d.sum() / n) if n > 0 else 0.0


def gspo_objective(seq_log_ratios: np.ndarray, advantages: np.ndarray, clip: float = 0.2) -> float:
    """Clipped surrogate at the sequence level (numpy reference; the torch loop mirrors this).

    L = -mean_i min( s_i * A_i,  clip(s_i, 1-eps, 1+eps) * A_i ),  s_i = exp(seq_log_ratio_i)
    """
    s = np.exp(np.asarray(seq_log_ratios, dtype=float))
    a = np.asarray(advantages, dtype=float)
    unclipped = s * a
    clipped = np.clip(s, 1 - clip, 1 + clip) * a
    return float(-np.minimum(unclipped, clipped).mean())


def reward_summary(rewards: np.ndarray, group_size: int) -> dict:
    r = np.asarray(rewards, dtype=float).reshape(-1, group_size)
    solved = (r >= 1.0).any(axis=1)
    return {"mean_reward": float(r.mean()), "groups": int(r.shape[0]), "group_size": group_size,
            "solve_rate": float(solved.mean()), "all_same_groups": int((r.std(axis=1) < 1e-6).sum()),
            "note": "groups where every sample got the same reward contribute zero advantage (no signal)"}
