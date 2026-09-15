"""Marker mining: which tokens / phrases are enriched in overspent or derailing thinking?

Contrast corpus:   NEED = text before the settle point in every probed trace
                   WASTE = text after the settle point (overspent + derailed) plus loop segments
                            plus whole traces that were wrong throughout
Statistic:         log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008),
                   reported as a z-score.  Phrases with z >= threshold form the penalty vocabulary.
This is the data-driven replacement for a hand-written "Wait/Hmm" list, and the input to
both the inference-time penalizer and the penalized-SFT loss.
"""
from __future__ import annotations
import math, re
from collections import Counter

from .trace import segment

_TOK = re.compile(r"[a-zA-Z][a-zA-Z'\-]*|[0-9]+|[^\sa-zA-Z0-9]")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text)]


def ngrams(toks: list[str], nmax: int = 3):
    for n in range(1, nmax + 1):
        for i in range(len(toks) - n + 1):
            yield " ".join(toks[i:i + n])


def split_need_waste(rows: list[dict], markers=None) -> tuple[list[str], list[str]]:
    need, waste = [], []
    for r in rows:
        s = r.get("settle", {})
        th = r["thinking"]
        cat = s.get("category")
        if cat in ("overspent", "derailed") and s.get("settle_char"):
            need.append(th[: s["settle_char"]]); waste.append(th[s["settle_char"]:])
        elif cat == "wrong":
            waste.append(th)
        elif cat == "tight":
            need.append(th)
        for seg in segment(th, markers):
            if seg.kind == "loop":
                waste.append(seg.text)
    return need, waste


def fightin_words(need: list[str], waste: list[str], nmax: int = 3, prior_scale: float = 1.0, min_count: int = 5) -> list[dict]:
    cn, cw = Counter(), Counter()
    for t in need:
        cn.update(ngrams(tokens(t), nmax))
    for t in waste:
        cw.update(ngrams(tokens(t), nmax))
    vocab = {g for g, c in cw.items() if c >= min_count} | {g for g, c in cn.items() if c >= min_count}
    n_n, n_w = sum(cn.values()), sum(cw.values())
    prior = Counter(); prior.update(cn); prior.update(cw)
    a0 = prior_scale * len(vocab) / max(1, n_n + n_w)
    out = []
    for g in vocab:
        a = a0 * prior[g] / max(1, len(vocab)) + 0.01
        yw, yn = cw[g], cn[g]
        lw = math.log((yw + a) / (n_w + a0 - yw - a + 1e-9))
        ln = math.log((yn + a) / (n_n + a0 - yn - a + 1e-9))
        delta = lw - ln
        var = 1.0 / (yw + a) + 1.0 / (yn + a)
        out.append({"phrase": g, "z": delta / math.sqrt(var), "count_waste": yw, "count_need": yn,
                    "rate_waste": yw / max(1, n_w), "rate_need": yn / max(1, n_n)})
    out.sort(key=lambda d: -d["z"])
    return out


def mine_markers(rows: list[dict], z_threshold: float = 3.0, nmax: int = 3, top: int = 100, markers=None) -> dict:
    need, waste = split_need_waste(rows, markers)
    ranked = fightin_words(need, waste, nmax=nmax)
    def _clean(ph: str) -> bool:
        ws = ph.split()
        return bool(re.search(r"[a-z]", ph)) and re.match(r"[a-z0-9]", ws[0]) is not None and re.match(r"[a-z0-9]", ws[-1]) is not None
    sel = [d for d in ranked if d["z"] >= z_threshold and _clean(d["phrase"])][:top]
    return {"n_need_docs": len(need), "n_waste_docs": len(waste), "need_tokens": sum(len(tokens(t)) for t in need),
            "waste_tokens": sum(len(tokens(t)) for t in waste), "z_threshold": z_threshold, "markers": sel,
            "phrases": [d["phrase"] for d in sel]}


def penalty_vocab_to_logit_bias(phrases: list[str], tokenizer, strength: float = -3.0, max_ids: int = 64) -> dict[int, float]:
    """Map mined phrases to token ids for an inference-time penalizer (vLLM/llama.cpp `logit_bias`).

    Only single-token surface forms (with and without a leading space) are biased, so a
    multi-token phrase is discouraged through its first distinctive token.
    """
    bias: dict[int, float] = {}
    for p in phrases:
        head = p.split()[0]
        for form in (head, " " + head, head.capitalize(), " " + head.capitalize()):
            ids = tokenizer.encode(form, add_special_tokens=False) if hasattr(tokenizer, "encode") else tokenizer(form)
            if len(ids) == 1:
                bias[int(ids[0])] = strength
        if len(bias) >= max_ids:
            break
    return bias
