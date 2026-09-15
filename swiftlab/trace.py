"""Thinking-trace parsing: split, segment, mark, detect loops.

The default marker lexicon is a starting point only; `swiftlab mine` replaces it with
tokens/phrases that are *empirically* enriched in overspent or wrong thinking for the
model under study (that is the part Swift's authors described as "finding the common
denominator tokens").
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from collections import Counter

DEFAULT_MARKERS = [
    "wait", "but wait", "hmm", "hold on", "actually", "let me double-check", "let me double check", "double-check",
    "let me verify", "let me re-check", "let me recheck", "let me reconsider", "let me re-examine", "let me re-read",
    "on second thought", "alternatively", "i need to make sure", "to be safe", "just to be sure", "let me make sure",
    "let me think again", "is that right", "am i sure", "let me confirm", "sanity check", "one more time",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|\n{2,}|\n(?=\s*(?:Step|\d+[.)]|-|\*))")


def split_thinking(text: str, open_tag: str = "<think>", close_tag: str = "</think>") -> tuple[str, str]:
    """Return (thinking, answer). Handles missing open tag (template pre-emits it) and unclosed traces."""
    if close_tag in text:
        pre, post = text.split(close_tag, 1)
        if open_tag in pre:
            pre = pre.split(open_tag, 1)[1]
        return pre.strip(), post.strip()
    if open_tag in text:
        return text.split(open_tag, 1)[1].strip(), ""       # never closed: whole thing is thinking
    return "", text.strip()


@dataclass
class Segment:
    idx: int
    start: int
    end: int
    text: str
    markers: list[str] = field(default_factory=list)
    kind: str = "productive"       # productive | reverify | loop
    n_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def compile_markers(markers: list[str]) -> list[tuple[str, re.Pattern]]:
    out = []
    for m in sorted(set(markers), key=len, reverse=True):
        out.append((m, re.compile(r"(?<![\w-])" + re.escape(m) + r"(?![\w-])", re.I)))
    return out


def segment(thinking: str, markers: list[str] | None = None, token_counter=None) -> list[Segment]:
    pats = compile_markers(markers or DEFAULT_MARKERS)
    count = token_counter or (lambda s: max(1, len(s.split())))
    segs: list[Segment] = []
    pos = 0
    for i, piece in enumerate(_SENT_SPLIT.split(thinking)):
        if piece is None:
            continue
        start = thinking.find(piece, pos)
        if start < 0:
            start = pos
        end = start + len(piece)
        pos = end
        txt = piece.strip()
        if not txt:
            continue
        hits = [m for m, p in pats if p.search(txt)]
        segs.append(Segment(len(segs), start, end, txt, hits, "reverify" if hits else "productive", count(txt)))
    _mark_loops(segs)
    return segs


def _mark_loops(segs: list[Segment], min_repeats: int = 3) -> None:
    """A segment repeated (after normalisation) >= min_repeats times in the trace is a loop."""
    norm = [re.sub(r"\W+", " ", s.text.lower()).strip() for s in segs]
    c = Counter(n for n in norm if len(n) > 20)
    for s, n in zip(segs, norm):
        if c.get(n, 0) >= min_repeats:
            s.kind = "loop"


def repeated_ngram_ratio(thinking: str, n: int = 8) -> float:
    """Fraction of n-gram windows that are exact repeats: a cheap 'anxiety loop' score."""
    toks = thinking.lower().split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    c = Counter(grams)
    return sum(v - 1 for v in c.values() if v > 1) / len(grams)


def marker_stats(thinking: str, markers: list[str] | None = None) -> dict:
    segs = segment(thinking, markers)
    total = sum(s.n_tokens for s in segs) or 1
    reverify = sum(s.n_tokens for s in segs if s.kind == "reverify")
    loop = sum(s.n_tokens for s in segs if s.kind == "loop")
    mc = Counter(m for s in segs for m in s.markers)
    return {"n_segments": len(segs), "tokens": total, "reverify_tokens": reverify, "loop_tokens": loop,
            "reverify_share": reverify / total, "loop_share": loop / total, "marker_counts": dict(mc),
            "repeat_ngram_ratio": repeated_ngram_ratio(thinking)}
