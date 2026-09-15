"""Overthinking directions: extraction, layer scoring and surgical cleaning.

  r_dirty[l] = mean_l(WASTE spans) - mean_l(NEED spans)                 (difference of means)
  atoms a_k[l] = mean_l(D_k+) - mean_l(D_k-)   for protected concepts    (coding, knowledge, format, ...)
  w_hat = argmin ||r - A w||^2 + lambda ||w||^2 ;   r_clean = r - A w_hat  (ridge residualization, SRA eq. 4-5)
  v[l] = r_clean / ||r_clean||

WASTE spans are the loop / re-verification segments and everything after the settle point;
NEED spans are productive segments before the settle point.  Both come from the model's own
rollouts, so the direction is on-distribution.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path

import numpy as np

from .backends import Backend
from .trace import segment

# --------------------------------------------------------------------- built-in concept atoms
ATOM_CONTRASTS: dict[str, tuple[list[str], list[str]]] = {
    "coding": (
        ["def parse(tokens):\n    return [t for t in tokens if t]\n", "for i in range(len(xs)):\n    total += xs[i] * w[i]\n",
         "class Node:\n    def __init__(self, val, nxt=None):\n        self.val, self.nxt = val, nxt\n", "import re\npattern = re.compile(r'\\d+')\n",
         "SELECT id, name FROM users WHERE active = 1 ORDER BY name;", "while lo < hi:\n    mid = (lo + hi) // 2\n",
         "try:\n    fh = open(path)\nexcept FileNotFoundError:\n    return None\n", "return sorted(items, key=lambda kv: kv[1], reverse=True)"],
        ["The weather this weekend should be mild with some clouds in the afternoon.", "She walked along the river and thought about the summer.",
         "Please remember to buy milk, bread and a birthday card on the way home.", "The committee will meet again next Tuesday to finalise the schedule.",
         "Mountains in the north receive far more rainfall than the coastal plains.", "He apologised for being late and took a seat by the window.",
         "The recipe calls for two eggs, a cup of flour and a pinch of salt.", "Tickets for the concert sold out within a few hours of release."]),
    "knowledge": (
        ["The capital of Australia is Canberra, not Sydney.", "Water boils at 100 degrees Celsius at sea level.", "The Treaty of Westphalia was signed in 1648.",
         "Mitochondria produce most of the cell's ATP.", "Light travels at about 299,792 km per second in vacuum.", "The Nile flows north into the Mediterranean.",
         "Python's dict preserves insertion order since version 3.7.", "HTTP status 404 means the resource was not found."],
        ["I guess it could be either, honestly it does not matter much.", "Maybe, maybe not, hard to say without more thought.",
         "Some people like it and some people do not.", "Whatever you prefer is fine with me.", "It depends on how you look at it, I suppose.",
         "That is an interesting question with no clear answer.", "Let us not worry about the details right now.", "Opinions differ a lot on this topic."]),
    "format": (
        ["Final answer: 42", "\\boxed{17}", "Final answer: Canberra", "```python\nprint(1)\n```", "Answer: B", "The result is \\boxed{3/4}.",
         "Final answer: O(log n)", "Answer: yes"],
        ["so that is roughly what we would expect to see here", "and then it kind of continues in the same way for a while",
         "which brings us to the next consideration about the setup", "there are several aspects worth keeping in mind",
         "as discussed, the situation is somewhat nuanced", "moving on to other parts of the argument", "this part is fairly standard",
         "one could also mention the historical background"]),
    "language": (
        ["Therefore the total is 12, so we move on to the next part.", "First compute the sum, then divide by the count.",
         "Hence the function returns the merged list.", "Thus x equals 7 and the proof is complete.",
         "Consequently the loop terminates after n steps.", "So the answer is the second option.",
         "Next, substitute back into the original equation.", "Finally, verify the boundary case n = 0."],
        ["Wait, let me double-check that.", "Hmm, actually maybe I should reconsider.", "Hold on, is that right?",
         "Let me re-examine the previous step once more.", "Actually, let me verify this again to be safe.",
         "But wait, am I sure about this?", "Let me think again about whether that is correct.", "Just to be sure, one more time."]),
}


def collect_spans(rows: list[dict], markers=None, max_per_class: int = 400, seed: int = 0) -> tuple[list, list]:
    """Return (waste, need) lists of (text, (start, end)) pairs drawn from probed rollouts."""
    waste, need = [], []
    for r in rows:
        th = r["thinking"]; s = r.get("settle", {}); sc = s.get("settle_char")
        for seg in segment(th, markers):
            if seg.kind in ("loop", "reverify"):
                waste.append((th, (seg.start, seg.end)))
            elif sc is not None and seg.end > sc and s.get("category") in ("overspent", "derailed"):
                waste.append((th, (seg.start, seg.end)))
            elif seg.kind == "productive" and (sc is None or seg.end <= sc) and s.get("category") != "wrong":
                need.append((th, (seg.start, seg.end)))
    rng = random.Random(seed)
    rng.shuffle(waste); rng.shuffle(need)
    return waste[:max_per_class], need[:max_per_class]


def capture_means(backend: Backend, items: list[tuple[str, tuple[int, int]]], layers: list[int]) -> dict[int, np.ndarray]:
    texts = [t for t, _ in items]; spans = [sp for _, sp in items]
    acts = backend.capture(texts, spans, layers)
    return {l: acts[l] for l in layers}


def diff_of_means(pos: dict[int, np.ndarray], neg: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {l: pos[l].mean(0) - neg[l].mean(0) for l in pos}


def separation_auc(pos: np.ndarray, neg: np.ndarray, d: np.ndarray) -> float:
    """AUC of projecting held-out spans on d (0.5 = no signal)."""
    u = d / (np.linalg.norm(d) + 1e-9)
    p, n = pos @ u, neg @ u
    wins = (p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean()
    return float(wins)


def ridge_clean(r: np.ndarray, atoms: list[np.ndarray], lam: float) -> tuple[np.ndarray, np.ndarray]:
    """SRA spectral residualization: remove the part of r predictable from protected atoms."""
    if not atoms:
        return r.copy(), np.zeros(0)
    A = np.stack([a / (np.linalg.norm(a) + 1e-9) for a in atoms], axis=1)     # [h, K]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ r)
    return r - A @ w, w


def cosines(r: np.ndarray, atoms: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: float(r @ a / (np.linalg.norm(r) * np.linalg.norm(a) + 1e-9)) for k, a in atoms.items()}


def atom_items(name: str, custom: dict | None = None) -> tuple[list, list]:
    if custom and name in custom:
        pos, neg = custom[name]
    else:
        pos, neg = ATOM_CONTRASTS[name]
    return [(t, (0, len(t))) for t in pos], [(t, (0, len(t))) for t in neg]


def build_bundle(backend: Backend, rows: list[dict], layers: list[int] | None = None, atoms: list[str] | None = None,
                 ridge_lambda: float = 1e-2, markers=None, holdout: float = 0.3, custom_atoms: dict | None = None,
                 max_per_class: int = 400, seed: int = 0) -> dict:
    """Extract dirty + cleaned overthinking directions for every candidate layer, with diagnostics."""
    L = backend.n_layers()
    layers = layers or list(range(max(1, L // 4), max(2, (3 * L) // 4) + 1))
    waste, need = collect_spans(rows, markers, max_per_class, seed)
    if len(waste) < 8 or len(need) < 8:
        raise RuntimeError(f"not enough spans (waste={len(waste)}, need={len(need)}); run settle on more rollouts")
    kw = int(len(waste) * (1 - holdout)); kn = int(len(need) * (1 - holdout))
    W = capture_means(backend, waste, layers); N = capture_means(backend, need, layers)
    r_dirty = diff_of_means({l: W[l][:kw] for l in layers}, {l: N[l][:kn] for l in layers})
    atom_vecs: dict[str, dict[int, np.ndarray]] = {}
    for a in (atoms or []):
        pi, ni = atom_items(a, custom_atoms)
        atom_vecs[a] = diff_of_means(capture_means(backend, pi, layers), capture_means(backend, ni, layers))
    out = {"layers": {}, "n_waste": len(waste), "n_need": len(need), "ridge_lambda": ridge_lambda, "atoms": list(atom_vecs)}
    for l in layers:
        r = r_dirty[l]
        ats = {a: atom_vecs[a][l] for a in atom_vecs}
        r_clean, w = ridge_clean(r, list(ats.values()), ridge_lambda)
        auc_dirty = separation_auc(W[l][kw:], N[l][kn:], r)
        auc_clean = separation_auc(W[l][kw:], N[l][kn:], r_clean)
        out["layers"][l] = {"v": (r_clean / (np.linalg.norm(r_clean) + 1e-9)).tolist(), "norm_dirty": float(np.linalg.norm(r)),
                            "norm_clean": float(np.linalg.norm(r_clean)), "auc_dirty": auc_dirty, "auc_clean": auc_clean,
                            "cos_atoms_dirty": cosines(r, ats), "cos_atoms_clean": cosines(r_clean, ats),
                            "energy_removed": float(1 - (np.linalg.norm(r_clean) / (np.linalg.norm(r) + 1e-9)) ** 2),
                            "mean_act_norm": float(np.linalg.norm(N[l][:kn], axis=1).mean())}
    out["best_layers"] = sorted(layers, key=lambda l: -out["layers"][l]["auc_clean"])[: max(1, len(layers) // 3)]
    return out


def gamma_weights(layers: list[int], center: float, width: float, gamma: float, kernel: str = "gaussian") -> dict[int, float]:
    if kernel == "flat":
        return {l: gamma for l in layers}
    return {l: gamma * math.exp(-((l - center) ** 2) / (2 * width * width)) for l in layers}


def save_bundle(bundle: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)


def load_bundle(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        b = json.load(f)
    b["layers"] = {int(k): v for k, v in b["layers"].items()}
    return b
