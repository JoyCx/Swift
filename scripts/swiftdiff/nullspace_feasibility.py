"""Feasibility of a null-space (AlphaEdit-style) loop edit, per module. CPU only.

Keys k are inputs to a linear map W. A rank-1 edit dW = r u^T changes the output by r * (u.k).
AlphaEdit restricts u to the null space of the protected keys: u = B z, where B spans eigenvectors of
C0 = E[k k^T] (random NORMAL thinking tokens) with eigenvalue < tau * max. Then normal tokens are barely moved.

For each module and tau:
  null_dim      dimension of the allowed subspace
  normal_leak   share of normal-key energy inside it (in-sample, sum of null eigenvalues / trace)
  diff_kept     ||P delta|| / ||delta||, delta = mean(loop keys) - mean(boundary keys), train split
  fit z by ridge so that (B z).k ~= 1 on train loop keys, then on held-out windows:
  resp_loop     mean response on test loop-entry keys (want ~1)
  resp_bound    mean response on test boundary keys (want ~0)
  resp_normal   RMS response on normal keys (from C0, want ~0)
  select        resp_loop / max(resp_normal, |resp_bound|)
  auc           separation of test loop vs test boundary by the response
Baseline row tau=none uses the full space (plain ridge edit, no protection).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(6)
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)

D = Path(sys.argv[1] if len(sys.argv) > 1 else "A:/swift/runs/swiftdiff/edit_keys")
TAUS = [None, 1e-2, 1e-3, 1e-4]
LAMBDAS = [1e-2, 1e-1, 1.0]


def auc(p, n):
    s = np.concatenate([p, n]); y = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
    r = np.empty(len(s)); r[np.argsort(s)] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def analyse(path):
    x = torch.load(path)
    pos, neg = x["pos"].double(), x["neg"].double()
    wp, wn = np.array(x["win_pos"]), np.array(x["win_neg"])
    wins = np.unique(np.concatenate([wp, wn]))
    cut = wins[int(len(wins) * 0.7)]
    ptr, pte = pos[torch.from_numpy(wp < cut)], pos[torch.from_numpy(wp >= cut)]
    ntr, nte = neg[torch.from_numpy(wn < cut)], neg[torch.from_numpy(wn >= cut)]
    C0 = x["C0"].double() / x["n0"]
    lam, U = torch.linalg.eigh(C0)
    lam = lam.clamp(min=0)
    trace = lam.sum()
    scale = (ptr.norm(dim=1) ** 2).mean()  # typical key energy, sets ridge scale
    delta = ptr.mean(0) - ntr.mean(0)
    rows = []
    for tau in TAUS:
        if tau is None:
            B = None
            null_dim, leak, kept = C0.shape[0], 1.0, 1.0
        else:
            mask = lam < tau * lam.max()
            B = U[:, mask]
            null_dim = int(mask.sum())
            if null_dim == 0:
                rows.append(dict(tau=tau, null_dim=0)); continue
            leak = float(lam[mask].sum() / trace)
            kept = float((B.T @ delta).norm() / delta.norm())
        Ztr = ptr if B is None else ptr @ B
        for lmb in LAMBDAS:
            G = Ztr @ Ztr.T + lmb * scale * torch.eye(len(Ztr), dtype=torch.float64)
            z = Ztr.T @ torch.linalg.solve(G, torch.ones(len(Ztr), dtype=torch.float64))
            u = z if B is None else B @ z
            rl, rb = (pte @ u).numpy(), (nte @ u).numpy()
            rn = float(torch.sqrt(u @ C0 @ u))
            rows.append(dict(tau="none" if tau is None else tau, lam=lmb, null_dim=null_dim, normal_leak=round(leak, 5),
                             diff_kept=round(kept, 3), resp_loop_train=round(float((ptr @ u).mean()), 3),
                             resp_loop=round(float(rl.mean()), 3), resp_bound=round(float(rb.mean()), 3),
                             resp_normal=round(rn, 3), select=round(float(rl.mean()) / max(rn, abs(float(rb.mean())), 1e-9), 2),
                             auc=round(auc(rl, rb), 3)))
    return dict(module=Path(path).stem, dim=int(C0.shape[0]), n_normal=int(x["n0"]), n_pos=[len(ptr), len(pte)],
                n_neg=[len(ntr), len(nte)], rows=rows)


def main():
    out = []
    for f in sorted(glob.glob(str(D / "*.pt"))):
        r = analyse(f)
        out.append(r)
        print(f"\n== {r['module']} dim={r['dim']} normal_keys={r['n_normal']} pos train/test={r['n_pos']} neg={r['n_neg']}", flush=True)
        print(f"{'tau':>6} {'lam':>5} {'nulldim':>7} {'leak':>8} {'dkept':>6} {'loop_tr':>7} {'loop':>6} {'bound':>6} {'normal':>6} {'select':>6} {'auc':>5}")
        for x in r["rows"]:
            if "lam" not in x:
                print(f"{x['tau']:>6} null space empty"); continue
            print(f"{str(x['tau']):>6} {x['lam']:>5} {x['null_dim']:>7} {x['normal_leak']:>8} {x['diff_kept']:>6} {x['resp_loop_train']:>7} "
                  f"{x['resp_loop']:>6} {x['resp_bound']:>6} {x['resp_normal']:>6} {x['select']:>6} {x['auc']:>5}")
    json.dump(out, open(D / "feasibility.json", "w"), indent=1)


if __name__ == "__main__":
    main()
