"""End-to-end pipeline on the mock model. Produces every artifact the real pipeline produces."""
from __future__ import annotations
import json
from pathlib import Path

from .tasks import TaskBank
from .backends.mock import MockBackend
from .rollout import run_rollouts
from .settle import probe_all, waste_summary
from .mining import mine_markers
from .scaling import scaling_curve
from .directions import build_bundle, save_bundle
from .edit import make_edit, apply_to_mock
from .search import run_search, kl_divergence
from .evaluate import run_arms, compare_all
from .report import write_report
from .util import write_jsonl, write_json, ensure_dir


def run_demo(out: str, n_tasks: int = 60, seeds: int = 3, trials: int = 20, quiet: bool = True) -> dict:
    out = ensure_dir(out)
    log = (lambda *a: None) if quiet else print
    bank, dropped = TaskBank.build([], synthetic_per_domain=n_tasks // 3, decontam_against=[], seed=0)
    bank.save(out / "bank.jsonl")
    splits = bank.split({"mine": 0.6, "calib": 0.2, "eval": 0.2})
    base = MockBackend(bank=bank)
    log(f"[bank] {len(bank.tasks)} tasks, splits={ {k: len(v) for k, v in splits.items()} }")

    # 1. rollouts on the mining split + settle probing
    rows = run_rollouts(base, splits["mine"], 2, out / "rollouts.jsonl", arm="base", resume=False)
    rows = probe_all(base, bank.by_id, rows, checkpoints=8, min_think_tokens=30)
    write_jsonl(out / "settled.jsonl", rows)
    waste = waste_summary(rows); log("[settle]", json.dumps(waste))

    # 2. mining + scaling
    markers = mine_markers(rows); write_json(out / "markers.json", markers)
    sizes = sorted({max(5, len(rows) // 8), len(rows) // 4, len(rows) // 2, len(rows)})
    scaling = scaling_curve(rows, sizes); write_json(out / "scaling.json", scaling)
    log("[mine]", markers["phrases"][:10]); log("[scale] recommended n =", scaling["recommended_n"])

    # 3. directions: dirty vs surgically cleaned
    bundle = build_bundle(base, rows, atoms=["coding", "knowledge", "format", "language"], ridge_lambda=1e-2, max_per_class=200)
    save_bundle(bundle, out / "bundle.json")
    dirty_bundle = build_bundle(base, rows, atoms=[], ridge_lambda=1e-2, max_per_class=200)
    save_bundle(dirty_bundle, out / "bundle_dirty.json")

    # 4. search gamma/layers on the calib split
    calib = splits["calib"]
    base_rows = run_rollouts(base, calib, 1, out / "search_base.jsonl", arm="base", resume=False)
    base_acc = sum(r["correct"] for r in base_rows) / len(base_rows); base_think = sum(r["think_tokens"] for r in base_rows)
    kl_prompts = [t.prompt for t in calib[:12]]; base_dist = base.next_token_dist(kl_prompts)

    def evaluate(edit):
        eb = apply_to_mock(base, edit)
        rs = run_rollouts(eb, calib, 1, out / "search_trial.jsonl", arm="trial", resume=False)
        return {"think_ratio": sum(r["think_tokens"] for r in rs) / base_think, "acc_delta": sum(r["correct"] for r in rs) / len(rs) - base_acc,
                "kl": kl_divergence(base_dist, eb.next_token_dist(kl_prompts))}

    search = run_search(bundle, evaluate, n_trials=trials, seed=0)
    write_json(out / "search.json", search)
    bp = search["best"]["params"]; log("[search]", json.dumps(bp), json.dumps(search["best"]["metrics"]))

    # 5. arms: base, dirty edit (same params, uncleaned direction), clean edit, "quant" of clean edit
    clean_edit = search["best_edit"]
    dirty_edit = make_edit(dirty_bundle, layers=search["layers"], gamma=bp["gamma"], kernel=bp["kernel"], center=bp["center"], width=bp["width"], matrices=bp["matrices"])
    arms = {"base": base, "edit_dirty": apply_to_mock(base, dirty_edit, "mock-dirty"), "edit_clean": apply_to_mock(base, clean_edit, "mock-clean")}
    q = arms["edit_clean"].clone(); q.state["acc_delta"] -= 0.02; q.state["drift"] += 0.01; q.state["variant"] = "mock-clean-q4"
    arms["edit_clean_q4"] = q
    eval_rows = run_arms(arms, splits["eval"], seeds, out / "eval")
    # settle-probe the base eval rows so coverage metrics are available
    settled_eval = probe_all(base, bank.by_id, eval_rows["base"], checkpoints=8, min_think_tokens=30)
    res = compare_all(eval_rows, base="base", settled_base=settled_eval, n_boot=500)
    run = {"name": "swiftlab mock demo", "model": "mock (planted overthinking direction)", "effort": "xhigh", "seeds": seeds, "n_tasks": len(splits["eval"]),
           "waste": waste, "scaling": scaling, "markers": markers, "bundle": {"layers": {str(k): v for k, v in bundle["layers"].items()}, "best_layers": bundle["best_layers"]},
           "search": {"method": search["method"], "best": search["best"], "trials": search["trials"]}, "eval": res,
           "quant": {"note": "mock arm 'edit_clean_q4' simulates a 4-bit quant of the clean edit (small drift); real runs use `swiftlab quant`"}}
    for k in run["bundle"]["layers"].values():
        k.pop("v", None)
    paths = write_report(run, out)
    for name, pr in res["pairs"].items():
        log(f"[eval] {name}: acc Δ={pr['accuracy_delta']['delta']:+.3f} [{pr['accuracy_delta']['lo']:+.3f},{pr['accuracy_delta']['hi']:+.3f}]  think ↓ {100 * pr['think_reduction_mean']:.1f}%  fixed/broken={pr['fixed']}/{pr['broken']}  coverage={pr.get('coverage')}")
    log("[report]", paths)
    return {"run": run, "paths": paths}
