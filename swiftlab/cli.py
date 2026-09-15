"""swiftlab command line.  Stages can be run one by one, or `demo` runs everything on the mock."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from .config import load_config
from .util import read_jsonl, write_jsonl, write_json, read_json, ensure_dir


def _cfg(args):
    ov = {}
    for kv in (args.set or []):
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        ov[k] = v
    return load_config(args.config, ov)


def _backend(cfg, bank=None):
    from .backends import make_backend
    return make_backend(cfg.backend, bank=bank)


def _tasks(args, cfg, split: str | None = None):
    from .tasks import TaskBank
    bank = TaskBank.load(args.bank)
    tasks = bank.split(cfg.bank.split)[split] if split else bank.tasks
    return bank, tasks


# --------------------------------------------------------------------------- stages
def cmd_bank(args):
    from .tasks import TaskBank
    cfg = _cfg(args)
    bank, dropped = TaskBank.build(cfg.bank.seeds, synthetic_per_domain=args.synthetic, domains=cfg.bank.domains,
                                   decontam_against=cfg.bank.decontam_against, ngram=cfg.bank.ngram, seed=args.seed)
    n = bank.save(args.out)
    splits = {k: len(v) for k, v in bank.split(cfg.bank.split).items()}
    print(json.dumps({"tasks": n, "dropped": len(dropped), "splits": splits, "out": args.out}))


def cmd_rollout(args):
    from .rollout import run_rollouts
    cfg = _cfg(args)
    bank, tasks = _tasks(args, cfg, args.split)
    if args.max_tasks:
        tasks = bank.__class__(tasks).sample(args.max_tasks, seed=0)
    be = _backend(cfg, bank)
    rows = run_rollouts(be, tasks, args.samples or cfg.rollout.samples_per_task, args.out, concurrency=cfg.backend.concurrency,
                        effort=cfg.backend.reasoning_effort, arm=args.arm)
    print(json.dumps({"rows": len(rows), "accuracy": sum(r["correct"] for r in rows) / max(1, len(rows)), "out": args.out}))


def cmd_settle(args):
    from .settle import probe_all, waste_summary
    cfg = _cfg(args)
    bank, _ = _tasks(args, cfg)
    be = _backend(cfg, bank)
    rows = read_jsonl(args.rollouts)
    rows = probe_all(be, bank.by_id, rows, checkpoints=cfg.settle.checkpoints, answer_max_tokens=cfg.settle.answer_max_tokens,
                     min_think_tokens=cfg.settle.min_think_tokens, mode=args.mode)
    write_jsonl(args.out, rows)
    print(json.dumps(waste_summary(rows), indent=1))


def cmd_mine(args):
    from .mining import mine_markers
    rows = read_jsonl(args.settled)
    m = mine_markers(rows, z_threshold=args.z, top=args.top)
    write_json(args.out, m)
    print(json.dumps({"n_markers": len(m["phrases"]), "top": m["phrases"][:20], "out": args.out}))


def cmd_scale(args):
    from .scaling import scaling_curve
    rows = read_jsonl(args.settled)
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else None
    sc = scaling_curve(rows, sizes, z_threshold=args.z)
    write_json(args.out, sc)
    for p in sc["points"]:
        print(f"n={p['n']:>5}  removable={p['removable_share']:.3f}  markers={p['markers_discovered']:>4}  loop-sigs={p['loop_signatures']:>4}/{p['loop_signatures_chao1']:.0f}  coverage={p['signature_coverage']:.2f}")
    print("recommended_n:", sc["recommended_n"])


def cmd_direction(args):
    from .directions import build_bundle, save_bundle
    cfg = _cfg(args)
    bank, _ = _tasks(args, cfg)
    be = _backend(cfg, bank)
    rows = read_jsonl(args.settled)
    atoms = [] if args.no_atoms else cfg.direction.atoms
    b = build_bundle(be, rows, layers=cfg.direction.layers or None, atoms=atoms, ridge_lambda=cfg.direction.ridge_lambda)
    save_bundle(b, args.out)
    for l, d in sorted(b["layers"].items()):
        print(f"layer {l:>3}  auc dirty={d['auc_dirty']:.3f} clean={d['auc_clean']:.3f}  energy removed={d['energy_removed']:.2f}  cos clean={d['cos_atoms_clean']}")
    print("best layers:", b["best_layers"])


def cmd_edit(args):
    from .directions import load_bundle
    from .edit import make_edit, apply_to_safetensors
    cfg = _cfg(args)
    bundle = load_bundle(args.bundle)
    if args.edit_json:
        edit = read_json(args.edit_json); edit["layers"] = {int(k): v for k, v in edit["layers"].items()}
    else:
        edit = make_edit(bundle, layers=cfg.direction.layers or None, gamma=cfg.direction.gamma, kernel=cfg.direction.gamma_kernel,
                         width=cfg.direction.kernel_width, matrices=cfg.direction.matrices)
    if cfg.backend.kind != "hf" and not args.model_dir:
        write_json(args.out, edit); print(json.dumps({"edit_spec": args.out, "layers": sorted(edit["layers"])})); return
    res = apply_to_safetensors(args.model_dir or cfg.backend.model, args.out, edit)
    print(json.dumps({"touched": len(res["touched"]), "out": res["out_dir"]}))


def cmd_search(args):
    from .directions import load_bundle
    from .search import run_search, kl_divergence
    from .rollout import run_rollouts
    from .edit import apply_in_place, restore_in_place, apply_to_mock
    cfg = _cfg(args)
    bank, tasks = _tasks(args, cfg, "calib")
    tasks = bank.__class__(tasks).sample(args.max_tasks or 40, seed=1)
    be = _backend(cfg, bank)
    bundle = load_bundle(args.bundle)
    tmp = ensure_dir(Path(args.out).parent / "search_tmp")
    base_rows = run_rollouts(be, tasks, 1, tmp / "base.jsonl", concurrency=cfg.backend.concurrency, arm="base")
    kl_prompts = [t.prompt for t in tasks[:16]]
    base_dist = be.next_token_dist(kl_prompts)
    base_acc = sum(r["correct"] for r in base_rows) / len(base_rows); base_think = sum(r["think_tokens"] for r in base_rows)
    n_trial = [0]

    def evaluate(edit):
        n_trial[0] += 1
        if cfg.backend.kind == "mock":
            eb = apply_to_mock(be, edit)
            rows = run_rollouts(eb, tasks, 1, tmp / f"t{n_trial[0]}.jsonl", arm="trial", resume=False)
            kl = kl_divergence(base_dist, eb.next_token_dist(kl_prompts))
        else:
            saved = apply_in_place(be.model, edit)
            try:
                rows = run_rollouts(be, tasks, 1, tmp / f"t{n_trial[0]}.jsonl", concurrency=1, arm="trial", resume=False)
                kl = kl_divergence(base_dist, be.next_token_dist(kl_prompts))
            finally:
                restore_in_place(be.model, saved)
        return {"think_ratio": sum(r["think_tokens"] for r in rows) / max(1, base_think), "acc_delta": sum(r["correct"] for r in rows) / len(rows) - base_acc, "kl": kl}

    res = run_search(bundle, evaluate, n_trials=args.trials, w_kl=args.w_kl, acc_tol=cfg.eval.accuracy_tolerance)
    write_json(args.out, res)
    b = res["best"]
    print(json.dumps({"method": res["method"], "best_params": b["params"], "best_metrics": b["metrics"], "out": args.out}))


def cmd_transfer(args):
    from .transfer import transfer
    res = transfer(args.target, args.donor, args.donor_base, args.out, alpha=args.alpha, keep=args.keep, only=args.only)
    print(json.dumps({"applied": len(res["applied"]), "skipped": len(res["skipped"]), "out": args.out}))


def cmd_quant(args):
    from .quant import build_calibration, write_calibration, gguf_commands, w4a16_script, size_table
    cfg = _cfg(args)
    bank, _ = _tasks(args, cfg)
    rows = read_jsonl(args.rollouts)
    samples = build_calibration(rows, bank.by_id, n=cfg.quant.calib_samples, max_chars=cfg.quant.calib_max_tokens * 4)
    out = ensure_dir(args.out)
    cal = write_calibration(samples, out)
    heldout = str(out / "heldout.txt")
    ev = bank.split(cfg.bank.split)["eval"]
    Path(heldout).write_text("\n\n".join(t.prompt for t in ev[:200]) + "\n", encoding="utf-8")
    cmds = gguf_commands(args.model_dir, str(out / "gguf"), cal["txt"], heldout, cfg.quant.gguf_types, cfg.quant.llamacpp_dir)
    (out / "quant_gguf.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(cmds) + "\n")
    (out / "quant_w4a16.py").write_text(w4a16_script(args.model_dir, str(out / "w4a16"), cal["jsonl"], cfg.quant.w4a16_scheme, cfg.quant.calib_samples))
    write_json(out / "sizes.json", size_table(args.params_b, cfg.quant.gguf_types + ["W4A16", "BF16"]))
    print(json.dumps({"calibration": cal, "scripts": [str(out / "quant_gguf.sh"), str(out / "quant_w4a16.py")]}))


def cmd_eval(args):
    from .evaluate import run_arms, compare_all
    from .backends import make_backend
    from .report import write_report
    cfg = _cfg(args)
    bank, tasks = _tasks(args, cfg, "eval")
    if args.max_tasks:
        tasks = bank.__class__(tasks).sample(args.max_tasks, seed=0)
    arms = {}
    for spec in args.arm:
        name, kind, model = spec.split(":", 2)
        url = None
        if kind == "openai" and "@" in model:
            model, url = model.split("@", 1)
        bc = load_config(args.config).backend
        bc.kind, bc.model = kind, model
        if url:
            bc.base_url = url
        arms[name] = make_backend(bc, bank=bank)
    rows = run_arms(arms, tasks, args.seeds or cfg.eval.seeds, args.out, effort=cfg.backend.reasoning_effort, concurrency=cfg.backend.concurrency)
    settled = read_jsonl(args.settled) if args.settled else None
    res = compare_all(rows, base=args.base, settled_base=settled, n_boot=cfg.eval.bootstrap)
    run = {"name": cfg.name, "model": cfg.backend.model, "effort": cfg.backend.reasoning_effort, "seeds": args.seeds or cfg.eval.seeds, "n_tasks": len(tasks), "eval": res}
    paths = write_report(run, args.out)
    print(json.dumps({"arms": {k: v["accuracy"] for k, v in res["arms"].items()}, "report": paths}))


def cmd_report(args):
    from .report import write_report
    run = read_json(args.run_json)
    print(json.dumps(write_report(run, args.out)))


def cmd_demo(args):
    from .demo import run_demo
    run_demo(args.out, n_tasks=args.n_tasks, seeds=args.seeds, trials=args.trials, quiet=False)


# --------------------------------------------------------------------------- parser
def main(argv=None):
    p = argparse.ArgumentParser(prog="swiftlab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, bank=True):
        sp.add_argument("--config", default=None)
        sp.add_argument("--set", action="append", help="override config key, e.g. backend.model=foo")
        if bank:
            sp.add_argument("--bank", default="runs/bank.jsonl")

    s = sub.add_parser("bank", help="build decontaminated task bank"); common(s, bank=False)
    s.add_argument("--synthetic", type=int, default=0); s.add_argument("--seed", type=int, default=0); s.add_argument("--out", default="runs/bank.jsonl"); s.set_defaults(fn=cmd_bank)
    s = sub.add_parser("rollout", help="generate traces"); common(s)
    s.add_argument("--split", default="mine"); s.add_argument("--samples", type=int); s.add_argument("--max-tasks", type=int, default=0)
    s.add_argument("--arm", default="base"); s.add_argument("--out", default="runs/rollouts.jsonl"); s.set_defaults(fn=cmd_rollout)
    s = sub.add_parser("settle", help="probe where answers settle; label overspent/derailed/wrong"); common(s)
    s.add_argument("--rollouts", required=True); s.add_argument("--mode", default="bisect", choices=["bisect", "scan"]); s.add_argument("--out", default="runs/settled.jsonl"); s.set_defaults(fn=cmd_settle)
    s = sub.add_parser("mine", help="mine overthinking markers"); common(s, bank=False)
    s.add_argument("--settled", required=True); s.add_argument("--z", type=float, default=3.0); s.add_argument("--top", type=int, default=100); s.add_argument("--out", default="runs/markers.json"); s.set_defaults(fn=cmd_mine)
    s = sub.add_parser("scale", help="dataset-size scaling curve"); common(s, bank=False)
    s.add_argument("--settled", required=True); s.add_argument("--sizes", default=None); s.add_argument("--z", type=float, default=3.0); s.add_argument("--out", default="runs/scaling.json"); s.set_defaults(fn=cmd_scale)
    s = sub.add_parser("direction", help="extract and clean overthinking directions"); common(s)
    s.add_argument("--settled", required=True); s.add_argument("--no-atoms", action="store_true"); s.add_argument("--out", default="runs/bundle.json"); s.set_defaults(fn=cmd_direction)
    s = sub.add_parser("edit", help="apply rank-one edit to safetensors (or write edit spec)"); common(s, bank=False)
    s.add_argument("--bundle", required=True); s.add_argument("--edit-json", default=None); s.add_argument("--model-dir", default=None); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_edit)
    s = sub.add_parser("search", help="search gamma/layers co-minimising tokens and KL"); common(s)
    s.add_argument("--bundle", required=True); s.add_argument("--trials", type=int, default=30); s.add_argument("--w-kl", type=float, default=2.0); s.add_argument("--max-tasks", type=int, default=0); s.add_argument("--out", default="runs/search.json"); s.set_defaults(fn=cmd_search)
    s = sub.add_parser("transfer", help="task-vector transfer from a donor fine-tune"); common(s, bank=False)
    s.add_argument("--target", required=True); s.add_argument("--donor", required=True); s.add_argument("--donor-base", required=True); s.add_argument("--out", required=True)
    s.add_argument("--alpha", type=float, default=0.5); s.add_argument("--keep", type=float, default=0.2); s.add_argument("--only", default=None); s.set_defaults(fn=cmd_transfer)
    s = sub.add_parser("quant", help="write calibration set + quantization scripts"); common(s)
    s.add_argument("--rollouts", required=True, help="rollouts of the EDITED model on the calib split"); s.add_argument("--model-dir", required=True)
    s.add_argument("--params-b", type=float, default=27.0); s.add_argument("--out", default="runs/quant"); s.set_defaults(fn=cmd_quant)
    s = sub.add_parser("eval", help="paired multi-arm evaluation"); common(s)
    s.add_argument("--arm", action="append", required=True, help="name:kind:model  (openai: model@base_url)"); s.add_argument("--base", default="base")
    s.add_argument("--seeds", type=int); s.add_argument("--max-tasks", type=int, default=0); s.add_argument("--settled", default=None); s.add_argument("--out", default="runs/eval"); s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("report", help="render report from run json"); s.add_argument("--run-json", required=True); s.add_argument("--out", default="runs/report"); s.set_defaults(fn=cmd_report)
    s = sub.add_parser("demo", help="full pipeline on the mock model"); s.add_argument("--out", default="runs/demo")
    s.add_argument("--n-tasks", type=int, default=60); s.add_argument("--seeds", type=int, default=3); s.add_argument("--trials", type=int, default=20); s.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
