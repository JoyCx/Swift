"""swiftlab command line. Run `swiftlab preflight` first to validate your real server and verifiers."""
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
    if cfg.backend.kind not in ("openai", "hf"):
        raise SystemExit(f"backend.kind must be 'openai' or 'hf' (got {cfg.backend.kind!r}). Set it in the config or with --set backend.kind=openai")
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
    from .edit import apply_in_place, restore_in_place
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


def cmd_sftdata(args):
    from .sftdata import build_sft_rows, write_sft
    cfg = _cfg(args)
    bank, _ = _tasks(args, cfg)
    be = _backend(cfg, bank) if not args.no_regen else None
    rows = build_sft_rows(read_jsonl(args.settled), bank.by_id, backend=be, max_per_task=args.max_per_task)
    print(json.dumps(write_sft(rows, args.out)))


def cmd_abliterate(args):
    from .abliterate import build_refusal_bundle, load_prompts, measure, score as ab_score
    from .directions import load_bundle, save_bundle
    from .edit import make_edit, apply_in_place, restore_in_place, apply_to_safetensors
    from .search import kl_divergence, run_search
    from .rollout import run_rollouts
    cfg = _cfg(args)
    bank, calib = _tasks(args, cfg, "calib")
    calib = bank.__class__(calib).sample(args.max_tasks or 24, seed=1)
    be = _backend(cfg, bank)
    harmful, harmless = load_prompts(args.harmful), load_prompts(args.harmless)
    protect = {}
    if args.protect_bundle:
        ob = load_bundle(args.protect_bundle)
        protect["overthink"] = {l: d["v"] for l, d in ob["layers"].items()}
    rb = build_refusal_bundle(be, harmful, harmless, layers=cfg.direction.layers or None, atoms=args.atoms.split(",") if args.atoms else [],
                              protect=protect, ridge_lambda=cfg.direction.ridge_lambda)
    save_bundle(rb, args.out_bundle)
    for l, d in sorted(rb["layers"].items()):
        print(f"layer {l:>3} auc dirty={d['auc_dirty']:.3f} clean={d['auc_clean']:.3f} energy removed={d['energy_removed']:.2f} cos={ {k: round(v, 2) for k, v in d['cos_atoms_clean'].items()} }")
    base_m = measure(be, harmful[-max(8, len(harmful) // 4):], effort=cfg.backend.reasoning_effort)
    base_rows = run_rollouts(be, calib, 1, Path(args.out).parent / "ab_base.jsonl", arm="base", resume=False)
    base_acc = sum(r["correct"] for r in base_rows) / len(base_rows); base_think = sum(r["think_tokens"] for r in base_rows)
    kl_prompts = [t.prompt for t in calib[:12]]; base_dist = be.next_token_dist(kl_prompts)
    print("base:", json.dumps(base_m))
    n = [0]

    def evaluate(edit):
        n[0] += 1
        saved = apply_in_place(be.model, edit)
        try:
            m = measure(be, harmful[-max(8, len(harmful) // 4):], effort=cfg.backend.reasoning_effort)
            rows = run_rollouts(be, calib, 1, Path(args.out).parent / "ab_trial.jsonl", concurrency=1, arm="trial", resume=False)
            kl = kl_divergence(base_dist, be.next_token_dist(kl_prompts))
        finally:
            restore_in_place(be.model, saved)
        return {"refusal_rate": m["refusal_rate"], "kl": kl, "think_ratio": sum(r["think_tokens"] for r in rows) / max(1, base_think),
                "acc_delta": sum(r["correct"] for r in rows) / len(rows) - base_acc}

    import swiftlab.search as S
    orig = S.score
    S.score = lambda metrics, w_kl=1.0, acc_tol=0.01, acc_penalty=10.0: ab_score(metrics, w_kl=w_kl, acc_tol=acc_tol)
    try:
        res = run_search(rb, evaluate, n_trials=args.trials, w_kl=args.w_kl, acc_tol=cfg.eval.accuracy_tolerance, space={"gamma": (0.5, 1.5)})
    finally:
        S.score = orig
    write_json(args.out, {"base": base_m, "best": res["best"], "trials": res["trials"], "best_edit": res["best_edit"], "method": res["method"]})
    print(json.dumps({"best_params": res["best"]["params"], "best_metrics": res["best"]["metrics"]}))
    if args.model_dir:
        r = apply_to_safetensors(args.model_dir, args.apply_out, res["best_edit"])
        print(json.dumps({"touched": len(r["touched"]), "out": r["out_dir"]}))


def cmd_report(args):
    from .report import write_report
    run = read_json(args.run_json)
    print(json.dumps(write_report(run, args.out)))


def cmd_preflight(args):
    from .preflight import run_preflight
    ok = run_preflight(_cfg(args), server=not args.local_only, n=args.n)
    raise SystemExit(0 if ok else 1)


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
    s = sub.add_parser("sftdata", help="build penalised-SFT dataset from settled base rollouts"); common(s)
    s.add_argument("--settled", required=True); s.add_argument("--max-per-task", type=int, default=2); s.add_argument("--no-regen", action="store_true", help="skip forced-answer regeneration for derailed traces")
    s.add_argument("--out", default="runs/sft.jsonl"); s.set_defaults(fn=cmd_sftdata)
    s = sub.add_parser("abliterate", help="surgical refusal ablation protected by the overthinking direction"); common(s)
    s.add_argument("--harmful", required=True); s.add_argument("--harmless", required=True); s.add_argument("--protect-bundle", default=None, help="bundle.json from `direction`")
    s.add_argument("--atoms", default="coding,knowledge,format"); s.add_argument("--trials", type=int, default=20); s.add_argument("--w-kl", type=float, default=1.0)
    s.add_argument("--max-tasks", type=int, default=0); s.add_argument("--out-bundle", default="runs/refusal_bundle.json"); s.add_argument("--out", default="runs/abliterate.json")
    s.add_argument("--model-dir", default=None); s.add_argument("--apply-out", default=None); s.set_defaults(fn=cmd_abliterate)
    s = sub.add_parser("report", help="render report from run json"); s.add_argument("--run-json", required=True); s.add_argument("--out", default="runs/report"); s.set_defaults(fn=cmd_report)
    s = sub.add_parser("preflight", help="validate the real server + verifiers before a long run"); common(s, bank=False)
    s.add_argument("--local-only", action="store_true", help="skip server checks, validate verifiers/decontam only")
    s.add_argument("--n", type=int, default=1, help="tasks per domain to probe through the server"); s.set_defaults(fn=cmd_preflight)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
