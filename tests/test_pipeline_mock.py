import numpy as np
from swiftlab.tasks import TaskBank
from swiftlab.backends.mock import MockBackend, MOCK_VOCAB
from swiftlab.backends.base import GenRequest
from swiftlab.rollout import run_rollouts, make_requests
from swiftlab.settle import probe_all, waste_summary
from swiftlab.mining import mine_markers
from swiftlab.scaling import scaling_curve, chao1
from swiftlab.directions import build_bundle, ridge_clean
from swiftlab.edit import make_edit, apply_to_mock, rank_one_project
from swiftlab.evaluate import run_arms, compare_all
from swiftlab.search import run_search, kl_divergence
from collections import Counter


def _setup(tmp_path, n=45):
    bank, _ = TaskBank.build([], synthetic_per_domain=n // 3, seed=0)
    base = MockBackend(bank=bank)
    rows = run_rollouts(base, bank.tasks, 2, tmp_path / "r.jsonl", arm="base", resume=False)
    rows = probe_all(base, bank.by_id, rows, checkpoints=8, min_think_tokens=30)
    return bank, base, rows


def test_seeds_are_model_independent():
    bank, _ = TaskBank.build([], synthetic_per_domain=3)
    a = make_requests(bank.tasks, 2); b = make_requests(bank.tasks, 2)
    assert [r.seed for r in a] == [r.seed for r in b]
    assert len({r.seed for r in a}) == len(a)


def test_settle_categories_and_mining(tmp_path):
    bank, base, rows = _setup(tmp_path)
    ws = waste_summary(rows)
    assert set(ws["categories"]) <= {"overspent", "tight", "wrong", "derailed", "short"}
    assert ws["overspent_share"] > 0
    for r in rows:
        s = r["settle"]
        if s["category"] == "overspent":
            assert s["settle_tokens"] < r["think_tokens"] and s["overspent_tokens"] > 0
    m = mine_markers(rows, z_threshold=3.0)
    assert m["phrases"], "no markers mined"
    assert any(p in ("re-deriving", "again", "double-check", "wait") for p in m["phrases"][:20])
    sc = scaling_curve(rows, [10, 30, len(rows)])
    assert [p["n"] for p in sc["points"]] == [10, 30, len(rows)]
    assert sc["points"][-1]["markers_discovered"] >= sc["points"][0]["markers_discovered"]
    assert chao1(Counter({"a": 1, "b": 1, "c": 2})) > 3


def test_rank_one_projection_removes_direction():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 8)); v = rng.standard_normal(16); v /= np.linalg.norm(v)
    W2 = rank_one_project(W, v, 1.0)
    assert np.allclose(v @ W2, 0, atol=1e-9)
    W3 = rank_one_project(W, v, 0.5)
    assert np.allclose(v @ W3, 0.5 * (v @ W))


def test_ridge_clean_removes_atom_component():
    rng = np.random.default_rng(1)
    d = rng.standard_normal(32); c = rng.standard_normal(32); c -= c @ d / (d @ d) * d
    r = 2 * d + 0.8 * c
    rc, w = ridge_clean(r, [c], 1e-3)
    assert abs(rc @ c) < 1e-2 * np.linalg.norm(c) * np.linalg.norm(rc) + 1e-6
    assert rc @ d > 0


def test_direction_recovery_and_edit(tmp_path):
    bank, base, rows = _setup(tmp_path, 60)
    dirty = build_bundle(base, rows, atoms=[], max_per_class=150)
    clean = build_bundle(base, rows, atoms=["coding", "language", "format"], max_per_class=150)
    cs_dirty, cs_clean = [], []
    for l in clean["layers"]:
        vd = np.array(dirty["layers"][l]["v"]); vc = np.array(clean["layers"][l]["v"])
        assert vd @ base.d_star > 0.4 and vc @ base.d_star > 0.4, "direction should find the planted d*"
        cs_dirty.append(abs(vd @ base.c_star)); cs_clean.append(abs(vc @ base.c_star))
    # cleaning shrinks the entangled capability component on average and at the selected layers
    assert np.mean(cs_clean) < np.mean(cs_dirty)
    for l in clean["best_layers"]:
        assert clean["layers"][l]["auc_clean"] > 0.6
    edit = make_edit(clean, gamma=1.0)
    m = apply_to_mock(base, edit)
    assert m.state["overthink_rate"] < base.state["overthink_rate"]
    assert base.state["overthink_rate"] == 0.6, "base must be untouched"
    req = GenRequest(bank.tasks[0].id, bank.tasks[0].prompt, 7, meta={"difficulty": 0.5})
    assert m.generate(req).think_tokens <= base.generate(req).think_tokens


def test_logit_bias_penalizer_shortens():
    bank, _ = TaskBank.build([], synthetic_per_domain=4)
    m = MockBackend(bank=bank)
    t = bank.tasks[0]
    a = m.generate(GenRequest(t.id, t.prompt, 3, meta={"difficulty": 0.9}))
    b = m.generate(GenRequest(t.id, t.prompt, 3, meta={"difficulty": 0.9}, logit_bias={MOCK_VOCAB["wait"]: -5.0}))
    assert b.think_tokens <= a.think_tokens


def test_paired_eval_and_search(tmp_path):
    bank, base, rows = _setup(tmp_path, 45)
    bundle = build_bundle(base, rows, atoms=["coding"], max_per_class=120)
    calib = bank.tasks[:12]
    base_rows = run_rollouts(base, calib, 1, tmp_path / "b.jsonl", arm="base", resume=False)
    bt = sum(r["think_tokens"] for r in base_rows); ba = sum(r["correct"] for r in base_rows) / len(base_rows)
    prompts = [t.prompt for t in calib[:5]]; bd = base.next_token_dist(prompts)

    def ev(edit):
        eb = apply_to_mock(base, edit)
        rs = run_rollouts(eb, calib, 1, tmp_path / "t.jsonl", arm="t", resume=False)
        return {"think_ratio": sum(r["think_tokens"] for r in rs) / bt, "acc_delta": sum(r["correct"] for r in rs) / len(rs) - ba, "kl": kl_divergence(bd, eb.next_token_dist(prompts))}

    s = run_search(bundle, ev, n_trials=6, seed=0)
    assert s["best"]["score"] == min(t["score"] for t in s["trials"])
    arms = {"base": base, "edit": apply_to_mock(base, s["best_edit"])}
    er = run_arms(arms, bank.tasks[:15], 2, tmp_path / "eval")
    res = compare_all(er, base="base", settled_base=probe_all(base, bank.by_id, er["base"], checkpoints=6, min_think_tokens=30), n_boot=100)
    p = res["pairs"]["base->edit"]
    assert p["n_pairs"] == 30 and 0 <= p["mcnemar_p"] <= 1
    assert p["think_reduction_mean"] >= 0
    assert "coverage" in p and p["coverage"]["overspend_removed"] is not None
    assert p["accuracy_delta"]["lo"] <= p["accuracy_delta"]["delta"] <= p["accuracy_delta"]["hi"]
