"""Analysis stages on data fixtures (mining, scaling, evaluate, sftdata) — no model."""
from swiftlab.mining import mine_markers, fightin_words, penalty_vocab_to_logit_bias
from swiftlab.scaling import scaling_curve, chao1, loop_signatures
from swiftlab.evaluate import compare, compare_all, summarize_arm
from swiftlab.sftdata import build_sft_rows
from swiftlab.settle import waste_summary, checkpoints_for
from swiftlab.tasks import Task
from collections import Counter


def test_mining_finds_loop_tokens(settled_rows):
    m = mine_markers(settled_rows, z_threshold=1.0)
    assert m["phrases"], "no markers mined from waste"
    joined = " ".join(m["phrases"])
    assert any(w in joined for w in ("wait", "double-check", "reconsider", "same", "again"))
    fw = fightin_words(["clean productive step"], ["wait wait double-check", "wait hmm"], nmax=2, min_count=1)
    assert fw[0]["z"] > 0


def test_penalty_vocab_maps_single_tokens():
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [1] if s.strip() in ("wait", "hmm") else [1, 2]
    bias = penalty_vocab_to_logit_bias(["wait", "but wait", "hmm"], Tok(), strength=-3.0)
    assert bias and all(v == -3.0 for v in bias.values())


def test_scaling_curve_monotone_and_chao1(settled_rows):
    sc = scaling_curve(settled_rows, [15, 30, 60])
    assert [p["n"] for p in sc["points"]] == [15, 30, 60]
    assert sc["points"][-1]["markers_discovered"] >= sc["points"][0]["markers_discovered"]
    assert 0 <= sc["points"][-1]["signature_coverage"] <= 1.0
    assert chao1(Counter({"a": 1, "b": 1, "c": 2})) > 3
    assert loop_signatures(settled_rows)


def test_waste_summary_and_checkpoints(settled_rows):
    w = waste_summary(settled_rows)
    assert w["overspent_share"] > 0 and 0 <= w["removable_share_upper_bound"] <= 1
    assert w["categories"]["overspent"] > 0
    cps = checkpoints_for("a. b. c. d. e. f. g. h.", 4)
    assert 1 <= len(cps) <= 4


def test_paired_compare_metrics(paired_rows, settled_rows):
    res = compare(paired_rows["base"], paired_rows["edited"], n_boot=200, settled_base=settled_rows)
    assert res["n_pairs"] == 60
    assert 0.29 < res["think_reduction_mean"] < 0.31          # every item cut to 70%
    assert res["share_items_shorter"] == 1.0
    assert res["fixed"] >= res["broken"]                       # derailments fixed
    assert 0 <= res["mcnemar_p"] <= 1
    assert res["coverage"]["overspend_removed"] is not None
    lo, hi = res["accuracy_delta"]["lo"], res["accuracy_delta"]["hi"]
    assert lo <= res["accuracy_delta"]["delta"] <= hi
    allres = compare_all(paired_rows, base="base", settled_base=settled_rows, n_boot=100)
    assert "base->edited" in allres["pairs"] and "domains" in allres["pairs"]["base->edited"]
    s = summarize_arm(paired_rows["base"])
    assert s["n"] == 60 and 0 <= s["accuracy"] <= 1 and "coding" in s["domains"]


def test_sftdata_truncates_overspent(settled_rows):
    tasks = {r["task_id"]: Task(r["task_id"], r["domain"], "p", {"kind": "regex", "patterns": ["42"]}) for r in settled_rows}
    sft = build_sft_rows(settled_rows, tasks, backend=None)          # backend=None drops derailed
    assert sft and all(r["category"] in ("tight", "overspent", "short") for r in sft)
    over = [r for r in sft if r["category"] == "overspent"]
    assert over and all(len(r["thinking"]) < len(settled_rows[0]["thinking"]) for r in over)
