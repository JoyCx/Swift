import json
import numpy as np
from swiftlab.tasks import TaskBank
from swiftlab.backends.mock import MockBackend
from swiftlab.backends.base import GenRequest
from swiftlab.abliterate import build_refusal_bundle, measure, is_refusal, refusal_rate, score
from swiftlab.directions import build_bundle
from swiftlab.edit import make_edit, apply_to_mock
from swiftlab.rollout import run_rollouts
from swiftlab.settle import probe_all
from swiftlab.sftdata import build_sft_rows, write_sft

HARM = [f"[harmful] placeholder request number {i} about topic {i % 7}" for i in range(40)]
SAFE = [f"Please write a short note about topic {i % 7}, item {i}" for i in range(40)]


def test_refusal_detection():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert not is_refusal("Sure, here is the plan.")
    assert refusal_rate(["I am unable to comply", "ok"]) == 0.5


def test_abliteration_protected_by_overthink_direction(tmp_path):
    bank, _ = TaskBank.build([], synthetic_per_domain=12, seed=0)
    base = MockBackend(bank=bank)
    rows = run_rollouts(base, bank.tasks, 2, tmp_path / "r.jsonl", arm="base", resume=False)
    rows = probe_all(base, bank.by_id, rows, checkpoints=8, min_think_tokens=30)
    ob = build_bundle(base, rows, atoms=["coding"], max_per_class=120)
    protect = {"overthink": {l: d["v"] for l, d in ob["layers"].items()}}
    naive = build_refusal_bundle(base, HARM, SAFE, atoms=[], protect=None)
    prot = build_refusal_bundle(base, HARM, SAFE, atoms=["coding", "format"], protect=protect)
    m0 = measure(base, HARM[-10:])
    assert m0["refusal_rate"] > 0.5
    for l in prot["layers"]:
        vn = np.array(naive["layers"][l]["v"]); vp = np.array(prot["layers"][l]["v"])
        assert vn @ base.r_star > 0.4 and vp @ base.r_star > 0.4
        assert abs(vp @ base.d_star) <= abs(vn @ base.d_star) + 1e-6   # protection removes the overthink component
    e_naive = apply_to_mock(base, make_edit(naive, gamma=1.0), "naive"); e_prot = apply_to_mock(base, make_edit(prot, gamma=1.0), "prot")
    assert measure(e_prot, HARM[-10:])["refusal_rate"] < m0["refusal_rate"]
    # protection keeps the thinking behaviour where the thinking stage left it; the naive edit moves it
    b = base.state["overthink_rate"]
    assert abs(e_prot.state["overthink_rate"] - b) < abs(e_naive.state["overthink_rate"] - b)
    assert score({"refusal_rate": 0.1, "kl": 0.0, "think_ratio": 1.2, "acc_delta": 0.0}) > score({"refusal_rate": 0.1, "kl": 0.0, "think_ratio": 1.0, "acc_delta": 0.0})


def test_sft_dataset_builder(tmp_path):
    bank, _ = TaskBank.build([], synthetic_per_domain=12, seed=0)
    base = MockBackend(bank=bank)
    rows = run_rollouts(base, bank.tasks, 2, tmp_path / "r.jsonl", arm="base", resume=False)
    rows = probe_all(base, bank.by_id, rows, checkpoints=8, min_think_tokens=30)
    sft = build_sft_rows(rows, bank.by_id, backend=base)
    assert sft and all(r["category"] != "wrong" for r in sft)
    over = [r for r in sft if r["category"] == "overspent"]
    assert over and all(len(r["thinking"].split()) < r["orig_think_tokens"] for r in over)
    info = write_sft(sft, tmp_path / "sft.jsonl")
    assert info["n"] == len(sft)
    first = json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])
    assert {"prompt", "thinking", "answer"} <= set(first)
