"""SRA / edit / abliteration linear algebra on numpy arrays (the substance of the surgery)."""
import numpy as np
from swiftlab.directions import ridge_clean, diff_of_means, separation_auc, cosines, gamma_weights
from swiftlab.edit import rank_one_project, make_edit
from swiftlab.abliterate import is_refusal, refusal_rate, score, load_prompts
from swiftlab.transfer import trim_topk


def test_ridge_clean_removes_atom_component():
    rng = np.random.default_rng(0)
    d = rng.standard_normal(48); c = rng.standard_normal(48); c -= (c @ d) / (d @ d) * d
    r = 2.0 * d + 0.9 * c
    rc, w = ridge_clean(r, [c], lam=1e-3)
    assert abs(rc @ c) < 0.05 * np.linalg.norm(c) * np.linalg.norm(rc)      # capability component gone
    assert rc @ d > 0                                                       # target component kept


def test_diff_of_means_and_auc_and_cos():
    pos = {0: np.array([[2.0, 0.0], [2.2, 0.1]])}; neg = {0: np.array([[0.0, 0.0], [-0.1, 0.0]])}
    d = diff_of_means(pos, neg)[0]
    assert d[0] > 1.5
    auc = separation_auc(pos[0], neg[0], d)
    assert auc == 1.0
    cs = cosines(np.array([1.0, 0.0]), {"x": np.array([1.0, 0.0]), "y": np.array([0.0, 1.0])})
    assert abs(cs["x"] - 1.0) < 1e-6 and abs(cs["y"]) < 1e-6


def test_rank_one_projection_and_gamma():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((12, 6)); v = rng.standard_normal(12); v /= np.linalg.norm(v)
    assert np.allclose(v @ rank_one_project(W, v, 1.0), 0, atol=1e-9)
    assert np.allclose(v @ rank_one_project(W, v, 0.4), 0.6 * (v @ W))
    gw = gamma_weights([2, 4, 6], center=4, width=2, gamma=1.0, kernel="gaussian")
    assert gw[4] == 1.0 and gw[2] < gw[4] and gw[6] < gw[4]


def test_make_edit_from_bundle():
    bundle = {"layers": {3: {"v": [1.0, 0.0, 0.0]}, 4: {"v": [0.0, 1.0, 0.0]}}, "best_layers": [3, 4]}
    e = make_edit(bundle, gamma=0.8, kernel="flat", matrices=["mlp.down_proj"])
    assert set(e["layers"]) == {3, 4} and e["layers"][3]["gamma"] == 0.8 and e["matrices"] == ["mlp.down_proj"]


def test_refusal_detection_and_score(tmp_path):
    assert is_refusal("I'm sorry, but I can't help with that.") and not is_refusal("Sure, here you go.")
    assert refusal_rate(["I am unable to comply", "here is the code"]) == 0.5
    # score penalises re-lengthened thinking and dropped accuracy
    assert score({"refusal_rate": 0.1, "kl": 0.0, "think_ratio": 1.3, "acc_delta": 0.0}) > score({"refusal_rate": 0.1, "kl": 0.0, "think_ratio": 1.0, "acc_delta": 0.0})
    assert score({"refusal_rate": 0.1, "kl": 0.0, "think_ratio": 1.0, "acc_delta": -0.1}) > score({"refusal_rate": 0.1, "kl": 0.0, "think_ratio": 1.0, "acc_delta": 0.0})
    f = tmp_path / "p.txt"; f.write_text("line one\nline two\n")
    assert load_prompts(f) == ["line one", "line two"]


def test_trim_topk_keeps_largest():
    torch = __import__("pytest").importorskip("torch")
    d = torch.tensor([0.1, -3.0, 0.2, 2.0, -0.05])
    t = trim_topk(d, 0.4)                     # keep top 2 by magnitude
    assert (t != 0).sum().item() == 2 and t[1] == -3.0 and t[3] == 2.0
