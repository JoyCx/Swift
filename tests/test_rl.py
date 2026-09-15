import numpy as np
from swiftlab.rl import (efficiency_reward, group_advantages, sequence_log_ratio, gspo_objective, reward_summary)


def test_efficiency_reward_correctness_dominates():
    # a wrong short answer never beats a correct one, whatever the length
    assert efficiency_reward(True, 1000, 100) > efficiency_reward(False, 10, 100)
    # correct + at/under target gets the full brevity bonus; overshooting decays it
    r_tight = efficiency_reward(True, 100, 100, brevity_coef=0.3)
    r_over = efficiency_reward(True, 400, 100, brevity_coef=0.3)
    assert 1.29 < r_tight <= 1.30 and 1.0 < r_over < r_tight
    # brevity never pushes a wrong answer positive past a correct one
    assert efficiency_reward(False, 10, 100) == 0.0
    assert efficiency_reward(True, 100, 100, format_ok=False) < efficiency_reward(True, 100, 100, format_ok=True)


def test_group_advantages_zero_mean_and_dead_groups():
    r = np.array([1.0, 1.0, 0.0, 0.0, 2.0, 2.0])          # 3 groups of 2
    a = group_advantages(r, 2)
    a = a.reshape(3, 2)
    assert np.allclose(a.mean(axis=1), 0.0)
    # a group where both samples tie -> ~0 advantage (no learning signal)
    tie = group_advantages(np.array([1.0, 1.0]), 2)
    assert np.all(np.abs(tie) < 1e-2)


def test_sequence_ratio_is_length_normalised():
    # doubling the sequence length with the same per-token gap gives the same sequence ratio
    new = np.array([0.0, -1.0, -1.0, -1.0]); old = np.array([0.0, -2.0, -2.0, -2.0]); mask = np.array([0, 1, 1, 1])
    r1 = sequence_log_ratio(new, old, mask)
    new2 = np.concatenate([new, new[1:]]); old2 = np.concatenate([old, old[1:]]); mask2 = np.concatenate([mask, mask[1:]])
    r2 = sequence_log_ratio(new2, old2, mask2)
    assert abs(r1 - r2) < 1e-9 and abs(r1 - 1.0) < 1e-9


def test_gspo_objective_clipping():
    # positive advantage with a large positive ratio is clipped (surrogate uses the min)
    adv = np.array([1.0]); big = np.array([np.log(2.0)])       # s = 2.0, clip to 1.2
    obj = gspo_objective(big, adv, clip=0.2)
    assert abs(-obj - 1.2) < 1e-9
    # negative advantage: min picks the more negative (unclipped) branch -> larger loss
    neg = gspo_objective(big, np.array([-1.0]), clip=0.2)
    assert -neg == -2.0


def test_reward_summary():
    s = reward_summary(np.array([1.0, 0.0, 0.0, 0.0]), 2)
    assert s["groups"] == 2 and 0 <= s["solve_rate"] <= 1 and s["all_same_groups"] == 1
