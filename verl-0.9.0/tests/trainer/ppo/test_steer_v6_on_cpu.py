# Copyright 2026 The verl team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""V6 score and boost-curve boundary cases."""

import pytest
import torch

from verl.trainer.ppo.core_algos import apply_q2_high_entropy_boost, compute_token_weights
from verl.workers.config import PolicyLossConfig


def test_square_curve_uses_only_selected_finite_scores():
    base = torch.tensor([[0.92, 0.85, 0.78, 1.0, 0.71, 0.0]])
    scores = torch.tensor([[0.005, 0.025, 0.045, float("inf"), 25.0, 0.0]], requires_grad=True)
    selected = torch.tensor([[True, True, True, True, False, False]])
    weights = apply_q2_high_entropy_boost(base, selected, scores, 0.2)
    torch.testing.assert_close(weights, torch.tensor([[1.2, 1.05075, 1.001, 1.0, 0.71, 0.0]]))
    torch.testing.assert_close(weights[~selected], base[~selected], rtol=0, atol=0)
    assert not weights.requires_grad


@pytest.mark.parametrize(
    ("scores", "selected", "expected"),
    [
        ([0.04, 0.02], [True, False], [1.2, 0.75]),
        ([0.04, 0.04], [True, True], [1.2, 1.2]),
        ([float("nan"), 0.04], [True, True], [1.0, 1.2]),
        ([float("nan"), float("inf")], [True, True], [1.0, 1.0]),
        ([0.04, 0.02], [False, False], [0.8, 0.75]),
    ],
)
def test_boost_degenerate_sets(scores, selected, expected):
    base = torch.tensor([0.8, 0.75])
    actual = apply_q2_high_entropy_boost(base, torch.tensor(selected), torch.tensor(scores), 0.2)
    torch.testing.assert_close(actual, torch.tensor(expected))


def test_empty_input_and_zero_strength_match_v5():
    base = torch.tensor([[0.85, 0.76, 0.0]])
    mask = torch.tensor([[True, False, False]])
    scores = torch.tensor([[0.005, 0.025, 0.045]])
    torch.testing.assert_close(
        apply_q2_high_entropy_boost(base, mask, scores, 0.0), base.masked_fill(mask, 1.0), rtol=0, atol=0
    )
    assert apply_q2_high_entropy_boost(base[:, :0], mask[:, :0], scores[:, :0], 0.2).shape == (1, 0)


@pytest.mark.parametrize("valid", [True, False])
def test_original_weights_and_raw_score_magnitude(valid):
    advantages = torch.tensor([[1.0, -1.0, 2.0]])
    entropy = torch.tensor([[1.0, 1.0, 0.4]])
    old_log_prob = torch.tensor([[-30.0, -2.0, float("nan")]])
    log_prob = torch.tensor([[-25.0, -2.0, -1.0]], requires_grad=True)
    mask = torch.full_like(log_prob, valid, dtype=torch.bool)
    args = (advantages, entropy, old_log_prob, log_prob, mask)
    base = compute_token_weights(*args, linear=False)
    weights, magnitude = compute_token_weights(*args, linear=False, return_score_magnitude=True)
    p = log_prob.detach().exp().clamp(1e-8, 1.0 - 1e-8)
    p_old = old_log_prob.exp().clamp(1e-8, 1.0)
    expected = ((advantages / p_old) * (p * (1 - p) * (p.log() + entropy))).abs()
    torch.testing.assert_close(weights, base, rtol=0, atol=0)
    torch.testing.assert_close(magnitude, expected, rtol=0, atol=0, equal_nan=True)
    assert not magnitude.requires_grad
    if valid:
        assert weights[0, 2] == 1.0 and magnitude[0, 2].isnan()


@pytest.mark.parametrize("value", [0.0, 0.001, 0.2])
def test_accepts_boost_strength(value):
    config = PolicyLossConfig(loss_mode="entropy_control", linear=False, q2_high_entropy_boost_strength=value)
    assert config.q2_high_entropy_boost_strength == value


@pytest.mark.parametrize("value", [-0.2, 0.0009, float("inf"), float("nan"), True, "0.2"])
def test_rejects_invalid_boost_strength(value):
    with pytest.raises(ValueError, match="q2_high_entropy_boost_strength"):
        PolicyLossConfig(q2_high_entropy_boost_strength=value)
