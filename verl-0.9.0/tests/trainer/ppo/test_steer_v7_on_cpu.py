# Copyright 2026 The verl team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""STEER V7 recovery thresholds, loss, and metric aggregation."""

import json
import math
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    apply_steer_recovery,
    compute_policy_loss_with_entropy,
    compute_steer_v4_weights,
    compute_token_weights,
    select_high_entropy_tokens,
)
from verl.trainer.ppo.steer_metrics import SteerV4Metrics
from verl.utils import tensordict_utils as tu
from verl.utils.metric import reduce_metrics
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss


def recovery_policy(**overrides):
    defaults = {
        "loss_mode": "entropy_control",
        "linear": False,
        "token_weight_min": 0.7,
        "token_weight_max": 1.0,
        "steer_quadrant_logging": True,
        "steer_recovery_quadrants": [1, 2, 3, 4],
    }
    return PolicyLossConfig(**(defaults | overrides))


def actor_config(**overrides):
    return ActorConfig(
        strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=1, policy_loss=recovery_policy(**overrides)
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("steer_recovery_quadrants", [1, 1]),
        ("steer_recovery_quadrants", [0]),
        ("steer_recovery_quadrants", [5]),
        ("steer_recovery_quadrants", [True]),
        ("steer_recovery_quadrants", ["2"]),
        ("steer_recovery_quadrants", (1, 2)),
        ("steer_recovery_top_ratio", 0),
        ("steer_recovery_top_ratio", 1.01),
        ("steer_recovery_top_ratio", float("nan")),
        ("steer_recovery_min_entropy", -0.001),
        ("steer_recovery_min_entropy", float("inf")),
        ("steer_recovery_rate", -0.01),
        ("steer_recovery_rate", 1.01),
        ("steer_recovery_rate", True),
    ],
)
def test_invalid_recovery_config(field, invalid):
    with pytest.raises(ValueError, match=field):
        recovery_policy(**{field: invalid})


@pytest.mark.parametrize(
    "overrides",
    [
        {"loss_mode": "vanilla"},
        {"entropy_control_mode": "asymmetric"},
        {"linear": True},
        {"q2_high_entropy_ratio": 0.2},
        {"q2_high_entropy_ratio": 0.2, "q2_high_entropy_boost_strength": 0.2},
    ],
)
def test_recovery_requires_original_symmetric_exponential_weights(overrides):
    with pytest.raises(ValueError, match="STEER recovery"):
        recovery_policy(**overrides)


@pytest.mark.parametrize(
    ("entropy", "at_cutoff", "selected"),
    [
        ([0.50, 0.51, 0.30, 0.29, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0, 1, 2], [1]),
        ([0.9, 0.8, 0.8, 0.79, 0.51, 0.50], [0, 1, 2], [0, 1, 2]),
        ([], [], []),
    ],
)
def test_recovery_combines_rank_and_strict_entropy_cutoffs(entropy, at_cutoff, selected):
    entropy = torch.tensor([entropy + [float("nan"), float("inf"), 100.0]])
    mask = torch.ones_like(entropy, dtype=torch.bool)
    mask[0, -1] = False
    quadrants = (mask, torch.zeros_like(mask), torch.zeros_like(mask), torch.zeros_like(mask))
    weights = torch.full_like(entropy, 0.70, requires_grad=True)
    high_entropy = select_high_entropy_tokens(entropy, mask, 0.20)
    assert high_entropy[0].nonzero(as_tuple=True)[0].tolist() == at_cutoff
    actual, recovered = apply_steer_recovery(weights, entropy, mask, quadrants, recovery_policy())
    assert recovered[0].nonzero(as_tuple=True)[0].tolist() == selected
    torch.testing.assert_close(actual[recovered], torch.full((len(selected),), 0.775))
    torch.testing.assert_close(actual[~recovered], weights[~recovered], rtol=0, atol=0)
    assert not actual.requires_grad


def test_recovery_restores_the_same_fraction_of_each_weight_reduction():
    weights = torch.tensor([[0.7, 0.8, 0.9, 1.0]])
    mask = torch.ones_like(weights, dtype=torch.bool)
    quadrants = tuple(position.unsqueeze(0) for position in torch.eye(4, dtype=torch.bool))
    actual, recovered = apply_steer_recovery(
        weights, torch.ones_like(weights), mask, quadrants, recovery_policy(steer_recovery_top_ratio=1.0)
    )
    assert recovered.all()
    torch.testing.assert_close(actual, torch.tensor([[0.775, 0.850, 0.925, 1.0]]))


def batch():
    log_prob = torch.tensor([[-1, -2, -2, -1, -2, -2, -2, -2, -2, -1, -1]], dtype=torch.float32)
    old = log_prob.clone()
    old[0, 4] -= 0.5  # PPO clips this high-entropy Q2 token.
    advantages = torch.tensor([[1, 20, -1, -1, 1, 1, 0, 1, -1, -1, 1]], dtype=torch.float32)
    entropy = torch.tensor([[1.5, 1.5, 1.5, 1.5, 1.5, 0.65, 2.0, 0.5, 0.1, 1.2, 99.0]])
    mask = torch.tensor([[True] * 10 + [False]])
    return old, log_prob, advantages, entropy, mask


def run_loss(data, config, collector=None):
    old, log_prob, advantages, entropy, mask = data
    leaf = log_prob.detach().clone().requires_grad_()
    loss, _ = compute_policy_loss_with_entropy(
        old, leaf, advantages, mask, config=config, entropys=entropy, steer_metrics=collector
    )
    return loss, torch.autograd.grad(loss, leaf)[0]


@pytest.mark.parametrize(("quadrants", "rate"), [([], 0.25), ([1, 2], 0)])
def test_recovery_baseline_is_exact(quadrants, rate):
    data = batch()
    original = run_loss(data, actor_config(steer_recovery_quadrants=[], steer_quadrant_logging=False))
    collector = SteerV4Metrics()
    config = actor_config(steer_recovery_quadrants=quadrants, steer_recovery_rate=rate)
    loss, gradient = run_loss(data, config, collector)
    torch.testing.assert_close(loss, original[0], rtol=0, atol=0)
    torch.testing.assert_close(gradient, original[1], rtol=0, atol=0)
    metrics = collector.flush()
    if quadrants:
        assert metrics["steer/q2_recovery_frac"] == 2 / 4
        assert metrics["steer/q2_recovery_delta_mean"] == 0.0
        assert metrics["steer/q3_recovery_frac"] == metrics["steer/q3_recovery_active_frac"] == 0.0
        assert "steer/q3_recovery_delta_mean" not in metrics
    else:
        assert not any("_recovery_" in tag or "_release_" in tag for tag in metrics)


@pytest.mark.parametrize("rate", [0.25, 1.0])
def test_loss_uses_full_steer_denominator_and_preserves_ppo_clipping(rate):
    old, log_prob, advantages, entropy, mask = data = batch()
    weights, quadrants, _ = compute_steer_v4_weights(
        advantages, entropy, old, log_prob, mask, token_weight_min=0.7, token_weight_max=1.0
    )
    assert weights[0, 1] == pytest.approx(0.7)
    exclude_max = mask.clone()
    exclude_max[0, 1] = False
    excluded = compute_token_weights(
        advantages, entropy, old, log_prob, exclude_max, token_weight_min=0.7, linear=False
    )
    assert not torch.isclose(excluded[0, 5], weights[0, 5])
    _, ppo_gradient = run_loss(data, actor_config(steer_recovery_quadrants=[], token_weight_min=1.0))
    ratio = (log_prob - old).exp()
    clipped = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2))
    ppo_losses = torch.where(advantages < 0, torch.minimum(-advantages * 3, clipped), clipped)
    gradients = []
    for selection, indices in [([1, 2], [0, 1, 4]), ([1, 2, 3, 4], [0, 1, 2, 3, 4])]:
        expected = weights.clone()
        expected[0, indices] += rate * (1.0 - expected[0, indices])
        config = actor_config(
            steer_recovery_quadrants=selection, steer_recovery_rate=rate, steer_quadrant_logging=False
        )
        loss, gradient = run_loss(data, config)
        torch.testing.assert_close(loss, (ppo_losses * expected * mask).sum() / mask.sum())
        torch.testing.assert_close(gradient, ppo_gradient * expected)
        assert gradient[0, 4] == 0
        gradients.append(gradient)
    q1_or_q2 = quadrants[0] | quadrants[1]
    torch.testing.assert_close(gradients[0][q1_or_q2], gradients[1][q1_or_q2], rtol=0, atol=0)


def logging_microbatches():
    data = batch()
    below_gate = tuple(tensor[:, 7:8] for tensor in data)
    low_scores = tuple(tensor[:, [1, 1]] for tensor in data)
    low_scores[2].fill_(0.01)
    return below_gate, low_scores, data


def check_recovery_metrics(metrics):
    low_score_weight = 0.7 ** (0.01 * (1 - math.exp(-2)) * 0.5 / 0.02)
    base_mean = (2 * low_score_weight + 0.7) / 3
    for quadrant, expected_frac in enumerate((1.0, 4 / 7, 0.5, 0.5), start=1):
        assert metrics[f"steer/q{quadrant}_recovery_frac"] == pytest.approx(expected_frac)
    assert metrics["steer/q2_recovery_active_frac"] == pytest.approx(3 / 6)
    assert metrics["steer/q2_recovery_delta_mean"] == pytest.approx(0.25 * (1 - base_mean))
    assert metrics["steer/q2_release_frac"] == metrics["steer/q2_recovery_frac"]
    assert metrics["steer/q2_release_active_frac"] == metrics["steer/q2_recovery_active_frac"]
    assert metrics["steer/q2_release_base_weight_mean"] == pytest.approx(base_mean)
    assert metrics["steer/q2_release_boost_mean"] == pytest.approx(0.75 * (base_mean - 1))


def test_v7_tensorboard_keeps_training_and_steer_metrics(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from verl.utils.tracking import Tracking

    collector = SteerV4Metrics()
    config = actor_config()
    for old, log_prob, advantages, entropy, mask in logging_microbatches():
        length = int(mask.sum())
        data = TensorDict(
            {
                "prompts": torch.ones(1, 1, dtype=torch.long),
                "responses": torch.ones_like(mask, dtype=torch.long),
                "attention_mask": torch.cat((torch.ones(1, 1, dtype=torch.long), mask.long()), dim=-1),
                "response_mask": mask,
                "old_log_probs": old,
                "advantages": advantages,
                "entropy": torch.nested.as_nested_tensor([entropy[0, :length]], layout=torch.jagged),
            },
            batch_size=[1],
        )
        tu.assign_non_tensor(
            data, dp_size=1, batch_num_tokens=length, global_batch_size=1, steer_local_mini_batch_size=1
        )
        model_log_prob = torch.cat((log_prob[0, :length], torch.zeros(1))).requires_grad_()
        loss, actor_metrics = ppo_loss(config, {"log_probs": model_log_prob}, data, steer_metrics=collector)
        loss.backward()
        assert collector.totals.shape == (33,)
        assert collector.totals.dtype == torch.float32
    metrics = collector.flush()
    check_recovery_metrics(metrics)

    logged = {**reduce_metrics(actor_metrics), **metrics, "val/accuracy": 0.5}
    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path))
    tracker = Tracking("steer", "v7", default_backend=["tensorboard"])
    tracker.log(logged, step=1)
    tracker.finish()
    event = EventAccumulator(str(tmp_path)).Reload()
    assert {"actor/pg_loss", "val/accuracy", "steer/q2_weight_mean", "steer/q2_release_frac"} <= logged.keys()
    assert set(event.Tags()["scalars"]) == logged.keys()
    for tag, value in logged.items():
        assert event.Scalars(tag)[0].value == pytest.approx(value)


def _distributed_recovery_worker(rank, directory):
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        below_gate, low_scores, data = logging_microbatches()
        collector = SteerV4Metrics()
        microbatches = (below_gate, low_scores) if rank == 0 else (data,)
        for microbatch in microbatches:
            run_loss(microbatch, actor_config(), collector)
        assert collector.totals.dtype == torch.float32
        metrics = collector.flush(dp_group=dist.group.WORLD)
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(metrics))
    finally:
        dist.destroy_process_group()


def test_two_process_recovery_statistics(tmp_path):
    mp.spawn(_distributed_recovery_worker, args=(str(tmp_path),), nprocs=2, join=True)
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert results[0] == results[1]
    check_recovery_metrics(results[0])
