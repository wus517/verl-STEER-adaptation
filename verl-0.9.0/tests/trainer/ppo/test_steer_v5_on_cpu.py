# Copyright 2026 The verl team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""STEER V5 selection, policy gradients, and logging regressions."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    compute_policy_loss_with_entropy,
    compute_steer_v4_weights,
    select_q2_high_entropy_tokens,
)
from verl.trainer.ppo.steer_metrics import SteerV4Metrics
from verl.utils import tensordict_utils as tu
from verl.utils.metric import reduce_metrics
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss


def actor_config(ratio, *, logging=True, weight_min=0.7):
    return ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        policy_loss=PolicyLossConfig(
            loss_mode="entropy_control",
            entropy_control_mode="symmetric",
            linear=False,
            token_weight_min=weight_min,
            token_weight_max=1.0,
            steer_quadrant_logging=logging,
            q2_high_entropy_ratio=ratio,
        ),
    )


def batch():
    # The two highest-entropy valid positions are 0 (active, also the maximum
    # absolute score) and 1 (clipped), both in Q2. Positions 6-10 cover other
    # quadrants, zero advantage and a padding token with the highest entropy.
    log_prob = torch.tensor([[-4.0, -4.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]])
    old = log_prob.clone()
    old[0, 1] -= 0.5
    advantages = torch.tensor([[25.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 9.0]])
    entropy = torch.tensor([[3.5, 3.0, 0.1, 0.2, 0.3, 0.4, 2.3, 0.4, 2.5, 2.8, 10.0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    return old, log_prob, advantages, entropy, mask


def policy_loss(data, config, collector=None):
    old, log_prob, advantages, entropy, mask = data
    return compute_policy_loss_with_entropy(
        old, log_prob, advantages, mask, config=config, entropys=entropy, steer_metrics=collector
    )


@pytest.mark.parametrize(
    ("valid_positions", "ratio", "selected_positions"),
    [
        (0, 0.2, []),
        (1, 0.2, [0]),
        (5, 0.2, []),
        (6, 0.2, [0, 2]),
        (6, 0.0, []),
        (6, 1.0, [0, 2, 3]),
    ],
)
def test_microbatch_entropy_cutoff_with_ties_and_padding(valid_positions, ratio, selected_positions):
    entropy = torch.tensor([2.0, 3.0, 2.0, 1.0, 0.5, 0.1, 100.0, float("nan")]).reshape(2, 4).t()
    q2 = torch.tensor([True, False, True, True, False, False, False, False]).reshape(2, 4).t()
    mask = ((torch.arange(8) < valid_positions) | (torch.arange(8) == 7)).reshape(2, 4).t()
    q2 = q2 & mask
    release = select_q2_high_entropy_tokens(entropy, q2, mask, ratio)
    assert release.shape == q2.shape
    assert release.dtype == torch.bool
    assert release.t().flatten().nonzero(as_tuple=True)[0].tolist() == selected_positions
    assert not select_q2_high_entropy_tokens(entropy, torch.zeros_like(q2), mask, ratio).any()


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf"), True, "0.2"])
def test_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="finite number in"):
        actor_config(ratio)


@pytest.mark.parametrize(
    "override", [{"loss_mode": "vanilla"}, {"linear": True}, {"entropy_control_mode": "asymmetric"}]
)
def test_release_requires_symmetric_exponential_steer(override):
    config = {"loss_mode": "entropy_control", "linear": False, "entropy_control_mode": "symmetric"}
    with pytest.raises(ValueError, match="requires entropy_control"):
        PolicyLossConfig(q2_high_entropy_ratio=0.2, **(config | override))


def test_release_preserves_base_denominator_and_token_gradients():
    old, log_prob, advantages, entropy, mask = batch()
    weight_options = {"token_weight_min": 0.7, "token_weight_max": 1.0}
    base_weights, quadrants, _ = compute_steer_v4_weights(advantages, entropy, old, log_prob, mask, **weight_options)
    release = select_q2_high_entropy_tokens(entropy, quadrants[1], mask, 0.2)
    assert release.nonzero(as_tuple=False).tolist() == [[0, 0], [0, 1]]
    assert base_weights[0, 0].item() == pytest.approx(0.7)
    excluded_weights, _, _ = compute_steer_v4_weights(
        advantages, entropy, old, log_prob, mask & ~release, **weight_options
    )
    assert (base_weights[mask & ~release] != excluded_weights[mask & ~release]).any()

    def run(ratio, logging=True, weight_min=0.7):
        leaf = log_prob.detach().clone().requires_grad_()
        loss, original_metrics = policy_loss(
            (old, leaf, advantages, entropy, mask), actor_config(ratio, logging=logging, weight_min=weight_min)
        )
        return loss, torch.autograd.grad(loss, leaf)[0], original_metrics

    v4_loss, v4_gradient, v4_metrics = run(0.0)
    baseline_loss, baseline_gradient, baseline_metrics = run(0.0, False)
    torch.testing.assert_close(v4_loss, baseline_loss, rtol=0, atol=0)
    torch.testing.assert_close(v4_gradient, baseline_gradient, rtol=0, atol=0)
    assert v4_metrics == baseline_metrics

    v5_loss, v5_gradient, _ = run(0.2, False)
    logged_loss, logged_gradient, _ = run(0.2)
    torch.testing.assert_close(v5_loss, logged_loss, rtol=0, atol=0)
    torch.testing.assert_close(v5_gradient, logged_gradient, rtol=0, atol=0)
    _, ppo_gradient, _ = run(0.0, weight_min=1.0)
    torch.testing.assert_close(v5_gradient, torch.where(release, ppo_gradient, v4_gradient), rtol=0, atol=0)
    assert v5_gradient[0, 0] != 0 and v5_gradient[0, 1] == 0

    ratio = (log_prob - old).exp()
    clipped = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2))
    token_loss = torch.where(advantages < 0, torch.minimum(-advantages * 3, clipped), clipped)
    weights = base_weights.masked_fill(release, 1)
    torch.testing.assert_close(v5_loss, (token_loss * weights * mask).sum() / mask.sum(), rtol=0, atol=0)


@pytest.mark.parametrize("index", [1, 6])
def test_release_metrics_skip_zero_denominators(index):
    # Index 1 is clipped Q2. Index 6 belongs to Q1.
    collector = SteerV4Metrics()
    policy_loss(tuple(value[:, index : index + 1] for value in batch()), actor_config(0.2), collector)
    metrics = collector.flush()
    if index == 1:
        assert metrics["steer/q2_release_frac"] == 1
        assert metrics["steer/q2_weight_mean"] == 1
    else:
        assert "steer/q2_release_frac" not in metrics
    assert "steer/q2_release_active_frac" not in metrics
    assert "steer/q2_release_base_weight_mean" not in metrics


def _microbatches():
    data = batch()
    return (tuple(value[:, :1] for value in data), data)


def _check_window(metrics):
    assert metrics["steer/q2_release_frac"] == pytest.approx(3 / 7)
    assert metrics["steer/q2_release_active_frac"] == pytest.approx(2 / 6)
    assert metrics["steer/q2_release_base_weight_mean"] == pytest.approx(0.7)
    old, log_prob, advantages, entropy, mask = batch()
    base, quadrants, _ = compute_steer_v4_weights(
        advantages, entropy, old, log_prob, mask, token_weight_min=0.7, token_weight_max=1.0
    )
    effective = base.masked_fill(select_q2_high_entropy_tokens(entropy, quadrants[1], mask, 0.2), 1)
    assert metrics["steer/q2_weight_mean"] == pytest.approx((effective[quadrants[1]].sum().item() + 1) / 7)
    assert metrics["steer/weight_mean"] == pytest.approx((effective.sum().item() + 1) / 11)


def test_v5_tensorboard_keeps_original_v4_and_release_metrics(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from verl.utils.tracking import Tracking

    collector = SteerV4Metrics()
    config = actor_config(0.2)
    for old, log_prob, advantages, entropy, mask in _microbatches():
        width = mask.shape[-1]
        length = int(mask.sum())
        data = TensorDict(
            {
                "prompts": torch.ones(1, 1, dtype=torch.long),
                "responses": torch.ones(1, width, dtype=torch.long),
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
        loss, original_metrics = ppo_loss(config, {"log_probs": model_log_prob}, data, steer_metrics=collector)
        loss.backward()
    metrics = collector.flush()
    _check_window(metrics)
    assert collector.totals.count_nonzero() == 0

    logged = {**reduce_metrics(original_metrics), **metrics, "val/accuracy": 0.5}
    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path))
    tracker = Tracking("steer", "v5", default_backend=["tensorboard"])
    tracker.log(logged, step=1)
    tracker.finish()
    event = EventAccumulator(str(tmp_path)).Reload()
    assert {"actor/pg_loss", "val/accuracy", "steer/q2_weight_mean", "steer/q2_release_frac"} <= logged.keys()
    assert set(event.Tags()["scalars"]) == logged.keys()
    for tag, value in logged.items():
        assert event.Scalars(tag)[0].value == pytest.approx(value)


def _distributed_worker(rank, directory):
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        collector = SteerV4Metrics()
        policy_loss(_microbatches()[rank], actor_config(0.2), collector)
        metrics = collector.flush(dp_group=dist.group.WORLD)
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(metrics))
    finally:
        dist.destroy_process_group()


def test_two_process_v5_statistics(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path),), nprocs=2, join=True)
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert results[0] == results[1]
    _check_window(results[0])
