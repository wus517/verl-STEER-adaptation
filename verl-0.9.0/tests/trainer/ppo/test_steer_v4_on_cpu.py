# Copyright 2026 The verl team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""STEER V4 masks, original-loss parity, and aggregated statistics."""

import inspect
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    agg_steer_loss,
    compute_policy_loss_with_entropy,
    compute_steer_v4_weights,
    compute_token_weights,
)
from verl.trainer.ppo.steer_metrics import SteerV4Metrics, compute_steer_v4_statistics
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.workers.config import ActorConfig, PolicyLossConfig


def actor_config(*, steer_quadrant_logging=True, **overrides):
    options = {
        "loss_mode": "entropy_control",
        "entropy_control_mode": "symmetric",
        "linear": False,
        "steer_quadrant_logging": steer_quadrant_logging,
    }
    options.update(overrides)
    return ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        clip_ratio_c=3.0,
        policy_loss=PolicyLossConfig(**options),
    )


def quadrant_batch(dtype=torch.float32):
    old = torch.full((1, 7), -1.0, dtype=dtype)
    log_prob = old.clone().requires_grad_()
    advantages = torch.tensor([[1, 1, -1, -1, 0, 1, -1]], dtype=dtype)
    boundary = torch.tensor([[0.4, -0.4, -0.4, 0.4, 0.2, 0.0, 0.0]], dtype=dtype)
    entropy = -log_prob.detach().exp().clamp(1e-8, 1 - 1e-8).log() + boundary
    response_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    return old, log_prob, advantages, entropy, response_mask


def _policy_loss(old, log_prob, advantages, entropy, mask, config, aggregation="token-mean", collector=None):
    return compute_policy_loss_with_entropy(
        old,
        log_prob,
        advantages,
        mask,
        aggregation,
        config,
        entropys=entropy,
        steer_metrics=collector,
    )


def test_four_quadrants_other_mask_and_padding():
    old, log_prob, advantages, entropy, mask = quadrant_batch()
    weights, quadrants, other = compute_steer_v4_weights(
        advantages,
        entropy,
        old,
        log_prob,
        mask,
        token_weight_min=0.7,
        token_weight_max=1.0,
    )
    assert [int(quadrant.sum()) for quadrant in quadrants] == [1, 1, 1, 1]
    assert int(other.sum()) == 2
    assert torch.all(sum(quadrant.to(torch.int8) for quadrant in quadrants) <= 1)
    assert torch.equal(other | quadrants[0] | quadrants[1] | quadrants[2] | quadrants[3], mask)
    assert weights[0, 4] == 1
    assert weights[0, 6] == 0


@pytest.mark.parametrize("magnitude", [0.005, 0.02, 0.08])
def test_v4_returns_original_steer_weights_exactly(magnitude):
    old, log_prob, advantages, entropy, mask = quadrant_batch()
    entropy = entropy.clone()
    entropy[0, 0] += magnitude
    actual, _, _ = compute_steer_v4_weights(
        advantages, entropy, old, log_prob, mask, token_weight_min=0.7, token_weight_max=1.0
    )
    original = compute_token_weights(
        advantages,
        entropy,
        old,
        log_prob,
        mask,
        token_weight_min=0.7,
        token_weight_max=1.0,
        linear=False,
        mode="symmetric",
    )
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert not actual.requires_grad


@pytest.mark.parametrize("max_score", [0.005, 0.02, 0.08])
def test_original_exponential_denominator_floor_is_preserved(max_score):
    dtype = torch.float32
    scores = torch.tensor([[max_score / 4, max_score / 2, -max_score]], dtype=dtype)
    advantages = torch.tensor([[1.0, 1.0, -1.0]], dtype=dtype)
    log_prob = torch.full_like(scores, -1.0, requires_grad=True)
    old_log_prob = log_prob.detach().clone()
    probability = log_prob.detach().exp().clamp(1e-8, 1.0 - 1e-8)
    old_probability = old_log_prob.exp().clamp(1e-8, 1.0)
    boundary = -scores * old_probability / (advantages * probability * (1.0 - probability))
    entropy = boundary - probability.log()
    mask = torch.ones_like(scores, dtype=torch.bool)

    weights, _, _ = compute_steer_v4_weights(
        advantages,
        entropy,
        old_log_prob,
        log_prob,
        mask,
        token_weight_min=0.7,
        token_weight_max=1.0,
    )
    denominator = max(max_score, 0.02)
    expected = torch.exp(torch.log(torch.tensor(0.7, dtype=dtype)) * scores.abs() / denominator)
    torch.testing.assert_close(weights, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_v4_classification_and_weights_preserve_input_dtype(dtype):
    old, log_prob, advantages, entropy, mask = quadrant_batch(dtype)
    if dtype in (torch.float64, torch.bfloat16):
        # The current original helper raises on these input dtypes when it assigns
        # the mapped weights into its FP32 output. V4 calls that helper unchanged.
        with pytest.raises(RuntimeError, match="dtypes match"):
            compute_steer_v4_weights(
                advantages, entropy, old, log_prob, mask, token_weight_min=0.7, token_weight_max=1.0
            )
        with pytest.raises(RuntimeError, match="dtypes match"):
            compute_token_weights(advantages, entropy, old, log_prob, mask, 0.7, 1.0, linear=False)
        return
    actual, quadrants, other = compute_steer_v4_weights(
        advantages, entropy, old, log_prob, mask, token_weight_min=0.7, token_weight_max=1.0
    )
    expected = compute_token_weights(advantages, entropy, old, log_prob, mask, 0.7, 1.0, linear=False, mode="symmetric")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sum(int(quadrant.sum()) for quadrant in quadrants) + int(other.sum()) == int(mask.sum())


@pytest.mark.parametrize(
    "aggregation", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]
)
def test_v4_loss_and_gradient_match_original_steer(aggregation):
    old, log_prob, advantages, entropy, mask = quadrant_batch()
    enabled_config = actor_config(token_weight_min=0.7, token_weight_max=1.0)
    original_config = actor_config(
        steer_quadrant_logging=False,
        token_weight_min=0.7,
        token_weight_max=1.0,
    )
    original_log_prob = log_prob.detach().clone().requires_grad_()
    v4_log_prob = log_prob.detach().clone().requires_grad_()
    original_loss, _ = _policy_loss(old, original_log_prob, advantages, entropy, mask, original_config, aggregation)
    collector = SteerV4Metrics()
    v4_loss, _ = _policy_loss(old, v4_log_prob, advantages, entropy, mask, enabled_config, aggregation, collector)
    original_gradient = torch.autograd.grad(original_loss, original_log_prob)[0]
    v4_gradient = torch.autograd.grad(v4_loss, v4_log_prob)[0]
    torch.testing.assert_close(v4_loss, original_loss, rtol=0, atol=0)
    torch.testing.assert_close(v4_gradient, original_gradient, rtol=0, atol=0)
    assert collector.totals.dtype == torch.float32


def test_one_optimizer_step_matches_original_steer():
    old, initial_log_prob, advantages, entropy, mask = quadrant_batch()
    original_log_prob = torch.nn.Parameter(initial_log_prob.detach().clone())
    v4_log_prob = torch.nn.Parameter(initial_log_prob.detach().clone())
    original_optimizer = torch.optim.AdamW([original_log_prob], lr=1e-3, weight_decay=0.01)
    v4_optimizer = torch.optim.AdamW([v4_log_prob], lr=1e-3, weight_decay=0.01)
    original_loss, _ = _policy_loss(
        old, original_log_prob, advantages, entropy, mask, actor_config(steer_quadrant_logging=False)
    )
    v4_loss, _ = _policy_loss(old, v4_log_prob, advantages, entropy, mask, actor_config())
    original_loss.backward()
    v4_loss.backward()
    original_optimizer.step()
    v4_optimizer.step()
    torch.testing.assert_close(v4_log_prob, original_log_prob, rtol=0, atol=0)
    torch.testing.assert_close(
        v4_optimizer.state[v4_log_prob]["exp_avg"],
        original_optimizer.state[original_log_prob]["exp_avg"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "aggregation", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]
)
def test_active_statistics_match_autograd_at_clip_boundaries(aggregation):
    differences = torch.tensor([-21, -20, -1.2, -0.4, 0, 0.4, 1.2, 20, 21], dtype=torch.float64)
    ratios = torch.cat([differences.exp(), torch.tensor([0.8, 1.2, 3.0], dtype=torch.float64)])
    log_prob = ratios.log().repeat(4, 1).requires_grad_()
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=log_prob.dtype)[:, None].expand_as(log_prob)
    mask = torch.ones_like(log_prob, dtype=torch.bool)
    mask[0, -1] = False
    difference = log_prob.clamp(-20, 20)
    ratio = difference.exp()
    losses1 = -advantages * ratio
    losses2 = -advantages * ratio.clamp(0.8, 1.2)
    clipped = torch.maximum(losses1, losses2)
    losses = torch.where(advantages < 0, torch.minimum(-advantages * 3, clipped), clipped)
    gradient = torch.autograd.grad(agg_steer_loss(losses, mask, aggregation), log_prob)[0]

    quadrants = tuple(torch.zeros_like(mask) for _ in range(4))
    for index, quadrant in enumerate(quadrants):
        quadrant[index] = mask[index]
    other_mask = torch.zeros_like(mask)
    stats = compute_steer_v4_statistics(
        advantages=advantages,
        ratio=ratio,
        log_prob_difference=log_prob,
        losses1=losses1,
        losses2=losses2,
        clipped=clipped,
        response_mask=mask,
        weights=torch.ones_like(log_prob),
        quadrants=quadrants,
        other_mask=other_mask,
        clip_low=0.2,
        clip_high=0.2,
        clip_ratio_c=3,
    )
    assert stats[0] == mask.sum()
    assert stats[1] == gradient.count_nonzero()
    for index, quadrant in enumerate(quadrants):
        assert stats[2 + index] == quadrant.sum()
        assert stats[6 + index] == ((gradient != 0) & quadrant).sum()


def _statistics_for_slice(old, log_prob, advantages, entropy, mask, index):
    selected = slice(index, index + 1)
    local_old = old[:, selected]
    local_log_prob = log_prob[:, selected]
    local_advantages = advantages[:, selected]
    local_entropy = entropy[:, selected]
    local_mask = mask[:, selected]
    weights, quadrants, other = compute_steer_v4_weights(
        local_advantages,
        local_entropy,
        local_old,
        local_log_prob,
        local_mask,
        token_weight_min=0.7,
        token_weight_max=1.0,
    )
    difference = (local_log_prob - local_old).clamp(-20, 20)
    ratio = difference.exp()
    losses1 = -local_advantages * ratio
    losses2 = -local_advantages * ratio.clamp(0.8, 1.2)
    clipped = torch.maximum(losses1, losses2)
    return compute_steer_v4_statistics(
        advantages=local_advantages,
        ratio=ratio,
        log_prob_difference=local_log_prob - local_old,
        losses1=losses1,
        losses2=losses2,
        clipped=clipped,
        response_mask=local_mask,
        weights=weights,
        quadrants=quadrants,
        other_mask=other,
        clip_low=0.2,
        clip_high=0.2,
        clip_ratio_c=3,
    )


def test_statistics_merge_microbatches_and_simulated_dp(monkeypatch):
    old, log_prob, advantages, entropy, mask = quadrant_batch()
    ranks = [SteerV4Metrics(), SteerV4Metrics()]
    for index in range(6):
        ranks[index % 2].add(_statistics_for_slice(old, log_prob, advantages, entropy, mask, index))
    remote = ranks[1].totals.clone()
    calls = []

    def all_reduce(tensor, op, group):
        calls.append((op, group))
        tensor.add_(remote)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    metrics = ranks[0].flush(dp_group="actor-dp")
    assert metrics["steer/valid_count"] == 6
    assert metrics["steer/q1_frac"] == pytest.approx(1 / 6)
    assert metrics["steer/q2_frac"] == pytest.approx(1 / 6)
    assert metrics["steer/q3_frac"] == pytest.approx(1 / 6)
    assert metrics["steer/q4_frac"] == pytest.approx(1 / 6)
    assert metrics["steer/other_frac"] == pytest.approx(1 / 3)
    assert metrics["steer/active_count"] == 5
    assert metrics["steer/other_active_frac"] == pytest.approx(1 / 5)
    q_fraction_sum = sum(metrics[f"steer/q{index}_frac"] for index in range(1, 5))
    assert q_fraction_sum + metrics["steer/other_frac"] == pytest.approx(1)
    active_fraction_sum = sum(metrics[f"steer/q{index}_active_frac"] for index in range(1, 5))
    assert active_fraction_sum + metrics["steer/other_active_frac"] == pytest.approx(1)
    assert calls == [(torch.distributed.ReduceOp.SUM, "actor-dp")]
    assert ranks[0].totals.count_nonzero() == 0


def test_actor_worker_flushes_v4_metrics_with_existing_actor_metrics():
    from verl.workers.engine_workers import ActorRolloutRefWorker

    old, log_prob, advantages, entropy, mask = quadrant_batch()
    collector = SteerV4Metrics()
    for index in range(6):
        collector.add(_statistics_for_slice(old, log_prob, advantages, entropy, mask, index))

    def train_mini_batch(data):
        return tu.get_tensordict(
            tensor_dict={},
            non_tensor_dict={"metrics": {"actor/pg_loss": -0.1, "actor/grad_norm": 0.5}},
        )

    worker = SimpleNamespace(
        config=SimpleNamespace(actor=actor_config(), rollout=SimpleNamespace(response_length=4096)),
        actor=SimpleNamespace(
            train_mini_batch=train_mini_batch,
            engine=SimpleNamespace(get_data_parallel_group=lambda: None),
        ),
        steer_metrics=collector,
    )
    data = TensorDict({"sample": torch.ones(1)}, batch_size=[1])
    output = inspect.unwrap(ActorRolloutRefWorker.update_actor)(worker, data)
    metrics = tu.get(output, "metrics")
    assert metrics["actor/pg_loss"] == -0.1
    assert metrics["actor/grad_norm"] == 0.5
    assert metrics["steer/valid_count"] == 6
    assert metrics["steer/q1_frac"] == pytest.approx(1 / 6)


def test_v1_trainer_keeps_steer_tags_unprefixed():
    from omegaconf import OmegaConf

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "actor": {
                        "ppo_mini_batch_size": 1,
                        "ppo_epochs": 1,
                        "calculate_entropy": False,
                        "entropy_coeff": 0.0,
                        "data_loader_seed": 42,
                        "shuffle": False,
                    },
                    "rollout": {"n": 1, "temperature": 1.0},
                }
            }
        ),
        distillation_config=None,
        actor_rollout_wg=SimpleNamespace(
            update_actor=lambda batch: tu.get_tensordict(
                tensor_dict={},
                non_tensor_dict={"metrics": {"steer/q1_frac": 0.25, "actor/pg_loss": [-0.1], "mfu": [0.0]}},
            )
        ),
    )
    batch = SimpleNamespace(extra_info={})
    metrics = {}
    PPOTrainer._update_actor(trainer, batch, metrics)
    assert metrics["steer/q1_frac"] == 0.25
    assert metrics["actor/pg_loss"] == pytest.approx(-0.1)
    assert "actor/steer/q1_frac" not in metrics


def test_v0_trainer_keeps_steer_tags_unprefixed(monkeypatch):
    from omegaconf import OmegaConf

    from verl.protocol import DataProto
    from verl.trainer.ppo import ray_trainer

    captured = []

    def update_actor(batch):
        captured.append(batch)
        return tu.get_tensordict(
            tensor_dict={},
            non_tensor_dict={"metrics": {"steer/q1_frac": 0.25, "actor/pg_loss": [-0.1], "mfu": [0.0]}},
        )

    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "actor": {
                        "calculate_entropy": False,
                        "entropy_coeff": 0.0,
                        "ppo_mini_batch_size": 1,
                        "ppo_epochs": 1,
                        "data_loader_seed": 42,
                        "shuffle": False,
                    },
                    "rollout": {"n": 1, "temperature": 1.0, "multi_turn": {"enable": False}},
                }
            }
        ),
        distillation_config=None,
        actor_rollout_wg=SimpleNamespace(update_actor=update_actor),
    )
    monkeypatch.setattr(ray_trainer, "left_right_2_no_padding", lambda batch: batch)
    batch = DataProto.from_single_dict({"sample": torch.ones(1)})
    result = ray_trainer.RayPPOTrainer._update_actor(trainer, batch)
    metrics = result.meta_info["metrics"]
    assert len(captured) == 1
    assert metrics["steer/q1_frac"] == 0.25
    assert metrics["actor/pg_loss"] == [-0.1]
    assert metrics["perf/mfu/actor"] == [0.0]
    assert "actor/steer/q1_frac" not in metrics


def test_zero_denominators_keep_counts_and_skip_ratios():
    collector = SteerV4Metrics()
    collector.add(torch.zeros(17, dtype=torch.float32))
    assert collector.flush() == {"steer/valid_count": 0, "steer/active_count": 0}


def test_invalid_and_empty_inputs_are_other_and_keep_original_weights():
    old, log_prob, advantages, entropy, mask = quadrant_batch()
    advantages[0, 4] = float("nan")
    entropy[0, 4] = float("inf")
    expected = compute_token_weights(advantages, entropy, old, log_prob, mask, 0.7, 1.0, False, "symmetric")
    weights, quadrants, other = compute_steer_v4_weights(
        advantages, entropy, old, log_prob, mask, token_weight_min=0.7, token_weight_max=1.0
    )
    assert other[0, 4]
    assert other[0, 6] == 0
    assert not any(quadrant[0, 4] for quadrant in quadrants)
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    empty = compute_token_weights(advantages, entropy, old, log_prob, torch.zeros_like(mask), 0.7, 1.0, False)
    empty_weights, _, empty_other = compute_steer_v4_weights(
        advantages, entropy, old, log_prob, torch.zeros_like(mask), token_weight_min=0.7, token_weight_max=1.0
    )
    assert empty.count_nonzero() == empty_weights.count_nonzero() == 0
    assert not empty_other.any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steer_quadrant_logging": True, "loss_mode": "vanilla"},
        {"steer_quadrant_logging": True, "entropy_control_mode": "asymmetric"},
        {"steer_quadrant_logging": True, "linear": True},
        {"steer_quadrant_logging": 1},
    ],
)
def test_invalid_v4_configuration(kwargs):
    options = {"loss_mode": "entropy_control", "entropy_control_mode": "symmetric", "linear": False}
    options.update(kwargs)
    with pytest.raises(ValueError, match="STEER"):
        PolicyLossConfig(**options)


def test_hydra_v4_configuration():
    from omegaconf import OmegaConf

    config_dir = str(Path(__file__).resolve().parents[3] / "verl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.policy_loss.loss_mode=entropy_control",
                "actor_rollout_ref.actor.policy_loss.entropy_control_mode=symmetric",
                "actor_rollout_ref.actor.policy_loss.linear=false",
                "actor_rollout_ref.actor.policy_loss.steer_quadrant_logging=true",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
            ],
        )
    policy = config.actor_rollout_ref.actor.policy_loss
    assert policy.entropy_control_mode == "symmetric"
    assert policy.linear is False
    assert policy.steer_quadrant_logging is True
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert actor.policy_loss.steer_quadrant_logging is True
    validate_config(config, use_reference_policy=False, use_critic=False)

    invalid = OmegaConf.create(OmegaConf.to_container(config))
    invalid.trainer.use_v1 = False
    with pytest.raises(ValueError, match="V1 sync trainer"):
        validate_config(invalid, use_reference_policy=False, use_critic=False)


def test_tensorboard_keeps_original_and_v4_metrics(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from verl.utils.tracking import Tracking

    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path))
    tracker = Tracking("steer", "v4", default_backend=["tensorboard"])
    tracker.log({"actor/pg_loss": -0.1, "val/accuracy": 0.5, "steer/q1_frac": 0.25}, step=1)
    tracker.finish()
    tags = EventAccumulator(str(tmp_path)).Reload().Tags()["scalars"]
    assert sorted(tags) == sorted(["actor/pg_loss", "val/accuracy", "steer/q1_frac"])


def _distributed_statistics_worker(rank, rendezvous, output_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        old, log_prob, advantages, entropy, mask = quadrant_batch()
        collector = SteerV4Metrics()
        for index in range(rank, 6, 2):
            collector.add(_statistics_for_slice(old, log_prob, advantages, entropy, mask, index))
        metrics = collector.flush(dp_group=dist.group.WORLD)
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(metrics))
    finally:
        dist.destroy_process_group()


def test_two_process_dp_statistics(tmp_path):
    mp.spawn(
        _distributed_statistics_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    outputs = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert outputs[0] == pytest.approx(outputs[1])
    assert outputs[0]["steer/valid_count"] == 6
    assert outputs[0]["steer/other_frac"] == pytest.approx(1 / 3)
