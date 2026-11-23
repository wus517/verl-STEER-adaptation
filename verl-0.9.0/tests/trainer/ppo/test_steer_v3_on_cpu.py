# Copyright 2026 The verl team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Quadrant weights, actual PPO derivatives, and device-side logging windows."""

import ast
import inspect
import json
from dataclasses import replace
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
    compute_steer_v3_weights,
)
from verl.trainer.ppo.steer_metrics import SteerV3Metrics, compute_steer_v3_statistics
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss


def actor_config(**policy):
    return ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="entropy_control", entropy_control_mode="quadrant_v3", **policy),
    )


def scored_batch(scores, advantages=None, ratio=None):
    """Choose entropy so the released raw proxy has the requested signed score."""
    scores = torch.tensor([scores], dtype=torch.float64)
    advantage = torch.ones_like(scores) if advantages is None else torch.tensor([advantages], dtype=scores.dtype)
    logp = torch.full_like(scores, -1.0, requires_grad=True)
    ratio = torch.ones_like(scores) if ratio is None else torch.tensor([ratio], dtype=scores.dtype)
    old = logp.detach() - ratio.log()
    p = logp.detach().exp()
    entropy = -logp.detach() - scores * old.exp() / (advantage * p * (1 - p))
    return old, logp, advantage, entropy, torch.ones_like(scores, dtype=torch.bool)


@pytest.mark.parametrize("magnitude", [0.005, 0.08])
def test_q1_q3_match_original_exponential(magnitude):
    source = Path(__file__).resolve().parents[4] / "STEER-main/verl/trainer/ppo/core_algos.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_token_weights"
    )
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    old, logp, advantage, entropy, mask = scored_batch(
        [-magnitude / 2, magnitude, -magnitude / 3, magnitude / 4], [1, 1, -1, -1]
    )
    # FP32 assignment is part of the released helper's supported calling contract.
    old, logp, advantage, entropy = (value.float() for value in (old, logp, advantage, entropy))
    expected = scope["compute_token_weights"](advantage, entropy, old, logp, mask, 0.7, 1.0, False, "symmetric")
    weights, _, quadrants = compute_steer_v3_weights(advantage, entropy, old, logp, mask, token_weight_min=0.7)
    selected = quadrants[0] | quadrants[2]
    assert selected.sum() == 2
    torch.testing.assert_close(weights[selected], expected[selected], rtol=0, atol=0)
    assert weights[quadrants[3]].item() == 1.0


def test_q2_raw_score_examples_and_independence():
    old, logp, advantage, entropy, mask = scored_batch([0.005, 0.020, 0.080, -0.4])
    weights, scores, quadrants = compute_steer_v3_weights(advantage, entropy, old, logp, mask, token_weight_min=0.7)
    torch.testing.assert_close(weights[quadrants[1]], torch.tensor([1.16, 1.10, 1.04]))
    torch.testing.assert_close(scores[0, :3], torch.tensor([0.005, 0.020, 0.080], dtype=scores.dtype))
    entropy[0, 3] += 100
    changed, _, _ = compute_steer_v3_weights(advantage, entropy, old, logp, mask, token_weight_min=0.7)
    torch.testing.assert_close(changed[0, :3], weights[0, :3], rtol=0, atol=0)
    assert not weights.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_boundaries_padding_nonfinite_and_empty(dtype):
    logp = torch.full((1, 7), -1.0, dtype=dtype)
    advantage = torch.tensor([[1, -1, 0, 1, 1, -1, 1]], dtype=dtype)
    entropy = torch.tensor([[1, 1, 2, 0, float("nan"), 2, 100]], dtype=dtype)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    weights, scores, quadrants = compute_steer_v3_weights(advantage, entropy, logp, logp, mask, token_weight_min=0.7)
    torch.testing.assert_close(weights[0, :3], torch.ones(3))
    assert weights[0, 4] == 1
    assert scores[0, 4] == 0
    assert weights[0, -1] == 0
    assert not any(quadrant[0, 4] for quadrant in quadrants)
    empty, empty_scores, empty_quadrants = compute_steer_v3_weights(
        advantage, entropy, logp, logp, torch.zeros_like(mask), token_weight_min=0.7
    )
    assert empty.count_nonzero() == empty_scores.count_nonzero() == 0
    assert not any(quadrant.any() for quadrant in empty_quadrants)


@pytest.mark.parametrize(
    "aggregation", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]
)
def test_statistics_match_autograd_at_clipping_boundaries(aggregation):
    # Float64 epsilon makes 1 +/- .25 and dual-clip=3 exact boundaries.
    differences = torch.tensor([-21, -20, -1.5, -0.5, 0, 0.5, 1.5, 20, 21], dtype=torch.float64)
    ratios = torch.cat([differences.exp(), torch.tensor([0.75, 1.25, 3.0], dtype=torch.float64)])
    logp = ratios.log().repeat(4, 1).requires_grad_()
    advantage = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=logp.dtype)[:, None].expand_as(logp)
    mask = torch.ones_like(logp, dtype=torch.bool)
    mask[0, -1] = False
    ratio = logp.clamp(-20, 20).exp()
    loss1 = -advantage * ratio
    loss2 = -advantage * ratio.clamp(0.75, 1.25)
    clipped = torch.maximum(loss1, loss2)
    losses = torch.where(advantage < 0, torch.minimum(-advantage * 3, clipped), clipped)
    scale = 0.25
    gradient = torch.autograd.grad(agg_steer_loss(losses, mask, aggregation) * scale, logp)[0]
    quadrants = tuple(torch.zeros_like(mask) for _ in range(4))
    for index, quadrant in enumerate(quadrants):
        quadrant[index] = mask[index]
    weights = torch.linspace(1.01, 1.19, logp.numel()).reshape(logp.shape)
    stats = compute_steer_v3_statistics(
        advantages=advantage,
        ratio=ratio,
        log_prob_difference=logp,
        losses1=loss1,
        losses2=loss2,
        clipped=clipped,
        response_mask=mask,
        weights=weights,
        scores=torch.zeros_like(logp),
        quadrants=quadrants,
        clip_low=0.25,
        clip_high=0.25,
        clip_ratio_c=3,
        loss_agg_mode=aggregation,
        accumulation_scale=scale,
        q2_boost_scale=0.02,
    )
    assert stats[0] == gradient.count_nonzero()
    for index, quadrant in enumerate(quadrants, start=1):
        assert stats[index] == ((gradient != 0) & quadrant).sum()
    q2 = quadrants[1]
    torch.testing.assert_close(stats[7].double(), gradient[q2].abs().sum(), rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(stats[6].double(), ((weights[q2] - 1) * gradient[q2].abs()).sum(), rtol=1e-6, atol=1e-8)


def test_document_logging_example_split_and_collective(monkeypatch):
    old, logp, advantage, entropy, mask = scored_batch([0.005, 0.020, 0.080, 0.04], [1, 3, 6, 1], [1, 1, 1, 1.5])
    _, scores, _ = compute_steer_v3_weights(advantage, entropy, old, logp, mask, token_weight_min=0.7)
    config = actor_config(token_weight_min=0.7, q2_boost_scale=scores[0, 1].item())
    complete = SteerV3Metrics(q2_boost_strength=0.2)
    loss, _ = compute_policy_loss_with_entropy(
        old, logp, advantage, mask, config=config, entropys=entropy, steer_metrics=complete
    )
    loss.backward()
    assert logp.grad[0, -1] == 0
    expected = complete.flush()
    assert expected["steer/q2_active_count"] == 3
    assert expected["steer/q2_boost_mean"] == pytest.approx(0.10, abs=1e-7)
    assert expected["steer/q2_effective_boost"] == pytest.approx(0.07, abs=1e-7)
    assert expected["steer/q2_clip_frac"] == 0.25
    assert expected["steer/q2_at_least_half_boost_frac"] == pytest.approx(2 / 3)
    ranks = [SteerV3Metrics(q2_boost_strength=0.2) for _ in range(2)]
    for index in range(4):
        compute_policy_loss_with_entropy(
            old[:, index : index + 1],
            logp[:, index : index + 1],
            advantage[:, index : index + 1],
            mask[:, index : index + 1],
            config=config,
            entropys=entropy[:, index : index + 1],
            steer_metrics=ranks[index % 2],
            accumulation_scale=0.25,
        )
    remote = ranks[1].totals.clone()
    calls = []

    def all_reduce(tensor, op, group):
        calls.append((op, group))
        tensor.add_(remote)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    assert ranks[0].flush(dp_group="test-dp") == pytest.approx(expected, abs=1e-7)
    assert calls == [(torch.distributed.ReduceOp.SUM, "test-dp")]
    assert ranks[0].totals.count_nonzero() == 0


@pytest.mark.parametrize("cached_key,nested", [("entropy", True), ("entropys", False)])
@pytest.mark.parametrize("dynamic", [True, False])
@pytest.mark.parametrize(
    "aggregation", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]
)
def test_bridge_effective_boost_uses_real_accumulated_gradients(cached_key, nested, dynamic, aggregation):
    config = replace(actor_config(token_weight_min=0.7), loss_agg_mode=aggregation, use_dynamic_bsz=dynamic)
    old, logp, advantage, entropy, mask = scored_batch([0.005, 0.02, 0.08], [1, 3, 6])
    collector = SteerV3Metrics(q2_boost_strength=0.2)
    gradient_sum = 0
    boosted_gradient_sum = 0
    for selected in (slice(0, 1), slice(1, 3)):
        cached = entropy[:, selected].detach().clone().requires_grad_()
        response_mask = mask[:, selected]
        length = response_mask.shape[1]
        model_logp = torch.cat([logp[0, selected].detach().clone(), torch.zeros(1)]).requires_grad_()
        data = TensorDict(
            {
                "prompts": torch.ones(1, 1, dtype=torch.long),
                "responses": torch.ones(1, length, dtype=torch.long),
                "attention_mask": torch.ones(1, length + 1, dtype=torch.long),
                "response_mask": response_mask,
                "old_log_probs": old[:, selected],
                "advantages": advantage[:, selected],
                cached_key: cached,
            },
            batch_size=[1],
        )
        if nested:
            for key in ("response_mask", "old_log_probs", "advantages", cached_key):
                data[key] = torch.nested.as_nested_tensor(list(data[key].unbind()), layout=torch.jagged)
        tu.assign_non_tensor(
            data,
            dp_size=1,
            batch_num_tokens=length,
            global_batch_size=2,
            steer_local_mini_batch_size=2,
            steer_response_length=3,
        )
        current_entropy = torch.ones_like(model_logp, requires_grad=True)
        loss, _ = ppo_loss(config, {"log_probs": model_logp, "entropy": current_entropy}, data, steer_metrics=collector)
        loss.backward()
        weights, _, _ = compute_steer_v3_weights(
            advantage[:, selected],
            cached,
            old[:, selected],
            model_logp[:length].unsqueeze(0),
            response_mask,
            token_weight_min=0.7,
        )
        original_gradient = model_logp.grad[:length].unsqueeze(0).abs() / weights
        gradient_sum += original_gradient.sum().item()
        boosted_gradient_sum += ((weights - 1) * original_gradient).sum().item()
        assert cached.grad is None
        assert current_entropy.grad is None
    metrics = collector.flush()
    assert metrics["steer/q2_effective_boost"] == pytest.approx(boosted_gradient_sum / gradient_sum, abs=1e-7)


def test_worker_logging_interval_and_final_flush():
    from verl.workers.engine_workers import ActorRolloutRefWorker

    collector = SteerV3Metrics(q2_boost_strength=0)
    config = actor_config(steer_log_interval=3, q2_boost_strength=0)

    def train_mini_batch(data):
        collector.add(torch.tensor([1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1], dtype=torch.float32))
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"loss": 0.0}})

    worker = SimpleNamespace(
        config=SimpleNamespace(actor=config, rollout=SimpleNamespace(response_length=4096)),
        actor=SimpleNamespace(
            train_mini_batch=train_mini_batch, engine=SimpleNamespace(get_data_parallel_group=lambda: None)
        ),
        steer_metrics=collector,
    )
    update = inspect.unwrap(ActorRolloutRefWorker.update_actor)
    for step in range(1, 6):
        data = TensorDict({"sample": torch.ones(1)}, batch_size=[1])
        tu.assign_non_tensor(data, steer_log_step=step, steer_log_flush=step == 5)
        output = update(worker, data)
        metrics = tu.get(output, "metrics")
        assert ("steer/q2_active_count" in metrics) == (step in (3, 5))
        if step in (3, 5):
            assert metrics["steer/q2_active_count"] == (3 if step == 3 else 2)
            assert "steer/q2_at_least_half_boost_frac" not in metrics


def test_zero_denominators_skip_ratios():
    collector = SteerV3Metrics(q2_boost_strength=0.2)
    collector.add(torch.zeros(11))
    assert collector.flush() == {"steer/q2_active_count": 0}


def test_active_probability_boundary_and_zero_boost():
    old, logp, advantage, entropy, mask = scored_batch([0.005, 0, -0.04, 0.03], [1, 1, -1, -1])
    config = actor_config(q2_boost_strength=0)
    collector = SteerV3Metrics(q2_boost_strength=0)
    loss, _ = compute_policy_loss_with_entropy(
        old, logp, advantage, mask, config=config, entropys=entropy, steer_metrics=collector
    )
    loss.backward()
    assert logp.grad.count_nonzero() == 4
    metrics = collector.flush()
    assert metrics["steer/q2_active_count"] == 1
    assert metrics["steer/q2_boost_mean"] == metrics["steer/q2_effective_boost"] == 0
    assert sum(metrics[f"steer/q{index}_active_frac"] for index in range(1, 5)) == 0.75
    assert "steer/q2_at_least_half_boost_frac" not in metrics


def _distributed_statistics_worker(rank, rendezvous, output_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        old, logp, advantage, entropy, mask = scored_batch([0.005, 0.020, 0.080, 0.04], [1, 3, 6, 1], [1, 1, 1, 1.5])
        _, scores, _ = compute_steer_v3_weights(advantage, entropy, old, logp, mask, token_weight_min=0.7)
        config = actor_config(q2_boost_scale=scores[0, 1].item())
        collector = SteerV3Metrics(q2_boost_strength=0.2)
        for index in range(rank, 4, 2):
            selected = slice(index, index + 1)
            compute_policy_loss_with_entropy(
                old[:, selected],
                logp[:, selected],
                advantage[:, selected],
                mask[:, selected],
                config=config,
                entropys=entropy[:, selected],
                steer_metrics=collector,
                accumulation_scale=0.25,
            )
        metrics = collector.flush(dp_group=dist.group.WORLD)
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(metrics))
    finally:
        dist.destroy_process_group()


def test_distributed_statistics_match_document_example(tmp_path):
    mp.spawn(_distributed_statistics_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    expected = {
        "steer/q1_active_frac": 0,
        "steer/q2_active_frac": 1,
        "steer/q3_active_frac": 0,
        "steer/q4_active_frac": 0,
        "steer/q2_active_count": 3,
        "steer/q2_boost_mean": 0.10,
        "steer/q2_effective_boost": 0.07,
        "steer/q2_clip_frac": 0.25,
        "steer/q2_at_least_half_boost_frac": 2 / 3,
    }
    for rank in range(2):
        assert json.loads((tmp_path / f"rank-{rank}.json").read_text()) == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("v1", [False, True])
def test_trainer_logging_metadata_and_top_level_tags(monkeypatch, v1):
    from omegaconf import OmegaConf

    from verl.protocol import DataProto
    from verl.trainer.ppo import ray_trainer

    captured = []

    def update_actor(data):
        captured.append(data)
        return tu.get_tensordict(
            tensor_dict={},
            non_tensor_dict={"metrics": {"steer/q2_active_count": 3, "mfu": [0.0], "actor/pg_loss": [-1.0]}},
        )

    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "actor": {
                        "ppo_mini_batch_size": 1,
                        "ppo_epochs": 1,
                        "calculate_entropy": False,
                        "entropy_coeff": 0,
                        "data_loader_seed": 42,
                        "shuffle": False,
                    },
                    "rollout": {"n": 1, "temperature": 1.0, "multi_turn": {"enable": False}},
                }
            }
        ),
        distillation_config=None,
        global_steps=5,
        total_training_steps=5,
        actor_rollout_wg=SimpleNamespace(update_actor=update_actor),
    )
    if v1:
        from verl.trainer.ppo.v1.trainer_base import PPOTrainer

        batch = SimpleNamespace(extra_info={})
        metrics = {}
        PPOTrainer._update_actor(trainer, batch, metrics)
        metadata = batch.extra_info
    else:
        monkeypatch.setattr(ray_trainer, "left_right_2_no_padding", lambda data: data)
        batch = DataProto.from_single_dict({"sample": torch.ones(1)})
        result = ray_trainer.RayPPOTrainer._update_actor(trainer, batch)
        metrics = result.meta_info["metrics"]
        metadata = {
            key: tu.get_non_tensor_data(captured[0], key, default=None) for key in ("steer_log_step", "steer_log_flush")
        }
    assert metadata["steer_log_step"] == 5
    assert metadata["steer_log_flush"] is True
    assert metrics["steer/q2_active_count"] == 3
    assert "actor/steer/q2_active_count" not in metrics
    assert "actor/pg_loss" in metrics


@pytest.mark.parametrize(
    "policy",
    [
        {"q2_boost_strength": -1},
        {"q2_boost_strength": float("inf")},
        {"q2_boost_strength": float("nan")},
        {"q2_boost_scale": 0},
        {"q2_boost_scale": -1},
        {"q2_boost_scale": float("inf")},
        {"q2_boost_scale": float("nan")},
        {"steer_log_interval": 0},
        {"steer_log_interval": 1.5},
        {"steer_log_interval": True},
    ],
)
def test_invalid_v3_configuration(policy):
    with pytest.raises(ValueError, match="STEER"):
        actor_config(**policy)


def test_hydra_v3_configuration():
    config_dir = str(Path(__file__).resolve().parents[3] / "verl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.policy_loss.loss_mode=entropy_control",
                "actor_rollout_ref.actor.policy_loss.entropy_control_mode=quadrant_v3",
                "actor_rollout_ref.actor.policy_loss.linear=false",
                "actor_rollout_ref.actor.policy_loss.q2_boost_strength=0.20",
                "actor_rollout_ref.actor.policy_loss.q2_boost_scale=0.02",
                "actor_rollout_ref.actor.policy_loss.steer_log_interval=3",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            ],
        )
    policy = omega_conf_to_dataclass(config.actor_rollout_ref.actor).policy_loss
    assert policy.entropy_control_mode == "quadrant_v3"
    assert policy.q2_boost_strength == 0.2
    assert policy.q2_boost_scale == 0.02
    assert policy.steer_log_interval == 3


@pytest.mark.parametrize("mode", ["quadrant_v3", "symmetric"])
def test_tensorboard_tags(tmp_path, monkeypatch, mode):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from verl.utils.tracking import Tracking

    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path))
    tracker = Tracking(
        "steer",
        "test",
        default_backend=["tensorboard"],
        config={
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "loss_mode": "entropy_control",
                        "entropy_control_mode": mode,
                    }
                }
            }
        },
    )
    tracker.log({"steer/q2_active_count": 3, "actor/pg_loss": -0.1, "val/accuracy": 0.5}, step=1)
    tracker.finish()
    tags = EventAccumulator(str(tmp_path)).Reload().Tags()["scalars"]
    expected = (
        ["steer/q2_active_count"]
        if mode == "quadrant_v3"
        else [
            "steer/q2_active_count",
            "actor/pg_loss",
            "val/accuracy",
        ]
    )
    assert sorted(tags) == sorted(expected)
