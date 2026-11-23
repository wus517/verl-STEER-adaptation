# Copyright 2026 The verl team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""STEER migration: numerical parity, gradients, and the actual loss bridge."""

import ast
import inspect
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    compute_policy_loss_with_entropy,
    compute_token_weights,
    get_policy_loss_fn,
)
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.metric import AggregationType
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import response_from_nested


def _config(**kwargs):
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        clip_ratio_c=10,
        policy_loss=PolicyLossConfig(loss_mode="entropy_control", **kwargs),
    )


def _batch():
    old = torch.tensor([[-1.0, -0.8, -0.4, -4.0], [-2.0, -2.0, -0.1, -1.0]])
    logp = (old + torch.tensor([[0.1, 0.5, -0.5, 3.0], [-0.1, 0.2, 0.0, 0.3]])).requires_grad_()
    advantage = torch.tensor([[1.0, 2.0, -1.0, -2.0], [-1.0, 0.5, 0.0, 100.0]])
    entropy = torch.tensor([[1.2, 0.4, 0.9, 1.4], [2.0, 1.0, 0.3, 2.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    return old, logp, advantage, entropy, mask


@pytest.fixture
def legacy():
    source = Path(__file__).resolve().parents[4] / "STEER-main/verl/trainer/ppo/core_algos.py"
    if not source.exists():
        pytest.skip("Original STEER reference checkout is not present")
    names = {"agg_loss", "compute_token_weights", "compute_policy_loss_with_entropy", "kl_penalty"}
    functions = [
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    for node in functions:
        node.decorator_list = []
    functional_source = source.parents[2] / "utils/torch_functional.py"
    reductions = [
        node
        for node in ast.parse(functional_source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in {"masked_sum", "masked_mean"}
    ]
    reduction_scope = {"torch": torch}
    exec(compile(ast.Module(body=reductions, type_ignores=[]), str(functional_source), "exec"), reduction_scope)
    scope = {
        "torch": torch,
        "verl_F": SimpleNamespace(**{name: reduction_scope[name] for name in ("masked_sum", "masked_mean")}),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    return scope


@pytest.mark.parametrize("linear", [True, False])
@pytest.mark.parametrize("mode", ["symmetric", "asymmetric"])
@pytest.mark.parametrize(
    "aggregation", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]
)
def test_released_implementation_parity(legacy, linear, mode, aggregation):
    old, logp, advantage, entropy, mask = _batch()
    config = _config(linear=linear, entropy_control_mode=mode, token_weight_min=0.8, token_weight_max=1.2)
    # Global 0.9.0 normalization must not leak into the old local loss.
    config.global_batch_info.update(dp_size=4, batch_num_tokens=100, global_batch_size=16)
    expected = legacy["compute_token_weights"](advantage, entropy, old, logp, mask, 0.8, 1.2, linear, mode)
    weights = compute_token_weights(advantage, entropy, old, logp, mask, 0.8, 1.2, linear, mode)
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    reference_loss = legacy["compute_policy_loss_with_entropy"](
        old,
        logp,
        advantage,
        mask,
        entropy,
        cliprange_low=0.2,
        cliprange_high=0.28,
        clip_ratio_c=10,
        token_weight_min=0.8,
        token_weight_max=1.2,
        linear=linear,
        entropy_control_mode=mode,
        loss_agg_mode=aggregation,
    )[0]
    loss, _ = compute_policy_loss_with_entropy(old, logp, advantage, mask, aggregation, config, entropys=entropy)
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(loss, logp)[0], torch.autograd.grad(reference_loss, logp)[0])
    assert not weights.requires_grad
    assert entropy.grad is None


def test_exponential_proxy_floor_and_direction():
    old = torch.tensor([[-1.0, -1.0]])
    logp = old.clone().requires_grad_()
    entropy = torch.ones_like(old, requires_grad=True) * 0.99
    mask = torch.ones_like(old, dtype=torch.bool)
    advantage = torch.tensor([[1.0, -1.0]])
    weights = compute_token_weights(advantage, entropy, old, logp, mask, 0.8, 1.0, False)
    metric = (1 - old.exp()) * 0.01
    torch.testing.assert_close(weights, torch.exp(torch.log(torch.tensor(0.8)) * metric / 0.02))
    assert weights.min() > 0.8
    directional = compute_token_weights(advantage, entropy, old, logp, mask, 0.8, 1.0, False, "asymmetric")
    assert directional[0, 0] == 1  # log p + H < 0, A > 0 -> entropy increase
    assert directional[0, 1] < 1


def test_defaults_match_old_actor_and_helper(legacy):
    actor_defaults = PolicyLossConfig(loss_mode="entropy_control")
    assert (actor_defaults.token_weight_min, actor_defaults.token_weight_max, actor_defaults.linear) == (
        0.95,
        1.05,
        True,
    )
    old, logp, advantage, entropy, mask = _batch()
    torch.testing.assert_close(
        compute_token_weights(advantage, entropy, old, logp, mask),
        legacy["compute_token_weights"](advantage, entropy, old, logp, mask),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("linear", [False, True])
def test_nonfinite_proxy_and_empty_mask_match_old(legacy, linear, capsys):
    old, logp, advantage, entropy, mask = _batch()
    entropy = entropy.detach().clone()
    entropy[0, 0] = torch.nan
    expected = legacy["compute_token_weights"](advantage, entropy, old, logp, mask, linear=linear)
    actual = compute_token_weights(advantage, entropy, old, logp, mask, linear=linear)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "non-finite" in capsys.readouterr().out
    empty_mask = torch.zeros_like(mask)
    assert compute_token_weights(advantage, entropy, old, logp, empty_mask).eq(0).all()
    # The released loss's min/max statistics do not support empty micro-batches.
    with pytest.raises(RuntimeError):
        compute_policy_loss_with_entropy(old, logp, advantage, empty_mask, config=_config(), entropys=entropy)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_input_dtype_math_matches_old(legacy, dtype):
    old, logp, advantage, entropy, mask = _batch()
    # Advantages are FP32 as in the old GRPO pipeline; low precision p/log(p)
    # is intentionally evaluated before it is promoted by the advantages.
    old, logp, entropy = [t.detach().to(dtype) for t in (old, logp, entropy)]
    actual = compute_token_weights(advantage, entropy, old, logp, mask, 0.8, 1.0, False)
    expected = legacy["compute_token_weights"](advantage, entropy, old, logp, mask, 0.8, 1.0, False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_unsupported_all_same_dtype_matches_old(legacy, dtype):
    tensors = [tensor.detach().to(dtype) for tensor in _batch()[:4]]
    old, logp, advantage, entropy = tensors
    mask = _batch()[4]
    # The old helper assigns to an FP32 output without casting. Preserve the
    # failure too; do not silently claim arbitrary input dtypes are supported.
    for function in (legacy["compute_token_weights"], compute_token_weights):
        with pytest.raises(RuntimeError, match="dtypes match"):
            function(advantage, entropy, old, logp, mask)


@pytest.mark.parametrize("cached_key,nested", [("entropy", True), ("entropys", False)])
def test_ppo_bridge_response_alignment_and_entropy_gradients(cached_key, nested):
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        entropy_coeff=0.01,
        policy_loss=PolicyLossConfig(loss_mode="entropy_control"),
    )
    # Full sequence length 5, prompt length 2, response length 3. Model
    # outputs at positions 1:4 predict the responses; cached entropy is already aligned.
    model_logp = torch.tensor([-0.1, -1.0, -0.7, -0.5, -0.2], requires_grad=True)
    current_entropy = torch.tensor([0.1, 1.0, 0.8, 0.5, 0.2], requires_grad=True)
    cached = torch.tensor([[1.5, 1.4, 1.3]], requires_grad=True)
    response = torch.ones(1, 3, dtype=torch.bool)
    data = TensorDict(
        {
            "prompts": torch.tensor([[1, 2]]),
            "responses": torch.tensor([[3, 4, 5]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "response_mask": response,
            "old_log_probs": torch.tensor([[-1.0, -0.8, -0.6]]),
            "advantages": torch.tensor([[1.0, -1.0, 1.0]]),
            cached_key: cached,
        },
        batch_size=[1],
    )
    if nested:
        for field in ("response_mask", "old_log_probs", "advantages", cached_key):
            data[field] = torch.nested.as_nested_tensor(list(data[field].unbind()), layout=torch.jagged)
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=3, global_batch_size=1, steer_local_mini_batch_size=1)
    loss, metrics = ppo_loss(config, {"log_probs": model_logp, "entropy": current_entropy}, data)
    expected, _ = compute_policy_loss_with_entropy(
        torch.tensor([[-1.0, -0.8, -0.6]]),
        model_logp[1:4].unsqueeze(0),
        torch.tensor([[1.0, -1.0, 1.0]]),
        response,
        config=config,
        entropys=cached,
    )
    torch.testing.assert_close(loss, expected - 0.01 * cached.mean())
    loss.backward()
    assert cached.grad is None
    assert current_entropy.grad is None
    assert metrics["actor/token_weights_min"].aggregation == AggregationType.MIN
    assert metrics["actor/token_weights_max"].aggregation == AggregationType.MAX
    with pytest.raises(ValueError, match="cached old-policy entropy"):
        ppo_loss(config, {"log_probs": model_logp}, data.exclude(cached_key))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"token_weight_min": 0},
        {"token_weight_min": 1.1},
        {"token_weight_min": float("nan")},
        {"token_weight_max": 0.7},
        {"entropy_control_mode": "typo"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError, match="STEER"):
        _config(**kwargs)


def test_variable_length_nested_old_entropy_round_trip():
    # Exercise the same full-sequence -> cached response -> loss conversion as
    # the V1 trainer, with different prompt and response lengths in one batch.
    nested = lambda rows: torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    prompts = nested([torch.tensor([1, 2]), torch.tensor([6, 7, 8])])
    responses = nested([torch.tensor([3, 4, 5]), torch.tensor([9, 10])])
    mask = nested([torch.ones(3, dtype=torch.bool), torch.ones(2, dtype=torch.bool)])
    old_full = nested([torch.tensor([-0.1, -1.0, -0.8, -0.6, -0.2]), torch.tensor([-0.1, -0.2, -1.2, -0.9, -0.3])])
    entropy_full = nested([torch.tensor([0.1, 1.5, 1.4, 1.3, 0.2]), torch.tensor([0.1, 0.2, 1.7, 1.6, 0.3])])
    old_response = response_from_nested(old_full, mask)
    cached = response_from_nested(entropy_full, mask)
    logp = torch.tensor([-0.1, -0.9, -0.7, -0.5, -0.2, -0.1, -0.2, -1.1, -0.8, -0.3], requires_grad=True)
    advantages = torch.tensor([[1.0, -1.0, 0.5], [-1.0, 1.0, 0.0]])
    data = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "response_mask": mask,
            "old_log_probs": old_response,
            "entropy": cached,
            "advantages": nested([advantages[0], advantages[1, :2]]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=5, global_batch_size=2, steer_local_mini_batch_size=2)
    config = _config()
    config.ppo_micro_batch_size_per_gpu = 2
    loss, _ = ppo_loss(config, {"log_probs": logp}, data)
    expected, _ = compute_policy_loss_with_entropy(
        torch.tensor([[-1.0, -0.8, -0.6], [-1.2, -0.9, 0.0]]),
        torch.stack([logp[1:4], torch.cat([logp[7:9], logp.new_zeros(1)])]),
        advantages,
        torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        config=config,
        entropys=torch.tensor([[1.5, 1.4, 1.3], [1.7, 1.6, 0.0]]),
    )
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, logp)[0], torch.autograd.grad(expected, logp)[0])


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "aggregation", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]
)
def test_loss_bridge_accumulated_gradients_match_old(legacy, dp_size, dynamic, aggregation):
    # Unequal response lengths expose the difference between the old local
    # micro-batch mean and 0.9.0's global token mean. Simulate FSDP's DP mean.
    lengths = [1, 3, 2, 3, 1, 2][: 3 * dp_size]
    old = torch.tensor([[-0.8, -1.2, -1.5]] * len(lengths))
    logp = (old + torch.tensor([[0.05, -0.1, 0.15]])).requires_grad_()
    advantages = torch.tensor([[1.0, -0.5, 0.75]] * len(lengths))
    cached = torch.tensor([[1.1, 1.8, 0.7]] * len(lengths))
    config = replace(
        _config(token_weight_min=0.8, token_weight_max=1.0, linear=False),
        use_dynamic_bsz=dynamic,
        loss_agg_mode=aggregation,
        entropy_coeff=0.03,
        use_kl_loss=True,
        kl_loss_type="low_var_kl",
        kl_loss_coef=0.02,
        loss_scale_factor=999,  # New normalization settings must not leak in.
    )
    actual, expected = [], []
    nested = lambda rows: torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    for rank in range(dp_size):
        groups = [[0, 1], [2]] if dynamic else [[0], [1], [2]]
        for group in groups:
            indices = [rank * 3 + index for index in group]
            rows = lambda values, indices=indices: [values[index, : lengths[index]] for index in indices]
            mask_rows = [torch.ones(lengths[index], dtype=torch.bool) for index in indices]
            data = TensorDict(
                {
                    "prompts": nested([torch.ones(1, dtype=torch.long) for _ in indices]),
                    "responses": nested([torch.ones(lengths[index], dtype=torch.long) for index in indices]),
                    "response_mask": nested(mask_rows),
                    "old_log_probs": nested(rows(old)),
                    "advantages": nested(rows(advantages)),
                    "entropy": nested(rows(cached)),
                    "ref_log_prob": nested(rows(old - 0.2)),
                },
                batch_size=[len(indices)],
            )
            tu.assign_non_tensor(
                data,
                dp_size=dp_size,
                batch_num_tokens=sum(lengths),
                global_batch_size=len(lengths),
                steer_local_mini_batch_size=3,
                steer_response_length=7,
            )
            model_logp = torch.cat([torch.cat([row, row.new_zeros(1)]) for row in rows(logp)])
            loss, _ = ppo_loss(config, {"log_probs": model_logp}, data)
            actual.append(loss / dp_size)
            pad = lambda tensors: torch.stack([torch.nn.functional.pad(row, (0, 7 - len(row))) for row in tensors])
            old_p, current_p, adv, entropy, mask, ref = [
                pad(tensors)
                for tensors in (rows(old), rows(logp), rows(advantages), rows(cached), mask_rows, rows(old - 0.2))
            ]
            reference = legacy["compute_policy_loss_with_entropy"](
                old_p,
                current_p,
                adv,
                mask,
                entropy,
                cliprange_low=0.2,
                cliprange_high=0.28,
                clip_ratio_c=10,
                token_weight_min=0.8,
                token_weight_max=1.0,
                linear=False,
                loss_agg_mode=aggregation,
            )[0]
            reference = reference - 0.03 * legacy["agg_loss"](entropy, mask, aggregation)
            reference = reference + 0.02 * legacy["agg_loss"](
                legacy["kl_penalty"](current_p, ref, "low_var_kl"), mask, aggregation
            )
            scale = len(indices) / 3 if dynamic else 1 / 3
            expected.append(reference * scale / dp_size)
    actual, expected = sum(actual), sum(expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, logp)[0], torch.autograd.grad(expected, logp)[0])


@pytest.mark.parametrize("dp_size", [1, 2])
def test_training_worker_records_actual_local_optimizer_batch(dp_size):
    from verl.workers.engine_workers import TrainingWorker

    captured = []
    worker = SimpleNamespace(
        engine=SimpleNamespace(
            get_data_parallel_size=lambda: dp_size,
            get_data_parallel_rank=lambda: 0,
            train_mode=lambda **_: nullcontext(),
            is_mp_src_rank_with_outputs=lambda: False,
        ),
        train_batch=lambda data: captured.append(data),
        profiler=SimpleNamespace(step=lambda: None),
    )
    data = TensorDict({"sample": torch.arange(6)}, batch_size=[6])
    tu.assign_non_tensor(data, mini_batch_size=3 * dp_size, epochs=2, steer_legacy_normalization=True)
    inspect.unwrap(TrainingWorker.train_mini_batch)(worker, data)
    assert len(captured) == 4
    for batch in captured:
        assert len(batch) == 3
        assert tu.get_non_tensor_data(batch, "steer_local_mini_batch_size", default=None) == 3


def test_actor_update_passes_old_entropy_and_horizon_metadata():
    from verl.workers.engine_workers import ActorRolloutRefWorker

    captured = []
    worker = SimpleNamespace(
        config=SimpleNamespace(actor=_config(), rollout=SimpleNamespace(response_length=17)),
        actor=SimpleNamespace(train_mini_batch=lambda data: captured.append(data)),
    )
    data = TensorDict({"sample": torch.arange(2)}, batch_size=[2])
    tu.assign_non_tensor(data, calculate_entropy=True)
    inspect.unwrap(ActorRolloutRefWorker.update_actor)(worker, data)
    assert captured == [data]
    assert tu.get_non_tensor_data(data, "steer_legacy_normalization", default=None)
    assert tu.get_non_tensor_data(data, "steer_response_length", default=None) == 17
    assert tu.get_non_tensor_data(data, "calculate_entropy", default=None) is False


@pytest.mark.parametrize("mode", ["symmetric", "quadrant_v3"])
def test_engine_backward_keeps_old_local_token_mean(monkeypatch, mode):
    from verl.trainer.ppo.steer_metrics import SteerV3Metrics
    from verl.workers.engine.fsdp import transformer_impl
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    # Real engine splitting and backward, with model/collectives replaced by CPU
    # fixtures. The 1-token and 3-token responses each receive half the update.
    engine = object.__new__(FSDPEngine)
    engine.engine_config = SimpleNamespace(grad_sync_every_micro_batch=False)
    engine.ulysses_sequence_parallel_size = 1
    engine.scaler = None
    engine.get_data_parallel_group = lambda: None
    engine.get_data_parallel_size = lambda: 1
    engine._gradient_sync_context = lambda **_: nullcontext()
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    config = _config(token_weight_min=1.0, token_weight_max=1.0, entropy_control_mode=mode, q2_boost_strength=0)
    collector = SteerV3Metrics(q2_boost_strength=0) if mode == "quadrant_v3" else None
    parameters = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], requires_grad=True)
    data = TensorDict(
        {
            "sample": torch.arange(2),
            "prompts": torch.ones(2, 1, dtype=torch.long),
            "responses": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
            "response_mask": torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool),
            "old_log_probs": parameters.detach().clone(),
            "advantages": torch.ones(2, 3),
            "entropys": torch.ones(2, 3),
            "loss_mask": torch.tensor([[1, 0, 0], [1, 1, 1]]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        global_batch_size=2,
        steer_local_mini_batch_size=2,
        steer_response_length=3,
    )

    def forward_step(batch, loss_function, forward_only):
        index = batch["sample"].item()
        length = int(batch["response_mask"].sum())
        model_logp = torch.cat([parameters[index, :length], parameters.new_zeros(1)])
        loss, metrics = ppo_loss(config, {"log_probs": model_logp}, batch, steer_metrics=collector)
        return loss, {"loss": loss.detach(), "metrics": metrics}

    engine.forward_step = forward_step
    engine.forward_backward_batch(data, loss_function=ppo_loss)
    torch.testing.assert_close(parameters.grad, torch.tensor([[-0.5, 0, 0], [-1 / 6, -1 / 6, -1 / 6]]))
    if collector is not None:
        assert collector.totals[0] == 4
        assert collector.flush()["steer/q2_active_count"] == 0


@pytest.mark.parametrize("steer", [True, False])
def test_v0_cached_input_dtypes(monkeypatch, steer):
    from verl.trainer.ppo import ray_trainer

    monkeypatch.setattr(ray_trainer, "left_right_2_no_padding", lambda data: data)
    monkeypatch.setattr(ray_trainer, "no_padding_2_padding", lambda tensor, _: tensor)
    output = tu.get_tensordict(
        {
            "log_probs": torch.full((1, 2), -1.0, dtype=torch.bfloat16),
            "entropy": torch.ones(1, 2, dtype=torch.bfloat16),
        },
        non_tensor_dict={"metrics": {"mfu": 0}},
    )
    policy = PolicyLossConfig(loss_mode="entropy_control" if steer else "vanilla")
    worker_group = SimpleNamespace(
        compute_log_prob=lambda _: output,
        compute_ref_log_prob=lambda _: output,
    )
    trainer = SimpleNamespace(
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=replace(_config(), policy_loss=policy))),
        actor_rollout_wg=worker_group,
        ref_policy_wg=worker_group,
        ref_in_actor=False,
    )
    batch = SimpleNamespace(to_tensordict=lambda: TensorDict({"sample": torch.ones(1)}, batch_size=[1]))
    old, _ = ray_trainer.RayPPOTrainer._compute_old_log_prob(trainer, batch)
    ref = ray_trainer.RayPPOTrainer._compute_ref_log_prob(trainer, batch)
    expected_dtype = torch.bfloat16 if steer else torch.float32
    for value in (old.batch["old_log_probs"], old.batch["entropys"], ref.batch["ref_log_prob"]):
        assert value.dtype == expected_dtype


def test_rollout_is_is_not_silently_applied():
    old, logp, advantage, entropy, mask = _batch()
    with pytest.raises(ValueError, match="rollout IS"):
        compute_policy_loss_with_entropy(
            old,
            logp,
            advantage,
            mask,
            config=_config(),
            entropys=entropy,
            rollout_is_weights=torch.ones_like(logp),
        )


def test_hydra_config_and_unsupported_combinations():
    config_dir = str(Path(__file__).resolve().parents[3] / "verl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.policy_loss.loss_mode=entropy_control",
                "actor_rollout_ref.actor.policy_loss.token_weight_min=0.7",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
            ],
        )
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert actor.policy_loss.token_weight_min == 0.7
    assert get_policy_loss_fn("entropy_control") is compute_policy_loss_with_entropy
    for key, value, message in [
        ("algorithm.rollout_correction.bypass_mode", True, "bypass_mode"),
        ("trainer.v1.trainer_mode", "separate_async", "sync"),
        ("distillation.enabled", True, "distillation"),
        ("algorithm.rollout_correction.rollout_is", "token", "rollout_is"),
        ("algorithm.rollout_correction.rollout_rs", "token_k1", "rollout_rs"),
        ("actor_rollout_ref.actor.strategy", "megatron", "fsdp"),
        ("actor_rollout_ref.actor.loss_agg_mode", "token-sum", "token-sum"),
    ]:
        from omegaconf import OmegaConf

        candidate = OmegaConf.create(OmegaConf.to_container(config))
        OmegaConf.update(candidate, key, value)
        with pytest.raises(ValueError, match=message):
            validate_config(candidate, use_reference_policy=False, use_critic=False)
