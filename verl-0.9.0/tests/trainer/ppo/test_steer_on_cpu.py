# Copyright 2026 The verl team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""STEER migration: numerical parity, gradients, and the actual loss bridge."""

import ast
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_policy_loss_vanilla,
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


@pytest.mark.parametrize("linear", [True, False])
@pytest.mark.parametrize("mode", ["symmetric", "asymmetric"])
def test_released_implementation_parity(linear, mode):
    # Load only the two original functions, never import the old verl package.
    source = Path(__file__).resolve().parents[4] / "STEER-main/verl/trainer/ppo/core_algos.py"
    if not source.exists():
        pytest.skip("Original STEER reference checkout is not present")
    tree = ast.parse(source.read_text())
    names = {"compute_token_weights", "compute_policy_loss_with_entropy"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list = []
    scope = {"torch": torch, "verl_F": verl_F, "agg_loss": agg_loss}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    old, logp, advantage, entropy, mask = _batch()
    config = _config(linear=linear, entropy_control_mode=mode, token_weight_max=1.2)
    expected = scope["compute_token_weights"](advantage, entropy, old, logp, mask, 0.8, 1.2, linear, mode)
    weights = compute_token_weights(advantage, entropy, old, logp, mask, 0.8, 1.2, linear, mode)
    torch.testing.assert_close(weights, expected)
    reference_loss = scope["compute_policy_loss_with_entropy"](
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
    )[0]
    loss, _ = compute_policy_loss_with_entropy(old, logp, advantage, mask, config=config, entropys=entropy)
    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(
        torch.autograd.grad(loss, logp)[0],
        torch.autograd.grad(reference_loss, logp)[0],
    )
    assert not weights.requires_grad
    assert entropy.grad is None


def test_exponential_proxy_floor_and_direction():
    old = torch.tensor([[-1.0, -1.0]])
    logp = old.clone().requires_grad_()
    entropy = torch.ones_like(old, requires_grad=True) * 0.99
    mask = torch.ones_like(old, dtype=torch.bool)
    advantage = torch.tensor([[1.0, -1.0]])
    weights = compute_token_weights(advantage, entropy, old, logp, mask)
    metric = (1 - old.exp()) * 0.01
    torch.testing.assert_close(weights, torch.exp(torch.log(torch.tensor(0.8)) * metric / 0.02))
    assert weights.min() > 0.8
    directional = compute_token_weights(advantage, entropy, old, logp, mask, mode="asymmetric")
    assert directional[0, 0] == 1  # log p + H < 0, A > 0 -> entropy increase
    assert directional[0, 1] < 1


@pytest.mark.parametrize("mode", ["token-mean", "token-sum", "seq-mean-token-sum", "seq-mean-token-mean"])
def test_unit_weights_equal_vanilla_with_is_and_global_scaling(mode):
    old, logp, advantage, entropy, mask = _batch()
    config = _config(token_weight_min=1)
    config.global_batch_info.update(dp_size=2, batch_num_tokens=15, global_batch_size=4)
    importance = torch.tensor([[0.5, 1.0, 2.0, 1.0], [1.0, 0.8, 1.5, 0.0]])
    loss, _ = compute_policy_loss_with_entropy(
        old,
        logp,
        advantage,
        mask,
        mode,
        config,
        importance,
        entropys=entropy,
    )
    reference, _ = compute_policy_loss_vanilla(old, logp, advantage, mask, mode, config, importance)
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(torch.autograd.grad(loss, logp)[0], torch.autograd.grad(reference, logp)[0])


def test_masked_nonfinite_values_and_empty_batch():
    old, logp, advantage, entropy, mask = _batch()
    clean, _ = compute_policy_loss_with_entropy(old, logp, advantage, mask, config=_config(), entropys=entropy)
    polluted = [torch.where(mask, t, torch.nan) for t in (old, logp, advantage, entropy)]
    loss, _ = compute_policy_loss_with_entropy(
        *polluted[:3],
        mask,
        config=_config(),
        entropys=polluted[3],
    )
    torch.testing.assert_close(loss, clean)
    assert torch.isfinite(torch.autograd.grad(loss, logp, retain_graph=True)[0]).all()
    empty_mask = torch.zeros_like(mask)
    loss, _ = compute_policy_loss_with_entropy(
        *polluted[:3],
        empty_mask,
        config=_config(),
        entropys=polluted[3],
    )
    assert loss.item() == 0
    assert torch.autograd.grad(loss, logp)[0].eq(0).all()
    with pytest.raises(ValueError, match="non-finite"):
        compute_token_weights(advantage, entropy * torch.nan, old, logp, mask)


def test_global_accumulation_with_fixed_weights():
    # Constant entropy makes the local proxy identical, so repartitioning does not
    # change weights. Other batches intentionally retain micro-batch dependence.
    old = torch.full((4, 2), -1.0)
    logp = torch.full((4, 2), -0.95, requires_grad=True)
    advantage = torch.ones_like(old)
    entropy = torch.full_like(old, 0.2)
    mask = torch.tensor([[1, 1], [1, 0], [1, 1], [1, 0]], dtype=torch.bool)
    config = _config()
    config.global_batch_info.update(batch_num_tokens=6, dp_size=1)
    whole, _ = compute_policy_loss_with_entropy(old, logp, advantage, mask, config=config, entropys=entropy)
    parts = sum(
        compute_policy_loss_with_entropy(
            old[i : i + 1],
            logp[i : i + 1],
            advantage[i : i + 1],
            mask[i : i + 1],
            config=config,
            entropys=entropy[i : i + 1],
        )[0]
        for i in range(4)
    )
    torch.testing.assert_close(parts, whole)
    torch.testing.assert_close(torch.autograd.grad(parts, logp)[0], torch.autograd.grad(whole, logp)[0])


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
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=3, global_batch_size=1)
    loss, metrics = ppo_loss(config, {"log_probs": model_logp, "entropy": current_entropy}, data)
    expected, _ = compute_policy_loss_with_entropy(
        torch.tensor([[-1.0, -0.8, -0.6]]),
        model_logp[1:4].unsqueeze(0),
        torch.tensor([[1.0, -1.0, 1.0]]),
        response,
        config=config,
        entropys=cached,
    )
    torch.testing.assert_close(loss, expected - 0.01 * current_entropy[1:4].mean())
    loss.backward()
    assert cached.grad is None
    torch.testing.assert_close(current_entropy.grad, torch.tensor([0.0, -0.01 / 3, -0.01 / 3, -0.01 / 3, 0.0]))
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
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=5, global_batch_size=2)
    config = _config()
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


def test_bfloat16_weights_use_fp32_and_preserve_gradient():
    old, logp, advantage, entropy, mask = _batch()
    old, logp, advantage, entropy = [t.detach().to(torch.bfloat16) for t in (old, logp, advantage, entropy)]
    logp.requires_grad_()
    weights = compute_token_weights(advantage, entropy, old, logp, mask)
    expected = compute_token_weights(advantage.float(), entropy.float(), old.float(), logp.float(), mask)
    assert weights.dtype == torch.float32
    assert weights.device == logp.device
    torch.testing.assert_close(weights, expected)
    loss, _ = compute_policy_loss_with_entropy(old, logp, advantage, mask, config=_config(), entropys=entropy)
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.autograd.grad(loss, logp)[0]).all()


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
    ]:
        from omegaconf import OmegaConf

        candidate = OmegaConf.create(OmegaConf.to_container(config))
        OmegaConf.update(candidate, key, value)
        with pytest.raises(ValueError, match=message):
            validate_config(candidate, use_reference_policy=False, use_critic=False)
