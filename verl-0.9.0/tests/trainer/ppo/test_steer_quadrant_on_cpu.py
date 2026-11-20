# Copyright 2026 The verl team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Four-quadrant specification examples and actor-loss integration."""

import ast
import math
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    compute_policy_loss_with_entropy,
    compute_quadrant_steer_weights,
    compute_token_weights,
    get_policy_loss_fn,
)
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss


def _policy(**kwargs):
    return PolicyLossConfig(
        loss_mode="entropy_control", entropy_control_mode="quadrant", linear=False, token_weight_min=0.7, **kwargs
    )


def _config(**kwargs):
    return ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        clip_ratio_c=3.0,
        policy_loss=_policy(**kwargs),
    )


def _scores(scores, signs):
    """Construct finite, positive entropy inputs with prescribed signed scores."""
    score = torch.tensor([scores], dtype=torch.float32)
    advantage = 4 * torch.tensor([signs], dtype=torch.float32)
    logp = torch.full_like(score, math.log(0.5))
    entropy = -logp - 2 * score / torch.where(advantage == 0, 1, advantage)
    mask = torch.ones_like(score, dtype=torch.bool)
    return advantage, entropy, logp.clone(), logp, mask


def _reference_weights(advantage, entropy, old, current, mask, *, wmin=0.7, d=0.2, strength=0.1, w3=1.0):
    """Independent scalar implementation of the specification, using Python math."""
    scores = []
    for a, h, old_p, p, valid in zip(
        advantage.flatten(), entropy.flatten(), old.flatten(), current.flatten(), mask.flatten(), strict=True
    ):
        if not valid:
            scores.append(0.0)
            continue
        probability = min(max(math.exp(float(p)), 1e-8), 1.0 - 1e-8)
        old_probability = min(max(math.exp(float(old_p)), 1e-8), 1.0)
        score = -float(a) / old_probability * probability * (1 - probability) * (math.log(probability) + float(h))
        scores.append(score if math.isfinite(score) else 0.0)
    denominator = max(max(map(abs, scores)), 0.02)
    weights = []
    for a, score, valid in zip(advantage.flatten(), scores, mask.flatten(), strict=True):
        z = abs(score) / denominator
        if not valid:
            weight = 0.0
        elif score > 0:
            if z < 0.5 - d:
                weight = 1 + strength * (0.5 - d - z) / (0.5 - d)
            elif z > 0.5 + d:
                weight = 1 - strength * (z - 0.5 - d) / (0.5 - d)
            else:
                weight = 1.0
        elif score < 0 and a > 0:
            weight = wmin**z
        elif score < 0 and a < 0:
            weight = w3
        else:
            weight = 1.0
        weights.append(weight)
    return current.new_tensor(weights, dtype=torch.float32).reshape(current.shape)


def _reduce(matrix, mask, mode):
    """Independent old actor reductions (not 0.9.0 global normalization)."""
    summed = (matrix * mask).sum(-1)
    if mode == "token-mean":
        return summed.sum() / (mask.sum() + 1e-8)
    if mode == "seq-mean-token-sum":
        return summed.mean()
    if mode == "seq-mean-token-mean":
        return (summed / mask.sum(-1)).mean()
    if mode == "seq-mean-token-sum-norm":
        return summed.sum() / mask.shape[-1]
    raise AssertionError(mode)


def _clipped(old, current, advantage):
    ratio = (current - old).clamp(-20, 20).exp()
    clipped = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.28))
    return torch.where(advantage < 0, torch.minimum(-3 * advantage, clipped), clipped)


def test_document_examples_and_quadrant_three_is_independent():
    inputs = _scores([-0.6, 0.1, 0.3, 0.5, -0.9, 0.7, 0.9, 1.0, 0.0], [1, 1, 1, 1, -1, -1, -1, 1, 0])
    weights = compute_quadrant_steer_weights(*inputs)
    expected = torch.tensor([[0.807344, 1.066667, 1, 1, 1, 1, 0.933333, 0.9, 1]])
    torch.testing.assert_close(weights, expected, rtol=1e-6, atol=1e-6)
    changed = compute_quadrant_steer_weights(*inputs, quadrant_3_weight=0.95)
    expected = weights.clone()
    expected[0, 4] = 0.95
    torch.testing.assert_close(changed, expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_quadrant_one_matches_baseline_exactly_with_shared_denominator(scale):
    inputs = _scores([scale * score for score in [-0.6, 0.1, -0.9, 1.0]], [1, 1, -1, -1])
    quadrant = compute_quadrant_steer_weights(*inputs)
    symmetric = compute_token_weights(*inputs, token_weight_min=0.7, token_weight_max=1.0, linear=False)
    # Maximum magnitude comes from IV, not I. When scale=0.01, the floor
    # limits z_max to 0.5 and prevents the high-amplitude suppression region.
    torch.testing.assert_close(quadrant[:, :1], symmetric[:, :1], rtol=0, atol=0)
    assert quadrant[0, 2] == 1.0
    assert quadrant[0, 3] == (0.9 if scale == 1.0 else 1.0)


def test_shared_max_in_quadrant_three_and_masked_extreme():
    inputs = _scores([-0.1, 0.1, -1.0, 5.0], [1, 1, -1, 1])
    inputs[-1][0, -1] = False
    weights = compute_quadrant_steer_weights(*inputs)
    torch.testing.assert_close(weights, torch.tensor([[0.7**0.1, 1.066666667, 1.0, 0.0]]))


@pytest.mark.parametrize("d,strength,w3", [(0.2, 0.1, 1.0), (0.0, 0.1, 0.95), (0.49, 0.0, 1.05)])
def test_independent_scalar_reference(d, strength, w3):
    torch.manual_seed(23)
    old = -torch.rand(4, 8) * 5 - 0.2
    current = old + 0.3 * torch.randn_like(old)
    advantage = torch.randn_like(old)
    entropy = torch.rand_like(old) * 4
    mask = torch.rand_like(old) > 0.2
    actual = compute_quadrant_steer_weights(
        advantage,
        entropy,
        old,
        current,
        mask,
        entropy_band_half_width=d,
        entropy_weight_strength=strength,
        quadrant_3_weight=w3,
    )
    expected = _reference_weights(advantage, entropy, old, current, mask, d=d, strength=strength, w3=w3)
    torch.testing.assert_close(actual, expected)


def test_document_three_token_sign_and_loss_example():
    distribution = torch.tensor([0.7, 0.2, 0.1])
    entropy = -(distribution * distribution.log()).sum()
    current = torch.tensor([[0.1]]).log().requires_grad_()
    advantage = torch.tensor([[0.2]])
    score = -(advantage / current.detach().exp()) * current.exp() * (1 - current.exp()) * (current + entropy)
    torch.testing.assert_close(score, torch.tensor([[0.270138]]), atol=1e-6, rtol=1e-6)
    # Add a III token that makes this token's normalized magnitude 0.2.
    denominator = float(score.detach()) / 0.2
    anchor_adv = -denominator / (0.9 * (-math.log(0.1) - float(entropy)))
    adv = torch.tensor([[0.2, anchor_adv]])
    logp = current.detach().expand(1, 2)
    weights = compute_quadrant_steer_weights(adv, entropy.expand(1, 2), logp, logp, torch.ones_like(adv))
    torch.testing.assert_close(weights[0, 0], torch.tensor(1.033333333))
    torch.testing.assert_close(-adv[0, 0] * weights[0, 0], torch.tensor(-0.206666667))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_reference_code_from_spec_and_no_grad(dtype):
    document = Path(__file__).resolve().parents[4] / "STEER_QUADRANT_ALGORITHM.md"
    source = document.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(compile(ast.parse(source), str(document), "exec"), namespace)
    inputs = list(_scores([-0.6, 0.1, 0.9, -1.0, 0.0], [1, 1, -1, -1, 0]))
    for index in range(4):
        inputs[index] = inputs[index].to(dtype).requires_grad_()
    actual = compute_quadrant_steer_weights(*inputs)
    expected = namespace["compute_quadrant_steer_weights"](*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32
    assert actual.device == inputs[3].device
    assert not actual.requires_grad


def test_zero_nonfinite_and_empty_helper():
    inputs = list(_scores([0.0, 0.0, 0.0, 0.0], [1, -1, 0, 1]))
    inputs[1][0, 0] = torch.nan
    inputs[1][0, 1] = torch.inf
    inputs[-1][0, -1] = False
    torch.testing.assert_close(compute_quadrant_steer_weights(*inputs), torch.tensor([[1.0, 1.0, 1.0, 0.0]]))
    inputs[-1].zero_()
    assert compute_quadrant_steer_weights(*inputs).eq(0).all()


@pytest.mark.parametrize("mode", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"])
@pytest.mark.parametrize("unit_weights", [False, True])
def test_clipped_policy_loss_and_gradient(mode, unit_weights):
    # Include clipped positive and dual-clipped negative tokens. Weights belong
    # after clipping, while their denominator includes all valid proxy scores.
    old = torch.tensor([[-2.0, -2.0, -4.0], [-1.0, -1.0, -2.0]])
    current = (old + torch.tensor([[0.8, -0.4, 3.0], [-0.2, 0.1, 0.4]])).requires_grad_()
    advantage = torch.tensor([[1.0, -1.0, -1.0], [0.5, -0.5, 0.0]], requires_grad=True)
    entropy = torch.tensor([[1.5, 1.5, 3.0], [2.0, 0.2, 1.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    config = _config()
    if unit_weights:
        config = replace(
            config,
            policy_loss=replace(
                config.policy_loss, token_weight_min=1.0, entropy_weight_strength=0.0, quadrant_3_weight=1.0
            ),
        )
        weights = torch.ones_like(current)
    else:
        weights = _reference_weights(advantage.detach(), entropy.detach(), old, current.detach(), mask)
    loss, _ = compute_policy_loss_with_entropy(old, current, advantage, mask, mode, config, entropys=entropy)
    expected = _reduce(_clipped(old, current, advantage) * weights, mask, mode)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, current, retain_graph=True)[0], torch.autograd.grad(expected, current)[0]
    )
    assert torch.autograd.grad(loss, entropy, allow_unused=True)[0] is None


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("mode", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"])
def test_loss_bridge_accumulation_and_entropy_source(nested, dynamic, dp_size, mode):
    config = replace(
        _config(entropy_band_half_width=0.1, entropy_weight_strength=0.2, quadrant_3_weight=0.95),
        use_dynamic_bsz=dynamic,
        loss_agg_mode=mode,
        entropy_coeff=0.03,
        use_kl_loss=True,
        kl_loss_type="mse",
        kl_loss_coef=0.02,
    )
    lengths = [1, 3, 2]
    old = torch.tensor([[-0.8, -1.2, -1.5]] * 3)
    current = (old + torch.tensor([[0.05, -0.1, 0.15]])).requires_grad_()
    advantage = torch.tensor([[1.0, -0.5, 0.75]] * 3)
    cached = torch.tensor([[1.1, 1.8, 0.7]] * 3, requires_grad=True)
    current_entropy = torch.ones(20, requires_grad=True)  # must be ignored
    actual, expected = [], []
    for group in [[0, 1], [2]] if dynamic else [[0], [1], [2]]:
        rows = lambda tensor, group=group: [tensor[i, : lengths[i]] for i in group]
        mask_rows = [torch.ones(lengths[i], dtype=torch.bool) for i in group]
        pad = lambda tensors, width: torch.stack(
            [torch.nn.functional.pad(row, (0, width - len(row))) for row in tensors]
        )
        response_fields = {
            "response_mask": mask_rows,
            "old_log_probs": rows(old),
            "advantages": rows(advantage),
            "entropy" if nested else "entropys": rows(cached),
            "ref_log_prob": rows(old - 0.2),
        }
        data = TensorDict(
            {
                key: torch.nested.as_nested_tensor(value, layout=torch.jagged) if nested else pad(value, 3)
                for key, value in response_fields.items()
            },
            batch_size=[len(group)],
        )
        data["prompts"] = (
            torch.nested.as_nested_tensor([torch.ones(1, dtype=torch.long) for _ in group], layout=torch.jagged)
            if nested
            else torch.ones(len(group), 1, dtype=torch.long)
        )
        data["responses"] = (
            torch.nested.as_nested_tensor(
                [torch.ones(lengths[i], dtype=torch.long) for i in group], layout=torch.jagged
            )
            if nested
            else torch.ones(len(group), 3, dtype=torch.long)
        )
        if not nested:
            data["attention_mask"] = torch.cat([torch.ones(len(group), 1), pad(mask_rows, 3)], dim=-1).long()
        tu.assign_non_tensor(
            data,
            dp_size=dp_size,
            batch_num_tokens=6 * dp_size,
            global_batch_size=3 * dp_size,
            steer_local_mini_batch_size=3,
            steer_response_length=7,
        )
        model_logp = torch.cat([torch.cat([row, row.new_zeros(1)]) for row in rows(current)])
        loss, _ = ppo_loss(config, {"log_probs": model_logp, "entropy": current_entropy}, data)
        actual.append(loss)
        old_p, new_p, adv, ent, ref, mask = [
            pad(tensors, 7)
            for tensors in (rows(old), rows(current), rows(advantage), rows(cached), rows(old - 0.2), mask_rows)
        ]
        weights = _reference_weights(adv, ent.detach(), old_p, new_p.detach(), mask, d=0.1, strength=0.2, w3=0.95)
        reference = _reduce(_clipped(old_p, new_p, adv) * weights, mask, mode)
        reference = reference - 0.03 * _reduce(ent.detach(), mask, mode)
        reference = reference + 0.02 * _reduce(0.5 * (new_p - ref).square(), mask, mode)
        scale = len(group) / 3 if dynamic else 1 / 3
        expected.append(reference * scale)
    # Each simulated DP rank has this local batch. DP averaging must not add
    # any extra factor, regardless of global token metadata above.
    actual, expected = sum(actual), sum(expected)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, current, retain_graph=True)[0], torch.autograd.grad(expected, current)[0]
    )
    assert torch.autograd.grad(actual, (cached, current_entropy), allow_unused=True) == (None, None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"linear": True},
        {"entropy_band_half_width": -0.01},
        {"entropy_band_half_width": 0.5},
        {"entropy_band_half_width": float("nan")},
        {"entropy_weight_strength": -0.1},
        {"entropy_weight_strength": 1.0},
        {"entropy_weight_strength": float("inf")},
        {"quadrant_3_weight": 0},
        {"quadrant_3_weight": -1},
        {"quadrant_3_weight": float("nan")},
    ],
)
def test_quadrant_config_validation(kwargs):
    fields = dict(loss_mode="entropy_control", entropy_control_mode="quadrant", linear=False)
    fields.update(kwargs)
    with pytest.raises(ValueError, match="STEER"):
        PolicyLossConfig(**fields)


def test_hydra_config_mode_and_parameters_reach_loss():
    directory = str(Path(__file__).resolve().parents[3] / "verl/trainer/config")
    with initialize_config_dir(config_dir=directory, version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.policy_loss.loss_mode=entropy_control",
                "actor_rollout_ref.actor.policy_loss.entropy_control_mode=quadrant",
                "actor_rollout_ref.actor.policy_loss.linear=false",
                "actor_rollout_ref.actor.policy_loss.token_weight_min=0.7",
                "actor_rollout_ref.actor.policy_loss.entropy_band_half_width=0.1",
                "actor_rollout_ref.actor.policy_loss.entropy_weight_strength=0.2",
                "actor_rollout_ref.actor.policy_loss.quadrant_3_weight=0.95",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
            ],
        )
    validate_config(config, use_reference_policy=False, use_critic=False)
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    inputs = _scores([-0.6, 0.1, -0.9, 1.0], [1, 1, -1, -1])
    advantage, entropy, old, current, mask = inputs
    loss_fn = get_policy_loss_fn(actor.policy_loss.loss_mode)
    assert loss_fn is compute_policy_loss_with_entropy
    actual, _ = loss_fn(old, current, advantage, mask, config=actor, entropys=entropy)
    weights = _reference_weights(*inputs[:2], old, current, mask, d=0.1, strength=0.2, w3=0.95)
    # All ratios are 1, so clipping settings do not enter this example.
    torch.testing.assert_close(actual, (-advantage * weights).mean())


def _server_script_overrides(filename, **environment):
    """Expand only the training command; never run installs, Ray, or training."""
    root = Path(__file__).resolve().parents[4]
    script = (root / filename).read_text()
    defaults = dict(re.findall(r"^([A-Z_][A-Z0-9_]*)=\$\{\1:-([^{}\n]*)\}$", script, re.MULTILINE))
    defaults.update(
        MODEL_PATH="/server/model/Qwen3_5_9B",
        TRAIN_FILE="/server/train.parquet",
        VAL_FILE_HIGH="/server/high.parquet",
        VAL_FILE_LOW="/server/low.parquet",
        REWARD_FILE="/server/reward.py",
        PROJECT_NAME="steer_check",
        EXPERIMENT_NAME="check",
        TRAIN_ROLLOUT_DIR="/server/train_rollouts",
        VAL_ROLLOUT_DIR="/server/val_rollouts",
        DEFAULT_LOCAL_DIR="/server/checkpoints",
        TRAINER_NNODES="2",
    )
    defaults.update(environment)
    command = (
        "python3 -m verl.trainer.main_ppo"
        + script.split("python3 -m verl.trainer.main_ppo", 1)[1].split("2>&1 |", 1)[0]
    )
    capture = 'python3() { printf "%s\\0" "$@"; };\n' + command
    result = subprocess.run(["bash", "-c", capture], env={**os.environ, **defaults}, capture_output=True, check=True)
    args = result.stdout.decode().rstrip("\0").split("\0")
    assert args[:2] == ["-m", "verl.trainer.main_ppo"]
    return args[2:]


def test_server_script_preserves_base_training_and_hydra_config():
    baseline = _server_script_overrides("run_steer_4096.sh")
    quadrant = _server_script_overrides("run_steer_quadrant_4096.sh")
    mapping = lambda args: dict(arg.split("=", 1) for arg in args)
    base_map, new_map = mapping(baseline), mapping(quadrant)
    assert len(quadrant) == len(new_map)  # No duplicate CLI overrides.
    differences = {key for key in base_map.keys() | new_map.keys() if base_map.get(key) != new_map.get(key)}
    prefix = "actor_rollout_ref.actor.policy_loss."
    assert differences == {
        prefix + key
        for key in (
            "entropy_control_mode",
            "entropy_band_half_width",
            "entropy_weight_strength",
            "quadrant_3_weight",
        )
    }
    directory = str(Path(__file__).resolve().parents[3] / "verl/trainer/config")
    with initialize_config_dir(config_dir=directory, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=quadrant)
    validate_config(config, use_reference_policy=True, use_critic=False)
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert actor.policy_loss.entropy_control_mode == "quadrant"
    assert actor.policy_loss.token_weight_min == 0.7
    assert actor.policy_loss.entropy_band_half_width == 0.2
    assert actor.policy_loss.entropy_weight_strength == 0.1
    assert actor.policy_loss.quadrant_3_weight == 1.0
    assert actor.policy_loss.linear is False
    assert actor.engine.grad_sync_every_micro_batch is True
    assert config.trainer.device == "npu"
    assert config.actor_rollout_ref.rollout.response_length == config.data.max_response_length == 4096
    assert config.data.max_prompt_length == 1024
    assert actor.kl_loss_coef == 0


def test_server_script_environment_overrides_and_symmetric_control():
    args = _server_script_overrides(
        "run_steer_quadrant_4096.sh",
        STEER_ENTROPY_CONTROL_MODE="symmetric",
        STEER_ENTROPY_BAND_HALF_WIDTH="0.1",
        STEER_ENTROPY_WEIGHT_STRENGTH="0.2",
        STEER_QUADRANT_3_WEIGHT="0.95",
    )
    directory = str(Path(__file__).resolve().parents[3] / "verl/trainer/config")
    with initialize_config_dir(config_dir=directory, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=args)
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert actor.policy_loss.entropy_control_mode == "symmetric"
    assert actor.policy_loss.entropy_band_half_width == 0.1
    assert actor.policy_loss.entropy_weight_strength == 0.2
    assert actor.policy_loss.quadrant_3_weight == 0.95


@pytest.mark.parametrize(
    "linear,half_width,valid", [("false", "0.2", True), ("true", "0.2", False), ("false", "0.5", False)]
)
def test_server_script_python_preflight(monkeypatch, linear, half_width, valid, capsys):
    script = (Path(__file__).resolve().parents[4] / "run_steer_quadrant_4096.sh").read_text()
    source = script.split("<<'PY_CHECK_QUADRANT'\n", 1)[1].split("\nPY_CHECK_QUADRANT", 1)[0]
    assert script.index("PY_CHECK_QUADRANT") < script.index("ray start --head")
    monkeypatch.setattr(sys, "argv", ["-", "0.7", "1.0", linear, "quadrant", half_width, "0.1", "1.0"])
    if valid:
        exec(compile(source, "server_script_preflight", "exec"), {})
        assert "Quadrant STEER loaded from:" in capsys.readouterr().out
    else:
        with pytest.raises(ValueError, match="STEER"):
            exec(compile(source, "server_script_preflight", "exec"), {})
