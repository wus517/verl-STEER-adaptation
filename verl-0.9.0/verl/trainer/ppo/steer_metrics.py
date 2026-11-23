# Copyright 2026 The verl team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Device-side sufficient statistics for STEER Quadrant V3 logging."""

from functools import cache

import torch
import torch.distributed as dist


@cache
def _clamp_endpoint_gradients(device, dtype):
    # Probe once per device/dtype because clamp endpoint derivatives vary across
    # PyTorch versions. This tiny graph is independent of the actor's graph.
    with torch.enable_grad():
        endpoints = torch.tensor([-1.0, 1.0], device=device, dtype=dtype, requires_grad=True)
        return torch.autograd.grad(endpoints.clamp(-1.0, 1.0).sum(), endpoints)[0].detach()


def _clamp_gradient(values, lower, upper):
    endpoints = _clamp_endpoint_gradients(values.device, values.dtype)
    return torch.where(
        values == lower,
        endpoints[0],
        torch.where(values == upper, endpoints[1], ((values > lower) & (values < upper)).to(values.dtype)),
    )


@torch.no_grad()
def compute_steer_v3_statistics(
    *,
    advantages,
    ratio,
    log_prob_difference,
    losses1,
    losses2,
    clipped,
    response_mask,
    weights,
    scores,
    quadrants,
    clip_low,
    clip_high,
    clip_ratio_c,
    loss_agg_mode,
    accumulation_scale,
    q2_boost_scale,
):
    """Match the unweighted PPO derivative, including PyTorch's tie gradients."""
    ratio_grad = _clamp_gradient(ratio, 1 - clip_low, 1 + clip_high)
    # maximum splits a tie equally, including a tie with the clamped branch.
    maximum_grad = torch.where(
        losses1 > losses2,
        1.0,
        torch.where(losses1 < losses2, ratio_grad, (1.0 + ratio_grad) * 0.5),
    )
    dual_bound = -advantages * clip_ratio_c
    minimum_grad = torch.where(clipped < dual_bound, 1.0, torch.where(clipped > dual_bound, 0.0, 0.5))
    dual_grad = torch.where(advantages < 0, minimum_grad, 1.0)
    difference_grad = _clamp_gradient(log_prob_difference, -20.0, 20.0)
    coefficient = advantages.abs() * ratio * maximum_grad * dual_grad * difference_grad
    active = response_mask.bool() & torch.isfinite(coefficient) & (coefficient > 0)

    # These are exactly the per-token coefficients of agg_steer_loss.
    if loss_agg_mode == "token-mean":
        reduction_scale = 1.0 / (response_mask.sum() + 1e-8)
    elif loss_agg_mode == "seq-mean-token-sum":
        reduction_scale = 1.0 / response_mask.shape[0]
    elif loss_agg_mode == "seq-mean-token-mean":
        reduction_scale = 1.0 / (response_mask.sum(dim=-1, keepdim=True).clamp_min(1) * response_mask.shape[0])
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        reduction_scale = 1.0 / response_mask.shape[-1]
    else:
        raise ValueError(f"Invalid STEER loss_agg_mode: {loss_agg_mode}")
    coefficient = coefficient * reduction_scale * accumulation_scale
    q2 = quadrants[1]
    q2_active = q2 & active
    boost = weights - 1.0
    effective_coefficient = torch.where(q2_active, coefficient, 0.0)
    clipped_by_high = (ratio > 1 + clip_high) & (advantages > 0)

    # Counts and sums stay on device until the whole logging window is reduced.
    return torch.stack(
        [
            active.sum(),
            *((quadrant & active).sum() for quadrant in quadrants),
            torch.where(q2_active, boost, 0.0).sum(),
            (torch.where(q2_active, boost, 0.0) * effective_coefficient).sum(),
            effective_coefficient.sum(),
            (q2 & clipped_by_high).sum(),
            q2.sum(),
            (q2_active & (scores <= q2_boost_scale)).sum(),
        ]
    ).float()


class SteerV3Metrics:
    """Accumulate microbatches locally, then reduce a logging window once over DP."""

    def __init__(self, *, q2_boost_strength):
        self.q2_boost_strength = q2_boost_strength
        self.totals = None

    def add(self, statistics):
        if self.totals is None:
            self.totals = statistics.detach().clone()
        else:
            self.totals.add_(statistics.detach())

    def flush(self, dp_group=None):
        if self.totals is None:
            return {}
        totals = self.totals.clone()
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=dp_group)
        (
            active,
            q1_active,
            q2_active,
            q3_active,
            q4_active,
            boost_sum,
            effective_boost_sum,
            effective_coefficient,
            q2_clipped,
            q2_valid,
            q2_half_boost,
        ) = totals.cpu().tolist()
        metrics = {"steer/q2_active_count": q2_active}
        if active > 0:
            for quadrant, count in enumerate((q1_active, q2_active, q3_active, q4_active), start=1):
                metrics[f"steer/q{quadrant}_active_frac"] = count / active
        if q2_active > 0:
            metrics["steer/q2_boost_mean"] = boost_sum / q2_active
            if self.q2_boost_strength > 0:
                metrics["steer/q2_at_least_half_boost_frac"] = q2_half_boost / q2_active
        if effective_coefficient > 0:
            metrics["steer/q2_effective_boost"] = effective_boost_sum / effective_coefficient
        if q2_valid > 0:
            metrics["steer/q2_clip_frac"] = q2_clipped / q2_valid
        self.totals.zero_()
        return metrics
