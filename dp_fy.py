"""Exact full-catalog structured-subset DP-FY loss."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _validate_inputs(
    scores: torch.Tensor,
    positive_indices: torch.Tensor,
    p: int,
    temperature: float,
) -> Tuple[int, int, int, float]:
    if scores.ndim != 2 or not scores.dtype.is_floating_point:
        raise ValueError("scores must be a floating tensor with shape [B, I]")
    if scores.dtype not in (torch.float32, torch.float64):
        raise ValueError("DP-FY expects float32 or float64 scores")
    if positive_indices.ndim != 2:
        raise ValueError("positive_indices must have shape [B, P]")

    batch_size, num_items = scores.shape
    p = int(p)
    temperature = float(temperature)
    if not 1 <= p <= num_items:
        raise ValueError("P must be between one and the catalog size")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if tuple(positive_indices.shape) != (batch_size, p):
        raise ValueError("positive_indices has the wrong shape")

    positive_indices = positive_indices.long()
    if positive_indices.numel():
        lower = int(positive_indices.min().item())
        upper = int(positive_indices.max().item())
        if lower < 0 or upper >= num_items:
            raise ValueError("positive_indices contains an item outside the catalog")
    if p > 1:
        ordered = torch.sort(positive_indices, dim=1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
            raise ValueError("Each target row must contain P distinct items")
    return batch_size, num_items, p, temperature


def log_partition_dp(
    scores: torch.Tensor,
    p: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute the exact log-partition over all size-P subsets in O(BIP).

    For a row ``s``, the returned value is

        log sum_{|S|=P} exp(sum_{i in S} s_i / temperature).

    The recurrence is evaluated with ``logcumsumexp``. It is algebraically
    equivalent to the two-branch include/exclude dynamic program while using
    only one catalog-length state tensor per cardinality level.
    """
    if scores.ndim != 2 or not scores.dtype.is_floating_point:
        raise ValueError("scores must be a floating tensor with shape [B, I]")
    if scores.dtype not in (torch.float32, torch.float64):
        raise ValueError("DP-FY expects float32 or float64 scores")
    p = int(p)
    temperature = float(temperature)
    if not 1 <= p <= scores.shape[1]:
        raise ValueError("P must be between one and the catalog size")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    scaled = scores / temperature
    prefix = torch.logcumsumexp(scaled, dim=1)
    if p == 1:
        return prefix[:, -1]

    # A finite sentinel avoids undefined gradients through an all--infinity
    # prefix while behaving as log(0) at the supported floating precisions.
    sentinel = scores.new_full((scores.shape[0], 1), -1.0e30)
    for _ in range(2, p + 1):
        previous = torch.cat((sentinel, prefix[:, :-1]), dim=1)
        prefix = torch.logcumsumexp(scaled + previous, dim=1)
    return prefix[:, -1]


def structured_surplus(
    scores: torch.Tensor,
    p: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Return temperature times the exact size-P log-partition."""
    return float(temperature) * log_partition_dp(scores, p, temperature)


def dp_fy_loss(
    scores: torch.Tensor,
    positive_indices: torch.Tensor,
    p: int,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return the mean structured-subset Fenchel--Young loss.

    ``scores`` contains one score for every catalog item. Each target row
    contains ``P`` distinct positive item indices. The optimized per-user loss
    is the exact structured surplus minus the sum of target scores.
    """
    _, _, p, temperature = _validate_inputs(
        scores, positive_indices, p, temperature
    )
    positive_indices = positive_indices.long()
    surplus = structured_surplus(scores, p, temperature)
    target_score_sum = scores.gather(1, positive_indices).sum(dim=1)
    losses = surplus - target_score_sum
    if not bool(torch.isfinite(losses).all().item()):
        raise FloatingPointError("DP-FY produced a non-finite loss")
    loss = losses.mean()
    diagnostics = {
        "loss": float(loss.detach().cpu()),
        "mean_surplus": float(surplus.mean().detach().cpu()),
        "mean_target_score_sum": float(target_score_sum.mean().detach().cpu()),
        "selection_mass": float(p),
        "temperature": temperature,
    }
    return loss, diagnostics


def exact_inclusion_marginals(
    scores: torch.Tensor,
    p: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute exact size-P inclusion marginals for inspection or testing.

    Training does not need this helper: differentiating ``dp_fy_loss``
    already propagates the marginal-minus-target gradient. This function uses
    a detached copy so that diagnostic calls cannot consume a training graph.
    """
    work_scores = scores.detach().clone().requires_grad_(True)
    surplus = structured_surplus(work_scores, p, temperature)
    marginals = torch.autograd.grad(surplus.sum(), work_scores)[0]
    return marginals.detach()
