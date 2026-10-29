"""Ensemble scores shared by train_afm.py (validation CRPS for model selection)
and evaluate.py (test CRPS). Both work batch by batch on torch tensors, so no
caller has to hold an ensemble for a whole split: at 17,520 test samples, 16
members and 200 x 200 pixels that is ~45 GB of float32."""

import torch


def crps_fair(ens, y):
    """Fair ensemble CRPS per sample, averaged over pixels.

    ens: (B, M, ...) members, y: (B, ...) truth with the same trailing shape.

        CRPS_fair = mean_i |x_i - y| - 1 / (2 M (M - 1)) * sum_{i,j} |x_i - x_j|

    The fair estimator is unbiased in M, so a 4-member validation score and a
    16-member test score estimate the same quantity. The pairwise term uses
    the sorted-member identity

        sum_{i,j} |x_i - x_j| = 2 * sum_k (2k - M - 1) x_(k),   k = 1..M,

    which costs one sort instead of M^2 differences. Returns a (B,) float32
    tensor on the input device.
    """
    M = ens.shape[1]
    if M < 2:
        raise ValueError(f"crps_fair needs at least 2 members, got {M}")
    ens = ens.float()
    skill = (ens - y.float().unsqueeze(1)).abs().mean(dim=1)
    xs, _ = ens.sort(dim=1)
    k = torch.arange(1, M + 1, device=ens.device, dtype=ens.dtype)
    w = (2 * k - M - 1).view(1, M, *([1] * (ens.ndim - 2)))
    pair_sum = 2 * (w * xs).sum(dim=1)
    crps = skill - pair_sum / (2 * M * (M - 1))
    return crps.flatten(1).mean(dim=1)


def ensemble_spread(ens):
    """Per-sample pixel mean of the ensemble standard deviation (ddof 0), the
    quantity evaluate.py has always reported as spread_mm."""
    return ens.float().std(dim=1, unbiased=False).flatten(1).mean(dim=1)


def rank_histogram(ens, y, generator=None):
    """Per-sample histogram of the truth's rank among the members (PLAN.md 3.4).

    ens: (B, M, ...) members, y: (B, ...) truth. The rank of a pixel is the
    number of members below the truth, 0..M; ties are broken uniformly at
    random, so a truth equal to k members takes each of the k + 1 ranks with
    equal probability. Returns (B, M + 1) int64 pixel counts. A calibrated
    ensemble has a flat histogram; a U shape is over-confidence.
    """
    M = ens.shape[1]
    ens, y = ens.float(), y.float().unsqueeze(1)
    below = (ens < y).sum(dim=1)
    ties = (ens == y).sum(dim=1)
    u = torch.rand(below.shape, device=ens.device, generator=generator)
    rank = below + torch.floor(u * (ties + 1)).long()
    rank = rank.clamp_(max=M).flatten(1)
    counts = torch.zeros(rank.shape[0], M + 1, dtype=torch.long, device=ens.device)
    return counts.scatter_add_(1, rank, torch.ones_like(rank))
