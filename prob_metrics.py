"""Ensemble scores shared by train_afm.py (validation CRPS for model selection)
and evaluate.py (test CRPS). Both work batch by batch on torch tensors, so no
caller has to hold an ensemble for a whole split: at 17,520 test samples, 16
members and 200 x 200 pixels that is ~45 GB of float32.

Spread-skill. For an exchangeable (calibrated) M-member ensemble with
ensemble mean xbar and unbiased member variance s^2 (ddof 1),

    E[(xbar - y)^2] = (M + 1) / M * E[s^2],

so the spread-skill ratio sqrt((M + 1) / M * mean s^2) / sqrt(mean (xbar - y)^2)
is 1 when calibrated, < 1 when over-confident. It needs ensemble_var and
ensemble_mean_se below; ensemble_spread (ddof 0, std averaged per pixel,
no (M + 1) / M) is kept only so old rows can be reproduced and is biased
towards "over-confident" at small M."""

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
    """DEPRECATED. Per-sample pixel mean of the ensemble standard deviation
    (ddof 0), the quantity evaluate.py has always reported as spread_mm /
    spread_std. Biased low and not comparable with an RMSE; use
    ensemble_var and ensemble_mean_se (module docstring)."""
    return ens.float().std(dim=1, unbiased=False).flatten(1).mean(dim=1)


def ensemble_var(ens):
    """Per-sample pixel mean of the unbiased member variance (ddof 1).
    The mean over samples (equal pixel counts) is mean s^2; its square root
    is the RMS spread. Returns (B,) float32."""
    if ens.shape[1] < 2:
        raise ValueError(f"ensemble_var needs at least 2 members, got {ens.shape[1]}")
    return ens.float().var(dim=1, unbiased=True).flatten(1).mean(dim=1)


def ensemble_mean_se(ens, y):
    """Per-sample pixel mean of the squared error of the ENSEMBLE MEAN,
    (xbar - y)^2: the skill term of the spread-skill ratio. Returns (B,)
    float32."""
    return ((ens.float().mean(dim=1) - y.float()) ** 2).flatten(1).mean(dim=1)


def _ranks(ens, y, generator=None):
    """(B, P) rank of the truth among the members, ties broken at random."""
    M = ens.shape[1]
    ens, y = ens.float(), y.float().unsqueeze(1)
    below = (ens < y).sum(dim=1)
    ties = (ens == y).sum(dim=1)
    u = torch.rand(below.shape, device=ens.device, generator=generator)
    rank = below + torch.floor(u * (ties + 1)).long()
    return rank.clamp_(max=M).flatten(1)


def _histogram(rank, M, weight=None):
    counts = torch.zeros(rank.shape[0], M + 1, dtype=torch.long, device=rank.device)
    w = torch.ones_like(rank) if weight is None else weight.flatten(1).to(rank.device).long()
    return counts.scatter_add_(1, rank, w)


def rank_histogram(ens, y, generator=None, mask=None):
    """Per-sample histogram of the truth's rank among the members (PLAN.md 3.4).

    ens: (B, M, ...) members, y: (B, ...) truth. The rank of a pixel is the
    number of members below the truth, 0..M; ties are broken uniformly at
    random, so a truth equal to k members takes each of the k + 1 ranks with
    equal probability. mask: optional bool (B, ...) of the pixels to count.
    Returns (B, M + 1) int64 pixel counts. A calibrated ensemble has a flat
    histogram; a U shape is over-confidence.
    """
    return _histogram(_ranks(ens, y, generator), ens.shape[1], mask)


def floored_rank_histograms(ens_mm, y_mm, floor_mm, generator=None):
    """Rank histograms after flooring members and truth at floor_mm (mm).

    A dry truth then equals every dry member exactly and its rank is drawn
    uniformly among the ties, instead of being decided by sub-floor scatter
    (log(P + 1e-5) resolves 1e-5 mm). Ranks are invariant under the monotone
    log, so no log is taken. Returns (all pixels, wet truth only): two
    (B, M + 1) int64 histograms from ONE draw of the tie-breaking, the second
    counting only pixels with y_mm >= floor_mm.
    """
    e = ens_mm.float().clamp(min=floor_mm)
    t = y_mm.float().clamp(min=floor_mm)
    rank = _ranks(e, t, generator)
    M = ens_mm.shape[1]
    return _histogram(rank, M), _histogram(rank, M, y_mm.float() >= floor_mm)
