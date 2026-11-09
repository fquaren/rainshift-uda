"""CPU check of prob_metrics against brute force. Run inside a job:
    sbatch --partition cpu --time 00:10:00 --mem 4G --chdir /scratch/fquareng \
        --output outputs/%j --error job_errors/%j \
        --wrap 'source <CODE_ROOT>/scripts/python_env.sh; cd <CODE_ROOT>; run_python test_prob_metrics.py'
"""

import itertools

import numpy as np
import torch

from prob_metrics import (crps_fair, ensemble_mean_se, ensemble_spread, ensemble_var,
                          floored_rank_histograms, rank_histogram)


def brute(ens, y):
    M = ens.shape[1]
    skill = (ens - y.unsqueeze(1)).abs().mean(dim=1)
    pair = sum((ens[:, i] - ens[:, j]).abs() for i, j in itertools.product(range(M), repeat=2))
    return (skill - pair / (2 * M * (M - 1))).flatten(1).mean(dim=1)


def old_evaluate_crps(ensemble, truth):
    """evaluate.py's former compute_crps (global mean, numpy)."""
    M = ensemble.shape[1]
    abs_err = np.abs(ensemble - truth[:, np.newaxis]).mean()
    spread = np.mean([np.abs(ensemble[:, i] - ensemble[:, j]).mean()
                      for i in range(M) for j in range(i + 1, M)])
    return float(abs_err - 0.5 * spread)


torch.manual_seed(0)
for M in (2, 3, 4, 8, 16):
    ens, y = torch.randn(6, M, 1, 7, 9), torch.randn(6, 1, 7, 9)
    got = crps_fair(ens, y)
    assert got.shape == (6,), got.shape
    assert torch.allclose(got, brute(ens, y), atol=1e-5), M
    assert abs(got.mean().item() - old_evaluate_crps(ens.numpy(), y.numpy())) < 1e-5, M
    sp = ensemble_spread(ens)
    assert np.allclose(sp.numpy(), ens.numpy().std(axis=1).reshape(6, -1).mean(axis=1), atol=1e-5), M
    va = ensemble_var(ens)
    assert va.shape == (6,) and np.allclose(va.numpy(), ens.numpy().var(axis=1, ddof=1).reshape(6, -1).mean(axis=1),
                                            atol=1e-5), M
    se = ensemble_mean_se(ens, y)
    assert np.allclose(se.numpy(), ((ens.numpy().mean(axis=1) - y.numpy()) ** 2).reshape(6, -1).mean(axis=1),
                       atol=1e-5), M

# Identical members: the pairwise term vanishes and CRPS is the MAE.
e = torch.randn(4, 1, 5, 5).unsqueeze(1).repeat(1, 8, 1, 1, 1)
t = torch.randn(4, 1, 5, 5)
assert torch.allclose(crps_fair(e, t), (e[:, 0] - t).abs().flatten(1).mean(1), atol=1e-6)

# Fair estimator: unbiased in M for a calibrated ensemble (members and truth
# from the same N(0, 1)); E[CRPS] = 1/sqrt(pi).
big = [crps_fair(torch.randn(20000, M, 1), torch.randn(20000, 1)).mean().item() for M in (2, 4, 16)]
assert all(abs(b - 1 / np.sqrt(np.pi)) < 0.02 for b in big), big

try:
    crps_fair(torch.randn(2, 1, 3), torch.randn(2, 3))
    raise AssertionError("M=1 should raise")
except ValueError:
    pass

# Rank histogram: brute-force ranks without ties, counts sum to the pixels,
# flat for a calibrated ensemble, all mass at the ends for a collapsed one,
# and ties split evenly.
ens, y = torch.randn(5, 4, 1, 6, 6), torch.randn(5, 1, 6, 6)
h = rank_histogram(ens, y)
assert h.shape == (5, 5) and (h.sum(1) == 36).all()
ref = (ens < y.unsqueeze(1)).sum(1).flatten(1)
assert all(torch.equal(h[b], torch.bincount(ref[b], minlength=5)) for b in range(5))
flat = rank_histogram(torch.randn(1, 7, 200000), torch.randn(1, 200000))[0].double()
assert (flat / flat.mean() - 1).abs().max() < 0.03, flat
lo = rank_histogram(torch.zeros(1, 3, 1000), torch.full((1, 1000), -1.0))[0]
assert lo[0] == 1000 and lo[1:].sum() == 0
tie = rank_histogram(torch.zeros(1, 3, 400000), torch.zeros(1, 400000))[0].double()
assert (tie / 100000 - 1).abs().max() < 0.02, tie

# Spread-skill: for members and truth i.i.d. N(0, 1), E[s^2] = 1 (ddof 1) and
# E[(xbar - y)^2] = 1 + 1/M, so the ratio with the (M + 1) / M factor is 1.
# Without the factor (or with ddof 0) it is biased low, most at small M.
for M in (2, 4, 16):
    e, t = torch.randn(4000, M, 50), torch.randn(4000, 50)
    v, s = ensemble_var(e).mean().item(), ensemble_mean_se(e, t).mean().item()
    assert abs(np.sqrt((M + 1) / M * v / s) - 1) < 0.02, (M, v, s)
    assert abs(v - 1) < 0.02 and abs(s - (1 + 1 / M)) < 0.03 * (1 + 1 / M), (M, v, s)
try:
    ensemble_var(torch.randn(2, 1, 3))
    raise AssertionError("ensemble_var with M=1 should raise")
except ValueError:
    pass

# Masked histogram: an all-true mask reproduces the unmasked one (same draws),
# an all-false mask counts nothing.
g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
ens, y = torch.randn(3, 4, 1, 8, 8), torch.randn(3, 1, 8, 8)
assert torch.equal(rank_histogram(ens, y, g1), rank_histogram(ens, y, g2, mask=torch.ones_like(y, dtype=torch.bool)))
assert rank_histogram(ens, y, mask=torch.zeros_like(y, dtype=torch.bool)).sum() == 0

# Floored ranks. Dry truth among all-dry members: uniform over the M + 1 ranks
# (the unfloored ranks would follow the sub-floor scatter instead), and no
# wet pixels. Wet truth above every member: rank M, counted in both.
M = 4
dry_e = torch.rand(1, M, 400000) * 0.05            # members < 0.1 mm
dry_t = torch.rand(1, 400000) * 0.05
h, hw = floored_rank_histograms(dry_e, dry_t, 0.1)
assert h.shape == (1, M + 1) and h.sum() == 400000 and hw.sum() == 0
assert ((h[0].double() / (400000 / (M + 1)) - 1).abs() < 0.02).all(), h
h, hw = floored_rank_histograms(dry_e[:, :, :1000], torch.full((1, 1000), 5.0), 0.1)
assert h[0, M] == 1000 and torch.equal(h, hw)
# Mixed: the wet histogram counts exactly the truth >= floor pixels, and
# equals the masked histogram of the floored fields under the same draws.
e, t = torch.rand(2, M, 5000) * 0.3, torch.rand(2, 5000) * 0.3
h, hw = floored_rank_histograms(e, t, 0.1, torch.Generator().manual_seed(3))
assert torch.equal(hw.sum(1), (t >= 0.1).sum(1)) and (h.sum(1) == 5000).all()
ref = rank_histogram(e.clamp(min=0.1), t.clamp(min=0.1), torch.Generator().manual_seed(3), mask=t >= 0.1)
assert torch.equal(hw, ref)

print("prob_metrics OK", [round(b, 4) for b in big], "expected", round(1 / np.sqrt(np.pi), 4))
