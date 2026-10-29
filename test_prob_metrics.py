"""CPU check of prob_metrics against brute force. Run inside a job:
    sbatch --partition cpu --time 00:10:00 --mem 4G --chdir /scratch/fquareng \
        --output outputs/%j --error job_errors/%j \
        --wrap 'source <CODE_ROOT>/scripts/python_env.sh; cd <CODE_ROOT>; run_python test_prob_metrics.py'
"""

import itertools

import numpy as np
import torch

from prob_metrics import crps_fair, ensemble_spread, rank_histogram


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

print("prob_metrics OK", [round(b, 4) for b in big], "expected", round(1 / np.sqrt(np.pi), 4))
