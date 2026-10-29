"""CPU checks of the binned conditional-shift test (conditional_shift_binned.py,
PLAN.md P1b). Run inside a job; scripts/run_cond_binned_test.sh runs it first.

1. The one-sort weighted statistic equals the exact W1 per bin, for every
   variant, and integer weights equal repetition (the bootstrap).
2. The conditional permutation keeps each label's fine-sub-bin counts.
3. Pure covariate shift INSIDE the coarse bins (same conditional, A sits at
   the high-CAPE end of every bin, B at the low end): the old one-domain week
   null flags shift, the conditional-permutation null does not.
4. A true conditional shift is flagged by the conditional-permutation null.
"""

import numpy as np

from conditional_shift_binned import (
    FLOORS_MM,
    WET_MM,
    CondPerm,
    Design,
    _binned_w1,
    _w1_exact,
    block_null_diagnostics,
    p_value,
    perm_null,
)
from evaluate import FLOORS_MM as EVAL_FLOORS

assert FLOORS_MM == EVAL_FLOORS, (FLOORS_MM, EVAL_FLOORS)
rng = np.random.default_rng(0)

# --- 1. exactness: one bin, all variants, and weights = repetition
n = 3000
pa = np.where(rng.random(n) < 0.4, 0.0, rng.lognormal(-1.0, 1.5, n))
pb = np.where(rng.random(n) < 0.6, 0.0, rng.lognormal(-0.5, 1.0, n))
cape, tcw = rng.random(2 * n), rng.random(2 * n)
t = np.tile(np.arange(n) * 7, 2)
lab = np.r_[np.ones(n, bool), np.zeros(n, bool)]
des = Design(cape, tcw, np.r_[pa, pb], t, lab, n_bins=1, min_weeks=1)
s = des.stats()
for k, f in FLOORS_MM.items():
    assert np.isclose(s[f"log_{k}"], _w1_exact(np.log(pa + f), np.log(pb + f)), rtol=1e-9), k
assert np.isclose(s["legacy_1e-5"], _w1_exact(np.log(pa + 1e-5), np.log(pb + 1e-5)), rtol=1e-9)
assert np.isclose(s["wet_w1"], _w1_exact(np.log(pa[pa >= WET_MM]), np.log(pb[pb >= WET_MM])), rtol=1e-9)
assert np.isclose(s["wet_frac"], abs((pa >= WET_MM).mean() - (pb >= WET_MM).mean()), rtol=1e-12)
assert np.isclose(s["wet_frac_signed"], (pa >= WET_MM).mean() - (pb >= WET_MM).mean(), rtol=1e-12)

m = rng.integers(0, 4, 2 * n)
sw = des.stats(w=m[des.idx].astype(float))
ma, mb = m[:n], m[n:]
assert np.isclose(sw["log_e01"], _w1_exact(np.repeat(np.log(pa + 0.1), ma), np.repeat(np.log(pb + 0.1), mb)),
                  rtol=1e-9), "integer weights = repetition"

# several bins: same edges and weighting as the legacy pixel-gated statistic
des5 = Design(cape, tcw, np.r_[pa, pb], t, lab, n_bins=3, min_weeks=1)
leg = _binned_w1(cape[:n], tcw[:n], np.log(pa + 0.1), cape[n:], tcw[n:], np.log(pb + 0.1), 3, 1)
assert des5.nseg == 9 and np.isclose(des5.stats()["log_e01"], leg[0], rtol=1e-9), (des5.stats(), leg)
print("exactness: one-sort W1 = exact W1 for every variant; weights = repetition")


# --- simulation: weeks, shared timestamps, week random effects
def simulate(cell_a, cell_b, cond_shift_b=0.0, n_weeks=60, fpw=3, pix=400, beta=6.0, tau=0.1, sig=0.2):
    """Pooled cape uniform on [0, 1]: within each 0.2-wide coarse cape bin,
    four 0.05-wide cells get A probabilities cell_a and B probabilities
    cell_b (cell_a + cell_b = 1/2 each, so the pooled density is flat and
    pooled quantile sub-bins sit on the cells). Inside a cell both are
    uniform. y = 2 + beta*cape + week effect + noise (+ shift for B)."""
    t = np.repeat((np.arange(n_weeks)[:, None] * 168 + np.arange(fpw) * 24).ravel(), pix)
    out = []
    for dom, cell, shift in ((0, cell_a, 0.0), (1, cell_b, cond_shift_b)):
        k = t.size
        c = 0.2 * rng.integers(0, 5, k) + 0.05 * (rng.choice(4, size=k, p=cell) + rng.random(k))
        e = rng.normal(0, tau, n_weeks)[t // 168]
        y = 2.0 + beta * c + e + rng.normal(0, sig, k) + shift
        out.append((c, rng.random(k), np.exp(y), t))
    return out


def pooled(dx, dy):
    cat = [np.concatenate([dx[i], dy[i]]) for i in range(4)]
    lab = np.r_[np.ones(dx[0].size, bool), np.zeros(dy[0].size, bool)]
    return Design(*cat, lab, n_bins=5, min_weeks=10)


tilt = np.array([0.35, 0.45, 0.55, 0.65]) / 2
flat = np.full(4, 0.25)

# --- 2. the permutation keeps fine composition per label
dx, dy = simulate(tilt, tilt[::-1])
des = pooled(dx, dy)
cp = CondPerm(des, 4)
plab, w, frac = cp.draw(rng)
na = np.bincount(cp.fb[plab], weights=w[plab], minlength=cp.nf)
nb = np.bincount(cp.fb[~plab], weights=w[~plab], minlength=cp.nf)
assert np.allclose(na, cp.na_obs) and np.allclose(nb, cp.nb_obs), "fine composition kept"
assert not np.array_equal(plab, des.lab)
print(f"permutation: fine-sub-bin counts per label kept (fallback pixels {frac:.1%})")

# --- 3. pure covariate shift inside the coarse bins
n_sim, n_perm = 10, 39
rej_new, flag_old = 0, 0
for _ in range(n_sim):
    dx, dy = simulate(tilt, tilt[::-1])
    des = pooled(dx, dy)
    obs = des.stats()["log_e01"]
    null, _, _ = perm_null(des, 4, n_perm, rng)
    nv = np.array([r["log_e01"] for r in null])
    rej_new += p_value(obs, nv) < 0.05
    old = block_null_diagnostics(dx, dy, "week", 10, 5, 10, rng)
    old_p95 = max(old["null_x"]["log_e01"]["p95"], old["null_y"]["log_e01"]["p95"])
    flag_old += obs > old_p95
    assert nv.mean() > old_p95, ("the permutation null keeps the within-bin mismatch", nv.mean(), old_p95)
print(f"pure covariate shift: old week null flags {flag_old}/{n_sim}, conditional permutation {rej_new}/{n_sim}")
assert flag_old >= 8, flag_old
assert rej_new <= 2, rej_new

# --- 4. true conditional shift, with and without covariate shift
for cells, shift in (((flat, flat), 0.15), ((tilt, tilt[::-1]), -0.15)):
    dx, dy = simulate(*cells, cond_shift_b=shift)
    des = pooled(dx, dy)
    obs = des.stats()["log_e01"]
    null, _, _ = perm_null(des, 4, n_perm, rng)
    p = p_value(obs, [r["log_e01"] for r in null])
    assert p < 0.05, (shift, obs, p)
    print(f"conditional shift {shift:+.2f}: W1 {obs:.3f}, p {p:.3f}")

print("test_conditional_shift_binned: all passed")
