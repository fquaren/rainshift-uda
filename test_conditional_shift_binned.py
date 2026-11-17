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


# --- 5. loader geometry (2026-11-13, COSMO C2): tiny synthetic stores
import tempfile
from pathlib import Path

import xarray as xr

from conditional_shift_binned import _load_matched, input_shift, input_marginals


def store(root, n_in, n_out, T=30, tp_is_block_mean=False):
    """data_in / data_out zarr pair as RainShift / cosmo_build write them."""
    p = rng.gamma(0.3, 2.0, (T, n_out, n_out)).astype(np.float32)
    b = n_out // n_in
    tp = p.reshape(T, n_in, b, n_in, b).mean(axis=(2, 4)) if tp_is_block_mean else rng.random((T, n_in, n_in))
    dims = ("time", "lat", "lon")
    xr.Dataset({"cape": (dims, rng.random((T, n_in, n_in))), "tcw": (dims, rng.random((T, n_in, n_in))),
                "tp": (dims, tp.astype(np.float32))},
               coords={"time": np.arange(T)}).to_zarr(root / "test_data_in.zarr")
    xr.Dataset({"precipitation": (dims, p)}, coords={"time": np.arange(T)}).to_zarr(root / "test_data_out.zarr")
    return p


tmp = Path(tempfile.mkdtemp(prefix="cond_binned_test_"))

# RainShift geometry: block scale reproduces the old hard-coded 40x40 loader.
(tmp / "rs").mkdir()
p_rs = store(tmp / "rs", 80, 200)
c, tc, y, tt = _load_matched(tmp / "rs", "test", 10, 3)
idx = np.sort(np.random.default_rng(3).choice(30, size=10, replace=False))
di = xr.open_zarr(tmp / "rs" / "test_data_in.zarr")
old_c = di["cape"].isel(time=idx).values.reshape(10, 40, 2, 40, 2).mean(axis=(2, 4))
old_p = np.clip(p_rs[idx].reshape(10, 40, 5, 40, 5).mean(axis=(2, 4)), 0, None)
assert np.allclose(c, old_c.ravel()) and np.allclose(y, old_p.ravel(), rtol=1e-6), "block = old loader"
assert np.array_equal(tt, np.repeat(idx, 1600))

# COSMO geometry, tp = block mean of the target: block scale makes y a function
# of the conditioning input (the degenerate case), fine scale does not.
(tmp / "cosmo").mkdir()
p_c = store(tmp / "cosmo", 20, 200, tp_is_block_mean=True)
tp_b, cape_b, y_b, _ = _load_matched(tmp / "cosmo", "test", 10, 3, cond=("tp", "cape"))
assert y_b.size == 10 * 400 and np.allclose(y_b, np.clip(tp_b, 0, None), rtol=1e-5), "block: y = tp"
tp_f, cape_f, y_f, t_f = _load_matched(tmp / "cosmo", "test", 10, 3, cond=("tp", "cape"), target_scale="fine")
assert y_f.size == 10 * 200 * 200 and np.allclose(y_f, p_c[idx].ravel(), rtol=1e-6), "fine: native pixels"
cell = tp_f.reshape(10, 200, 200)[:, ::10, ::10]          # one pixel per 10x10 cell
assert np.allclose(tp_f.reshape(10, 20, 10, 20, 10), cell[:, :, None, :, None]), "covariates constant per cell"
assert np.allclose(y_f.reshape(10, 20, 10, 20, 10).mean(axis=(2, 4)), cell, rtol=1e-5), "cell mean of y = tp"
# Subsample: a store whose target value IS the flat pixel position (exact in
# float32 below 2^24), so the kept positions can be read off the values.
(tmp / "pos").mkdir()
T, dims = 30, ("time", "lat", "lon")
xr.Dataset({"cape": (dims, rng.random((T, 20, 20))), "tcw": (dims, rng.random((T, 20, 20)))},
           coords={"time": np.arange(T)}).to_zarr(tmp / "pos" / "test_data_in.zarr")
posf = np.broadcast_to(np.arange(200 * 200, dtype=np.float32).reshape(200, 200), (T, 200, 200))
xr.Dataset({"precipitation": (dims, np.array(posf))},
           coords={"time": np.arange(T)}).to_zarr(tmp / "pos" / "test_data_out.zarr")
k = 7
cape_k, _, y_k, t_k = _load_matched(tmp / "pos", "test", 10, 3, target_scale="fine", fine_per_cell=k)
assert y_k.size == 10 * 400 * k and t_k.size == y_k.size
pk = y_k.reshape(10, 400 * k).astype(np.int64)
assert (pk == pk[0]).all(), "same fine positions in every frame"
assert np.unique(pk[0]).size == 400 * k, "no position twice"
cell_of = (pk[0] // 200) // 10 * 20 + (pk[0] % 200) // 10
assert np.all(np.bincount(cell_of, minlength=400) == k), "k pixels in every cell"
cape_cells = xr.open_zarr(tmp / "pos" / "test_data_in.zarr")["cape"].isel(time=idx).values.reshape(10, 400)
assert np.allclose(cape_k.reshape(10, -1), cape_cells[:, cell_of]), "each pixel carries its cell's covariates"

# input shift: identical domains give 0, a shifted copy gives the shift.
m = input_marginals(tmp / "cosmo", "test", 10, 3)
assert set(m) == {"cape", "tcw", "tp"}
assert all(v["w1"] == 0 for v in input_shift(m, m).values())
sh = input_shift({"cape": m["cape"]}, {"cape": m["cape"] + 0.5})["cape"]
assert np.isclose(sh["w1"], 0.5) and np.isclose(sh["w1_per_sd_x"], 0.5 / m["cape"].std())
print(f"loader: block = old 40x40 loader; COSMO block degenerate (y = tp), fine native, "
      f"{k}/cell subsample fixed across frames; input W1 exact")

print("test_conditional_shift_binned: all passed")
