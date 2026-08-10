"""
Matched-input conditional-shift diagnostic (model-free, binning-based).

NOTE ON NAMING: the repository's `conditional_shift.py` currently holds the
MODEL-BASED diagnostic (it compares trained per-domain models in the full input
space). This file is the model-free binning diagnostic; keep the two separate.

WHAT THIS MEASURES
------------------
Whether P(y | x) differs between two domains, using observed samples only: no
model, no training, no predictions. The construction is a conditional
two-sample test. Bin the input space on a few chosen coordinates using shared
quantile edges from the pooled data, so that within a bin the two domains have
approximately the same values of those coordinates. Then, within each bin,
compare the two domains' distributions of the OBSERVED target y with the exact
empirical Wasserstein-1 distance, and average over bins weighted by occupancy:

    W1_cond = sum_b w_b * W1( P_A(y | b), P_B(y | b) )

Under pure covariate shift, matched inputs imply matched targets and W1_cond
falls to the noise floor. Under conditional shift the same atmospheric state
produces different precipitation and W1_cond stays well above it.

WHAT CHANGED, AND WHY (read before using previous numbers)
----------------------------------------------------------
1. W1_cond is now the PRIMARY statistic. The earlier version reported the ratio
   r = W1_cond / W1_uncond as the headline. That ratio is unstable: its
   denominator can be arbitrarily small for reasons unrelated to conditional
   shift, e.g. when a difference in P(x) offsets a difference in P(y|x) so the
   two output marginals happen to coincide. A small denominator inflates r at
   perfectly adequate support, so a large r is NOT by itself evidence of
   conditional shift. The ratio is still reported, as context only.

2. A NULL BASELINE is now computed. Empirical W1 between two finite samples is
   positive even when the distributions are identical, and positive again from
   residual covariate variation WITHIN a bin (binning matches inputs only to
   the bin width, and only on the binned coordinates). Both inflate W1_cond by
   an unknown amount. We therefore split ONE domain at random into two halves
   and run the identical pipeline between them. That yields the value the
   statistic takes under a true null, and W1_cond is only interpretable
   relative to it. A W1_cond not clearly above the null is not evidence of
   anything.

3. Common support is reported and gates everything, as before.

KNOWN LIMITATIONS (state these when reporting)
----------------------------------------------
- Pixel-wise vs field-wise conditional. Inputs and target are coarsened to a
  common grid and pixels are treated as exchangeable samples, so what is tested
  is P(y_pixel | cape_pixel, tcw_pixel). The downscaling model instead learns
  P(Y_field | X_field), where precipitation at a pixel depends on the
  surrounding atmospheric state. Two domains could share the field-wise
  conditional yet differ pixel-wise if their spatial context differs. This is a
  genuine mismatch between what is measured and what the model uses.
- Effective sample size. Pixels within a frame are strongly spatially
  correlated, so the number of INDEPENDENT samples per bin is far below the raw
  count and `min_count` is more permissive than it appears. The null baseline
  partly absorbs this, since it inherits the same correlation structure.
- One-sided error from the projection. Conditioning on only a few coordinates
  can only ever make conditional shift look LARGER (a difference in an unbinned
  variable appears as different targets at "matched" inputs). So a LOW W1_cond
  is conservative and trustworthy; a HIGH W1_cond is an upper bound.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import xarray as xr

_LOG_EPS = 1e-5


def _w1_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Exact 1-D empirical Wasserstein-1 via the CDF-difference integral."""
    if a.size == 0 or b.size == 0:
        return np.nan
    a = np.sort(a)
    b = np.sort(b)
    allv = np.sort(np.concatenate([a, b]))
    d = np.diff(allv)
    ca = np.searchsorted(a, allv[:-1], side="right") / a.size
    cb = np.searchsorted(b, allv[:-1], side="right") / b.size
    return float(np.sum(np.abs(ca - cb) * d))


def _load_matched(domain_path: Path, split: str, n_frames: int, seed: int):
    """Return per-pixel (cape, tcw, log-precip) on a common 40x40 grid.

    Inputs are 80x80 and the target 200x200; both are block-averaged to 40x40
    so each input pixel has one collocated target value.
    """
    pre = "test" if split == "test" else "train"
    di = xr.open_zarr(domain_path / f"{pre}_data_in.zarr")
    do = xr.open_zarr(domain_path / f"{pre}_data_out.zarr")

    n = di.sizes["time"]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=min(n_frames, n), replace=False))

    cape = np.nan_to_num(di["cape"].isel(time=idx).values, nan=0.0)
    tcw = np.nan_to_num(di["tcw"].isel(time=idx).values, nan=0.0)
    p = np.nan_to_num(do["precipitation"].isel(time=idx).values,
                      nan=0.0, posinf=0.0, neginf=0.0)

    F = p.shape[0]
    p = p.reshape(F, 40, 5, 40, 5).mean(axis=(2, 4))
    cape = cape.reshape(F, 40, 2, 40, 2).mean(axis=(2, 4))
    tcw = tcw.reshape(F, 40, 2, 40, 2).mean(axis=(2, 4))
    p = np.log(np.clip(p, 0.0, None) + _LOG_EPS)

    di.close()
    do.close()
    return cape.ravel(), tcw.ravel(), p.ravel()


def _binned_w1(a_cape, a_tcw, a_p, b_cape, b_tcw, b_p, n_bins, min_count):
    """Occupancy-weighted within-bin W1 plus support fractions.

    Bin edges come from the POOLED data so both sets occupy the same bins.
    """
    ce = np.quantile(np.concatenate([a_cape, b_cape]), np.linspace(0, 1, n_bins + 1))
    te = np.quantile(np.concatenate([a_tcw, b_tcw]), np.linspace(0, 1, n_bins + 1))
    ce[0] -= 1e-6; ce[-1] += 1e-6
    te[0] -= 1e-6; te[-1] += 1e-6

    aci = np.digitize(a_cape, ce) - 1
    ati = np.digitize(a_tcw, te) - 1
    bci = np.digitize(b_cape, ce) - 1
    bti = np.digitize(b_tcw, te) - 1

    w, v = [], []
    a_in = b_in = 0
    for ci, ti in itertools.product(range(n_bins), range(n_bins)):
        am = (aci == ci) & (ati == ti)
        bm = (bci == ci) & (bti == ti)
        na, nb = int(am.sum()), int(bm.sum())
        if na < min_count or nb < min_count:
            continue
        w.append(na + nb)
        v.append(_w1_exact(a_p[am], b_p[bm]))
        a_in += na
        b_in += nb

    if not w:
        return np.nan, np.nan, 0, 0.0
    w = np.asarray(w, float)
    v = np.asarray(v, float)
    cond = float((w * v).sum() / w.sum())
    uncond = _w1_exact(a_p, b_p)
    support = min(a_in / max(a_p.size, 1), b_in / max(b_p.size, 1))
    return cond, uncond, len(w), support


def null_baseline(cape, tcw, p, n_bins, min_count, seed):
    """W1_cond under a TRUE null: one domain split at random into two halves.

    Runs the identical pipeline, so the returned value contains exactly the
    same finite-sample noise and residual within-bin covariate variation as a
    real cross-domain comparison. It is the floor W1_cond must clear.
    """
    rng = np.random.default_rng(seed)
    m = rng.permutation(p.size)
    h = p.size // 2
    i, j = m[:h], m[h:]
    cond, _, nb, _ = _binned_w1(cape[i], tcw[i], p[i],
                                cape[j], tcw[j], p[j], n_bins, min_count)
    return cond, nb


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--domains", nargs="+",
                    default=["europe_west", "horn-of-africa", "melanesia"])
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n_frames", type=int, default=300)
    ap.add_argument("--n_bins", type=int, default=8)
    ap.add_argument("--min_count", type=int, default=200)
    ap.add_argument("--support_threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="conditional_shift_binned.json")
    a = ap.parse_args()

    root = Path(a.data_root)
    data = {d: _load_matched(root / d, a.split, a.n_frames, a.seed) for d in a.domains}

    # Null floor, one per domain: identical pipeline on a random split.
    print("=== null baseline (same domain split in half; W1_cond under a true null) ===")
    nulls = {}
    for d in a.domains:
        c, t, p = data[d]
        nv, nb = null_baseline(c, t, p, a.n_bins, a.min_count, a.seed)
        nulls[d] = nv
        print(f"  {d:22s} W1_cond_null = {nv:.4f}   ({nb} bins)")
    null_med = float(np.nanmedian(list(nulls.values())))
    print(f"  median null floor = {null_med:.4f}\n")

    print("=== matched-input conditional discrepancy ===")
    print(f"{'pair':34s} {'W1_cond':>8} {'/null':>7} {'W1_unc':>8} "
          f"{'ratio':>7} {'support':>8} {'bins':>5}  verdict")
    res = {}
    for x, y in itertools.combinations(a.domains, 2):
        cond, unc, nb, sup = _binned_w1(*data[x], *data[y], a.n_bins, a.min_count)
        floor = np.nanmean([nulls[x], nulls[y]])
        excess = cond / floor if floor and np.isfinite(floor) and floor > 0 else np.nan
        ratio = cond / unc if unc and np.isfinite(unc) and unc > 0 else np.nan

        # Verdict is driven by W1_cond RELATIVE TO THE NULL, never by the ratio.
        if sup < a.support_threshold:
            verdict = f"LOW SUPPORT ({sup:.0%}) - not interpretable"
        elif not np.isfinite(excess):
            verdict = "null undefined"
        elif excess < 1.5:
            verdict = "at the noise floor -> covariate shift"
        elif excess < 3.0:
            verdict = "weak excess -> inconclusive"
        else:
            verdict = "clear excess -> conditional shift"

        res[f"{x}__{y}"] = {"w1_cond": cond, "w1_uncond": unc,
                            "null_floor": floor, "excess_over_null": excess,
                            "ratio_cond_over_uncond": ratio,
                            "support_min": sup, "n_bins_used": nb,
                            "verdict": verdict}
        print(f"{x+' <-> '+y:34s} {cond:8.4f} {excess:7.2f} {unc:8.4f} "
              f"{ratio:7.2f} {sup:8.1%} {nb:5d}  {verdict}")

    Path(a.out).write_text(json.dumps({"nulls": nulls, "pairs": res}, indent=2))
    print(f"\nWrote {a.out}")
    print(
        "\nReading the table:\n"
        "  W1_cond is the statistic. Compare it to the null floor (/null column);\n"
        "  the floor is what this estimator returns when the two samples come\n"
        "  from the SAME distribution, so it absorbs finite-sample noise and the\n"
        "  residual covariate variation left inside each bin.\n"
        "  The ratio W1_cond/W1_uncond is context only: its denominator can be\n"
        "  small for reasons unrelated to conditional shift, so a large ratio is\n"
        "  not on its own evidence.\n"
        "  Support gates everything. And because binning conditions on only\n"
        "  (cape, tcw), a LOW W1_cond is conservative while a HIGH one is an\n"
        "  upper bound."
    )


if __name__ == "__main__":
    main()
