"""
Matched-input conditional-shift diagnostic (model-free, binning-based).

NOTE ON NAMING: the repository's `conditional_shift.py` holds the MODEL-BASED
diagnostic (PLAN.md P2: trained per-domain models in the full input space).
This file is the model-free binning diagnostic (P1, claim 1); keep them apart.

WHAT THIS MEASURES
------------------
Whether P(y | cape, tcw) differs between two domains, from observed samples
only: no model, no training. Inputs and target are block-averaged to a common
40 x 40 grid, so every input pixel has one collocated target value. The input
plane is cut into shared quantile bins of (cape, tcw) from the POOLED pixels
of both domains, and within each bin the two domains' target distributions
are compared with the exact empirical Wasserstein-1 distance, averaged over
bins by occupancy:

    W1_cond = sum_b w_b * W1( P_A(y | b), P_B(y | b) )

STATISTICS (all computed on the same bins, all reported)
--------------------------------------------------------
  log_e01      W1 of log(P + 0.1 mm/h). The headline (PLAN.md D1 floor).
  log_e001     W1 of log(P + 0.01 mm/h). Floor check.
  wet_frac     |wet fraction A - wet fraction B| (P >= 0.1 mm/h), the dry/wet
               part of the conditional on its own. Signed (A - B) also kept.
  wet_w1       W1 of log P over wet pixels only (P >= 0.1), weighted by wet
               counts: the amount part on its own.
  legacy_1e-5  W1 of log(P + 1e-5), the statistic of every run before P1b.
               A dry pixel sits at -11.5 there, so it is ~9.5 x |delta dry
               fraction| plus the wet part (audit A4): legacy column only.
"Wet" and the floors apply to 5 x 5 block means of the 200 x 200 target, not
to native pixels.

SUPPORT
-------
A bin counts when EACH domain has pixels from at least --min_weeks distinct
weeks (168-h blocks of the split) in it. Pixels within a frame, and frames
within a week, are strongly dependent, so a pixel count says little (the old
min_count = 200 pixels was 1/8 of one frame: one storm made a bin
"supported"). Support = the smaller of the two domains' pixel fractions in
counted bins, and gates the verdict. The within-bin W1 of cape and of tcw
between the domains is reported (raw and per pooled within-bin sd), over the
coarse bins and over the fine sub-bins of the null below: the coarse value is
the covariate mismatch the binning leaves; the fine value is what the
conditional-permutation null erases, i.e. the size of its liberal bias.

NULLS: WHAT EACH ONE HOLDS FIXED
--------------------------------
Empirical W1 is positive even between identical distributions, and binning
matches inputs only to the bin width. W1_cond is read against a null only.

  conditional permutation (PRIMARY). Pool A and B. Inside each coarse bin cut
    --n_fine x --n_fine pooled-quantile sub-bins of (cape, tcw). Domain labels
    are permuted by whole (domain, week) units: a random set of as many units
    as A has is labelled A, everywhere at once, so each unit's spatial field
    and its coherence across bins stay together. Each relabelled pixel is
    then weighted so that, in every fine sub-bin, each label carries exactly
    the pixel count the real domain has there. Held fixed: each domain's
    (cape, tcw) composition down to the fine sub-bins (hence the coarse-bin
    covariate mismatch A4 says the old null lacks), the sample sizes, the
    weeks, and all dependence inside a week. Destroyed: the domain label
    given fine-sub-bin covariates, which is exactly H0 (shared conditional at
    fine-bin resolution). Where a relabelling leaves a sub-bin without a
    label that the real data has there, that sub-bin falls back to a pixel
    permutation of its observed labels (liberal; the pixel fraction is
    reported). Covariate mismatch inside a fine sub-bin is erased by the
    permutation, so the null is slightly liberal by the "fine" mismatch
    reported above. p = (1 + #{null >= observed}) / (1 + --n_perm).

  one-domain splits (DIAGNOSTICS, kept from P1). One domain split into two
    halves: "pixel" at random (neighbours in both halves: far too low),
    "week" by random halves of the weeks, "year" first test year against the
    second (also carries interannual variability). Held fixed: one domain's
    conditional AND its covariate distribution, so both halves have the same
    position inside every coarse bin. Under pure covariate shift A and B can
    sit at opposite ends of a wide bin and give W1 > 0 that this null never
    contains (audit A4): it is biased towards finding shift and decides
    nothing. To make sizes equal it is reported next to "cross_half", the
    same split applied across domains (A half vs B other half: 300 vs 300
    frames like the null, not 600 vs 600).

UNCERTAINTY AND VERDICT
-----------------------
Week-block bootstrap of W1_cond: weeks drawn with replacement, jointly for
both domains (they share timestamps), the counted bins held fixed. W1 is
biased upwards and more so with fewer distinct weeks, so the bootstrap
interval leans high; it is for magnitude, not the test. Verdict on log_e01:
LOW SUPPORT below --support_threshold; else "within the null" if p >= alpha;
"clear excess" if p < alpha and the CI's lower end clears the permutation
null's 95th percentile; "weak" otherwise. No fixed multiples of a null mean.

KNOWN LIMITATIONS (state these when reporting)
----------------------------------------------
- Pixel-wise vs field-wise conditional. What is tested is
  P(y_pixel | cape_pixel, tcw_pixel). The downscaling model learns
  P(Y_field | X_field). Two domains could share the field-wise conditional yet
  differ pixel-wise if their spatial context differs.
- Projection. Conditioning on two coordinates makes conditional shift look
  LARGER (a difference in an unbinned input shows up as different targets at
  "matched" inputs). A W1_cond inside the null is conservative evidence for a
  shared conditional on (cape, tcw); a W1_cond above it is an upper bound on
  conditional shift, not a measurement of it. P2 is the 9-D counterpart.
- The permutation null assumes (domain, week) units are exchangeable given
  the fine sub-bin. Weeks of the two domains share timestamps but come from
  different regions, so this is plausible, not guaranteed.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import xarray as xr

# Floors of log(P + floor) in mm/h, as evaluate.FLOORS_MM (PLAN.md P14,
# decision D1). Defined here so this CPU script does not import torch;
# test_conditional_shift_binned.py checks the two agree.
FLOORS_MM = {"e001": 0.01, "e01": 0.1}
WET_MM = 0.1
_LEGACY_EPS = 1e-5
WEEK_H = 168
YEAR_H = 8760
HEADLINE = "log_e01"
STATS = ("log_e01", "log_e001", "wet_frac", "wet_w1", "legacy_1e-5")


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
    """Return per-pixel (cape, tcw, precipitation in mm/h) on a common 40x40
    grid, and each pixel's time index in the split (hours from its start).

    Inputs are 80x80 and the target 200x200; both are block-averaged to 40x40
    so each input pixel has one collocated target value. The frame draw
    depends on (n, seed) only, so domains with the same split length get the
    same timestamps.
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
    p = np.clip(p, 0.0, None)

    di.close()
    do.close()
    return (cape.ravel().astype(np.float64), tcw.ravel().astype(np.float64),
            p.ravel().astype(np.float64), np.repeat(idx, 40 * 40))


def _binned_w1(a_cape, a_tcw, a_p, b_cape, b_tcw, b_p, n_bins, min_count):
    """LEGACY (pre-P1b) statistic, kept only to tie a new run to the old
    numbers: occupancy-weighted within-bin W1 of the values given (log(P +
    1e-5) then), bins counted at >= min_count PIXELS per domain."""
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


# ---------------------------------------------------------------------------
#  Weighted within-segment W1 on one sort
# ---------------------------------------------------------------------------


def _cdf_gap(wa, wb, seg, nseg):
    """For values sorted by (segment, value): |F_a - F_b| at every position
    but the last, F the within-segment weighted empirical CDFs, and each
    segment's total weight per sample. Multiplying by the value increments
    (zeroed across segment boundaries) and summing per segment gives the
    exact weighted W1 of each segment, for any labels or weights, without
    re-sorting."""
    ta = np.bincount(seg, wa, nseg)
    tb = np.bincount(seg, wb, nseg)
    oa = np.concatenate([[0.0], np.cumsum(ta)[:-1]])
    ob = np.concatenate([[0.0], np.cumsum(tb)[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        fa = (np.cumsum(wa) - oa[seg]) / ta[seg]
        fb = (np.cumsum(wb) - ob[seg]) / tb[seg]
    return np.nan_to_num(np.abs(fa - fb)[:-1]), ta, tb


def _agg(per_seg, weight):
    """Weighted mean over the segments where the value is finite."""
    ok = np.isfinite(per_seg) & (weight > 0)
    if not ok.any():
        return np.nan
    return float((per_seg[ok] * weight[ok]).sum() / weight[ok].sum())


def _edges(v, n):
    e = np.quantile(v, np.linspace(0, 1, n + 1))
    e[0] -= 1e-6
    e[-1] += 1e-6
    return e


class Design:
    """Two labelled samples (lab True = A) pooled for the binned statistic.

    Coarse bins: n_bins x n_bins quantile bins of (cape, tcw) from the pooled
    pixels. A bin is counted when each sample has pixels from >= min_weeks
    distinct weeks in it; pixels outside counted bins are dropped. The kept
    pixels are sorted once by (bin, P); every statistic below is a function
    of labels and weights on that order.
    """

    def __init__(self, cape, tcw, p, t, lab, n_bins, min_weeks):
        lab = np.asarray(lab, bool)
        t = np.asarray(t)
        ce, te = _edges(cape, n_bins), _edges(tcw, n_bins)
        nb = n_bins * n_bins
        cb = (np.digitize(cape, ce) - 1) * n_bins + (np.digitize(tcw, te) - 1)
        wk = t // WEEK_H
        span = int(wk.max()) + 1

        def weeks(m):
            u = np.unique(cb[m] * span + wk[m])
            return np.bincount(u // span, minlength=nb)

        self.weeks_a, self.weeks_b = weeks(lab), weeks(~lab)
        self.bins = np.flatnonzero((self.weeks_a >= min_weeks) & (self.weeks_b >= min_weeks))
        self.nseg = len(self.bins)
        keep = np.isin(cb, self.bins)
        self.support = (float(min(keep[lab].mean(), keep[~lab].mean()))
                        if lab.any() and (~lab).any() else 0.0)

        seg_of_bin = np.full(nb, -1)
        seg_of_bin[self.bins] = np.arange(self.nseg)
        idx = np.flatnonzero(keep)
        seg = seg_of_bin[cb[idx]]
        o = np.lexsort((p[idx], seg))
        self.idx = idx[o]
        self.seg = seg[o]
        self.n = len(self.idx)
        self.start = np.searchsorted(self.seg, np.arange(self.nseg + 1))
        ps = p[self.idx]
        cut = self.seg[1:] != self.seg[:-1]

        def inc(v):
            d = np.diff(v)
            d[cut] = 0.0
            return d

        self.d = {f"log_{k}": inc(np.log(ps + f)) for k, f in FLOORS_MM.items()}
        self.d["legacy_1e-5"] = inc(np.log(ps + _LEGACY_EPS))
        self.d["wet_w1"] = inc(np.log(np.maximum(ps, WET_MM)))
        self.wet = ps >= WET_MM
        self.lab = lab[self.idx]
        self.wk = wk[self.idx]
        self.cape = np.asarray(cape)[self.idx]
        self.tcw = np.asarray(tcw)[self.idx]

    def stats(self, lab=None, w=None):
        """All STATS for labels `lab` and pixel weights `w` (sorted order;
        default the observed labels, unit weights)."""
        if self.nseg == 0:
            return {k: np.nan for k in STATS + ("wet_frac_signed",)}
        lab = self.lab if lab is None else lab
        w = np.ones(self.n) if w is None else w
        wa, wb = w * lab, w * ~lab
        sp = self.seg[:-1]
        out = {}

        gap, ta, tb = _cdf_gap(wa, wb, self.seg, self.nseg)
        valid = (ta > 0) & (tb > 0)
        for k in ("log_e01", "log_e001", "legacy_1e-5"):
            per = np.bincount(sp, gap * self.d[k], self.nseg)
            per[~valid] = np.nan
            out[k] = _agg(per, ta + tb)

        wwa, wwb = wa * self.wet, wb * self.wet
        gapw, twa, twb = _cdf_gap(wwa, wwb, self.seg, self.nseg)
        per = np.bincount(sp, gapw * self.d["wet_w1"], self.nseg)
        per[~((twa > 0) & (twb > 0))] = np.nan
        out["wet_w1"] = _agg(per, twa + twb)

        with np.errstate(invalid="ignore", divide="ignore"):
            diff = twa / ta - twb / tb
        diff[~valid] = np.nan
        out["wet_frac"] = _agg(np.abs(diff), ta + tb)
        out["wet_frac_signed"] = _agg(diff, ta + tb)
        return out

    def fine_bins(self, k):
        """Fine sub-bin of every kept pixel: k x k pooled-quantile sub-bins of
        (cape, tcw) inside its coarse bin. Ids are global (seg * k^2 + ...)."""
        fb = np.empty(self.n, np.int64)
        for s in range(self.nseg):
            sl = slice(self.start[s], self.start[s + 1])
            c = np.searchsorted(np.quantile(self.cape[sl], np.linspace(0, 1, k + 1))[1:-1],
                                self.cape[sl], side="right")
            t = np.searchsorted(np.quantile(self.tcw[sl], np.linspace(0, 1, k + 1))[1:-1],
                                self.tcw[sl], side="right")
            fb[sl] = s * k * k + c * k + t
        return fb

    def covariate_mismatch(self, groups=None):
        """Occupancy-weighted within-group W1 of cape and tcw between the
        samples (observed labels), raw and per pooled within-group sd.
        Groups: the coarse bins, or e.g. fine_bins(k)."""
        g = self.seg if groups is None else groups
        _, g = np.unique(g, return_inverse=True)
        ng = int(g.max()) + 1 if g.size else 0
        res = {}
        for name, v in (("cape", self.cape), ("tcw", self.tcw)):
            o = np.lexsort((v, g))
            s, vs, la = g[o], v[o], self.lab[o]
            d = np.diff(vs)
            d[s[1:] != s[:-1]] = 0.0
            gap, ta, tb = _cdf_gap(la.astype(float), (~la).astype(float), s, ng)
            per = np.bincount(s[:-1], gap * d, ng)
            per[~((ta > 0) & (tb > 0))] = np.nan
            cnt = np.bincount(s, minlength=ng)
            mu = np.bincount(s, vs, ng) / np.maximum(cnt, 1)
            sd = np.sqrt(np.maximum(np.bincount(s, vs * vs, ng) / np.maximum(cnt, 1) - mu ** 2, 0.0))
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = per / sd
            res[f"w1_{name}"] = _agg(per, ta + tb)
            res[f"w1_{name}_per_sd"] = _agg(rel, ta + tb)
        return res


# ---------------------------------------------------------------------------
#  Primary null: conditional permutation by (domain, week) units
# ---------------------------------------------------------------------------


class CondPerm:
    """Draws of the conditional-permutation null for a Design (see the module
    docstring). draw() returns (labels, weights, fallback pixel fraction) to
    pass to Design.stats."""

    def __init__(self, design, n_fine):
        self.D = design
        self.fb = design.fine_bins(n_fine)
        self.nf = max(design.nseg * n_fine * n_fine, 1)
        lab = design.lab
        self.na_obs = np.bincount(self.fb[lab], minlength=self.nf).astype(float)
        self.nb_obs = np.bincount(self.fb[~lab], minlength=self.nf).astype(float)
        uw, winv = np.unique(design.wk, return_inverse=True)
        self.unit = lab.astype(np.int64) * len(uw) + winv
        self.n_units = 2 * len(uw)
        self.present = np.unique(self.unit)
        self.n_a_units = int((self.present >= len(uw)).sum())

    def draw(self, rng):
        lab_u = np.zeros(self.n_units, bool)
        lab_u[rng.permutation(self.present)[: self.n_a_units]] = True
        plab = lab_u[self.unit]
        na = np.bincount(self.fb[plab], minlength=self.nf)
        nb = np.bincount(self.fb[~plab], minlength=self.nf)
        bad = ((self.na_obs > 0) & (na == 0)) | ((self.nb_obs > 0) & (nb == 0))
        frac = 0.0
        if bad.any():
            ii = np.flatnonzero(bad[self.fb])
            frac = ii.size / self.D.n
            obs = ii[np.lexsort((~self.D.lab[ii], self.fb[ii]))]
            rnd = ii[np.lexsort((rng.random(ii.size), self.fb[ii]))]
            plab[rnd] = self.D.lab[obs]
            na = np.bincount(self.fb[plab], minlength=self.nf)
            nb = np.bincount(self.fb[~plab], minlength=self.nf)
        with np.errstate(invalid="ignore", divide="ignore"):
            ra, rb = self.na_obs / na, self.nb_obs / nb
        w = np.where(plab, ra[self.fb], rb[self.fb])
        return plab, np.nan_to_num(w, nan=0.0, posinf=0.0), frac


def week_bootstrap(design, n_boot, rng):
    """W1 statistics under week-block resampling (weeks drawn jointly for both
    samples, counted bins fixed)."""
    uw, winv = np.unique(design.wk, return_inverse=True)
    out = []
    for _ in range(n_boot):
        m = rng.multinomial(len(uw), np.full(len(uw), 1.0 / len(uw)))
        out.append(design.stats(w=m[winv].astype(float)))
    return out


def perm_null(design, n_fine, n_perm, rng):
    cp = CondPerm(design, n_fine)
    vals, fr = [], []
    for _ in range(n_perm):
        lab, w, f = cp.draw(rng)
        vals.append(design.stats(lab, w))
        fr.append(f)
    return vals, float(np.mean(fr)) if fr else np.nan, cp


def p_value(obs, null):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan
    return float((1 + (null >= obs).sum()) / (1 + null.size))


def _col(rows, k):
    return np.array([r[k] for r in rows], float)


def _summ(v):
    v = np.asarray(v, float)
    if not np.isfinite(v).any():
        return {"mean": None, "sd": None, "p95": None, "n": int(v.size)}
    return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
            "p95": float(np.nanpercentile(v, 95)), "n": int(v.size)}


def verdict(obs, null, ci, support, alpha, support_threshold):
    if support < support_threshold:
        return f"LOW SUPPORT ({support:.0%}) - not interpretable"
    p = p_value(obs, null)
    if not np.isfinite(p):
        return "null undefined"
    if p >= alpha:
        return f"within the conditional-permutation null (p={p:.3f}) -> no evidence beyond covariate shift"
    if ci[0] > np.nanpercentile(null, 95):
        return f"clear excess (p={p:.3f}, CI above null p95) -> conditional shift on (cape, tcw), upper bound"
    return f"weak excess (p={p:.3f}, CI reaches the null's upper tail) -> inconclusive"


# ---------------------------------------------------------------------------
#  Diagnostic nulls: one domain split in halves, with a size-matched cross
# ---------------------------------------------------------------------------

NULL_MODES = ("pixel", "week", "year")


def split_masks(t, mode, rng, n_rep):
    """Boolean half masks over pixels: random pixels, random halves of the
    weeks, or the first test year (one split)."""
    if mode == "year":
        h = t < YEAR_H
        if h.all() or not h.any():
            raise ValueError("null mode year: the split does not span two years")
        return [h]
    out = []
    for _ in range(n_rep):
        if mode == "pixel":
            h = np.zeros(t.size, bool)
            h[rng.permutation(t.size)[: t.size // 2]] = True
        else:
            ub = np.unique(t // WEEK_H)
            if len(ub) < 2:
                raise ValueError("null mode week: fewer than two weeks")
            h = np.isin(t // WEEK_H, ub[rng.permutation(len(ub))[: len(ub) // 2]])
        out.append(h)
    return out


def _pair_design(dx, mx, dy, my, n_bins, min_weeks):
    cat = [np.concatenate([dx[i][mx], dy[i][my]]) for i in range(4)]
    lab = np.r_[np.ones(int(mx.sum()), bool), np.zeros(int(my.sum()), bool)]
    return Design(*cat, lab, n_bins, min_weeks)


def block_null_diagnostics(dx, dy, mode, n_rep, n_bins, min_weeks, rng):
    """One-domain split nulls of x and y and the same split across domains
    (x half vs y other half), all at half size."""
    rows = {"null_x": [], "null_y": [], "cross_half": []}
    for h in split_masks(dx[3], mode, rng, n_rep):
        rows["null_x"].append(_pair_design(dx, h, dx, ~h, n_bins, min_weeks).stats())
        rows["null_y"].append(_pair_design(dy, h, dy, ~h, n_bins, min_weeks).stats())
        c = [_pair_design(dx, h, dy, ~h, n_bins, min_weeks).stats(),
             _pair_design(dx, ~h, dy, h, n_bins, min_weeks).stats()]
        rows["cross_half"].append({k: float(np.nanmean([ci[k] for ci in c])) for k in c[0]})
    return {g: {k: _summ(_col(r, k)) for k in STATS} for g, r in rows.items()}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--domains", nargs="+",
                    default=["europe_west", "horn-of-africa", "melanesia"])
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n_frames", type=int, default=600)
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--n_fine", type=int, default=4, help="fine sub-bins per coarse bin and axis (permutation null)")
    ap.add_argument("--min_weeks", type=int, default=10, help="distinct weeks per domain for a bin to count")
    ap.add_argument("--min_count", type=int, default=200, help="pixels; legacy statistic only")
    ap.add_argument("--support_threshold", type=float, default=0.5)
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--null_modes", nargs="*", default=list(NULL_MODES), choices=NULL_MODES,
                    help="one-domain split nulls, diagnostics only")
    ap.add_argument("--null_reps", type=int, default=20,
                    help="random splits per diagnostic null mode (\"year\" has one)")
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists. Not overwriting.")

    root = Path(a.data_root)
    data = {d: _load_matched(root / d, a.split, a.n_frames, a.seed) for d in a.domains}
    t0 = data[a.domains[0]][3]
    for d in a.domains:
        if not np.array_equal(data[d][3], t0):
            raise SystemExit(f"{d}: frame times differ from {a.domains[0]}; the week units assume shared timestamps")
    rng = np.random.default_rng(a.seed)

    res = {}
    print(f"{'pair':34s} {'stat':11s} {'obs':>8} {'null mu':>8} {'null95':>8} {'p':>6} "
          f"{'boot 95% CI':>19} {'sup':>5} {'bins':>4}")
    for x, y in itertools.combinations(a.domains, 2):
        dx, dy = data[x], data[y]
        des = _pair_design(dx, np.ones(dx[0].size, bool), dy, np.ones(dy[0].size, bool),
                           a.n_bins, a.min_weeks)
        obs = des.stats()
        null, fallback, cp = perm_null(des, a.n_fine, a.n_perm, rng)
        boot = week_bootstrap(des, a.n_boot, rng)
        legacy = _binned_w1(dx[0], dx[1], np.log(dx[2] + _LEGACY_EPS),
                            dy[0], dy[1], np.log(dy[2] + _LEGACY_EPS), a.n_bins, a.min_count)

        r = {"support_min": des.support, "n_bins_used": int(des.nseg),
             "weeks_per_bin": {"a": des.weeks_a.tolist(), "b": des.weeks_b.tolist()},
             "observed": obs,
             "legacy_p1": {"w1_cond_1e-5_min_count_pixels": legacy[0], "w1_uncond": legacy[1],
                           "n_bins_used": legacy[2], "support_min": legacy[3]},
             "perm_null": {}, "bootstrap_ci": {}, "excess_over_null_mean": {}, "verdict": {},
             "perm_fallback_pixel_frac": fallback, "perm_units": int(cp.present.size),
             "covariate_mismatch": {"coarse": des.covariate_mismatch(),
                                    "fine": des.covariate_mismatch(cp.fb)}}
        for k in STATS:
            nv, bv = _col(null, k), _col(boot, k)
            ci = (np.nanpercentile(bv, [2.5, 97.5]).tolist() if np.isfinite(bv).any()
                  else [np.nan, np.nan])
            r["perm_null"][k] = {**_summ(nv), "p": p_value(obs[k], nv), "values": nv.tolist()}
            r["bootstrap_ci"][k] = ci
            mu = r["perm_null"][k]["mean"]
            r["excess_over_null_mean"][k] = (obs[k] - mu if mu is not None else None)
            r["verdict"][k] = verdict(obs[k], nv, ci, des.support, a.alpha, a.support_threshold)
            print(f"{x + ' <-> ' + y:34s} {k:11s} {obs[k]:8.4f} {np.nanmean(nv):8.4f} "
                  f"{np.nanpercentile(nv, 95):8.4f} {r['perm_null'][k]['p']:6.3f} "
                  f"[{ci[0]:8.4f},{ci[1]:8.4f}] {des.support:5.0%} {des.nseg:4d}")
        r["headline"] = {"stat": HEADLINE, "verdict": r["verdict"][HEADLINE]}
        mc = r["covariate_mismatch"]
        print(f"  verdict ({HEADLINE}): {r['verdict'][HEADLINE]}")
        print(f"  covariate W1/sd inside bins: coarse cape {mc['coarse']['w1_cape_per_sd']:.3f} "
              f"tcw {mc['coarse']['w1_tcw_per_sd']:.3f}; fine cape {mc['fine']['w1_cape_per_sd']:.3f} "
              f"tcw {mc['fine']['w1_tcw_per_sd']:.3f}; fallback pixels {fallback:.1%}; "
              f"legacy W1_cond(1e-5) {legacy[0]:.4f}")

        r["block_nulls"] = {}
        for mode in a.null_modes:
            bn = block_null_diagnostics(dx, dy, mode, a.null_reps, a.n_bins, a.min_weeks, rng)
            r["block_nulls"][mode] = bn
            h = HEADLINE
            print(f"  diag {mode:5s} ({h}, half size): null {x} {bn['null_x'][h]['mean']}, "
                  f"null {y} {bn['null_y'][h]['mean']}, cross-half {bn['cross_half'][h]['mean']}")
        res[f"{x}__{y}"] = r
        print()

    Path(a.out).write_text(json.dumps({"args": vars(a), "floors_mm": FLOORS_MM, "wet_mm": WET_MM,
                                       "headline": HEADLINE, "pairs": res}, indent=2, default=float))
    print(f"Wrote {a.out}")
    print(
        "\nReading the table:\n"
        "  obs is W1_cond on the counted bins; the null is the conditional permutation, which\n"
        "  keeps each domain's (cape, tcw) composition down to the fine sub-bins and permutes\n"
        "  whole (domain, week) units. p is one-sided. The bootstrap CI (weeks) leans high.\n"
        "  wet_frac and wet_w1 split the headline into its dry/wet and amount parts.\n"
        "  Support gates everything; a value above the null is an upper bound on conditional\n"
        "  shift (two binned inputs), a value inside it is conservative evidence against.\n"
        "  The one-domain block nulls are diagnostics: they hold the covariate distribution\n"
        "  fixed and miss the within-bin covariate mismatch (audit A4). Compare them only with\n"
        "  cross_half, which has the same size."
    )


if __name__ == "__main__":
    main()
