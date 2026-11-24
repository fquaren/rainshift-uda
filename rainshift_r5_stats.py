"""
R5 (PLAN.md, 2026-11-17): normalisation stats and target checks for RainShift
regions that have no normalization_stats.json, written into OUR overlay root,
never into the shared raw_data/rainshift copy (CLAUDE.md rule 3).

Stats follow data/dataset_zarr._compute_and_cache_stats exactly: per variable,
nan/inf -> 0, tp/cp x1000, log(clip(., 0) + 1e-5) for tp, cp, precipitation
and z, then the global mean and (population) std in float64 over the TRAINING
store (validation chunks included, as the cached files were made). lsm gets no
entry. The original loads a whole store into memory (~45 GB float64 for the
target); this streams 200-step chunks and merges per-chunk (mean, M2) with
Chan's parallel update, which equals the one-shot result to float64 rounding.

--verify_region recomputes an existing region from the shared copy and
compares it with that region's cached file (rtol 1e-6), before anything is
written. A mismatch stops the job.

For every region it also writes target checks on train and test:
non-finite pixel count, maximum, mean (mm/h), wet fraction (>= 0.1 mm/h),
number of steps, and the store shapes.

Outputs:
  <overlay_root>/<region>/normalization_stats.json   (refused if it exists)
  <check_out>                                         (JSON, refused if it exists)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from data.dataset_zarr import (_LOG_EPS, _LOG_VARS, _PASSTHROUGH_VARS, _UNIT_SCALE, DEFAULT_INPUT_VARS,
                               STATIC_VARS, TARGET_VAR)

CHUNK = 200
WET_MM = 0.1


def _transform(a, var):
    a = np.nan_to_num(a.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if var in _UNIT_SCALE:
        a = a * _UNIT_SCALE[var]
    if var in _LOG_VARS:
        a = np.log(np.clip(a, 0.0, None) + _LOG_EPS)
    return a


class Running:
    """Chan et al. merge of (n, mean, M2); population std at the end."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def add(self, a):
        nb = a.size
        if nb == 0:
            return
        mb = float(a.mean())
        m2b = float(((a - mb) ** 2).sum())
        n = self.n + nb
        d = mb - self.mean
        self.mean += d * nb / n
        self.m2 += m2b + d * d * self.n * nb / n
        self.n = n

    def result(self):
        return [self.mean, float(np.sqrt(self.m2 / self.n))]


def region_stats(root):
    di = xr.open_zarr(root / "train_data_in.zarr")
    do = xr.open_zarr(root / "train_data_out.zarr")
    nt = do.sizes["time"]
    if di.sizes["time"] != nt:
        raise SystemExit(f"{root}: input/output time lengths differ ({di.sizes['time']} vs {nt})")
    acc = {v: Running() for v in DEFAULT_INPUT_VARS if v not in _PASSTHROUGH_VARS}
    acc[TARGET_VAR] = Running()
    for t0 in range(0, nt, CHUNK):
        sl = slice(t0, t0 + CHUNK)
        for v in acc:
            src = do if v == TARGET_VAR else di
            acc[v].add(_transform(src[v].isel(time=sl).values, v))
    di.close()
    do.close()
    stats = {v: acc[v].result() for v in DEFAULT_INPUT_VARS if v in acc}
    stats[TARGET_VAR] = acc[TARGET_VAR].result()
    st = xr.open_dataset(root / "static_variables.nc")
    for v in STATIC_VARS:
        if v in _PASSTHROUGH_VARS:
            continue
        r = Running()
        r.add(_transform(st[v].values, v))
        stats[v] = r.result()
    st.close()
    return stats


def target_check(root):
    out = {}
    for split in ("train", "test"):
        di = xr.open_zarr(root / f"{split}_data_in.zarr")
        do = xr.open_zarr(root / f"{split}_data_out.zarr")
        y = do[TARGET_VAR]
        nt = y.sizes["time"]
        nonfinite, mx, s, wet, npix = 0, -np.inf, 0.0, 0, 0
        for t0 in range(0, nt, CHUNK):
            a = y.isel(time=slice(t0, t0 + CHUNK)).values
            fin = np.isfinite(a)
            nonfinite += int((~fin).sum())
            a = np.where(fin, a, 0.0).astype(np.float64)
            mx = max(mx, float(a.max()))
            s += float(a.sum())
            wet += int((a >= WET_MM).sum())
            npix += a.size
        out[split] = {"n_steps": int(nt), "target_shape": list(y.shape),
                      "input_shapes": {v: list(di[v].shape) for v in DEFAULT_INPUT_VARS if v in di},
                      "nonfinite_pixels": nonfinite, "max_mm": mx, "mean_mm": s / npix,
                      "wet_frac": wet / npix,
                      "time_start": str(do["time"].values[0]), "time_end": str(do["time"].values[-1])}
        di.close()
        do.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlay_root", required=True)
    ap.add_argument("--new_regions", nargs="+", required=True, help="regions to compute stats for")
    ap.add_argument("--check_regions", nargs="+", required=True, help="regions to run target checks on")
    ap.add_argument("--verify_region", required=True, help="existing region whose cached file is the check")
    ap.add_argument("--check_out", required=True)
    a = ap.parse_args()
    ov = Path(a.overlay_root)
    check_out = Path(a.check_out)
    if check_out.exists():
        raise SystemExit(f"{check_out} exists; not overwriting.")
    for r in a.new_regions:
        if (ov / r / "normalization_stats.json").exists():
            raise SystemExit(f"{ov / r / 'normalization_stats.json'} exists; not overwriting.")

    ref = json.loads((ov / a.verify_region / "normalization_stats.json").read_text())
    got = region_stats(ov / a.verify_region)
    worst = 0.0
    for v, (m, s) in ref.items():
        gm, gs = got[v]
        rel = max(abs(gm - m) / max(abs(m), 1e-12), abs(gs - s) / max(abs(s), 1e-12))
        worst = max(worst, rel)
        print(f"verify {a.verify_region:14s} {v:14s} mean {m:+.6e} vs {gm:+.6e}  std {s:.6e} vs {gs:.6e}  rel {rel:.1e}")
    if set(ref) != set(got) or worst > 1e-6:
        raise SystemExit(f"verification failed (worst rel {worst:.1e}, keys {sorted(set(ref) ^ set(got))})")
    print(f"verification passed, worst rel {worst:.1e}")

    for r in a.new_regions:
        stats = region_stats(ov / r)
        p = ov / r / "normalization_stats.json"
        if p.exists() or p.is_symlink():
            raise SystemExit(f"{p} appeared; not overwriting.")
        p.write_text(json.dumps(stats, indent=2))
        print(f"wrote {p}")

    checks = {r: target_check(ov / r) for r in a.check_regions}
    check_out.parent.mkdir(parents=True, exist_ok=True)
    check_out.write_text(json.dumps({"verify_worst_rel": worst, "regions": checks}, indent=2))
    for r, c in checks.items():
        for sp, d in c.items():
            print(f"{r:18s} {sp:5s} steps {d['n_steps']:6d}  nonfinite {d['nonfinite_pixels']:9d}  "
                  f"max {d['max_mm']:8.2f}  mean {d['mean_mm']:.4f}  wet {d['wet_frac']:.3f}  "
                  f"{d['time_start'][:10]}..{d['time_end'][:10]}")
    print(f"wrote {check_out}")


if __name__ == "__main__":
    main()
