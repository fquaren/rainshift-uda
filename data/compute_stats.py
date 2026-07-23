"""
Precompute per-domain normalization statistics for the RainShift Zarr stores.

Writes {domain}/normalization_stats.json with per-variable [mean, std] computed
in the SAME transformed space the training loss uses:
    units (tp, cp x1000) -> log(clip(x, 0) + 1e-5) for _LOG_VARS -> mean/std.

Design notes
------------
- Streaming (chunk-by-chunk) Welford accumulation: never materialises a full
  variable array, so it does not OOM on large targets (melanesia's full target
  is ~56 GB). This also avoids the failure mode that produced a NaN stat —
  computing mean/std over a full array containing a few non-finite values.
- Non-finite scrubbing: every block is passed through
  np.nan_to_num(nan=0, posinf=0, neginf=0) in physical space before the log,
  matching convert_zarr_to_npy. RainShift targets contain a tiny number of
  non-finite pixels (IMERG gaps); scrubbing to zero is safe at that scale.
- lsm is passthrough (no stat entry); z (geopotential) is log-transformed and
  DOES get a stat entry.

Run one domain per SLURM array task (see compute_stats.sh).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


DEFAULT_INPUT_VARS = ["cape", "cp", "sp", "tclw", "tcw", "tisr", "tp", "u", "v"]
TARGET_VAR = "precipitation"
STATIC_VARS = ["lsm", "z"]

_LOG_VARS = {"tp", "cp", "precipitation", "z"}
_PASSTHROUGH_VARS = {"lsm"}
_UNIT_SCALE = {"tp": 1000.0, "cp": 1000.0}
_LOG_EPS = 1e-5

_CHUNK = 200  # time-chunk size in the RainShift stores


def _transform_block(a: np.ndarray, var: str) -> np.ndarray:
    """units -> scrub non-finite -> log (for _LOG_VARS). Returns float64.
    Scrubbing happens in physical space, before the log, matching the
    converter. Non-finite values become 0 (i.e. 'no precipitation' / neutral)."""
    a = a.astype(np.float64)
    if var in _UNIT_SCALE:
        a = a * _UNIT_SCALE[var]
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if var in _LOG_VARS:
        a = np.log(np.clip(a, 0.0, None) + _LOG_EPS)
    return a


class _Welford:
    """Streaming mean/variance over arbitrarily many blocks (numerically
    stable, single pass). Tracks count, mean, M2."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, block: np.ndarray):
        b = block.reshape(-1)
        bn = b.size
        if bn == 0:
            return
        bmean = float(b.mean())
        bvar = float(b.var())  # population variance of the block
        # Chan et al. parallel/streaming combination of two sample sets.
        delta = bmean - self.mean
        tot = self.n + bn
        self.mean += delta * bn / tot
        self.M2 += bvar * bn + delta * delta * self.n * bn / tot
        self.n = tot

    def result(self):
        if self.n == 0:
            return float("nan"), float("nan")
        var = self.M2 / self.n
        return float(self.mean), float(np.sqrt(max(var, 0.0)))


def _stat_streaming(da: xr.DataArray, var: str, n_time: int) -> tuple:
    """Stream over time-chunks, accumulating Welford stats on the transformed,
    scrubbed block. Never holds more than one chunk in memory."""
    w = _Welford()
    n_bad = 0
    for t0 in range(0, n_time, _CHUNK):
        t1 = min(t0 + _CHUNK, n_time)
        raw = da.isel(time=slice(t0, t1)).values
        n_bad += int((~np.isfinite(raw)).sum())
        w.update(_transform_block(raw, var))
    return w.result(), n_bad


def _stat_static(arr: np.ndarray, var: str) -> tuple:
    """Static fields are small (200x200); transform+scrub whole, then stat."""
    t = _transform_block(arr, var)
    return float(t.mean()), float(np.sqrt(max(t.var(), 0.0)))


def compute_domain(domain_path: Path, input_vars) -> dict:
    di = xr.open_zarr(domain_path / "train_data_in.zarr")
    do = xr.open_zarr(domain_path / "train_data_out.zarr")
    ds_static = xr.open_dataset(domain_path / "static_variables.nc")

    stats = {}
    total_bad = 0

    n_time = di.sizes["time"]
    for v in input_vars:
        if v in _PASSTHROUGH_VARS:
            continue
        (m, s), nb = _stat_streaming(di[v], v, n_time)
        stats[v] = [m, s]
        total_bad += nb
        print(f"  {v:6s} mean={m:12.5f} std={s:12.5f}  non-finite={nb}", flush=True)

    (m, s), nb = _stat_streaming(do[TARGET_VAR], TARGET_VAR, do.sizes["time"])
    stats[TARGET_VAR] = [m, s]
    total_bad += nb
    print(f"  {TARGET_VAR:6s} mean={m:12.5f} std={s:12.5f}  non-finite={nb}", flush=True)

    for v in STATIC_VARS:
        if v in _PASSTHROUGH_VARS:
            continue
        m, s = _stat_static(ds_static[v].values, v)
        stats[v] = [m, s]
        print(f"  {v:6s} mean={m:12.5f} std={s:12.5f}  (static)", flush=True)

    di.close()
    do.close()
    ds_static.close()

    # Guard: refuse to write a stat file containing any non-finite entry.
    for k, (m, s) in stats.items():
        if not (np.isfinite(m) and np.isfinite(s)):
            raise ValueError(f"Non-finite stat for '{k}': [{m}, {s}] — aborting.")

    print(f"  total non-finite values scrubbed: {total_bad}", flush=True)
    return stats


def main():
    p = argparse.ArgumentParser(description="Precompute RainShift normalization stats (streaming, scrubbed).")
    p.add_argument("--domain_path", required=True, help="path to one domain dir")
    p.add_argument("--input_vars", nargs="+", default=DEFAULT_INPUT_VARS)
    p.add_argument("--force", action="store_true", help="overwrite an existing normalization_stats.json")
    args = p.parse_args()

    root = Path(args.domain_path)
    out = root / "normalization_stats.json"
    if out.exists() and not args.force:
        print(f"{out} exists; use --force to overwrite. Skipping.")
        return

    print(f"Computing stats for {root.name} ...", flush=True)
    stats = compute_domain(root, args.input_vars)
    out.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()