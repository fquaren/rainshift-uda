"""
Per-channel input ranges, in the normalised units the model sees, for a set
of (region, split) stores normalised by ONE stats region. Tests the working
hypothesis of PLAN.md R5 "Read 2026-11-24": the west-africa in-domain UNet
blows up (up to 1e5 mm/h) because some test-period inputs lie outside the
west-africa training range.

Each store is streamed one 200-step time chunk at a time (a dask reduction
over a whole store OOMs, job 20391536) and transformed with
data.dataset_zarr._transform_channel, i.e. exactly as the dataset does:
nan_to_num in physical space, tp/cp x1000, log for tp/cp, z-score with the
stats region's (mean, std).

Writes to --output_dir (refused if it exists):
  ranges.csv          per (entry, var): physical min/max, z min/max, z
                      quantiles 1e-6 .. 1-1e-6 (histogram, 0.01 bins), the
                      fraction of non-finite raw pixels, and for log vars the
                      smallest positive decoded value (the dry floor set by
                      the int16 packing, which differs per store)
  exceed.csv          per (entry, var) for every entry that is not the stats
                      region's train store: the fraction of pixels and of
                      timesteps beyond the train store's z min / max
  <entry>_tmax.npy    per-timestep (n_time, n_var, 2) min and max z, for
                      linking to the timesteps where a model blows up later
The "train" store holds the validation chunks too, so the train range is an
upper bound on what the model was fitted on, and the exceedances are
conservative.

Usage (inside a job, see scripts/rainshift_input_ranges.sh):
  rainshift_input_ranges.py --data_root .../data_rainshift_r5 \
      --stats_region west-africa --output_dir ... \
      --entry west-africa:train --entry west-africa:test --entry melanesia:test
The first entry must be <stats_region>:train.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr

from data.dataset_zarr import DEFAULT_INPUT_VARS, _LOG_VARS, _UNIT_SCALE, _transform_channel

CHUNK = 200
EDGES = np.arange(-60.0, 60.0 + 1e-9, 0.01)   # z bins; anything beyond goes to the end bins
QS = (1e-6, 1e-4, 1e-2, 0.5, 1 - 1e-2, 1 - 1e-4, 1 - 1e-6)


def _quantiles(hist, lo_out, hi_out):
    """Quantiles from a histogram plus under/overflow counts (bin left edges)."""
    counts = np.concatenate([[lo_out], hist, [hi_out]]).astype(np.float64)
    cum = np.cumsum(counts) / counts.sum()
    edges = np.concatenate([[-np.inf], EDGES[:-1], [EDGES[-1]]])
    return [float(edges[np.searchsorted(cum, q)]) for q in QS]


def scan(store, var_names, stats, tag, out_dir):
    ds = xr.open_zarr(store, chunks=None)
    n = ds.sizes["time"]
    nv = len(var_names)
    tmax = np.zeros((n, nv, 2), dtype=np.float32)
    acc = {v: dict(pmin=np.inf, pmax=-np.inf, zmin=np.inf, zmax=-np.inf, nonfin=0, npix=0,
                   minpos=np.inf, hist=np.zeros(len(EDGES) - 1, np.int64), lo=0, hi=0)
           for v in var_names}
    for i in range(0, n, CHUNK):
        for j, v in enumerate(var_names):
            raw = np.asarray(ds[v][i:i + CHUNK].values)
            a = acc[v]
            fin = np.isfinite(raw)
            a["nonfin"] += int((~fin).sum())
            a["npix"] += raw.size
            phys = np.where(fin, raw, 0.0).astype(np.float64)
            if v in _UNIT_SCALE:
                phys = phys * _UNIT_SCALE[v]
            a["pmin"] = min(a["pmin"], float(phys.min()))
            a["pmax"] = max(a["pmax"], float(phys.max()))
            if v in _LOG_VARS and (phys > 0).any():
                a["minpos"] = min(a["minpos"], float(phys[phys > 0].min()))
            m, s = stats[v]
            z = _transform_channel(raw, v, m, s)
            a["zmin"] = min(a["zmin"], float(z.min()))
            a["zmax"] = max(a["zmax"], float(z.max()))
            a["lo"] += int((z < EDGES[0]).sum())
            a["hi"] += int((z >= EDGES[-1]).sum())
            a["hist"] += np.histogram(z, bins=EDGES)[0]
            zz = z.reshape(z.shape[0], -1)
            tmax[i:i + zz.shape[0], j, 0] = zz.min(axis=1)
            tmax[i:i + zz.shape[0], j, 1] = zz.max(axis=1)
        print(f"  {tag}: {min(i + CHUNK, n)}/{n}", flush=True)
    ds.close()
    np.save(out_dir / f"{tag}_tmax.npy", tmax)
    return acc, tmax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="read only")
    p.add_argument("--stats_region", required=True)
    p.add_argument("--output_dir", required=True, help="refused if it exists")
    p.add_argument("--entry", action="append", required=True, help="region:split, repeatable")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; not overwriting.")
    entries = [e.split(":", 1) for e in args.entry]
    if entries[0] != [args.stats_region, "train"]:
        raise SystemExit(f"first --entry must be {args.stats_region}:train")
    for r, split in entries:
        if split not in ("train", "test"):
            raise SystemExit(f"split {split!r}: expected train or test")
    data_root = Path(args.data_root)
    stats = json.loads((data_root / args.stats_region / "normalization_stats.json").read_text())
    var_names = list(DEFAULT_INPUT_VARS)
    out_dir.mkdir(parents=True)

    rows, ex_rows, ref = [], [], None
    for r, split in entries:
        tag = f"{r}_{split}"
        print(f"=== {tag} in {args.stats_region} units", flush=True)
        acc, tmax = scan(data_root / r / f"{split}_data_in.zarr", var_names, stats, tag, out_dir)
        for v in var_names:
            a = acc[v]
            qs = _quantiles(a["hist"], a["lo"], a["hi"])
            rows.append(dict(entry=tag, stats=args.stats_region, var=v,
                             phys_min=a["pmin"], phys_max=a["pmax"],
                             min_pos=a["minpos"] if np.isfinite(a["minpos"]) else "",
                             z_min=a["zmin"], z_max=a["zmax"],
                             **{f"z_q{q:g}": x for q, x in zip(QS, qs)},
                             frac_nonfinite=a["nonfin"] / a["npix"]))
        if ref is None:
            ref = {v: (acc[v]["zmin"], acc[v]["zmax"]) for v in var_names}
            continue
        for j, v in enumerate(var_names):
            lo, hi = ref[v]
            a = acc[v]
            # pixel fractions from the histogram at 0.01 resolution: bins wholly
            # beyond the train range (conservative by up to one bin)
            below = a["lo"] + int(a["hist"][EDGES[1:] <= lo].sum())
            above = a["hi"] + int(a["hist"][EDGES[:-1] > hi].sum())
            ex_rows.append(dict(entry=tag, var=v, train_z_min=lo, train_z_max=hi,
                                frac_pix_below=below / a["npix"], frac_pix_above=above / a["npix"],
                                frac_steps_below=float((tmax[:, j, 0] < lo).mean()),
                                frac_steps_above=float((tmax[:, j, 1] > hi).mean()),
                                z_max_excess=a["zmax"] - hi, z_min_excess=lo - a["zmin"]))

    for name, rs in (("ranges.csv", rows), ("exceed.csv", ex_rows)):
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
