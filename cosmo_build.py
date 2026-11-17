"""COSMO perfect-model arm, C1: build RainShift-format stores (PLAN_COSMO.md B2, B3).

Two steps:

  month  --month YYYYMM  : read every hour of one month once, cut the five
         regions, write one staging .npz per region to --stage_dir. A
         region-month already staged is skipped (resume), never rewritten.
  merge  --region R      : concatenate the staged months of one region into
         {out_root}/{R}/train_data_{in,out}.zarr (2011-2019) and
         test_data_{in,out}.zarr (2020-2021), write static_variables.nc and
         normalization_stats.json (train split only, by the dataset's own
         _compute_and_cache_stats). Refuses an existing region directory.

Conventions (all fixed here, documented in the store attributes):

  time       valid time t, hourly, "hours since 1900-01-01" as RainShift.
  target     precipitation [mm/h] at 2.2 km, 200x200: sum of the 12 5-min
             TOT_PREC files with stamps in (t-1h, t]. A 5-min file stamped s
             covers (previous stamp, s] (time_bnds, read 2026-11-09); stamps
             snap to the 20 s model step, so files are binned by parsed time,
             never by constructed names. Exactly 12 per hour or the task fails.
  inputs     10x block mean of 200x200 -> 20x20, RainShift names and units:
             tp    block mean of the target / 1000          [m]   (ERA5 tp)
             cape  CAPE_ML                                   [J/kg]
             tcw   TQV+TQC+TQI+TQR+TQS+TQG                   [kg/m2] (ERA5 tcw is total column water)
             tclw  TQC                                       [kg/m2]
             sp    PS                                        [Pa]
             u, v  U_10M, V_10M (10 m, rotated-grid relative) [m/s] -- NOT 700 hPa
             tisr  ASOD_T * 3600 (hour mean -> hour integral) [J/m2]
             cp    0 everywhere: a convection-permitting model has no
                   parameterised convective precipitation. Physically exact,
                   keeps the 9-channel models, needs no trainer change.
             Instantaneous fields are at t (cell_methods "time: point").
  statics    from data_cosmo_pm10/_statics/statics.npz (cosmo_statics.py):
             z = hsurf * g [m2/s2] (geopotential, as RainShift), lsm = land (0/1).
  grid       native rotated grid, no reprojection. latitude/longitude are the
             1-D rotated coordinates rlat/rlon; true lat/lon are lat2d/lon2d in
             static_variables.nc.

Region boxes: 200x200 native points centred on the grid point nearest to the
pre-registered centre (REGIONS); a box closer than --margin points to the
domain edge is an error (lateral relaxation zone).
"""

import argparse
import datetime as dt
import json
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ARCHIVE = Path("/reference/FGSE/climate_simulations")
RUN_DIRS = {"present": "cosmo6_2km_climate", "pgw": "cosmo6_2km_climate_PGW_MPI_HR"}
G = 9.80665
BOX, FACTOR = 200, 10
EPOCH = dt.datetime(1900, 1, 1)

# Pre-registered centres (PLAN_COSMO.md B2), fixed before any data is looked at.
REGIONS = {
    "scandinavia_south": (58.0, 10.5),
    "spain_central": (40.0, -4.0),
    "alps": (46.5, 10.0),
    "british_isles": (52.5, -2.0),
    "poland": (52.0, 19.0),
}
INPUT_VARS = ["cape", "cp", "sp", "tclw", "tcw", "tisr", "tp", "u", "v"]
TQ_ALL = ["TQV", "TQC", "TQI", "TQR", "TQS", "TQG"]
TRAIN_YEARS, TEST_YEARS = range(2011, 2020), range(2020, 2022)
STAMP = re.compile(r"lffd(\d{14})\.nz$")


def region_boxes(grid_npz, margin):
    g = np.load(grid_npz)
    lat, lon = g["lat"], g["lon"]
    ny, nx = lat.shape
    boxes = {}
    for name, (la, lo) in REGIONS.items():
        d = (lat - la) ** 2 + ((lon - lo) * np.cos(np.deg2rad(la))) ** 2
        i, j = np.unravel_index(np.argmin(d), d.shape)
        i0, j0 = i - BOX // 2, j - BOX // 2
        if min(i0, j0) < margin or i0 + BOX > ny - margin or j0 + BOX > nx - margin:
            raise SystemExit(f"ERROR: {name} box [{i0}:{i0+BOX}, {j0}:{j0+BOX}] within {margin} of the edge")
        boxes[name] = (int(i0), int(j0))
    return boxes


def block_mean(a):
    n = a.shape[0]
    return a.reshape(n, BOX // FACTOR, FACTOR, BOX // FACTOR, FACTOR).mean(axis=(2, 4))


def list_5min(run, month):
    """{stamp datetime: path} for the month and the hour before it."""
    d = ARCHIVE / RUN_DIRS[run] / "5min_2D"
    first = dt.datetime.strptime(month, "%Y%m")
    prev = (first - dt.timedelta(hours=1)).strftime("%Y%m")
    prefixes = ("lffd" + prev, "lffd" + month)
    out = {}
    for e in os.scandir(d):   # one pass over the 1.18 M entries (~8 s, C0)
        if e.name.startswith(prefixes):
            m = STAMP.match(e.name)
            if m:
                out[dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")] = e.path
    return out


def read_full(path, names):
    with Dataset(path) as ds:
        return {v: np.ma.filled(ds[v][0].astype(np.float32), np.nan) for v in names}


def one_hour(args):
    t, run, files5, boxes = args
    hpath = ARCHIVE / RUN_DIRS[run] / "1h_2D" / f"lffd{t:%Y%m%d%H}0000.nz"
    f = read_full(hpath, ["CAPE_ML", "PS", "TQC", "U_10M", "V_10M", "ASOD_T"] + TQ_ALL)
    prec = None
    for p in files5:
        a = read_full(p, ["TOT_PREC"])["TOT_PREC"]
        prec = a if prec is None else prec + a
    res = {}
    for name, (i0, j0) in boxes.items():
        sl = (slice(i0, i0 + BOX), slice(j0, j0 + BOX))
        y = prec[sl]
        cut = {k: v[sl] for k, v in f.items()}
        tcw = sum(cut[k] for k in TQ_ALL)
        dyn = {
            "cape": cut["CAPE_ML"], "cp": np.zeros_like(y), "sp": cut["PS"], "tclw": cut["TQC"],
            "tcw": tcw, "tisr": cut["ASOD_T"] * 3600.0, "tp": y / 1000.0,
            "u": cut["U_10M"], "v": cut["V_10M"],
        }
        x = np.stack([block_mean(dyn[v][None])[0] for v in INPUT_VARS])  # (9, 20, 20)
        res[name] = (x.astype(np.float32), y.astype(np.float32))
    return t, res


def cmd_month(a):
    stage = Path(a.stage_dir) / a.run
    stage.mkdir(parents=True, exist_ok=True)
    boxes = region_boxes(a.grid, a.margin)
    todo = [r for r in REGIONS if not (stage / f"{r}_{a.month}.npz").exists()]
    if not todo:
        print(f"{a.month}: all regions staged; nothing to do")
        return
    boxes = {r: boxes[r] for r in todo}
    first = dt.datetime.strptime(a.month, "%Y%m")
    nxt = (first + dt.timedelta(days=32)).replace(day=1)
    hours = [first + dt.timedelta(hours=h) for h in range(int((nxt - first).total_seconds() // 3600))]
    stamps = list_5min(a.run, a.month)
    sorted_st = sorted(stamps)
    jobs = []
    for t in hours:
        w = [stamps[s] for s in sorted_st if t - dt.timedelta(hours=1) < s <= t]
        if len(w) != 12:
            raise SystemExit(f"ERROR: {t}: {len(w)} 5-min files in (t-1h, t], expected 12")
        jobs.append((t, a.run, w, boxes))
    t0 = time.time()
    with Pool(a.workers) as pool:
        out = sorted(pool.imap(one_hour, jobs, chunksize=4), key=lambda r: r[0])
    th = np.array([int((t - EPOCH).total_seconds() // 3600) for t, _ in out], dtype=np.int32)
    if np.any(np.diff(th) != 1):
        raise SystemExit("ERROR: hours not contiguous")
    for r in todo:
        x = np.stack([res[r][0] for _, res in out])
        y = np.stack([res[r][1] for _, res in out])
        bad = int((~np.isfinite(x)).sum() + (~np.isfinite(y)).sum())
        if bad:
            raise SystemExit(f"ERROR: {r} {a.month}: {bad} non-finite values")
        # by construction: block mean of the target / 1000 == tp
        # float32 rounding: tolerance relative to the largest tp (2026-11-09: a 3.7e-9 m
        # absolute miss in 2015-07 scandinavia_south failed an absolute 1e-9 check)
        tp = x[:, INPUT_VARS.index("tp")]
        err = np.abs(block_mean(y).astype(np.float64) / 1000.0 - tp).max()
        if err > 1e-5 * max(float(np.abs(tp).max()), 1e-6):
            raise SystemExit(f"ERROR: {r}: tp != block mean of target ({err})")
        tmp = stage / f".{r}_{a.month}.npz.tmp.npz"
        np.savez(tmp, x=x, y=y, time=th)
        tmp.rename(stage / f"{r}_{a.month}.npz")
        print(f"  {r} {a.month}: {len(th)} h, mean {y.mean():.4f} mm/h, wet>0.1 {(y > 0.1).mean():.3f}, "
              f"max {y.max():.1f}", flush=True)
    print(f"{a.month}: {len(hours)} h in {time.time() - t0:.0f} s")


def cmd_merge(a):
    import xarray as xr
    from data.dataset_zarr import _compute_and_cache_stats

    root = Path(a.out_root) / a.region
    if root.exists():
        raise SystemExit(f"ERROR: {root} exists; not overwriting.")
    stage = Path(a.stage_dir) / a.run
    boxes = region_boxes(a.grid, a.margin)
    i0, j0 = boxes[a.region]
    g = np.load(a.grid)
    st = np.load(a.statics)
    sl = (slice(i0, i0 + BOX), slice(j0, j0 + BOX))
    rlat = g["rlat"][sl[0]].astype(np.float32)
    rlon = g["rlon"][sl[1]].astype(np.float32)
    attrs = {"source": f"COSMO-CLM 2.2 km {RUN_DIRS[a.run]}, perfect-model 10x block mean",
             "region": a.region, "box_rows_cols": [i0, j0], "centre": list(REGIONS[a.region]),
             "grid": "native rotated pole (latitude/longitude are rlat/rlon)",
             "built_by": "cosmo_build.py", "conventions": "see cosmo_build.py docstring"}
    root.mkdir(parents=True)

    for split, years in (("train", TRAIN_YEARS), ("test", TEST_YEARS)):
        months = [f"{y}{m:02d}" for y in years for m in range(1, 13)]
        missing = [m for m in months if not (stage / f"{a.region}_{m}.npz").exists()]
        if missing:
            raise SystemExit(f"ERROR: {a.region}: {len(missing)} months not staged, e.g. {missing[:3]}")
        last = None
        for k, m in enumerate(months):
            d = np.load(stage / f"{a.region}_{m}.npz")
            if last is not None and d["time"][0] != last + 1:
                raise SystemExit(f"ERROR: gap before {m}")
            last = int(d["time"][-1])
            tc = xr.Variable("time", d["time"], {"units": "hours since 1900-01-01", "calendar": "gregorian"})
            dsi = xr.Dataset({v: (("time", "latitude", "longitude"), d["x"][:, c]) for c, v in enumerate(INPUT_VARS)},
                             coords={"time": tc})
            dso = xr.Dataset({"precipitation": (("time", "latitude", "longitude"), d["y"])},
                             coords={"time": tc, "latitude": rlat, "longitude": rlon})
            for ds, name in ((dsi, "in"), (dso, "out")):
                path = root / f"{split}_data_{name}.zarr"
                if k == 0:
                    ds.attrs.update(attrs)
                    enc = {v: {"chunks": (200,) + ds[v].shape[1:]} for v in ds.data_vars}
                    enc["time"] = {"dtype": "int32"}   # units/calendar live in attrs, as RainShift
                    ds.to_zarr(path, mode="w-", encoding=enc, zarr_format=2, consolidated=True)
                else:
                    ds.to_zarr(path, append_dim="time", zarr_format=2, consolidated=True)
            print(f"  {a.region} {split} {m}", flush=True)

    xr.Dataset(
        {"lsm": (("latitude", "longitude"), st["land"][sl].astype(np.float64)),
         "z": (("latitude", "longitude"), (st["hsurf"][sl] * G).astype(np.float64)),
         "lat2d": (("latitude", "longitude"), g["lat"][sl].astype(np.float32)),
         "lon2d": (("latitude", "longitude"), g["lon"][sl].astype(np.float32))},
        coords={"latitude": rlat, "longitude": rlon}, attrs=attrs,
    ).to_netcdf(root / "static_variables.nc")
    stats = _compute_and_cache_stats(root, INPUT_VARS)
    print(json.dumps(stats))
    print(f"written: {root}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    base = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_cosmo_pm10"
    for name in ("month", "merge"):
        s = sub.add_parser(name)
        s.add_argument("--run", default="present", choices=list(RUN_DIRS))
        s.add_argument("--stage_dir", default="/scratch/fquareng/cosmo_pm10_stage")
        s.add_argument("--grid", default=f"{base}/_c0/grid.npz")
        s.add_argument("--margin", type=int, default=60)
    sub.choices["month"].add_argument("--month", required=True)
    sub.choices["month"].add_argument("--workers", type=int, default=4)
    sub.choices["merge"].add_argument("--region", required=True, choices=list(REGIONS))
    sub.choices["merge"].add_argument("--out_root", default=base)
    sub.choices["merge"].add_argument("--statics", default=f"{base}/_statics/statics.npz")
    a = p.parse_args()
    for path in (getattr(a, "out_root", ""), a.stage_dir):
        if str(path).startswith("/reference"):
            raise SystemExit("ERROR: refusing to write under /reference")
    {"month": cmd_month, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
