"""COSMO perfect-model arm: static fields from the model's own output.

The archive (/reference/FGSE/climate_simulations, read-only) carries no
constants file: no HSURF, no FR_LAND (C0, PLAN_COSMO.md B5). Both are
recovered here from the hourly single-level output, so the statics are the
model's own and need no external DEM remapped to the rotated grid.

Surface height. COSMO's PMSL is PS reduced to sea level through its own
surface height. Inverting the hypsometric equation
    z = (R_d / g) * T_m * ln(PMSL / PS),   T_m = T_2M + gamma * z / 2
(gamma = 6.5 K/km, three fixed-point iterations) recovers that height up to
the reduction's temperature assumption, which differs from T_2M under
inversions and over high terrain. Done for --n_times instants spread over
one year (both 00 and 12 UTC); the median is the field and the per-point
spread across instants is its error, reported, not hidden.

Land mask. RUNOFF_G (subsurface runoff) is written only at land points and
carries _FillValue elsewhere; land = not filled. Checked to be identical at
every instant, and checked against the height (sea points should invert to
|z| of a few metres).

Writes statics.npz (hsurf, hsurf_spread, land, lat, lon) and statics.json to a
new --out_dir; refuses an existing one. Reads nothing but 4 variables of
--n_times hourly files.
"""

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ARCHIVE = Path("/reference/FGSE/climate_simulations/cosmo6_2km_climate/1h_2D")
R_D, G, GAMMA = 287.05, 9.80665, 0.0065


def instants(year, n):
    """n instants spread evenly over the year, alternating 00 and 12 UTC."""
    start = dt.datetime(year, 1, 1)
    days = np.linspace(0, 364, n).round().astype(int)
    return [start + dt.timedelta(days=int(d), hours=12 * (i % 2)) for i, d in enumerate(days)]


def read(path, names):
    with Dataset(path) as ds:
        out = {}
        for v in names:
            a = ds[v][0]
            out[v] = np.ma.getmaskarray(a) if v == "RUNOFF_G" else np.ma.filled(a.astype(np.float64), np.nan)
        lat, lon = ds["lat"][:], ds["lon"][:]
    return out, np.asarray(lat), np.asarray(lon)


def height(ps, pmsl, t2m, iters=3):
    z = np.zeros_like(ps)
    for _ in range(iters):
        z = (R_D / G) * (t2m + GAMMA * z / 2) * np.log(pmsl / ps)
    return z


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--year", type=int, default=2015)
    p.add_argument("--n_times", type=int, default=96)
    args = p.parse_args()

    out = Path(args.out_dir)
    if str(out.resolve()).startswith("/reference"):
        raise SystemExit("ERROR: refusing to write under /reference")
    if out.exists():
        raise SystemExit(f"ERROR: {out} exists; not overwriting.")

    t0 = time.time()
    zs, masks, lat, lon, used = [], [], None, None, []
    for t in instants(args.year, args.n_times):
        f = ARCHIVE / f"lffd{t:%Y%m%d%H}0000.nz"
        if not f.exists():
            print(f"  missing {f.name}", flush=True)
            continue
        d, la, lo = read(f, ["PS", "PMSL", "T_2M", "RUNOFF_G"])
        if lat is None:
            lat, lon = la, lo
        zs.append(height(d["PS"], d["PMSL"], d["T_2M"]).astype(np.float32))
        masks.append(d["RUNOFF_G"])
        used.append(f.name)
    if not zs:
        raise SystemExit("ERROR: no files read")

    Z = np.stack(zs)                       # (T, 1542, 1542)
    hsurf = np.median(Z, axis=0)
    q75, q25 = np.percentile(Z, [75, 25], axis=0)
    spread = (q75 - q25).astype(np.float32)
    filled = np.stack(masks)
    land = ~filled[0]
    mask_stable = bool((filled == filled[0]).all())

    def pct(a):
        a = a[np.isfinite(a)]
        return {k: float(v) for k, v in zip(("p1", "p50", "p99", "max"), np.percentile(a, [1, 50, 99, 100]))}

    summary = {
        "year": args.year, "n_used": len(used), "files_first_last": [used[0], used[-1]],
        "method": "z = R_d/g * (T_2M + gamma z/2) * ln(PMSL/PS), median over instants",
        "hsurf": {"min": float(np.nanmin(hsurf)), "max": float(np.nanmax(hsurf)), "nonfinite": int((~np.isfinite(hsurf)).sum())},
        "spread_iqr_land": pct(spread[land]),
        "spread_iqr_land_above_1500m": pct(spread[land & (hsurf > 1500)]),
        "land_fraction": float(land.mean()),
        "land_mask_identical_at_all_instants": mask_stable,
        "sea_abs_height": pct(np.abs(hsurf[~land])),
        "land_below_minus10m": int((land & (hsurf < -10)).sum()),
        "seconds": round(time.time() - t0, 1),
    }
    out.mkdir(parents=True)
    np.savez_compressed(out / "statics.npz", hsurf=hsurf.astype(np.float32), hsurf_spread=spread,
                        land=land, lat=lat.astype(np.float32), lon=lon.astype(np.float32))
    (out / "statics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
