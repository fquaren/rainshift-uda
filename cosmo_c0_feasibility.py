"""COSMO perfect-model arm, step C0: feasibility (PLAN_COSMO.md B3). No training.

Reads a handful of files from the COSMO-CLM 2.2 km archive (read-only) and
answers what C1's preprocessing depends on. Writes only to --out_dir.

  inventory   file counts per directory and year; missing hours (1h) and
              missing 5-min steps, from file names only; non-standard names.
  schema      variables, dims, shapes, dtypes, units / long_name /
              cell_methods / grid_mapping of one 1h file, one 5-min file and
              the stray .nc file; global attributes; rotated-pole parameters.
  precip      TOT_PREC semantics. In a 200x200 central window: the 13 5-min
              files of one hour (hh:00 .. hh+1:00), 4 hourly files, and the
              files either side of a day, month and year boundary. Reports the
              window mean and max per file, so an accumulation since model
              start (large, non-decreasing) is told apart from per-step amounts
              (small, independent). Also whether TOT_PREC is in the 1h files.
  constants   whether HSURF / FR_LAND (or any static field) is present in
              any inspected file, and a scan for constant-file names.
  timing      per-file open time and read time of one full field and of one
              200x200 hyperslab, for the variables C1 needs, over 5 files of
              each kind (first read is cold; reported separately).
  grid        2-D lat/lon if present, saved with any static fields to
              grid.npz, for region placement in C1.

Output: <out_dir>/c0.json and grid.npz.
"""

import argparse
import json
import os
import re
import socket
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path("/reference/FGSE/climate_simulations")
RUNS = ["cosmo6_2km_climate", "cosmo6_2km_climate_PGW_MPI_HR"]
NAME_RE = re.compile(r"^lffd(\d{14})\.nz$")
NEEDED = ["TOT_PREC", "CAPE_ML", "TQV", "TQC", "PS", "U_10M", "V_10M", "ASOD_T", "TD_2M"]
STATIC_GUESS = ["HSURF", "FR_LAND", "SOILTYP", "FIS", "lat", "lon", "rlat", "rlon"]
WIN = 200

try:
    import netCDF4
except ImportError:      # recorded in the output; h5py is the fallback
    netCDF4 = None
try:
    import h5py
except ImportError:
    h5py = None


# ---------------------------------------------------------------------------
#  Reading helpers (netCDF4 if available, else h5py)
# ---------------------------------------------------------------------------


class File:
    def __init__(self, path):
        self.path = str(path)
        if netCDF4 is not None:
            self.kind = "netCDF4"
            self.ds = netCDF4.Dataset(self.path, "r")
        elif h5py is not None:
            self.kind = "h5py"
            self.ds = h5py.File(self.path, "r")
        else:
            raise SystemExit("ERROR: neither netCDF4 nor h5py importable")

    def close(self):
        self.ds.close()

    def varnames(self):
        if self.kind == "netCDF4":
            return list(self.ds.variables)
        out = []
        self.ds.visititems(lambda n, o: out.append(n) if isinstance(o, h5py.Dataset) else None)
        return out

    def info(self, name):
        v = self.ds.variables[name] if self.kind == "netCDF4" else self.ds[name]
        if self.kind == "netCDF4":
            attrs = {k: _js(v.getncattr(k)) for k in v.ncattrs()}
            dims = list(v.dimensions)
            filt = v.filters()
            chunk = v.chunking()
        else:
            attrs = {k: _js(v.attrs[k]) for k in v.attrs if not k.startswith("_") and k != "DIMENSION_LIST"}
            dims = []
            filt = {"compression": v.compression, "opts": _js(v.compression_opts)}
            chunk = list(v.chunks) if v.chunks else "contiguous"
        return {"dims": dims, "shape": list(v.shape), "dtype": str(v.dtype), "attrs": attrs,
                "filters": _js(filt), "chunking": _js(chunk)}

    def global_attrs(self):
        if self.kind == "netCDF4":
            return {k: _js(self.ds.getncattr(k)) for k in self.ds.ncattrs()}
        return {k: _js(self.ds.attrs[k]) for k in self.ds.attrs}

    def read(self, name, window=None):
        """Last two dims are (y, x); leading dims (time, level) take index 0."""
        v = self.ds.variables[name] if self.kind == "netCDF4" else self.ds[name]
        if self.kind == "netCDF4":
            v.set_auto_mask(False)
        lead = (0,) * (v.ndim - 2)
        if window is None:
            sl = lead + (slice(None), slice(None))
        else:
            (y0, y1), (x0, x1) = window
            sl = lead + (slice(y0, y1), slice(x0, x1))
        a = np.asarray(v[sl], dtype=np.float64)
        if self.kind == "netCDF4":
            sf = getattr(v, "scale_factor", 1.0) if "scale_factor" in v.ncattrs() else 1.0
            ao = getattr(v, "add_offset", 0.0) if "add_offset" in v.ncattrs() else 0.0
            fv = v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else None
            if fv is not None:
                a[a == fv] = np.nan
            a = a * sf + ao
        return a

    def time_value(self):
        for n in ("time", "Time"):
            if n in self.varnames():
                v = self.ds.variables[n] if self.kind == "netCDF4" else self.ds[n]
                units = (v.getncattr("units") if self.kind == "netCDF4" and "units" in v.ncattrs()
                         else _js(v.attrs.get("units")) if self.kind == "h5py" else None)
                return {"value": _js(np.asarray(v[:]).ravel()[:3]), "units": units}
        return None


def _js(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 64 else f"array{list(x.shape)}"
    if isinstance(x, bytes):
        return x.decode(errors="replace")
    if isinstance(x, (list, tuple)):
        return [_js(i) for i in x]
    if isinstance(x, dict):
        return {str(k): _js(v) for k, v in x.items()}
    return x if isinstance(x, (str, int, float, bool, type(None))) else str(x)


def fname(t):
    return f"lffd{t:%Y%m%d%H%M%S}.nz"


# ---------------------------------------------------------------------------
#  Sections
# ---------------------------------------------------------------------------


def inventory(run_dir):
    res = {}
    for sub, step in (("1h_2D", timedelta(hours=1)), ("5min_2D", timedelta(minutes=5))):
        d = run_dir / sub
        t0 = time.time()
        times, odd = [], []
        with os.scandir(d) as it:
            for e in it:
                m = NAME_RE.match(e.name)
                if m:
                    times.append(datetime.strptime(m.group(1), "%Y%m%d%H%M%S"))
                else:
                    odd.append(e.name)
        times.sort()
        per_year = Counter(t.year for t in times)
        gaps = []
        for a, b in zip(times[:-1], times[1:]):
            if b - a != step:
                gaps.append([a.isoformat(), b.isoformat(), (b - a) / step])
        res[sub] = {
            "n_files": len(times), "first": times[0].isoformat() if times else None,
            "last": times[-1].isoformat() if times else None,
            "per_year": dict(sorted(per_year.items())),
            "n_gaps": len(gaps), "gaps_first_50": gaps[:50],
            "nonstandard_names": odd[:50], "scan_seconds": round(time.time() - t0, 1),
        }
    return res


def schema(paths):
    out = {}
    for label, p in paths.items():
        if not Path(p).exists():
            out[label] = {"missing": str(p)}
            continue
        f = File(p)
        names = f.varnames()
        out[label] = {"path": str(p), "reader": f.kind, "size_bytes": os.path.getsize(p),
                      "global_attrs": f.global_attrs(), "time": f.time_value(),
                      "variables": {n: f.info(n) for n in names}}
        f.close()
    return out


def window_of(shape):
    ny, nx = shape[-2:]
    y0, x0 = ny // 2 - WIN // 2, nx // 2 - WIN // 2
    return (y0, y0 + WIN), (x0, x0 + WIN)


def precip(run_dir, hour):
    """TOT_PREC window statistics over an hour of 5-min files, 4 hourly files
    and day/month/year boundaries."""
    res = {"window": None, "five_min": [], "hourly": [], "boundaries": []}

    def stat(path):
        if not path.exists():
            return {"file": path.name, "missing": True}
        f = File(path)
        if "TOT_PREC" not in f.varnames():
            f.close()
            return {"file": path.name, "has_TOT_PREC": False}
        shape = f.info("TOT_PREC")["shape"]
        w = window_of(shape)
        res["window"] = [list(w[0]), list(w[1])]
        a = f.read("TOT_PREC", w)
        f.close()
        return {"file": path.name, "has_TOT_PREC": True, "mean": float(np.nanmean(a)),
                "max": float(np.nanmax(a)), "min": float(np.nanmin(a)),
                "frac_gt0": float(np.mean(a > 0)), "nan_frac": float(np.mean(~np.isfinite(a)))}

    for k in range(13):
        res["five_min"].append(stat(run_dir / "5min_2D" / fname(hour + timedelta(minutes=5 * k))))
    for k in range(4):
        res["hourly"].append(stat(run_dir / "1h_2D" / fname(hour + timedelta(hours=k))))
    for label, t in (("day", datetime(2015, 7, 2)), ("month", datetime(2015, 8, 1)),
                     ("year", datetime(2016, 1, 1))):
        for sub, step in (("5min_2D", timedelta(minutes=5)), ("1h_2D", timedelta(hours=1))):
            for tt in (t - 2 * step, t - step, t, t + step):
                r = stat(run_dir / sub / fname(tt))
                r.update({"boundary": label, "dir": sub})
                res["boundaries"].append(r)
    return res


def constants(schema_out, run_dir):
    found = defaultdict(list)
    for label, s in schema_out.items():
        for n in s.get("variables", {}):
            if n in STATIC_GUESS or n.upper() in STATIC_GUESS:
                found[n].append(label)
    names = []
    for sub in ("1h_2D", "5min_2D"):
        with os.scandir(run_dir / sub) as it:
            names += [f"{sub}/{e.name}" for e in it if "c" in e.name[4:].lower() or not e.name.startswith("lffd")]
    parent = [e.name for e in os.scandir(run_dir)] + [e.name for e in os.scandir(ROOT)]
    return {"static_vars_found": dict(found), "candidate_constant_files": names[:50],
            "run_dir_entries": parent}


def timing(run_dir, sub, start, step, n=5):
    rows = []
    for k in range(n):
        p = run_dir / sub / fname(start + k * step)
        if not p.exists():
            rows.append({"file": p.name, "missing": True})
            continue
        t0 = time.time()
        f = File(p)
        t_open = time.time() - t0
        have = [v for v in NEEDED if v in f.varnames()]
        r = {"file": p.name, "open_s": round(t_open, 4), "vars": {}}
        for v in have:
            shape = f.info(v)["shape"]
            t0 = time.time()
            f.read(v, window_of(shape))
            t_win = time.time() - t0
            t0 = time.time()
            f.read(v)
            t_full = time.time() - t0
            r["vars"][v] = {"window_s": round(t_win, 4), "full_s": round(t_full, 4)}
        f.close()
        rows.append(r)
    return rows


def grid(paths, out):
    keep = {}
    for p in paths:
        if not Path(p).exists():
            continue
        f = File(p)
        for n in f.varnames():
            if n in STATIC_GUESS and n not in keep:
                try:
                    a = f.read(n) if f.info(n)["shape"].__len__() >= 2 else np.asarray(
                        f.ds.variables[n][:] if f.kind == "netCDF4" else f.ds[n][:])
                    keep[n] = np.asarray(a, dtype=np.float32)
                except Exception as e:          # recorded, not fatal
                    keep[f"{n}__error"] = np.array(str(e))
        f.close()
    if keep:
        np.savez_compressed(out / "grid.npz", **keep)
    return {k: list(v.shape) for k, v in keep.items()}


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hour", default="2015-07-01T12:00", help="hour for the TOT_PREC probe (summer, convective)")
    p.add_argument("--skip_inventory", action="store_true", help="smoke test: skip the 1.18 M-entry scans")
    args = p.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        raise SystemExit(f"ERROR: {out} exists; not overwriting.")
    if str(out.resolve()).startswith(str(ROOT)):
        raise SystemExit("ERROR: refusing to write under the read-only archive")
    out.mkdir(parents=True)

    hour = datetime.fromisoformat(args.hour)
    run = ROOT / RUNS[0]
    res = {"host": socket.gethostname(), "netCDF4": getattr(netCDF4, "__version__", None),
           "h5py": getattr(h5py, "__version__", None), "hour_probe": args.hour}

    def save():
        (out / "c0.json").write_text(json.dumps(res, indent=2, default=str))

    print(f"=== C0 on {res['host']} | netCDF4 {res['netCDF4']} | h5py {res['h5py']}", flush=True)

    paths = {"1h": run / "1h_2D" / fname(hour), "5min": run / "5min_2D" / fname(hour),
             "stray_nc": run / "1h_2D" / "lffd20111231000000.nc"}
    t0 = time.time()
    res["schema"] = schema(paths)
    save()
    print(f"schema done ({time.time() - t0:.0f} s)", flush=True)
    for k, s in res["schema"].items():
        print(f"  {k}: {len(s.get('variables', {}))} variables: {sorted(s.get('variables', {}))[:80]}")
    for k in ("1h", "5min"):
        tp = res["schema"][k].get("variables", {}).get("TOT_PREC")
        print(f"  TOT_PREC in {k}: {'yes ' + json.dumps(tp['attrs']) if tp else 'NO'}")

    res["precip"] = precip(run, hour)
    save()
    print("precip probe (window mean / max):", flush=True)
    for sec in ("five_min", "hourly", "boundaries"):
        for r in res["precip"][sec]:
            print(f"  {sec:10s} {r.get('boundary', ''):5s} {r['file']}: "
                  + (f"mean={r['mean']:.4g} max={r['max']:.4g} frac>0={r['frac_gt0']:.3f}"
                     if r.get("has_TOT_PREC") else json.dumps(r)))

    res["timing_1h"] = timing(run, "1h_2D", hour, timedelta(hours=1))
    res["timing_5min"] = timing(run, "5min_2D", hour, timedelta(minutes=5))
    save()
    print("timing done", flush=True)

    res["grid_saved"] = grid([paths["1h"], paths["stray_nc"], paths["5min"]], out)
    res["constants"] = constants(res["schema"], run)
    save()
    print(f"grid.npz: {res['grid_saved']}\nconstants: {json.dumps(res['constants']['static_vars_found'])}"
          f"\ncandidate constant files: {res['constants']['candidate_constant_files'][:10]}", flush=True)

    if not args.skip_inventory:
        res["inventory"] = {r: inventory(ROOT / r) for r in RUNS}
        save()
        for r, inv in res["inventory"].items():
            for sub, s in inv.items():
                print(f"  {r}/{sub}: {s['n_files']} files {s['first']} .. {s['last']}, "
                      f"{s['n_gaps']} gaps ({s['scan_seconds']} s)")
    print(f"written: {out / 'c0.json'}")


if __name__ == "__main__":
    main()
