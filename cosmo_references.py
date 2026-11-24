"""
Trivial references for COSMO C3 (PLAN_COSMO.md C3), scored like evaluate.py.

For each target region, on the test split (2020-2021), four predictors that
use no learning:
  zero          0 mm/h everywhere
  climatology   the region's per-pixel mean precipitation over the train split
                (2011-2019, validation chunks included), the same field at
                every hour
  block_tp      the coarse input `tp` (m/h, x1000 -> mm/h) repeated over its
                10 x 10 block. In the perfect model this is the target's exact
                block mean, so its bias is 0 by construction (checked).
  bicubic_tp    `tp` bicubic-upsampled 20 -> 200 (torch, align_corners=False,
                as DualEncoderUNet.forward), clipped at 0
A reference depends only on the target region, so there are 5 x 4 rows; every
cell (source -> target) of the transfer matrix shares its target's rows.

RainShift (MILESTONES M1, L4; PLAN.md R5 step 4) uses the same script on its
stores: input `tp` is ERA5's own precipitation (m, hourly, x1000 -> mm/h) on
80 x 80, the target IMERG on 200 x 200. 200 is not a multiple of 80, so
block_tp is not defined and is left out (its block error column is NaN);
bicubic_tp is then a weak reference (report/aies/NOTES.md). COSMO, where the
output is a multiple of the input, is unchanged.

Metrics as evaluate.py: per-sample mean over pixels, then mean over samples,
for mse_log_e01 / mse_log_e001 (log(mm + floor)); mae_mm, rmse_mm, bias_mm
over all pixels. Added: wet fraction (>= 0.1 mm/h) and maximum of the
prediction, and the truth's. Truth is read from the store in mm/h, clipped at
0; evaluate.py's truth goes through the log transform and back, which differs
by float32 rounding only.

Reads by chunks of 200 hours (a dask max over a whole store OOMs, CLAUDE.md).
Writes <out> (CSV); refuses if it exists.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

FLOORS_MM = {"e001": 0.01, "e01": 0.1}
WET_MM = 0.1
CHUNK = 200
REFS = ("zero", "climatology", "block_tp", "bicubic_tp")


def train_climatology(root):
    do = xr.open_zarr(root / "train_data_out.zarr")
    n = do.sizes["time"]
    acc, cnt = None, 0
    for t0 in range(0, n, CHUNK):
        y = np.nan_to_num(do["precipitation"].isel(time=slice(t0, t0 + CHUNK)).values.astype(np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        y = np.clip(y, 0.0, None)
        acc = y.sum(0) if acc is None else acc + y.sum(0)
        cnt += y.shape[0]
    do.close()
    return (acc / cnt).astype(np.float32)


def predictions(name, tp_mm, clim, n_out):
    T, n_in = tp_mm.shape[0], tp_mm.shape[1]
    if name == "zero":
        return np.zeros((T, n_out, n_out), np.float32)
    if name == "climatology":
        return np.broadcast_to(clim, (T, n_out, n_out))
    if name == "block_tp":
        b = n_out // n_in
        return np.repeat(np.repeat(tp_mm, b, axis=1), b, axis=2)
    if name == "bicubic_tp":
        x = torch.from_numpy(np.ascontiguousarray(tp_mm))[:, None]
        up = F.interpolate(x, size=(n_out, n_out), mode="bicubic", align_corners=False)
        return np.clip(up[:, 0].numpy(), 0.0, None)
    raise ValueError(name)


def score_region(root):
    clim = train_climatology(root)
    di = xr.open_zarr(root / "test_data_in.zarr")
    do = xr.open_zarr(root / "test_data_out.zarr")
    n = do.sizes["time"]
    n_in, n_out_full = di["tp"].shape[-1], do["precipitation"].shape[-1]
    blocks = n_out_full % n_in == 0
    refs = REFS if blocks else tuple(r for r in REFS if r != "block_tp")
    acc = {r: {"se": 0.0, "ae": 0.0, "bias": 0.0, "wet": 0, "max": 0.0,
               **{f"log_{f}": [] for f in FLOORS_MM}} for r in refs}
    t_wet, t_max, n_pix = 0, 0.0, 0
    block_err = 0.0 if blocks else float("nan")
    for t0 in range(0, n, CHUNK):
        sl = slice(t0, t0 + CHUNK)
        y = np.clip(np.nan_to_num(do["precipitation"].isel(time=sl).values.astype(np.float32),
                                  nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        tp = np.nan_to_num(di["tp"].isel(time=sl).values.astype(np.float32), nan=0.0) * 1000.0
        n_out = y.shape[1]
        if blocks:
            b = n_out // tp.shape[1]
            ym = y.reshape(y.shape[0], tp.shape[1], b, tp.shape[1], b).mean(axis=(2, 4))
            block_err = max(block_err, float(np.abs(ym - tp).max()))
        t_wet += int((y >= WET_MM).sum())
        t_max = max(t_max, float(y.max()))
        n_pix += y.size
        for r in refs:
            p = predictions(r, tp, clim, n_out).astype(np.float64)
            d = p - y
            a = acc[r]
            a["se"] += float((d ** 2).sum())
            a["ae"] += float(np.abs(d).sum())
            a["bias"] += float(d.sum())
            a["wet"] += int((p >= WET_MM).sum())
            a["max"] = max(a["max"], float(p.max()))
            for f, v in FLOORS_MM.items():
                a[f"log_{f}"].append(((np.log(p + v) - np.log(y + v)) ** 2).mean(axis=(1, 2)))
    di.close()
    do.close()
    rows = []
    for r in refs:
        a = acc[r]
        rows.append({"reference": r, "n_samples": n,
                     "rmse_mm": np.sqrt(a["se"] / n_pix), "mae_mm": a["ae"] / n_pix,
                     "bias_mm": a["bias"] / n_pix,
                     **{f"mse_log_{f}": float(np.concatenate(a[f"log_{f}"]).mean()) for f in FLOORS_MM},
                     "wet_frac_pred": a["wet"] / n_pix, "max_pred_mm": a["max"],
                     "wet_frac_true": t_wet / n_pix, "max_true_mm": t_max,
                     "block_tp_max_abs_err_mm": block_err})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True)
    p.add_argument("--regions", nargs="+", required=True)
    p.add_argument("--out", required=True, help="CSV, refused if it exists")
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting.")
    rows = []
    for r in a.regions:
        for row in score_region(Path(a.data_root) / r):
            rows.append({"target": r, **row})
            print(f"{r:18s} {row['reference']:12s} mse_log_e01 {row['mse_log_e01']:.4f}  "
                  f"mae {row['mae_mm']:.4f}  bias {row['bias_mm']:+.4f}  wet {row['wet_frac_pred']:.3f} "
                  f"(true {row['wet_frac_true']:.3f})  max {row['max_pred_mm']:.1f}", flush=True)
        if rows[-1]["block_tp_max_abs_err_mm"] > 1e-3:  # False for NaN (no blocks)
            print(f"WARNING {r}: block mean of target differs from tp by "
                  f"{rows[-1]['block_tp_max_abs_err_mm']:.2e} mm/h")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
