"""
Field-mean check for models evaluated on one target domain (default europe_west).

WHY
---
Every model scored on europe_west, the oracle included, has tgt_bias_mm =
-tgt_mae_mm to four decimals (0.0974), the signature of predicting almost no
rain. Yet tgt_mse_std differs by up to 0.2 between methods (FDA horn->europe
-0.20 against source-only). mse_std is computed on z-scored log(mm + 1e-5), so
a dry pixel sits at log(1e-5) = -11.5 and the loss can move a lot through
dry/drizzle discrimination far below any rain that shows up in mm. This script
says where in the intensity distribution each model's error lives, and whether
the predicted field mean is anywhere near the observed one.

WHAT IT COMPUTES, per checkpoint, on the target's TEST split
-----------------------------------------------------------
- observed vs predicted field mean (mm), bias, MAE
- wet fraction, observed and predicted, at 0.01 / 0.1 / 1 mm
- mse_std (reproduces results.csv tgt_mse_std, cross-checked), and the same
  error in UN-normalised log space, mse_log = mse_std * sd_src^2, which is
  comparable across source domains (mse_std is not: target data is z-scored
  with the SOURCE stats)
- mse_log with the dry floor raised: log(mm + 0.01) and log(mm + 0.1). If a
  method's gain vanishes there, it lives below 0.01 / 0.1 mm.
- per OBSERVED-intensity bin: pixel share, mean log error, its contribution
  to the total (contributions sum to mse_log), MAE, observed and predicted mean
- 2x2 wet/dry contingency at 0.1 mm with the log error in each cell
- time-mean maps, log-intensity histograms, per-sample field means (npz)
- reference predictors from the observations alone: all-zero (pred = 0 mm)
  and the best constant in log space (mse = Var of observed log)

Models: every `*__to__{target}__*/best.pt` in unet_seed<N>/ (phase 2 and the
in-domain `{target}__to__{target}__none`), and every
`*__to__{target}__none/best.pt` in unet_seed<N>_oracle/, labelled `oracle`.
Models are grouped by source; each group shares one pass over the test data,
normalised with that source's stats exactly as evaluate.py does.
--tree_suffix _mask_geo reads unet_seed<N>_mask_geo/ and its _oracle/ instead
(input-set ablation, PLAN.md P19); a masked checkpoint loads with its mask.

Input quantile mapping (PLAN.md P15, --input_qm tp precip all): each
off-diagonal source-only model is scored a second time per arm, on target
inputs mapped to the source's marginals channel by channel (quantile_map.py,
fitted on both training splits). Its label gains "+qm_<arm>", method
"none+qm_<arm>", tree "qm". tp is RainShift's choice, precip adds cp, all maps
the nine dynamic channels; statics are never mapped. No target labels, no
training. One all-channel fit per pair; its quantiles go into fields.npz.

Output: results/seed<N>/field_mean_check/<target>/ (with --input_qm:
field_mean_check_qm/<target>/). Refuses to run if it exists.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import DEFAULT_INPUT_VARS
from data.dataset_zarr import ClimateSRDatasetZarr, _LOG_EPS
from evaluate import load_model, load_stats, parse_exp_name
from quantile_map import fit_qm_from_zarr

TARGET_VAR = "precipitation"
BIN_EDGES_MM = [0.0, 0.01, 0.1, 1.0, 5.0, 20.0, float("inf")]
BIN_NAMES = ["<0.01", "0.01-0.1", "0.1-1", "1-5", "5-20", ">=20"]
WET_THRESHOLDS_MM = [0.01, 0.1, 1.0]
CONTINGENCY_MM = 0.1
FLOORS_MM = {"e001": 0.01, "e01": 0.1}   # raised dry floors for log(mm + floor)
HIST_LO, HIST_HI, HIST_BINS = -11.6, 5.0, 166   # log(mm + 1e-5), 0.1 wide
LOG_FLOOR = float(np.log(_LOG_EPS))


# ---------------------------------------------------------------------------
#  Accumulators
# ---------------------------------------------------------------------------


class Acc:
    """Running sums in float64 on the GPU for one model (or the observations).
    `pred_log` is log(mm + 1e-5) (the model's own space before z-scoring)."""

    def __init__(self, device, hw):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        nb = len(BIN_NAMES)
        self.n = 0
        self.s = {k: z(()) for k in ("pred_mm", "abs_mm", "sq_mm", "sq_log", "log", "log2",
                                     *(f"sq_log_{f}" for f in FLOORS_MM))}
        self.bin = {k: z(nb) for k in ("cnt", "sq_log", "abs_mm", "pred_mm", "true_mm",
                                       "true_log", "true_log2",
                                       *(f"sq_log_{f}" for f in FLOORS_MM))}
        self.wet = z(len(WET_THRESHOLDS_MM))
        self.cont_cnt = z(4)     # (obs wet, pred wet): 00 01 10 11
        self.cont_sq = z(4)
        self.map = z(hw, hw)
        self.hist = z(HIST_BINS)
        self.sample_mean = []

    @torch.no_grad()
    def update(self, p_mm, p_log, t_mm, t_log, t_bin, t_wet):
        p_mm, p_log = p_mm.double(), p_log.double()
        d_mm = p_mm - t_mm
        sq_log = (p_log - t_log) ** 2
        self.n += p_mm.numel()
        self.s["pred_mm"] += p_mm.sum()
        self.s["abs_mm"] += d_mm.abs().sum()
        self.s["sq_mm"] += (d_mm ** 2).sum()
        self.s["sq_log"] += sq_log.sum()
        self.s["log"] += p_log.sum()
        self.s["log2"] += (p_log ** 2).sum()
        sq_floor = {}
        for f, v in FLOORS_MM.items():
            sq_floor[f] = (torch.log(p_mm + v) - torch.log(t_mm + v)) ** 2
            self.s[f"sq_log_{f}"] += sq_floor[f].sum()

        nb = len(BIN_NAMES)
        idx = t_bin.reshape(-1)
        add = lambda key, val: self.bin[key].index_add_(0, idx, val.reshape(-1))
        self.bin["cnt"] += torch.bincount(idx, minlength=nb).double()
        add("sq_log", sq_log)
        add("abs_mm", d_mm.abs())
        add("pred_mm", p_mm)
        add("true_mm", t_mm)
        add("true_log", t_log)
        add("true_log2", t_log ** 2)
        for f in FLOORS_MM:
            add(f"sq_log_{f}", sq_floor[f])

        for i, thr in enumerate(WET_THRESHOLDS_MM):
            self.wet[i] += (p_mm >= thr).sum()
        cell = (t_wet.long() * 2 + (p_mm >= CONTINGENCY_MM).long()).reshape(-1)
        self.cont_cnt += torch.bincount(cell, minlength=4).double()
        self.cont_sq.index_add_(0, cell, sq_log.reshape(-1))

        self.map += p_mm.sum(dim=(0, 1))
        self.hist += torch.histc(p_log.float().clamp(HIST_LO, HIST_HI - 1e-6),
                                 bins=HIST_BINS, min=HIST_LO, max=HIST_HI).double()
        self.sample_mean.append(p_mm.mean(dim=(1, 2, 3)).float().cpu().numpy())


def _bins(t_mm):
    edges = torch.tensor(BIN_EDGES_MM[1:-1], dtype=t_mm.dtype, device=t_mm.device)
    return torch.bucketize(t_mm, edges, right=True)


# ---------------------------------------------------------------------------
#  Model discovery
# ---------------------------------------------------------------------------


def find_models(results_root, seed, target, qm_arms=(), tree_suffix=""):
    """[(label, source, method, tree, ckpt, qm_arm)] for every model scored on
    target; qm_arm is "" except for the QM rescoring of source-only models.
    tree_suffix reads a side tree instead, e.g. "_mask_geo" for
    unet_seed<N>_mask_geo/ and its _oracle/ (PLAN.md P19)."""
    out = []
    phase2 = results_root / f"unet_seed{seed}{tree_suffix}"
    oracle = results_root / f"unet_seed{seed}{tree_suffix}_oracle"
    for ckpt in sorted(phase2.glob(f"*__to__{target}__*/best.pt")):
        src, tgt, method = parse_exp_name(ckpt.parent.name)
        tree = "in_domain" if src == tgt else "none"
        if method == "adabn":
            # best.pt is the unadapted model; the target-adapted one is
            # best_adabn.pt (evaluate.py target_checkpoint()). Scoring best.pt
            # would repeat source-only.
            ckpt = ckpt.parent / "best_adabn.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"{ckpt} missing: cannot score AdaBN on its target")
        out.append((f"{src}->{tgt}:{method}", src, method, tree, ckpt, ""))
        if method == "none" and src != tgt:
            for arm in qm_arms:
                out.append((f"{src}->{tgt}:none+qm_{arm}", src, f"none+qm_{arm}", "qm", ckpt, arm))
    if oracle.is_dir():
        for ckpt in sorted(oracle.glob(f"*__to__{target}__none/best.pt")):
            src, tgt, _ = parse_exp_name(ckpt.parent.name)
            out.append((f"{src}->{tgt}:oracle", src, "oracle", "oracle", ckpt, ""))
    return out


def csv_mse_std(results_root, seed, tree, src, tgt, method):
    """tgt_mse_std from the evaluate.py results.csv, for the cross-check."""
    sub = "oracle" if tree == "oracle" else "none"
    m = "none" if method == "oracle" else method
    path = results_root / "results" / f"seed{seed}" / "unet" / sub / "results.csv"
    if not path.exists():
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            if (row["source"], row["target"], row["method"], row["transform"]) == (src, tgt, m, "none"):
                return float(row["tgt_mse_std"])
    return None


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def summarise(label, source, method, tree, acc, sd_src, obs, csv_ref):
    n = acc.n
    obs_mean = obs.s["pred_mm"].item() / n
    pred_mean = acc.s["pred_mm"].item() / n
    mse_log = acc.s["sq_log"].item() / n
    row = {
        "label": label, "source": source, "method": method, "tree": tree,
        "n_pixels": n,
        "obs_mean_mm": obs_mean, "pred_mean_mm": pred_mean,
        "mean_ratio": pred_mean / obs_mean if obs_mean > 0 else float("nan"),
        "bias_mm": pred_mean - obs_mean,
        "mae_mm": acc.s["abs_mm"].item() / n,
        "rmse_mm": (acc.s["sq_mm"].item() / n) ** 0.5,
        "mse_std": mse_log / (sd_src + 1e-8) ** 2,
        "csv_tgt_mse_std": csv_ref if csv_ref is not None else "",
        "mse_log": mse_log,
        **{f"mse_log_{f}": acc.s[f"sq_log_{f}"].item() / n for f in FLOORS_MM},
        "pred_log_mean": acc.s["log"].item() / n,
        "pred_log_var": acc.s["log2"].item() / n - (acc.s["log"].item() / n) ** 2,
    }
    for i, thr in enumerate(WET_THRESHOLDS_MM):
        row[f"obs_wet_{thr:g}"] = obs.wet[i].item() / n
        row[f"pred_wet_{thr:g}"] = acc.wet[i].item() / n
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target", default="europe_west")
    ap.add_argument("--results_root", required=True,
                    help="results_rainshift_uda/: holds unet_seed<N>/, unet_seed<N>_oracle/, results/")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output_dir", default=None,
                    help="default: <results_root>/results/seed<N>/field_mean_check/<target>")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_batches", type=int, default=None, help="smoke test only")
    ap.add_argument("--input_qm", nargs="*", default=[], choices=["tp", "precip", "all"],
                    help="also score each off-diagonal source-only model on quantile-mapped target inputs")
    ap.add_argument("--qm_n_quantiles", type=int, default=1000)
    ap.add_argument("--qm_fit_chunks", type=int, default=40,
                    help="training chunks per domain for the QM fit, spread over the split")
    ap.add_argument("--tree_suffix", default="",
                    help="score unet_seed<N><suffix>/ and its _oracle/ instead (P19, e.g. _mask_geo); "
                         "output goes to field_mean_check<suffix>/")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    data_root = Path(args.data_root)
    sub = ("field_mean_check_qm" if args.input_qm else "field_mean_check") + args.tree_suffix
    out_dir = Path(args.output_dir) if args.output_dir else (
        results_root / "results" / f"seed{args.seed}" / sub / args.target)
    if out_dir.exists():
        raise SystemExit(f"ERROR: {out_dir} exists. Not overwriting; move it to _trash/ first.")

    device = torch.device("cuda")
    models = find_models(results_root, args.seed, args.target, args.input_qm, args.tree_suffix)
    if not models:
        raise SystemExit(f"ERROR: no checkpoints scored on {args.target} for seed {args.seed}")
    groups = {}
    for m in models:
        groups.setdefault(m[1], []).append(m)
    print(f"seed {args.seed}, target {args.target}: {len(models)} checkpoints, "
          f"{len(groups)} source groups -> {out_dir}")
    for src, ms in groups.items():
        print(f"  {src}: {', '.join(m[0].split(':')[1] + '/' + m[3] for m in ms)}")

    rows, bin_rows, cont_rows, ref_rows = [], [], [], []
    arrays = {}
    obs_means = {}

    for src, ms in groups.items():
        t0 = time.time()
        src_stats = load_stats(str(data_root / src))
        mu, sd = src_stats[TARGET_VAR]
        # The checkpoint's own record of its source stats must agree with the
        # data root's, or evaluate.py's numbers were made in a different space.
        for label, _, _, _, ckpt, _ in ms:
            rec = ckpt.parent / "src_stats.json"
            if rec.exists():
                rmu, rsd = json.loads(rec.read_text())[TARGET_VAR]
                if abs(rmu - mu) > 1e-6 or abs(rsd - sd) > 1e-6:
                    print(f"  WARNING {label}: src_stats.json precipitation ({rmu}, {rsd}) "
                          f"!= data root ({mu}, {sd})")

        loader = DataLoader(
            ClimateSRDatasetZarr(str(data_root / args.target), "test", stats=src_stats),
            args.batch_size, shuffle=False,
            num_workers=1,   # per-sample arrays are in time order only at 1 worker
            pin_memory=True,
        )
        # One net per checkpoint; a QM row reuses its source-only net.
        nets = {}
        for m in ms:
            if m[4] not in nets:
                nets[m[4]] = load_model("unet", str(m[4]), device)
        accs = [Acc(device, 200) for _ in ms]
        obs = Acc(device, 200)

        # QM for this (source, target) pair: one fit on all channels, then
        # each arm maps its subset of them.
        qms = {}
        arms = sorted({m[5] for m in ms if m[5]})
        if arms:
            tq = time.time()
            full = fit_qm_from_zarr(data_root / src, data_root / args.target, src_stats,
                                    n_quantiles=args.qm_n_quantiles, channels=None,
                                    n_chunks=args.qm_fit_chunks)
            qms = {arm: full.restrict(arm) for arm in arms}
            pair = f"{src}__to__{args.target}"
            arrays[f"qm__{pair}__src_quantiles"] = full.src_quantiles
            arrays[f"qm__{pair}__tgt_quantiles"] = full.tgt_quantiles
            mid = args.qm_n_quantiles // 2
            print(f"  {src}: fitted QM in {time.time() - tq:.0f} s. Per channel, in source-z units: "
                  "median src / tgt, max |quantile gap|")
            for c, v in enumerate(DEFAULT_INPUT_VARS):
                sq, tq_ = full.src_quantiles[c], full.tgt_quantiles[c]
                print(f"    {v:5s} {sq[mid]:7.3f} / {tq_[mid]:7.3f}   {np.abs(sq - tq_).max():7.3f}")

        with torch.no_grad():
            for b, (x, s, y) in enumerate(loader):
                if args.max_batches is not None and b >= args.max_batches:
                    break
                x, s, y = (t.to(device, non_blocking=True) for t in (x, s, y))
                xs = {"": x, **{arm: qm.transform(x) for arm, qm in qms.items()}}
                t_log = y.double() * (sd + 1e-8) + mu
                t_mm = (torch.exp(t_log) - _LOG_EPS).clamp_min(0.0)
                t_bin = _bins(t_mm)
                t_wet = t_mm >= CONTINGENCY_MM
                obs.update(t_mm, t_log, t_mm, t_log, t_bin, t_wet)
                for m, acc in zip(ms, accs):
                    # Same forward as evaluate.py: bf16 autocast, then float.
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        pred = nets[m[4]](xs[m[5]], s)
                    if not torch.isfinite(pred).all():
                        raise RuntimeError(f"non-finite prediction in group {src}")
                    p_log = pred.double() * (sd + 1e-8) + mu
                    p_mm = (torch.exp(p_log) - _LOG_EPS).clamp_min(0.0)
                    acc.update(p_mm, p_log, t_mm, t_log, t_bin, t_wet)
        print(f"  {src}: {obs.n // (200 * 200)} samples, {time.time() - t0:.0f} s")
        obs_means[src] = obs.s["pred_mm"].item() / obs.n

        # Reference predictors, from the observations alone. In log space they
        # do not depend on the source; mse_std does, via sd.
        n = obs.n
        e_t, e_t2 = obs.s["log"].item() / n, obs.s["log2"].item() / n
        zero_mse_log = LOG_FLOOR ** 2 - 2 * LOG_FLOOR * e_t + e_t2
        const_mse_log = e_t2 - e_t ** 2
        for name, mse_log in (("all_zero", zero_mse_log), ("best_const_log", const_mse_log)):
            ref_rows.append({"source_stats": src, "reference": name, "mse_log": mse_log,
                             "mse_std": mse_log / (sd + 1e-8) ** 2})

        # Observed per-bin quantities, plus the all-zero predictor per bin.
        bc = obs.bin["cnt"].cpu().numpy()
        btl = obs.bin["true_log"].cpu().numpy()
        btl2 = obs.bin["true_log2"].cpu().numpy()
        zero_bin_sq = LOG_FLOOR ** 2 * bc - 2 * LOG_FLOOR * btl + btl2

        for (label, _, method, tree, ckpt, _), acc in zip(ms, accs):
            src_, tgt_, _ = parse_exp_name(ckpt.parent.name)
            # results.csv describes the main trees only; a side tree has no
            # evaluate.py row to cross-check against.
            ref = None if args.tree_suffix else csv_mse_std(results_root, args.seed, tree, src_, tgt_, method)
            row = summarise(label, src, method, tree, acc, sd, obs, ref)
            if ref is not None and args.max_batches is None and abs(row["mse_std"] - ref) > 5e-3:
                print(f"  WARNING {label}: mse_std {row['mse_std']:.4f} != results.csv {ref:.4f}")
            rows.append(row)

            for k, name in enumerate(BIN_NAMES):
                c = bc[k]
                g = lambda key: acc.bin[key][k].item()
                bin_rows.append({
                    "label": label, "bin_mm": name, "pixel_share": c / acc.n,
                    "mse_log_in_bin": g("sq_log") / c if c else float("nan"),
                    "mse_log_contrib": g("sq_log") / acc.n,
                    **{f"mse_log_{f}_contrib": g(f"sq_log_{f}") / acc.n for f in FLOORS_MM},
                    "zero_pred_mse_log_contrib": zero_bin_sq[k] / acc.n,
                    "mae_mm_in_bin": g("abs_mm") / c if c else float("nan"),
                    "obs_mean_mm": g("true_mm") / c if c else float("nan"),
                    "pred_mean_mm": g("pred_mm") / c if c else float("nan"),
                })
            for k, name in enumerate(("obs_dry_pred_dry", "obs_dry_pred_wet",
                                      "obs_wet_pred_dry", "obs_wet_pred_wet")):
                cont_rows.append({
                    "label": label, "cell": name, "threshold_mm": CONTINGENCY_MM,
                    "pixel_share": acc.cont_cnt[k].item() / acc.n,
                    "mse_log_contrib": acc.cont_sq[k].item() / acc.n,
                })

            key = label.replace("->", "__to__").replace(":", "__")
            arrays[f"{key}__map_mm"] = (acc.map / (acc.n / (200 * 200))).float().cpu().numpy()
            arrays[f"{key}__hist_log"] = acc.hist.cpu().numpy()
            arrays[f"{key}__sample_mean_mm"] = np.concatenate(acc.sample_mean)

        arrays[f"observed__stats_{src}__map_mm"] = (obs.map / (obs.n / (200 * 200))).float().cpu().numpy()
        arrays[f"observed__stats_{src}__hist_log"] = obs.hist.cpu().numpy()
        arrays[f"observed__stats_{src}__sample_mean_mm"] = np.concatenate(obs.sample_mean)

        del nets
        torch.cuda.empty_cache()

    # Every group must have read the same observations.
    vals = list(obs_means.values())
    if max(vals) - min(vals) > 1e-6 * max(abs(v) for v in vals):
        print(f"  WARNING: observed field mean differs between source groups: {obs_means}")

    out_dir.mkdir(parents=True, exist_ok=False)
    for name, rs in (("summary.csv", rows), ("bins.csv", bin_rows),
                     ("contingency.csv", cont_rows), ("reference_predictors.csv", ref_rows)):
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
    np.savez_compressed(out_dir / "fields.npz",
                        hist_edges_log=np.linspace(HIST_LO, HIST_HI, HIST_BINS + 1),
                        bin_edges_mm=np.array(BIN_EDGES_MM), **arrays)
    (out_dir / "meta.json").write_text(json.dumps({
        "seed": args.seed, "target": args.target, "split": "test",
        "max_batches": args.max_batches, "batch_size": args.batch_size,
        "log_eps": _LOG_EPS, "floors_mm": FLOORS_MM, "bins_mm": BIN_NAMES,
        "wet_thresholds_mm": WET_THRESHOLDS_MM, "contingency_mm": CONTINGENCY_MM,
        "checkpoints": [str(m[4]) for m in models],
        "input_qm": args.input_qm, "qm_n_quantiles": args.qm_n_quantiles,
        "qm_fit_chunks": args.qm_fit_chunks, "tree_suffix": args.tree_suffix,
    }, indent=2))

    print("\nlabel                                   mean_ratio  pred_wet_0.1  mse_std  csv     "
          "mse_log  mse_log_e001  mse_log_e01")
    for r in rows:
        print(f"{r['label']:40s} {r['mean_ratio']:9.3f}  {r['pred_wet_0.1']:11.4f}  "
              f"{r['mse_std']:7.4f}  {str(r['csv_tgt_mse_std']):7s} {r['mse_log']:7.3f}  "
              f"{r['mse_log_e001']:11.4f}  {r['mse_log_e01']:10.4f}")
    print(f"observed wet fraction at 0.1 mm: {rows[0]['obs_wet_0.1']:.4f}, "
          f"field mean {rows[0]['obs_mean_mm']:.4f} mm")
    for r in ref_rows:
        print(f"reference {r['reference']:15s} (stats {r['source_stats']}): "
              f"mse_log {r['mse_log']:.3f}  mse_std {r['mse_std']:.4f}")
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
