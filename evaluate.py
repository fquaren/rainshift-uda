"""
Evaluate trained UNet / AFM models on target domains.

Modes:
  single:  evaluate one checkpoint on one target domain
  batch:   scan experiment directory, evaluate all found checkpoints

Metrics (physical space, mm):
  Deterministic: RMSE, MAE, bias
  Probabilistic: CRPS, spread (AFM only, when --n_ensemble > 0)
  Standardized:  mse_std (exact optimizer loss space); crps_std and
                 spread_std for AFM ensembles. crps_mm is kept as a record
                 only: like rmse_mm it is dominated by members that blow up
                 when converted back to mm.

Optional test-time input transforms (--input_transform):
  none       no transform (default)
  qm_tp      quantile mapping on total precipitation (RainShift paper baseline)
  qm_precip  quantile mapping on both precipitation channels (tp, cp)
  qm_all     quantile mapping on all 9 input channels
  ot         joint-distribution Sinkhorn OT on all channels

All transforms require --src_path to fit on source training data.
The script evaluates the model on both the source domain (no transform)
and the target domain (with transform, if specified), saving metrics for both.

Usage:
  # JDOT
  python evaluate.py single --input_transform ot \
      --checkpoint experiments/europe_west__to__melanesia__none/best.pt \
      --src_path /data/rainshift_npy/europe_west \
      --target_path /data/rainshift_npy/melanesia
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ClimateSRDatasetNPY, inverse_transform, load_domain_stats
from data.dataset_zarr import ClimateSRDatasetZarr, _compute_and_cache_stats


def load_stats(domain_path):
    """Load normalization_stats.json for a domain, computing+caching from the
    training Zarr if absent. Matches the trainer's stats loader and the Zarr
    dataset's stats file (NOT the NPY-era stats.json)."""
    import json as _json
    from pathlib import Path as _P

    p = _P(domain_path) / "normalization_stats.json"
    if p.exists():
        return _json.loads(p.read_text())
    return _compute_and_cache_stats(_P(domain_path), None)
from models.afm import AFMModel
from models.unet import DualEncoderUNet
from prob_metrics import crps_fair, ensemble_spread, rank_histogram
from quantile_map import QuantileMap, fit_qm_from_zarr
from sinkhorn import SinkhornOTMap, fit_ot_from_npy


_TRANSFORM_CHOICES = ["none", "qm_tp", "qm_precip", "qm_all", "ot"]
# Raised dry floors for log(mm + floor), as in field_mean_check.FLOORS_MM
# (PLAN.md P14, decision D1). Training stays in log(mm + 1e-5).
FLOORS_MM = {"e001": 0.01, "e01": 0.1}


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------


def compute_metrics(pred, true):
    d = pred - true
    return {
        "rmse_mm": float(np.sqrt(np.mean(d**2))),
        "mae_mm": float(np.mean(np.abs(d))),
        "bias_mm": float(np.mean(d)),
        "n_samples": int(pred.shape[0]),
    }


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------


def _infer_model_type(exp_dir_name):
    return "afm" if exp_dir_name.startswith("afm_") else "unet"


def run_base_features(run_dir, default):
    """
    Width the run was trained at. train_afm.py records base_features in
    config.json (AFM trains at 64); train_unet.py hard-codes 32 and does not
    record it, so UNet runs fall back to `default`. Building an AFM model at
    the CLI default of 32 fails load_state_dict on every AFM checkpoint.
    """
    cfg = Path(run_dir) / "config.json"
    if cfg.exists():
        bf = json.loads(cfg.read_text()).get("base_features")
        if bf is not None:
            return int(bf)
    return default


def load_model(model_type, checkpoint, device, base_features=32):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    clean = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
    if model_type == "unet":
        # A masked-input checkpoint (PLAN.md P19) carries its keep vectors.
        keep = {k: clean[k] for k in ("dyn_keep", "stat_keep") if k in clean}
        model = DualEncoderUNet(9, 2, 1, base_features, **keep)
    else:
        model = AFMModel(9, 2, 1, base_features, encoder_loss_weight=0.1)
    model.load_state_dict(clean)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
#  Input transform loading / fitting with cache
# ---------------------------------------------------------------------------


def _transform_cache_path(args, src, tgt):
    suffix = ".npz"
    return Path(args.transform_cache_dir) / args.input_transform / f"{src}__to__{tgt}{suffix}"


def _load_or_fit_transform(src_path, tgt_path, args, device):
    """
    Return a fitted transform object (with .transform(x)->x) or None.
    Caches per (transform_type, src, tgt).
    """
    kind = args.input_transform
    if kind == "none":
        return None

    src_name = Path(src_path).stem
    tgt_name = Path(tgt_path).stem
    cache = _transform_cache_path(args, src_name, tgt_name)

    if cache.exists() and not args.transform_refit:
        print(f"  Loading cached {kind}: {cache}")
        if kind == "ot":
            return SinkhornOTMap.load(cache, device=device)
        return QuantileMap.load(cache)

    print(f"  Fitting {kind}: {src_name} -> {tgt_name}")
    cache.parent.mkdir(parents=True, exist_ok=True)

    if kind == "ot":
        tf = fit_ot_from_npy(
            src_path=src_path,
            tgt_path=tgt_path,
            n_anchors=args.ot_n_anchors,
            reg=args.ot_reg,
            n_iter=args.ot_n_iter,
            k=args.ot_k,
            device=device,
            subset=args.transform_fit_subset,
        )
    else:
        # QM variants. The data are Zarr; fit_qm_from_npy read train_x.npy,
        # which no longer exists, from the first `subset` samples (one season).
        channel_spec = {"qm_tp": "tp", "qm_precip": "precip", "qm_all": None}[kind]
        tf = fit_qm_from_zarr(
            src_path=src_path,
            tgt_path=tgt_path,
            src_stats=load_stats(src_path),
            n_quantiles=args.qm_n_quantiles,
            channels=channel_spec,
            n_chunks=max(1, args.transform_fit_subset // 200),
        )

    tf.save(cache)
    return tf


# ---------------------------------------------------------------------------
#  Evaluation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model, model_type, loader, device, stats, n_ensemble=0, sample_steps=20, transform=None,
    desc="Evaluating", collect_fields=True,
):
    """collect_fields=False streams the metrics without concatenating the
    prediction/truth fields. Required for the TRAINING split: at ~122k samples
    of 200x200 float32 the two arrays would be ~40 GB. Physical-space metrics
    are then accumulated as running sums instead, and no example fields can be
    saved for that split."""
    var = "precipitation"
    all_pred, all_true = [], []
    # Ensemble scores, one value per sample, accumulated batch by batch.
    ens_scores = {"crps_std": [], "spread_std": [], "crps_mm": [], "spread_mm": [],
                  **{f"crps_log_{f}": [] for f in FLOORS_MM}}
    rank_hists = []
    # Fixed generators: the same ensemble noise and rank tie-breaking on every
    # run, so re-scoring a checkpoint reproduces its row.
    rank_gen = torch.Generator(device=device).manual_seed(0) if n_ensemble > 0 else None
    ens_gen = torch.Generator(device=device).manual_seed(1) if n_ensemble > 0 else None

    sum_loss = 0.0
    sum_abs = 0.0
    n_elements = 0
    n_nonfinite_batches = 0
    # Running physical-space accumulators, used when collect_fields=False.
    s_se = s_ae = s_bias = 0.0
    n_phys = 0
    per_sample_mse_mm, per_sample_mae_mm = [], []
    per_sample_log_floor = {f: [] for f in FLOORS_MM}
    # Per-sample loss in STANDARDISED space, one scalar per test sample.
    # Kept separately from the aggregate because downstream diagnostics
    # (OODiag-style analog matching, regime stratification, error-distribution
    # comparisons) need the sample-wise error, not its mean.
    per_sample_std = []
    # MAE in standardised space: a point forecast's CRPS, so the UNet's
    # mae_std is what the AFM's crps_std is compared with (PLAN.md 3.5,
    # acceptance).
    per_sample_mae_std = []

    for batch in tqdm(loader, desc=desc):
        x, s, y = (t.to(device, non_blocking=True) for t in batch)

        if transform is not None:
            x = transform.transform(x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x, s) if model_type == "unet" else model.deterministic_predict(x, s)

        # Flag non-finite predictions (e.g. a checkpoint trained on NaN loss).
        # Do NOT silently accumulate them into a NaN mse_std.
        if not torch.isfinite(pred).all():
            n_nonfinite_batches += 1

        # Accumulate exact MSE in standardized space
        sq = (pred - y) ** 2
        ab = (pred.float() - y.float()).abs()
        sum_loss += torch.sum(sq).item()
        sum_abs += torch.sum(ab).item()
        n_elements += pred.numel()
        # Mean over the spatial/channel dims -> one value per sample.
        per_sample_std.append(sq.float().mean(dim=tuple(range(1, sq.ndim))).cpu().numpy())
        per_sample_mae_std.append(ab.mean(dim=tuple(range(1, ab.ndim))).cpu().numpy())

        p_mm = inverse_transform(pred.float().cpu().numpy(), var, stats)
        t_mm = inverse_transform(y.float().cpu().numpy(), var, stats)
        ax = tuple(range(1, p_mm.ndim))
        per_sample_mse_mm.append(np.mean((p_mm - t_mm) ** 2, axis=ax))
        per_sample_mae_mm.append(np.mean(np.abs(p_mm - t_mm), axis=ax))
        for f, v in FLOORS_MM.items():
            per_sample_log_floor[f].append(np.mean((np.log(p_mm + v) - np.log(t_mm + v)) ** 2, axis=ax))
        if collect_fields:
            all_pred.append(p_mm)
            all_true.append(t_mm)
        else:
            s_se += float(((p_mm - t_mm) ** 2).sum())
            s_ae += float(np.abs(p_mm - t_mm).sum())
            s_bias += float((p_mm - t_mm).sum())
            n_phys += p_mm.size

        if model_type == "afm" and n_ensemble > 0:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                samples = model.sample(x, s, n_samples=n_ensemble, steps=sample_steps, generator=ens_gen)
            # Scored per batch: the ensemble for a whole split does not fit
            # in memory (~45 GB at 17,520 samples and 16 members).
            ens_scores["crps_std"].append(crps_fair(samples, y).cpu().numpy())
            ens_scores["spread_std"].append(ensemble_spread(samples).cpu().numpy())
            ens_mm = torch.from_numpy(np.stack(
                [inverse_transform(samples[:, i].float().cpu().numpy(), var, stats) for i in range(n_ensemble)], axis=1
            ))
            t_mm_t = torch.from_numpy(t_mm)
            ens_scores["crps_mm"].append(crps_fair(ens_mm, t_mm_t).numpy())
            ens_scores["spread_mm"].append(ensemble_spread(ens_mm).numpy())
            # Members and truth clipped at 0 mm by inverse_transform, then
            # put in each raised-floor log space.
            for f, v in FLOORS_MM.items():
                ens_scores[f"crps_log_{f}"].append(
                    crps_fair(torch.log(ens_mm + v), torch.log(t_mm_t + v)).numpy())
            rank_hists.append(rank_histogram(samples, y, rank_gen).cpu().numpy())

    if not per_sample_std:
        raise RuntimeError(
            f"{desc}: no batches evaluated (empty loader). Cannot compute metrics."
        )

    pred = np.concatenate(all_pred) if collect_fields else None
    true = np.concatenate(all_true) if collect_fields else None

    if n_nonfinite_batches > 0:
        # A non-finite prediction almost always means the checkpoint was
        # trained on a NaN loss (e.g. unscrubbed non-finite input pixels).
        # Fail loudly rather than write a silent NaN row to results.csv.
        raise RuntimeError(
            f"{desc}: {n_nonfinite_batches} batch(es) produced non-finite "
            f"predictions. The checkpoint is likely corrupt (trained on NaN "
            f"loss). Retrain this configuration rather than trusting NaN metrics."
        )

    if collect_fields:
        metrics = compute_metrics(pred, true)
    else:
        n_phys = max(n_phys, 1)
        metrics = {
            "rmse_mm": float(np.sqrt(s_se / n_phys)),
            "mae_mm": float(s_ae / n_phys),
            "bias_mm": float(s_bias / n_phys),
            "n_samples": int(sum(len(a) for a in per_sample_std)),
        }
    metrics["mse_std"] = float(sum_loss / max(n_elements, 1))
    metrics["mae_std"] = float(sum_abs / max(n_elements, 1))

    # Sample-wise errors, aligned by index (see the ordering note in
    # save_results).
    per_sample = {
        "mse_std": np.concatenate(per_sample_std),
        "mae_std": np.concatenate(per_sample_mae_std),
        "mse_mm": np.concatenate(per_sample_mse_mm),
        "mae_mm": np.concatenate(per_sample_mae_mm),
    }
    # Every sample has the same pixel count, so the sample mean is the pixel
    # mean.
    for f in FLOORS_MM:
        per_sample[f"mse_log_{f}"] = np.concatenate(per_sample_log_floor[f])
        metrics[f"mse_log_{f}"] = float(per_sample[f"mse_log_{f}"].mean())

    if ens_scores["crps_std"]:
        for k, v in ens_scores.items():
            v = np.concatenate(v)
            metrics[k] = float(v.mean())
            per_sample[k] = v
        # (n, M + 1) pixel counts per sample; not a scalar, so not in metrics.
        per_sample["rank_hist"] = np.concatenate(rank_hists)

    return metrics, pred, true, per_sample


def save_results(metrics, src_pred, src_true, tgt_pred, tgt_true, out_dir, n_save=5,
                 src_per_sample=None, tgt_per_sample=None, train_per_sample=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Sample-wise losses on the test sets, one row per sample, in test-set
    # order so they can be joined to the inputs by index. Saved for every
    # evaluated configuration, independent of n_save (which only controls how
    # many example FIELDS are stored).
    if any(d is not None for d in (src_per_sample, tgt_per_sample, train_per_sample)):
        ps = {}
        for tag, d in (("src", src_per_sample), ("tgt", tgt_per_sample),
                       ("train", train_per_sample)):
            if d is None:
                continue
            for k, v in d.items():
                v = np.asarray(v)
                # Counts (rank_hist) stay integer; losses are stored as float32.
                ps[f"{tag}_{k}"] = v.astype(np.int32 if v.dtype.kind in "iu" else np.float32)
        if ps:
            # Ordering contract. These arrays are in DataLoader emission order,
            # which equals the Zarr time order ONLY because the evaluation
            # loaders use num_workers<=1. With num_workers>1 the IterableDataset
            # shards chunks strided across workers and the DataLoader
            # round-robins their batches, so emission order becomes a
            # permutation of time order (e.g. with 2 workers, batch_size 64 and
            # 200-sample chunks, position 64 is time index 200). Do not raise
            # num_workers here without also emitting an explicit index.
            n_src = len(ps.get("src_mse_std", []))
            n_tgt = len(ps.get("tgt_mse_std", []))
            if n_src:
                ps["src_time_index"] = np.arange(n_src, dtype=np.int32)
            if n_tgt:
                ps["tgt_time_index"] = np.arange(n_tgt, dtype=np.int32)
            # No train_time_index. The TRAIN split shuffles its chunk order,
            # streams through a shuffle buffer and skips validation and purged
            # chunks, so emission order is not time order even at
            # num_workers=1. The train per-sample losses are usable as a
            # distribution (OODiag), not for per-timestep joins.
            np.savez_compressed(out_dir / "per_sample_loss.npz", **ps)

    if n_save > 0:
        out_dict = {}

        # Save Source Predictions
        if src_pred.shape[0] >= n_save:
            s_mse = np.mean((src_pred - src_true) ** 2, axis=(1, 2, 3))
            s_idx = np.argsort(s_mse)
            out_dict.update(
                {
                    "src_best_pred": src_pred[s_idx[:n_save]],
                    "src_best_true": src_true[s_idx[:n_save]],
                    "src_worst_pred": src_pred[s_idx[-n_save:]],
                    "src_worst_true": src_true[s_idx[-n_save:]],
                }
            )

        # Save Target Predictions
        if tgt_pred.shape[0] >= n_save:
            t_mse = np.mean((tgt_pred - tgt_true) ** 2, axis=(1, 2, 3))
            t_idx = np.argsort(t_mse)
            out_dict.update(
                {
                    "tgt_best_pred": tgt_pred[t_idx[:n_save]],
                    "tgt_best_true": tgt_true[t_idx[:n_save]],
                    "tgt_worst_pred": tgt_pred[t_idx[-n_save:]],
                    "tgt_worst_true": tgt_true[t_idx[-n_save:]],
                }
            )

        if out_dict:
            np.savez_compressed(out_dir / "predictions.npz", **out_dict)


# A row is uniquely identified by this key. Re-evaluating the same experiment
# must REPLACE its row, not add a second one.
_ROW_KEY = ("model", "source", "target", "method", "transform")


def append_csv(csv_path, row):
    """Write `row`, replacing any existing row with the same identity key.

    The previous implementation opened in append mode unconditionally, so every
    re-run of evaluate.sh duplicated all rows (observed: 6 oracle rows written
    twice). Duplicates silently corrupt any downstream aggregation that means
    or counts rows, e.g. compute_ngg_all.
    """
    csv_path = Path(csv_path)
    fields = list(row.keys())
    key = tuple(str(row.get(k, "")) for k in _ROW_KEY)

    kept = []
    if csv_path.exists():
        with open(csv_path) as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        if lines:
            header = lines[0].split(",")
            idx = [header.index(k) for k in _ROW_KEY if k in header]
            for ln in lines[1:]:
                parts = ln.split(",")
                if len(idx) == len(_ROW_KEY) and len(parts) == len(header):
                    if tuple(parts[i] for i in idx) == key:
                        continue  # drop the stale row for this experiment
                kept.append(ln)
            fields = header  # preserve the existing column order

    with open(csv_path, "w") as f:
        f.write(",".join(fields) + "\n")
        for ln in kept:
            f.write(ln + "\n")
        f.write(",".join(str(row.get(k, "")) for k in fields) + "\n")


def parse_exp_name(name):
    name = name.removeprefix("afm_")
    m = re.match(r"(.+?)__to__(.+?)__(.+)", name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None


def target_checkpoint(ckpt, method):
    """
    Checkpoint to score on the TARGET domain. AdaBN keeps per-domain
    BatchNorm statistics: best.pt holds the source statistics, and both
    trainers write the target-adapted model to best_adabn.pt. Scoring best.pt
    on the target (as evaluate.py did until 2026-10-02) evaluates an unadapted
    model, i.e. a second source-only run. Returns None for an AdaBN run
    without best_adabn.pt.
    """
    ckpt = Path(ckpt)
    if method != "adabn":
        return ckpt
    adapted = ckpt.parent / "best_adabn.pt"
    return adapted if adapted.exists() else None


# ---------------------------------------------------------------------------
#  Single evaluation
# ---------------------------------------------------------------------------


def cmd_single(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Source Setup
    src_stats = load_domain_stats(args.src_path)
    src_loader = DataLoader(
        ClimateSRDatasetNPY(args.src_path, "test", stats=src_stats),
        args.batch_size,
        shuffle=False,
        num_workers=1,   # see note in save_results: >1 permutes sample order
        pin_memory=True,
    )

    # Target Setup
    tgt_stats = load_domain_stats(args.target_path)
    tgt_loader = DataLoader(
        ClimateSRDatasetNPY(args.target_path, "test", stats=tgt_stats),
        args.batch_size,
        shuffle=False,
        num_workers=1,   # see note in save_results: >1 permutes sample order
        pin_memory=True,
    )

    model = load_model(args.model, args.checkpoint, device, args.base_features)
    # Same target-checkpoint rule as batch mode (best_adabn.pt for AdaBN).
    _, _, method = parse_exp_name(Path(args.checkpoint).parent.stem)
    tgt_ckpt = target_checkpoint(args.checkpoint, method)
    if tgt_ckpt is None:
        raise FileNotFoundError(f"adabn run without best_adabn.pt next to {args.checkpoint}")
    tgt_model = model if tgt_ckpt == Path(args.checkpoint) else load_model(
        args.model, str(tgt_ckpt), device, args.base_features)

    transform = None
    if args.input_transform != "none":
        transform = _load_or_fit_transform(args.src_path, args.target_path, args, device)

    # 1. Evaluate Source (No Transform)
    src_metrics, src_pred, src_true, src_per_sample = evaluate_model(
        model,
        args.model,
        src_loader,
        device,
        src_stats,
        args.n_ensemble,
        args.sample_steps,
        transform=None,
        desc="Evaluating Source",
    )

    # 2. Evaluate Target (With Transform)
    tgt_metrics, tgt_pred, tgt_true, tgt_per_sample = evaluate_model(
        tgt_model,
        args.model,
        tgt_loader,
        device,
        tgt_stats,
        args.n_ensemble,
        args.sample_steps,
        transform=transform,
        desc="Evaluating Target",
    )

    # Combine Metrics
    combined_metrics = {f"src_{k}": v for k, v in src_metrics.items()}
    combined_metrics.update({f"tgt_{k}": v for k, v in tgt_metrics.items()})

    tag_base = Path(args.checkpoint).parent.stem
    tag = f"{tag_base}____{args.input_transform}" if args.input_transform != "none" else tag_base
    res_dir = Path(args.output_dir) / "results" / tag

    save_results(combined_metrics, src_pred, src_true, tgt_pred, tgt_true, res_dir, args.save_samples,
                     src_per_sample=src_per_sample, tgt_per_sample=tgt_per_sample)

    src, tgt, method = parse_exp_name(tag_base)
    append_csv(
        Path(args.output_dir) / "results" / "results.csv",
        {
            "model": args.model,
            "source": src,
            "target": tgt,
            "method": method,
            "transform": args.input_transform,
            **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in combined_metrics.items()},
        },
    )

    for k, v in combined_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------------
#  Batch evaluation
# ---------------------------------------------------------------------------


def cmd_batch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_root = Path(args.exp_root)
    data_root = Path(args.data_root)
    csv_path = Path(args.output_dir) / "results.csv"

    experiments = sorted(exp_root.glob(f"*/{args.ckpt_name}"))
    print(f"Found {len(experiments)} checkpoints in {exp_root}")

    # --only: score the named runs and nothing else, one checkpoint per array
    # task (PLAN.md 3.6 step 5). Each task then writes its row to its own
    # rows/<tag>.csv, because append_csv rewrites results.csv and concurrent
    # tasks would lose each other's rows; `evaluate.py merge` builds
    # results.csv from rows/ once every task is done.
    if args.only:
        found = {c.parent.name: c for c in experiments}
        missing = [t for t in args.only if t not in found]
        if missing:
            raise SystemExit(f"--only: no {exp_root}/<tag>/{args.ckpt_name} for {missing}")
        experiments = [found[t] for t in args.only]
        csv_path = Path(args.output_dir) / "rows" / f"{'+'.join(args.only)}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    for ckpt in experiments:
        exp_name = ckpt.parent.stem
        model_type = _infer_model_type(exp_name)
        src, tgt, method = parse_exp_name(exp_name)
        if tgt is None:
            print(f"  Skipping {exp_name} (cannot parse)")
            continue

        tgt_path = data_root / tgt
        src_path = data_root / src

        if not (tgt_path / "normalization_stats.json").exists() or not (src_path / "normalization_stats.json").exists():
            print(f"  Skipping {exp_name} (missing normalization_stats.json)")
            continue

        tag = f"{exp_name}__{args.input_transform}" if args.input_transform != "none" else exp_name
        res_dir = Path(args.output_dir) / tag
        if (res_dir / "metrics.json").exists() and not args.force:
            print(f"  Skipping {tag} (already evaluated)")
            continue

        print(f"  Evaluating {tag} ...")

        # SOURCE stats normalize BOTH source and target: the model lives in
        # source-normalized space, so target inputs must be normalized by, and
        # target predictions inverse-transformed by, source stats. Using target
        # stats here would give physically wrong mm values (the same bug that
        # was fixed in the trainers). load_stats reads normalization_stats.json.
        src_stats = load_stats(str(src_path))

        tgt_loader = DataLoader(
            ClimateSRDatasetZarr(str(tgt_path), "test", stats=src_stats),
            args.batch_size,
            shuffle=False,
            num_workers=1,   # see note in save_results: >1 permutes sample order
            pin_memory=True,
        )
        src_loader = DataLoader(
            ClimateSRDatasetZarr(str(src_path), "test", stats=src_stats),
            args.batch_size,
            shuffle=False,
            num_workers=1,   # see note in save_results: >1 permutes sample order
            pin_memory=True,
        )

        transform = None
        if args.input_transform != "none":
            transform = _load_or_fit_transform(str(src_path), str(tgt_path), args, device)

        base_features = run_base_features(ckpt.parent, args.base_features)
        model = load_model(model_type, str(ckpt), device, base_features)

        # Target is scored with target_checkpoint() (best_adabn.pt for AdaBN);
        # source and train splits keep best.pt and its source statistics.
        tgt_ckpt = target_checkpoint(ckpt, method)
        if tgt_ckpt is None:
            print(f"  Skipping {exp_name} (adabn run without best_adabn.pt)")
            del model
            continue
        tgt_model = model if tgt_ckpt == ckpt else load_model(model_type, str(tgt_ckpt), device, base_features)

        # 1. Evaluate Source (source stats)
        src_metrics, src_pred, src_true, src_per_sample = evaluate_model(
            model,
            model_type,
            src_loader,
            device,
            src_stats,
            args.n_ensemble,
            args.sample_steps,
            transform=None,
            desc="Evaluating Source",
        )

        # 2. Evaluate Target (SOURCE stats — model outputs live in source space)
        tgt_metrics, tgt_pred, tgt_true, tgt_per_sample = evaluate_model(
            tgt_model,
            model_type,
            tgt_loader,
            device,
            src_stats,
            args.n_ensemble,
            args.sample_steps,
            transform=transform,
            desc="Evaluating Target",
        )

        # Combine Metrics
        combined_metrics = {f"src_{k}": v for k, v in src_metrics.items()}
        combined_metrics.update({f"tgt_{k}": v for k, v in tgt_metrics.items()})

        # ---- TRAINING-split pass -------------------------------------
        # OODiag-style diagnostics reference the training loss L_train and
        # need it sample-wise, not just as an aggregate. We therefore also
        # score the model on its own SOURCE TRAINING split. Fields are not
        # collected (collect_fields=False): at ~122k samples of 200x200 the
        # prediction and truth arrays would be ~40 GB. Only metrics and
        # per-sample losses are produced -- no example fields are saved for
        # this split, which is what "no predictions" means here.
        train_per_sample = None
        if args.eval_train:
            train_loader = DataLoader(
                ClimateSRDatasetZarr(str(src_path), "train", stats=src_stats,
                                     subset_chunks=args.train_subset_chunks),
                args.batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )
            train_metrics, _, _, train_per_sample = evaluate_model(
                model, model_type, train_loader, device, src_stats,
                args.n_ensemble, args.sample_steps, transform=None,
                desc="Evaluating Train", collect_fields=False,
            )
            for k, v in train_metrics.items():
                combined_metrics[f"train_{k}"] = v

        save_results(combined_metrics, src_pred, src_true, tgt_pred, tgt_true, res_dir, args.save_samples,
                     src_per_sample=src_per_sample, tgt_per_sample=tgt_per_sample,
                     train_per_sample=train_per_sample)

        append_csv(
            csv_path,
            {
                "model": model_type,
                "source": src,
                "target": tgt,
                "method": method,
                "transform": args.input_transform,
                **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in combined_metrics.items()},
            },
        )

        for k, v in combined_metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")

        del model, tgt_model
        torch.cuda.empty_cache()

    print(f"\nResults -> {csv_path}")


def cmd_merge(args):
    """Build results.csv from the per-task rows/ files of `batch --only`.
    Refuses to touch an existing results.csv (CLAUDE.md rule 4); move it to a
    _trash/ folder first if a re-merge is really wanted."""
    out_dir = Path(args.output_dir)
    out = out_dir / "results.csv"
    if out.exists():
        raise SystemExit(f"{out} exists. Not overwriting.")
    files = sorted((out_dir / "rows").glob("*.csv"))
    if not files:
        raise SystemExit(f"no row files in {out_dir / 'rows'}")
    header, rows = None, []
    for f in files:
        lines = [ln.rstrip("\n") for ln in f.open() if ln.strip()]
        if header is None:
            header = lines[0]
        elif lines[0] != header:
            raise SystemExit(f"{f}: header differs from {files[0]}")
        rows += lines[1:]
    keys = [tuple(r.split(",")[header.split(",").index(k)] for k in _ROW_KEY) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate (model, source, target, method, transform) rows in rows/")
    out.write_text("\n".join([header, *rows]) + "\n")
    print(f"{len(rows)} rows from {len(files)} files -> {out}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def _add_common_args(p):
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--base_features", type=int, default=32)
    p.add_argument("--n_ensemble", type=int, default=0)
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--save_samples", type=int, default=5)

    # input transforms (shared)
    p.add_argument(
        "--eval_train", action="store_true",
        help="also score the model on its own SOURCE TRAINING split (metrics + "
             "per-sample losses only, no fields). Needed for OODiag-style "
             "diagnostics that reference L_train."
    )
    p.add_argument(
        "--train_subset_chunks", type=int, default=None,
        help="cap the training-split chunks scored by --eval_train "
             "(200 samples per chunk). None = the full split."
    )
    p.add_argument(
        "--input_transform", default="none", choices=_TRANSFORM_CHOICES, help="test-time input transform to apply"
    )
    p.add_argument("--transform_cache_dir", default="./experiments/transforms")
    p.add_argument("--transform_refit", action="store_true")
    p.add_argument("--transform_fit_subset", type=int, default=8000,
                   help="training samples used for fitting; QM reads them as subset/200 chunks spread over the split")

    # QM-specific
    p.add_argument("--qm_n_quantiles", type=int, default=1000)

    # OT-specific
    p.add_argument("--ot_n_anchors", type=int, default=5000)
    p.add_argument("--ot_reg", type=float, default=0.1)
    p.add_argument("--ot_n_iter", type=int, default=300)
    p.add_argument("--ot_k", type=int, default=10)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("single")
    s.add_argument("--model", required=True, choices=["unet", "afm"])
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--target_path", required=True)
    s.add_argument("--src_path", required=True, help="source dataset dir (now required to compute source metrics)")
    s.add_argument("--output_dir", default="./experiments")
    _add_common_args(s)

    b = sub.add_parser("batch")
    b.add_argument("--exp_root", required=True)
    b.add_argument("--data_root", required=True)
    b.add_argument("--output_dir", default="./experiments/results")
    b.add_argument("--force", action="store_true")
    b.add_argument("--ckpt_name", default="best.pt",
                   help="checkpoint file in each run directory (last.pt for the few-shot runs, P16)")
    b.add_argument("--only", nargs="+", default=None, metavar="TAG",
                   help="score only these run directories under --exp_root; "
                        "the row goes to <output_dir>/rows/, see `merge`")
    _add_common_args(b)

    m = sub.add_parser("merge", help="results.csv from the rows/ of batch --only")
    m.add_argument("--output_dir", required=True)

    args = p.parse_args()
    {"single": cmd_single, "batch": cmd_batch, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
