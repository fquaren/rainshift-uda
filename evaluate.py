"""
Evaluate trained UNet / AFM models on target domains.

Modes:
  single:  evaluate one checkpoint on one target domain
  batch:   scan experiment directory, evaluate all found checkpoints

Metrics (physical space, mm):
  Deterministic: RMSE, MAE, bias
  Probabilistic: CRPS, spread (AFM only, when --n_ensemble > 0)
  Standardized:  mse_std (exact optimizer loss space)

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
from models.afm import AFMModel
from models.unet import DualEncoderUNet
from quantile_map import QuantileMap, fit_qm_from_npy
from sinkhorn import SinkhornOTMap, fit_ot_from_npy


_TRANSFORM_CHOICES = ["none", "qm_tp", "qm_precip", "qm_all", "ot"]


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


def compute_crps(ensemble, truth):
    M = ensemble.shape[1]
    abs_err = np.abs(ensemble - truth[:, np.newaxis]).mean()
    spread, count = 0.0, 0
    for i in range(M):
        for j in range(i + 1, M):
            spread += np.abs(ensemble[:, i] - ensemble[:, j]).mean()
            count += 1
    spread /= max(count, 1)
    return float(abs_err - 0.5 * spread)


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------


def _infer_model_type(exp_dir_name):
    return "afm" if exp_dir_name.startswith("afm_") else "unet"


def load_model(model_type, checkpoint, device, base_features=32):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    clean = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
    if model_type == "unet":
        model = DualEncoderUNet(9, 2, 1, base_features)
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
        # QM variants
        channel_spec = {"qm_tp": "tp", "qm_precip": "precip", "qm_all": None}[kind]
        tf = fit_qm_from_npy(
            src_path=src_path,
            tgt_path=tgt_path,
            n_quantiles=args.qm_n_quantiles,
            channels=channel_spec,
            subset=args.transform_fit_subset,
        )

    tf.save(cache)
    return tf


# ---------------------------------------------------------------------------
#  Evaluation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model, model_type, loader, device, stats, n_ensemble=0, sample_steps=20, transform=None, desc="Evaluating"
):
    var = "precipitation"
    all_pred, all_true, all_ens = [], [], []

    sum_loss = 0.0
    n_elements = 0

    for batch in tqdm(loader, desc=desc):
        x, s, y = (t.to(device, non_blocking=True) for t in batch)

        if transform is not None:
            x = transform.transform(x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x, s) if model_type == "unet" else model.deterministic_predict(x, s)

        # Accumulate exact MSE in standardized space
        sum_loss += torch.sum((pred - y) ** 2).item()
        n_elements += pred.numel()

        all_pred.append(inverse_transform(pred.float().cpu().numpy(), var, stats))
        all_true.append(inverse_transform(y.float().cpu().numpy(), var, stats))

        if model_type == "afm" and n_ensemble > 0:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                samples = model.sample(x, s, n_samples=n_ensemble, steps=sample_steps)
            ens_np = np.stack(
                [inverse_transform(samples[:, i].float().cpu().numpy(), var, stats) for i in range(n_ensemble)], axis=1
            )
            all_ens.append(ens_np)

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    metrics = compute_metrics(pred, true)
    metrics["mse_std"] = float(sum_loss / max(n_elements, 1))

    if all_ens:
        ens = np.concatenate(all_ens)
        metrics["crps_mm"] = compute_crps(ens, true)
        metrics["spread_mm"] = float(ens.std(axis=1).mean())

    return metrics, pred, true


def save_results(metrics, src_pred, src_true, tgt_pred, tgt_true, out_dir, n_save=5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

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


def append_csv(csv_path, row):
    csv_path = Path(csv_path)
    header_needed = not csv_path.exists()
    with open(csv_path, "a") as f:
        if header_needed:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")


def parse_exp_name(name):
    name = name.removeprefix("afm_")
    m = re.match(r"(.+?)__to__(.+?)__(.+)", name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None


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
        num_workers=2,
        pin_memory=True,
    )

    # Target Setup
    tgt_stats = load_domain_stats(args.target_path)
    tgt_loader = DataLoader(
        ClimateSRDatasetNPY(args.target_path, "test", stats=tgt_stats),
        args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = load_model(args.model, args.checkpoint, device, args.base_features)

    transform = None
    if args.input_transform != "none":
        transform = _load_or_fit_transform(args.src_path, args.target_path, args, device)

    # 1. Evaluate Source (No Transform)
    src_metrics, src_pred, src_true = evaluate_model(
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
    tgt_metrics, tgt_pred, tgt_true = evaluate_model(
        model,
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

    save_results(combined_metrics, src_pred, src_true, tgt_pred, tgt_true, res_dir, args.save_samples)

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

    experiments = sorted(exp_root.glob("*/best.pt"))
    print(f"Found {len(experiments)} checkpoints in {exp_root}")

    for ckpt in experiments:
        exp_name = ckpt.parent.stem
        model_type = _infer_model_type(exp_name)
        src, tgt, method = parse_exp_name(exp_name)
        if tgt is None:
            print(f"  Skipping {exp_name} (cannot parse)")
            continue

        tgt_path = data_root / tgt
        src_path = data_root / src

        if not (tgt_path / "stats.json").exists() or not (src_path / "stats.json").exists():
            print(f"  Skipping {exp_name} (missing domain stats)")
            continue

        tag = f"{exp_name}__{args.input_transform}" if args.input_transform != "none" else exp_name
        res_dir = Path(args.output_dir) / tag
        if (res_dir / "metrics.json").exists() and not args.force:
            print(f"  Skipping {tag} (already evaluated)")
            continue

        print(f"  Evaluating {tag} ...")

        # Load Data
        tgt_stats = load_domain_stats(str(tgt_path))
        tgt_loader = DataLoader(
            ClimateSRDatasetNPY(str(tgt_path), "test", stats=tgt_stats),
            args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        src_stats = load_domain_stats(str(src_path))
        src_loader = DataLoader(
            ClimateSRDatasetNPY(str(src_path), "test", stats=src_stats),
            args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        transform = None
        if args.input_transform != "none":
            transform = _load_or_fit_transform(str(src_path), str(tgt_path), args, device)

        model = load_model(model_type, str(ckpt), device, args.base_features)

        # 1. Evaluate Source
        src_metrics, src_pred, src_true = evaluate_model(
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

        # 2. Evaluate Target
        tgt_metrics, tgt_pred, tgt_true = evaluate_model(
            model,
            model_type,
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

        save_results(combined_metrics, src_pred, src_true, tgt_pred, tgt_true, res_dir, args.save_samples)

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

        del model
        torch.cuda.empty_cache()

    print(f"\nResults -> {csv_path}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def _add_common_args(p):
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--base_features", type=int, default=64)
    p.add_argument("--n_ensemble", type=int, default=0)
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--save_samples", type=int, default=5)

    # input transforms (shared)
    p.add_argument(
        "--input_transform", default="none", choices=_TRANSFORM_CHOICES, help="test-time input transform to apply"
    )
    p.add_argument("--transform_cache_dir", default="./experiments/transforms")
    p.add_argument("--transform_refit", action="store_true")
    p.add_argument("--transform_fit_subset", type=int, default=2000, help="cap on training samples used for fitting")

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
    _add_common_args(b)

    args = p.parse_args()
    (cmd_single if args.cmd == "single" else cmd_batch)(args)


if __name__ == "__main__":
    main()
