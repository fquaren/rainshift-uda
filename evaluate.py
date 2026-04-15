"""
Evaluate trained UNet / AFM models on target domains.

Modes:
  single:  evaluate one checkpoint on one target domain
  batch:   scan experiment directory, evaluate all found checkpoints

Metrics (physical space, mm):
  Deterministic: RMSE, MAE, bias
  Probabilistic: CRPS, spread (AFM only, when --n_ensemble > 0)

Usage:
  # Single model
  python evaluate.py single \
      --checkpoint experiments/europe_west__to__melanesia__coral/best.pt \
      --target_path /data/rainshift_npy/melanesia \
      --output_dir experiments

  # Batch (scan all experiments)
  python evaluate.py batch \
      --exp_root experiments/unet \
      --output_dir experiments/results
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ClimateSRDatasetNPY, inverse_transform, load_domain_stats
from models.unet import DualEncoderUNet
from models.afm import AFMModel


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred, true):
    d = pred - true
    return {
        "rmse_mm": float(np.sqrt(np.mean(d ** 2))),
        "mae_mm": float(np.mean(np.abs(d))),
        "bias_mm": float(np.mean(d)),
        "n_samples": int(pred.shape[0]),
    }


def compute_crps(ensemble, truth):
    """ensemble: (N, M, ...), truth: (N, ...)."""
    M = ensemble.shape[1]
    abs_err = np.abs(ensemble - truth[:, np.newaxis]).mean()
    spread = 0.0
    count = 0
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
    """Guess model type from directory name."""
    return "afm" if exp_dir_name.startswith("afm_") else "unet"


def load_model(model_type, checkpoint, device, base_features=64):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    clean = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}

    if model_type == "unet":
        model = DualEncoderUNet(9, 2, 1, base_features)
    else:
        model = AFMModel(9, 2, 1, base_features, encoder_loss_weight=0.1)

    model.load_state_dict(clean)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, model_type, loader, device, stats,
                   n_ensemble=0, sample_steps=20):
    var = "precipitation"
    all_pred, all_true, all_ens = [], [], []

    for batch in loader:
        x, s, y = (t.to(device, non_blocking=True) for t in batch)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if model_type == "unet":
                pred = model(x, s)
            else:
                pred = model.deterministic_predict(x, s)

        all_pred.append(inverse_transform(pred.float().cpu().numpy(), var, stats))
        all_true.append(inverse_transform(y.float().cpu().numpy(), var, stats))

        if model_type == "afm" and n_ensemble > 0:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                samples = model.sample(x, s, n_samples=n_ensemble, steps=sample_steps)
            ens_np = np.stack([
                inverse_transform(samples[:, i].float().cpu().numpy(), var, stats)
                for i in range(n_ensemble)
            ], axis=1)
            all_ens.append(ens_np)

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    metrics = compute_metrics(pred, true)

    if all_ens:
        ens = np.concatenate(all_ens)
        metrics["crps_mm"] = compute_crps(ens, true)
        metrics["spread_mm"] = float(ens.std(axis=1).mean())

    return metrics, pred, true


def save_results(metrics, pred, true, out_dir, n_save=5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if n_save > 0 and pred.shape[0] >= n_save:
        mse = np.mean((pred - true) ** 2, axis=(1, 2, 3))
        idx = np.argsort(mse)
        np.savez_compressed(
            out_dir / "predictions.npz",
            best_pred=pred[idx[:n_save]], best_true=true[idx[:n_save]],
            worst_pred=pred[idx[-n_save:]], worst_true=true[idx[-n_save:]],
        )


def append_csv(csv_path, row):
    """Append a dict as a CSV row, creating header if needed."""
    csv_path = Path(csv_path)
    header_needed = not csv_path.exists()
    with open(csv_path, "a") as f:
        if header_needed:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")


# ---------------------------------------------------------------------------
#  Parse experiment directory name -> (source, target, method)
# ---------------------------------------------------------------------------

def parse_exp_name(name):
    """
    Parse 'europe_west__to__melanesia__coral' or
          'afm_europe_west__to__melanesia__coral'.
    """
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
    tgt_stats = load_domain_stats(args.target_path)
    loader = DataLoader(
        ClimateSRDatasetNPY(args.target_path, "test", stats=tgt_stats),
        args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = load_model(args.model, args.checkpoint, device, args.base_features)

    metrics, pred, true = evaluate_model(
        model, args.model, loader, device, tgt_stats,
        args.n_ensemble, args.sample_steps)

    tag = Path(args.checkpoint).parent.stem
    res_dir = Path(args.output_dir) / "results" / tag
    save_results(metrics, pred, true, res_dir, args.save_samples)

    src, tgt, method = parse_exp_name(tag)
    append_csv(Path(args.output_dir) / "results" / "results.csv", {
        "model": args.model, "source": src, "target": tgt, "method": method,
        **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()},
    })

    for k, v in metrics.items():
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

    # find all experiments with a best.pt
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
        if not (tgt_path / "stats.json").exists():
            print(f"  Skipping {exp_name} (missing stats at {tgt_path})")
            continue

        # skip if already evaluated
        res_dir = Path(args.output_dir) / exp_name
        if (res_dir / "metrics.json").exists() and not args.force:
            print(f"  Skipping {exp_name} (already evaluated)")
            continue

        print(f"  Evaluating {exp_name} ...")
        tgt_stats = load_domain_stats(str(tgt_path))
        loader = DataLoader(
            ClimateSRDatasetNPY(str(tgt_path), "test", stats=tgt_stats),
            args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = load_model(model_type, str(ckpt), device, args.base_features)
        metrics, pred, true = evaluate_model(
            model, model_type, loader, device, tgt_stats,
            args.n_ensemble, args.sample_steps)

        save_results(metrics, pred, true, res_dir, args.save_samples)

        append_csv(csv_path, {
            "model": model_type, "source": src, "target": tgt, "method": method,
            **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()},
        })

        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")

        del model
        torch.cuda.empty_cache()

    print(f"\nResults -> {csv_path}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("single")
    s.add_argument("--model", required=True, choices=["unet", "afm"])
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--target_path", required=True)
    s.add_argument("--output_dir", default="./experiments")
    s.add_argument("--batch_size", type=int, default=128)
    s.add_argument("--base_features", type=int, default=64)
    s.add_argument("--n_ensemble", type=int, default=0)
    s.add_argument("--sample_steps", type=int, default=20)
    s.add_argument("--save_samples", type=int, default=5)

    b = sub.add_parser("batch")
    b.add_argument("--exp_root", required=True, help="dir containing experiment subdirs")
    b.add_argument("--data_root", required=True, help="dir containing region npy dirs")
    b.add_argument("--output_dir", default="./experiments/results")
    b.add_argument("--batch_size", type=int, default=128)
    b.add_argument("--base_features", type=int, default=64)
    b.add_argument("--n_ensemble", type=int, default=0)
    b.add_argument("--sample_steps", type=int, default=20)
    b.add_argument("--save_samples", type=int, default=5)
    b.add_argument("--force", action="store_true", help="re-evaluate existing")

    args = p.parse_args()
    if args.cmd == "single":
        cmd_single(args)
    else:
        cmd_batch(args)


if __name__ == "__main__":
    main()
