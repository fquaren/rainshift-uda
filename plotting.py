"""
Generate physical spatial plots for test samples of one trained run.
Retrieves inputs, static fields and targets from both the source and the
target test splits.

Both splits are read from the Zarr data and normalised with the SOURCE
stats, as in evaluate.py batch: the model lives in source-normalised space,
so the target inputs are normalised, and every field (target prediction
included) is inverse-transformed, with the source stats. The source panels
use best.pt; the target panels use evaluate.target_checkpoint() (best_adabn.pt
for AdaBN, best.pt otherwise). Only finished runs (evaluate.run_finished) are
plotted, and --output_dir is refused if it exists.

Ported on 2026-10-29 from the NPY version, which normalised the target with
its own stats and drew best_adabn.pt for the source panels too.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from torch.utils.data import DataLoader

from data.dataset import inverse_transform
from data.dataset_zarr import ClimateSRDatasetZarr
from evaluate import (_infer_model_type, load_model, load_stats, parse_exp_name, run_base_features,
                      run_finished, target_checkpoint)

DEFAULT_INPUT_VARS = ["cape", "cp", "sp", "tclw", "tcw", "tisr", "tp", "u", "v"]
DEFAULT_STATIC_VARS = ["lsm", "z"]


def plot_sample(
    x_src, s_src, y_src, y_pred_src, x_tgt, s_tgt, y_tgt, y_pred_tgt, stats, names, sample_idx, out_dir
):
    """stats: the SOURCE stats, for both domains. names: (source, target)."""
    src_name, tgt_name = names
    # 4 rows, 9 columns to fit all variables across both domains
    fig, axes = plt.subplots(4, 9, figsize=(32, 16))
    used_axes = []

    # 1. Plot Dynamic Inputs (Row 0: Source, Row 1: Target)
    for i, var_name in enumerate(DEFAULT_INPUT_VARS):
        # Source
        ax_src = axes[0, i]
        field_src = inverse_transform(x_src[i], var_name, stats)
        im_src = ax_src.imshow(field_src, cmap="viridis", origin="upper")
        ax_src.set_title(f"SRC {src_name}: {var_name}")
        fig.colorbar(im_src, ax=ax_src, fraction=0.046, pad=0.04)
        used_axes.append(ax_src)

        # Target
        ax_tgt = axes[1, i]
        field_tgt = inverse_transform(x_tgt[i], var_name, stats)
        im_tgt = ax_tgt.imshow(field_tgt, cmap="viridis", origin="upper")
        ax_tgt.set_title(f"TGT {tgt_name}: {var_name}")
        fig.colorbar(im_tgt, ax=ax_tgt, fraction=0.046, pad=0.04)
        used_axes.append(ax_tgt)

    # 2. Plot Static Inputs (Row 2)
    for i, var_name in enumerate(DEFAULT_STATIC_VARS):
        cmap = "terrain" if var_name == "z" else "ocean"

        # Source
        ax_src = axes[2, i]
        field_src = inverse_transform(s_src[i], var_name, stats)
        im_src = ax_src.imshow(field_src, cmap=cmap, origin="upper")
        ax_src.set_title(f"SRC Static: {var_name}")
        fig.colorbar(im_src, ax=ax_src, fraction=0.046, pad=0.04)
        used_axes.append(ax_src)

        # Target
        ax_tgt = axes[2, i + 2]
        field_tgt = inverse_transform(s_tgt[i], var_name, stats)
        im_tgt = ax_tgt.imshow(field_tgt, cmap=cmap, origin="upper")
        ax_tgt.set_title(f"TGT Static: {var_name}")
        fig.colorbar(im_tgt, ax=ax_tgt, fraction=0.046, pad=0.04)
        used_axes.append(ax_tgt)

    # 3. Plot Targets and Predictions (Row 3), all in mm via the source stats
    src_true_field = inverse_transform(y_src[0], "precipitation", stats)
    src_pred_field = inverse_transform(y_pred_src[0], "precipitation", stats)
    tgt_true_field = inverse_transform(y_tgt[0], "precipitation", stats)
    tgt_pred_field = inverse_transform(y_pred_tgt[0], "precipitation", stats)

    vmin = 0.1
    vmax = max(src_true_field.max(), src_pred_field.max(), tgt_true_field.max(), tgt_pred_field.max(), 1.0)

    # Clip arrays to prevent matplotlib LogNorm crashes
    src_true_field = np.clip(src_true_field, a_min=vmin, a_max=None)
    src_pred_field = np.clip(src_pred_field, a_min=vmin, a_max=None)
    tgt_true_field = np.clip(tgt_true_field, a_min=vmin, a_max=None)
    tgt_pred_field = np.clip(tgt_pred_field, a_min=vmin, a_max=None)

    panels = [(src_true_field, "SRC Target: precip"), (src_pred_field, "SRC Prediction (best.pt)"),
              (tgt_true_field, "TGT Target: precip"), (tgt_pred_field, "TGT Prediction")]
    for k, (field, title) in enumerate(panels):
        ax = axes[3, k]
        im = ax.imshow(field, cmap="Blues", norm=LogNorm(vmin=vmin, vmax=vmax), origin="upper")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        used_axes.append(ax)

    # Disable all unused axes systematically
    for ax in axes.flatten():
        if ax not in used_axes:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / f"sample_{sample_idx:05d}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _predict(model, model_type, x, s):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(x, s) if model_type == "unet" else model.deterministic_predict(x, s)
    return torch.clamp(pred.float(), min=-50.0, max=50.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="finished run directory (best.pt + config.json)")
    p.add_argument("--data_root", required=True, help="Zarr RainShift root (read-only)")
    p.add_argument("--output_dir", required=True, help="PDFs go here; refused if it exists")
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--every", type=int, default=1000,
                   help="plot test samples 0, every, 2*every, ... (time order) rather than the first hours")
    p.add_argument("--base_features", type=int, default=32)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; not overwriting.")
    if not run_finished(run_dir):
        raise SystemExit(f"{run_dir} is not finished (no config.json newer than its checkpoints).")
    exp_name = run_dir.name
    model_type = _infer_model_type(exp_name)
    src, tgt, method = parse_exp_name(exp_name)
    if tgt is None:
        raise SystemExit(f"cannot parse {exp_name} as <src>__to__<tgt>__<method>")

    # SOURCE stats for both domains (CLAUDE.md: the model lives in
    # source-normalised space).
    data_root = Path(args.data_root)
    src_stats = load_stats(str(data_root / src))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = run_dir / "best.pt"
    tgt_ckpt = target_checkpoint(ckpt, method)
    if tgt_ckpt is None:
        raise SystemExit(f"{exp_name}: adabn run without best_adabn.pt")
    base_features = run_base_features(run_dir, args.base_features)
    model = load_model(model_type, str(ckpt), device, base_features)
    tgt_model = model if tgt_ckpt == ckpt else load_model(model_type, str(tgt_ckpt), device, base_features)
    print(f"{exp_name}: source panels {ckpt.name}, target panels {tgt_ckpt.name}")

    loaders = [
        DataLoader(ClimateSRDatasetZarr(str(data_root / d), "test", stats=src_stats),
                   batch_size=1, shuffle=False, num_workers=1)
        for d in (src, tgt)
    ]

    out_dir.mkdir(parents=True)
    n_done = 0
    print(f"Plotting {args.n_samples} samples (every {args.every}) for {exp_name} -> {out_dir}")
    with torch.no_grad():
        for idx, (src_batch, tgt_batch) in enumerate(zip(*loaders)):
            if idx % args.every:
                continue

            x_src, s_src, y_src = (t.to(device) for t in src_batch)
            x_tgt, s_tgt, y_tgt = (t.to(device) for t in tgt_batch)
            pred_src = _predict(model, model_type, x_src, s_src)
            pred_tgt = _predict(tgt_model, model_type, x_tgt, s_tgt)

            plot_sample(
                x_src.squeeze(0).cpu().numpy(),
                s_src.squeeze(0).cpu().numpy(),
                y_src.squeeze(0).cpu().numpy(),
                pred_src.squeeze(0).cpu().numpy(),
                x_tgt.squeeze(0).cpu().numpy(),
                s_tgt.squeeze(0).cpu().numpy(),
                y_tgt.squeeze(0).cpu().numpy(),
                pred_tgt.squeeze(0).cpu().numpy(),
                src_stats,
                (src, tgt),
                idx,
                out_dir,
            )
            n_done += 1
            if n_done >= args.n_samples:
                break
    print(f"{n_done} plots written")


if __name__ == "__main__":
    main()
