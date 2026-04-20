"""
Generate physical spatial plots for samples of a trained model.
Retrieves inputs, static fields, and targets directly from both source and target datasets.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from torch.utils.data import DataLoader

from data.dataset import ClimateSRDatasetNPY, inverse_transform, load_domain_stats
from models.afm import AFMModel
from models.unet import DualEncoderUNet

DEFAULT_INPUT_VARS = ["cape", "cp", "sp", "tclw", "tcw", "tisr", "tp", "u", "v"]
DEFAULT_STATIC_VARS = ["lsm", "z"]


def _infer_model_type(exp_dir_name):
    return "afm" if "afm" in exp_dir_name else "unet"


def load_model(model_type, checkpoint, device, base_features=32):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    clean = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
    if model_type == "unet":
        model = DualEncoderUNet(9, 2, 1, base_features)
    else:
        model = AFMModel(9, 2, 1, base_features, encoder_loss_weight=0.1)
    model.load_state_dict(clean)
    return model.to(device).eval()


def plot_sample(
    x_src, s_src, y_src, y_pred_src, x_tgt, s_tgt, y_tgt, y_pred_tgt, src_stats, tgt_stats, sample_idx, out_dir
):
    # 4 rows, 9 columns to fit all variables across both domains
    fig, axes = plt.subplots(4, 9, figsize=(32, 16))
    used_axes = []

    # 1. Plot Dynamic Inputs (Row 0: Source, Row 1: Target)
    for i, var_name in enumerate(DEFAULT_INPUT_VARS):
        # Source
        ax_src = axes[0, i]
        field_src = inverse_transform(x_src[i], var_name, src_stats)
        im_src = ax_src.imshow(field_src, cmap="viridis", origin="upper")
        ax_src.set_title(f"SRC: {var_name}")
        fig.colorbar(im_src, ax=ax_src, fraction=0.046, pad=0.04)
        used_axes.append(ax_src)

        # Target
        ax_tgt = axes[1, i]
        field_tgt = inverse_transform(x_tgt[i], var_name, tgt_stats)
        im_tgt = ax_tgt.imshow(field_tgt, cmap="viridis", origin="upper")
        ax_tgt.set_title(f"TGT: {var_name}")
        fig.colorbar(im_tgt, ax=ax_tgt, fraction=0.046, pad=0.04)
        used_axes.append(ax_tgt)

    # 2. Plot Static Inputs (Row 2)
    for i, var_name in enumerate(DEFAULT_STATIC_VARS):
        cmap = "terrain" if var_name == "z" else "ocean"

        # Source
        ax_src = axes[2, i]
        field_src = inverse_transform(s_src[i], var_name, src_stats)
        im_src = ax_src.imshow(field_src, cmap=cmap, origin="upper")
        ax_src.set_title(f"SRC Static: {var_name}")
        fig.colorbar(im_src, ax=ax_src, fraction=0.046, pad=0.04)
        used_axes.append(ax_src)

        # Target
        ax_tgt = axes[2, i + 2]
        field_tgt = inverse_transform(s_tgt[i], var_name, tgt_stats)
        im_tgt = ax_tgt.imshow(field_tgt, cmap=cmap, origin="upper")
        ax_tgt.set_title(f"TGT Static: {var_name}")
        fig.colorbar(im_tgt, ax=ax_tgt, fraction=0.046, pad=0.04)
        used_axes.append(ax_tgt)

    # 3. Plot Targets and Predictions (Row 3)
    src_true_field = inverse_transform(y_src[0], "precipitation", src_stats)
    src_pred_field = inverse_transform(y_pred_src[0], "precipitation", src_stats)
    tgt_true_field = inverse_transform(y_tgt[0], "precipitation", tgt_stats)
    tgt_pred_field = inverse_transform(y_pred_tgt[0], "precipitation", tgt_stats)

    vmin = 0.1
    vmax = max(src_true_field.max(), src_pred_field.max(), tgt_true_field.max(), tgt_pred_field.max(), 1.0)

    # Clip arrays to prevent matplotlib LogNorm crashes
    src_true_field = np.clip(src_true_field, a_min=vmin, a_max=None)
    src_pred_field = np.clip(src_pred_field, a_min=vmin, a_max=None)
    tgt_true_field = np.clip(tgt_true_field, a_min=vmin, a_max=None)
    tgt_pred_field = np.clip(tgt_pred_field, a_min=vmin, a_max=None)

    ax_src_true = axes[3, 0]
    im_src_true = ax_src_true.imshow(src_true_field, cmap="Blues", norm=LogNorm(vmin=vmin, vmax=vmax), origin="upper")
    ax_src_true.set_title("SRC Target: precip")
    fig.colorbar(im_src_true, ax=ax_src_true, fraction=0.046, pad=0.04)
    used_axes.append(ax_src_true)

    ax_src_pred = axes[3, 1]
    im_src_pred = ax_src_pred.imshow(src_pred_field, cmap="Blues", norm=LogNorm(vmin=vmin, vmax=vmax), origin="upper")
    ax_src_pred.set_title("SRC Prediction: precip")
    fig.colorbar(im_src_pred, ax=ax_src_pred, fraction=0.046, pad=0.04)
    used_axes.append(ax_src_pred)

    ax_tgt_true = axes[3, 2]
    im_tgt_true = ax_tgt_true.imshow(tgt_true_field, cmap="Blues", norm=LogNorm(vmin=vmin, vmax=vmax), origin="upper")
    ax_tgt_true.set_title("TGT Target: precip")
    fig.colorbar(im_tgt_true, ax=ax_tgt_true, fraction=0.046, pad=0.04)
    used_axes.append(ax_tgt_true)

    ax_tgt_pred = axes[3, 3]
    im_tgt_pred = ax_tgt_pred.imshow(tgt_pred_field, cmap="Blues", norm=LogNorm(vmin=vmin, vmax=vmax), origin="upper")
    ax_tgt_pred.set_title("TGT Prediction: precip")
    fig.colorbar(im_tgt_pred, ax=ax_tgt_pred, fraction=0.046, pad=0.04)
    used_axes.append(ax_tgt_pred)

    # Disable all unused axes systematically
    for ax in axes.flatten():
        if ax not in used_axes:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / f"sample_{sample_idx:03d}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", required=True, help="Path to specific experiment result directory")
    p.add_argument("--data_root", required=True, help="Root path for npy datasets")
    p.add_argument("--n_samples", type=int, default=5)
    args = p.parse_args()

    res_dir = Path(args.result_dir)

    exp_name = res_dir.name
    transform_type = res_dir.parent.name
    model_type = res_dir.parent.parent.name
    exp_root = res_dir.parent.parent.parent.parent

    # Recover base model directory
    if transform_type != "none" and exp_name.endswith(f"__{transform_type}"):
        base_model_name = exp_name[: -len(f"__{transform_type}")]
    else:
        base_model_name = exp_name

    exp_dir = exp_root / model_type / base_model_name
    cfg_path = exp_dir / "config.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at {cfg_path}. Parsed base model: {base_model_name}")

    cfg = json.loads(cfg_path.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Target Dataloader
    tgt_domain = Path(cfg["target_path"]).name
    tgt_path = Path(args.data_root) / tgt_domain
    tgt_stats = load_domain_stats(str(tgt_path))
    tgt_loader = DataLoader(ClimateSRDatasetNPY(str(tgt_path), "test", stats=tgt_stats), batch_size=1, shuffle=False)

    # Setup Source Dataloader
    src_domain = Path(cfg["source_path"]).name
    src_path = Path(args.data_root) / src_domain
    src_stats = load_domain_stats(str(src_path))
    src_loader = DataLoader(ClimateSRDatasetNPY(str(src_path), "test", stats=src_stats), batch_size=1, shuffle=False)

    # Load appropriate checkpoint
    ckpt_path = exp_dir / "best_adabn.pt" if (exp_dir / "best_adabn.pt").exists() else exp_dir / "best.pt"
    model = load_model(model_type, ckpt_path, device, cfg.get("base_features", 32))

    out_dir = res_dir / "pdf_plots"
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Plotting {args.n_samples} samples for {res_dir.name}...")
    with torch.no_grad():
        for idx, (src_batch, tgt_batch) in enumerate(zip(src_loader, tgt_loader)):
            if idx >= args.n_samples:
                break

            x_src, s_src, y_src = (t.to(device) for t in src_batch)
            x_tgt, s_tgt, y_tgt = (t.to(device) for t in tgt_batch)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Prediction is done on both source and target domains
                if model_type == "unet":
                    pred_src = model(x_src, s_src)
                    pred_tgt = model(x_tgt, s_tgt)
                else:
                    pred_src = model.deterministic_predict(x_src, s_src)
                    pred_tgt = model.deterministic_predict(x_tgt, s_tgt)

            pred_src = torch.clamp(pred_src, min=-50.0, max=50.0)
            pred_tgt = torch.clamp(pred_tgt, min=-50.0, max=50.0)

            plot_sample(
                x_src.squeeze(0).cpu().numpy(),
                s_src.squeeze(0).cpu().numpy(),
                y_src.squeeze(0).cpu().numpy(),
                pred_src.squeeze(0).float().cpu().numpy(),
                x_tgt.squeeze(0).cpu().numpy(),
                s_tgt.squeeze(0).cpu().numpy(),
                y_tgt.squeeze(0).cpu().numpy(),
                pred_tgt.squeeze(0).float().cpu().numpy(),
                src_stats,
                tgt_stats,
                idx,
                out_dir,
            )


if __name__ == "__main__":
    main()
