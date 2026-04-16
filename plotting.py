"""
Generate physical spatial plots for 5 samples of a trained model.
Retrieves inputs, static fields, and targets directly from the test dataset.
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

def load_model(model_type, checkpoint, device, base_features=64):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    clean = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
    if model_type == "unet":
        model = DualEncoderUNet(9, 2, 1, base_features)
    else:
        model = AFMModel(9, 2, 1, base_features, encoder_loss_weight=0.1)
    model.load_state_dict(clean)
    return model.to(device).eval()

def plot_sample(x, s, y_true, y_pred, stats, sample_idx, out_dir):
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    
    # 1. Plot Dynamic Inputs
    for i, var_name in enumerate(DEFAULT_INPUT_VARS):
        field = inverse_transform(x[i], var_name, stats)
        ax = axes[i]
        im = ax.imshow(field, cmap="viridis", origin="lower")
        ax.set_title(f"Input: {var_name}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
        
    # 2. Plot Static Inputs
    offset = len(DEFAULT_INPUT_VARS)
    for i, var_name in enumerate(DEFAULT_STATIC_VARS):
        field = inverse_transform(s[i], var_name, stats)
        ax = axes[offset + i]
        cmap = "terrain" if var_name == "z" else "ocean"
        im = ax.imshow(field, cmap=cmap, origin="lower")
        ax.set_title(f"Static: {var_name}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
        
    # 3. Plot Target and Prediction
    pred_field = inverse_transform(y_pred[0], "precipitation", stats)
    true_field = inverse_transform(y_true[0], "precipitation", stats)
    
    vmin = 0.1
    vmax = max(pred_field.max(), true_field.max(), 1.0)
    
    ax_true = axes[11]
    im_true = ax_true.imshow(true_field, cmap="Blues", norm=LogNorm(vmin=vmin, vmax=vmax), origin="lower")
    ax_true.set_title("Target: precipitation")
    fig.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)
    ax_true.axis("off")
    
    ax_pred = axes[12]
    im_pred = ax_pred.imshow(pred_field, cmap="Blues", norm=LogNorm(vmin=vmin, vmax=vmax), origin="lower")
    ax_pred.set_title("Prediction: precipitation")
    fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
    ax_pred.axis("off")
    
    # Disable unused axes
    for j in range(13, 16):
        axes[j].axis("off")
        
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
    cfg_path = res_dir / "config.json"
    if not cfg_path.exists():
        # Fallback to experiment dir if evaluating from results/
        exp_name = res_dir.name
        exp_dir = Path("./experiments/unet") / exp_name
        if not exp_dir.exists():
            exp_dir = Path("./experiments/afm") / exp_name
        cfg_path = exp_dir / "config.json"
        
    cfg = json.loads(cfg_path.read_text())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgt_domain = Path(cfg["target_path"]).name
    tgt_path = Path(args.data_root) / tgt_domain
    
    stats = load_domain_stats(str(tgt_path))
    loader = DataLoader(
        ClimateSRDatasetNPY(str(tgt_path), "test", stats=stats),
        batch_size=1, shuffle=False
    )
    
    model_type = _infer_model_type(res_dir.name)
    ckpt_path = exp_dir / "best.pt" if 'exp_dir' in locals() else res_dir / "best.pt"
    
    model = load_model(model_type, ckpt_path, device, cfg.get("base_features", 64))
    
    out_dir = res_dir / "pdf_plots"
    out_dir.mkdir(exist_ok=True)
    
    print(f"Plotting {args.n_samples} samples for {res_dir.name}...")
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.n_samples:
                break
                
            x, s, y = (t.to(device) for t in batch)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if model_type == "unet":
                    pred = model(x, s)
                else:
                    pred = model.deterministic_predict(x, s)
                    
            x_np = x.squeeze(0).cpu().numpy()
            s_np = s.squeeze(0).cpu().numpy()
            y_np = y.squeeze(0).cpu().numpy()
            p_np = pred.squeeze(0).float().cpu().numpy()
            
            plot_sample(x_np, s_np, y_np, p_np, stats, idx, out_dir)

if __name__ == "__main__":
    main()