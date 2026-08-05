"""
UNet training with UDA for climate super-resolution.
GH200-optimised. Baseline training and UDA application workflow.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import inverse_transform, load_domain_stats
from data.dataset_zarr import ClimateSRDatasetZarr, JointZarrDataset
from models.unet import DualEncoderUNet
from uda import (
    DomainDiscriminator,
    apply_adabn,
    coral_loss,
    dann_grl_schedule,
    dann_loss,
    fda_transfer,
    jdot_loss,
    lambda_uda_schedule,
    mmd_loss,
    mmd_multiscale_loss,
    spectral_density_loss,
)

_NEEDS_TARGET_FORWARD = {"coral", "mmd", "mmd_ms", "spectral", "dann", "joint_ot"}
_NEEDS_FEATURES = {"coral", "mmd", "mmd_ms", "dann", "joint_ot"}


def _tag(src, tgt):
    return f"{Path(src).stem}__to__{Path(tgt).stem}"


def _base_hp_path(out_dir, src, tgt):
    return Path(out_dir) / "base_hp" / f"{_tag(src, tgt)}.json"


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------


def build_loaders(args, src_stats, tgt_stats, batch_size=None):
    bs = batch_size or args.batch_size
    nw = args.num_workers
    dl_kw = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)

    # SOURCE stats normalize every dataset — source and target alike. The model
    # lives in source-normalized space; normalizing the target by its own stats
    # would silently erase the covariate shift the framework measures.
    def mk(path, split):
        return ClimateSRDatasetZarr(
            path,
            split,
            stats=src_stats,  # <-- source stats everywhere
            val_chunks=args.val_chunks,
            shuffle_buffer_chunks=args.shuffle_buffer_chunks,
            subset_chunks=args.subset_chunks,
        )

    if args.joint_training:
        joint = JointZarrDataset(mk(args.source_path, "train"), mk(args.target_path, "train"))  # both use src_stats
        train_loader = DataLoader(joint, bs, **dl_kw)
    else:
        train_loader = DataLoader(mk(args.source_path, "train"), bs, **dl_kw)

    return (
        train_loader,
        DataLoader(mk(args.source_path, "validation"), bs, **dl_kw),
        DataLoader(mk(args.target_path, "train"), bs, **dl_kw),
        DataLoader(mk(args.target_path, "test"), bs, **dl_kw),
        # Target VALIDATION split. Used only for oracle (joint-training) model
        # selection, where target labels are legitimately available. Never the
        # target TEST set: that would be test-set leakage.
        DataLoader(mk(args.target_path, "validation"), bs, **dl_kw),
    )


# ---------------------------------------------------------------------------
#  UDA setup
# ---------------------------------------------------------------------------


def build_uda(method, device, mmd_levels):
    if method == "coral":
        return {"loss_fn": coral_loss}, []
    if method == "mmd":
        return {"loss_fn": mmd_loss}, []
    if method == "mmd_ms":
        return {"loss_fn": mmd_multiscale_loss, "levels": mmd_levels}, []
    if method == "spectral":
        return {"loss_fn": spectral_density_loss}, []
    if method == "joint_ot":
        # JDOT is parameter-free: the coupling is solved per batch.
        return {"loss_fn": jdot_loss}, []
    if method == "dann":
        # in_dim = UNet bottleneck channels (512 for base_features=32).
        # hidden_dim=64 (lightweight) per Wang et al. 2026 for stability.
        disc = DomainDiscriminator(in_dim=512, hidden_dim=64).to(device)
        return {"disc": disc}, list(disc.parameters())
    return {}, []


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model,
    src_loader,
    tgt_loader,
    optimiser,
    criterion,
    uda_comp,
    method,
    lambda_eff,
    fda_beta,
    device,
    epoch,
    total_epochs,
    dann_weight=5e-4,
    jdot_alpha=1.0,
    jdot_reg=0.1,
):
    model.train()
    sum_task, sum_uda, n = 0.0, 0.0, 0
    use_feat = method in _NEEDS_FEATURES
    alpha = dann_grl_schedule(epoch, total_epochs) if method == "dann" else 1.0
    tgt_iter = iter(tgt_loader)

    for src_batch in src_loader:
        try:
            tgt_batch = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            tgt_batch = next(tgt_iter)

        x_s, s_s, y_s = (t.to(device, non_blocking=True) for t in src_batch)
        x_t, s_t, _ = (t.to(device, non_blocking=True) for t in tgt_batch)

        if method == "fda":
            x_s = fda_transfer(x_s, x_t, beta=fda_beta)

        if use_feat:
            pred_s, feats_s = model(x_s, s_s, extract_features=True)
        else:
            pred_s = model(x_s, s_s)

        task_loss = criterion(pred_s, y_s)
        uda_loss = torch.tensor(0.0, device=device)

        if method == "coral":
            _, feats_t = model(x_t, s_t, extract_features=True)
            uda_loss = uda_comp["loss_fn"](feats_s["bottleneck"], feats_t["bottleneck"])
        elif method == "mmd":
            _, feats_t = model(x_t, s_t, extract_features=True)
            uda_loss = uda_comp["loss_fn"](feats_s["bottleneck"], feats_t["bottleneck"])
        elif method == "mmd_ms":
            _, feats_t = model(x_t, s_t, extract_features=True)
            uda_loss = uda_comp["loss_fn"](feats_s, feats_t, levels=uda_comp["levels"])
        elif method == "dann":
            _, feats_t = model(x_t, s_t, extract_features=True)
            uda_loss = dann_loss(uda_comp["disc"], feats_s["bottleneck"], feats_t["bottleneck"], alpha)
        elif method == "joint_ot":
            # Joint alignment needs the target PREDICTION as well as target
            # features: the label-transfer cost couples source labels to the
            # model's output at target inputs.
            pred_t, feats_t = model(x_t, s_t, extract_features=True)
            uda_loss = uda_comp["loss_fn"](
                feats_s["bottleneck"], feats_t["bottleneck"], y_s, pred_t,
                alpha=jdot_alpha, reg=jdot_reg,
            )
        elif method == "spectral":
            pred_t = model(x_t, s_t)
            uda_loss = uda_comp["loss_fn"](pred_s, pred_t)

        # Weighting. DANN uses a SEPARATE small weight: the GRL alpha already
        # ramps the adversarial reversal 0->1, so the loss-weight on top must
        # be small (Wang et al. 2026 best alpha ~5e-4, vs the shared 0.1 used
        # by moment-matching methods). A 0.1 weight on DANN is ~200x too large
        # and collapses the discriminator to ln(2). Other methods keep lambda_eff.
        if method == "dann":
            loss = task_loss + dann_weight * uda_loss
        else:
            loss = task_loss + lambda_eff * uda_loss

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        sum_task += task_loss.item()
        sum_uda += uda_loss.item()
        n += 1

    return {"task": sum_task / n, "uda": sum_uda / n}


@torch.no_grad()
def evaluate(model, loader, criterion, device, stats, var="precipitation"):
    model.eval()
    sum_loss, sum_mae, n = 0.0, 0.0, 0
    for batch in loader:
        x, s, y = (t.to(device, non_blocking=True) for t in batch)

        if torch.isnan(y).any() or torch.isnan(x).any():
            print("WARNING: NaN detected in target batch. Skipping.")
            continue

        pred = model(x, s)
        sum_loss += criterion(pred, y).item()
        p = inverse_transform(pred.float().cpu().numpy(), var, stats)
        t = inverse_transform(y.float().cpu().numpy(), var, stats)
        sum_mae += float(np.abs(p - t).mean())
        n += 1
    return {"loss": sum_loss / n, "mae_mm": sum_mae / n}


# ---------------------------------------------------------------------------
#  Core run
# ---------------------------------------------------------------------------


def run_training(args, device, lr=None, lambda_uda=None, batch_size=None, fda_beta=None, weight_decay=None):
    _lr = lr or args.lr
    _lam = lambda_uda if lambda_uda is not None else args.lambda_uda
    _bs = batch_size or args.batch_size
    if args.uda_method == "dann":
        _bs = min(_bs, 128)  # or 64 if 128 still OOMs
    _beta = fda_beta if fda_beta is not None else args.fda_beta
    _wd = weight_decay if weight_decay is not None else args.weight_decay

    src_stats = load_domain_stats(args.source_path)
    tgt_stats = load_domain_stats(args.target_path)
    src_tr, src_val, tgt_tr, tgt_te, tgt_val = build_loaders(args, src_stats, tgt_stats, _bs)

    model = DualEncoderUNet(
        dynamic_channels=9, static_channels=2, out_channels=1, base_features=32
    ).to(device)

    uda_comp, extra_params = build_uda(args.uda_method, device, args.mmd_levels)
    opt = torch.optim.AdamW(list(model.parameters()) + extra_params, lr=_lr, weight_decay=_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.MSELoss()

    tag = f"{_tag(args.source_path, args.target_path)}__{args.uda_method}"
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    best, wait = float("inf"), 0
    for epoch in tqdm(range(1, args.epochs + 1), desc=f"Training {tag}", unit="epoch"):
        t0 = time.time()

        # Standard schedule: fixed by default, optional linear/sigmoid ramp.
        # When uda_method is 'none', 'fda', or 'adabn' there is no auxiliary
        # loss term and lambda_eff is ignored inside the inner loop anyway.
        lambda_eff = lambda_uda_schedule(epoch, args.epochs, _lam, kind=args.lambda_schedule)

        tr = train_one_epoch(
            model,
            src_tr,
            tgt_tr,
            opt,
            criterion,
            uda_comp,
            args.uda_method,
            lambda_eff,
            _beta,
            device,
            epoch,
            args.epochs,
            dann_weight=args.dann_weight,
            jdot_alpha=args.jdot_alpha,
            jdot_reg=args.jdot_reg,
        )
        sched.step()
        val = evaluate(model, src_val, criterion, device, src_stats)
        # Target outputs live in source-normalized space -> source stats.
        tgt = evaluate(model, tgt_te, criterion, device, src_stats)

        # Model-selection criterion.
        #
        # UDA runs: source validation only. Target labels must not influence
        # selection or the protocol is no longer unsupervised.
        #
        # ORACLE (joint) runs: the oracle estimates
        #     lambda = min_h [ R_S(h) + R_T(h) ]
        # so selecting on R_S alone returns a model good on the source and
        # possibly bad on the target. That UNDERSTATES the oracle's target
        # capability, understates the addressable budget and overstates the
        # irreducible risk -- a bias pointing towards the conditional-shift
        # conclusion, i.e. exactly the wrong direction to be sloppy in. The
        # oracle legitimately holds target labels, so we select on the joint
        # criterion using the target VALIDATION split (never the test split).
        if args.joint_training:
            val_tgt = evaluate(model, tgt_val, criterion, device, src_stats)
            select = 0.5 * (val["loss"] + val_tgt["loss"])
            sel_str = f"  sel={select:.4f} (joint: src {val['loss']:.4f} / tgt {val_tgt['loss']:.4f})"
        else:
            select = val["loss"]
            sel_str = ""

        print(
            f"[{epoch:3d}/{args.epochs}] {time.time()-t0:5.1f}s  "
            f"task={tr['task']:.4f}  uda={tr['uda']:.4f}  lam={lambda_eff:.4f}  "
            f"val={val['loss']:.4f}  tgt_mae={tgt['mae_mm']:.3f}mm" + sel_str
        )

        if select < best:
            best = select
            wait = 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            wait += 1

        if args.patience > 0 and wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if args.uda_method == "adabn":
        if (out / "best.pt").exists():
            model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
        apply_adabn(model, tgt_tr, device)
        # Source stats: the model lives in source-normalised space, so target
        # predictions must be inverse-transformed with source stats to give
        # physically correct mm (this branch was still using tgt_stats).
        tgt = evaluate(model, tgt_te, criterion, device, src_stats)
        print(f"  AdaBN -> tgt_mae={tgt['mae_mm']:.3f}mm")
        # NOTE: `best` is deliberately NOT updated from the target metric.
        # Folding a target-test loss into the selection criterion would leak
        # target labels into an unsupervised protocol and make best_val_loss
        # incomparable across methods.
        torch.save(model.state_dict(), out / "best_adabn.pt")

    cfg = {**vars(args), "lr": _lr, "bs": _bs, "wd": _wd, "lambda": _lam, "beta": _beta, "best_val_loss": best}

    if args.uda_method == "none":
        hp_path = _base_hp_path(args.output_dir, args.source_path, args.target_path)
        hp_path.parent.mkdir(parents=True, exist_ok=True)
        hp_path.write_text(json.dumps(cfg, indent=2))
    else:
        # Filename must match the Phase 2 skip-guard in run_exp_unet.sh:
        #   best_hp/{src}__to__{tgt}__{method}.json
        # `tag` already contains the method suffix, so build the combo name
        # from the bare pair tag to avoid a doubled __{method}__{method}.
        combo = (
            Path(args.output_dir) / "best_hp" / f"{_tag(args.source_path, args.target_path)}__{args.uda_method}.json"
        )
        combo.parent.mkdir(parents=True, exist_ok=True)
        combo.write_text(json.dumps(cfg, indent=2))

    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "src_stats.json").write_text(json.dumps(src_stats))
    (out / "tgt_stats.json").write_text(json.dumps(tgt_stats))

    return best


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_path", required=True)
    p.add_argument("--target_path", required=True)
    p.add_argument("--output_dir", default="./experiments")
    p.add_argument("--data_format", type=str, default="npy")
    p.add_argument(
        "--uda_method",
        default="none",
        choices=["none", "coral", "mmd", "mmd_ms", "spectral", "fda", "dann", "adabn", "joint_ot"],
    )
    p.add_argument(
        "--lambda_uda",
        type=float,
        default=0.1,
        help="Asymptotic UDA loss weight (constant when " "--lambda_schedule fixed).",
    )
    p.add_argument(
        "--lambda_schedule",
        default="fixed",
        choices=["fixed", "linear", "sigmoid"],
        help="How lambda_uda evolves across epochs. 'fixed' is the "
        "reproducible default; 'sigmoid' mirrors the DANN GRL "
        "ramp.",
    )
    p.add_argument(
        "--mmd_levels",
        nargs="+",
        default=["enc2", "enc3", "enc4", "bottleneck"],
        help="Feature levels for multi-scale MMD (uda_method=mmd_ms).",
    )
    p.add_argument("--fda_beta", type=float, default=0.01)
    p.add_argument("--jdot_alpha", type=float, default=1.0,
                   help="JDOT weight on the feature term relative to the label-transfer term.")
    p.add_argument("--jdot_reg", type=float, default=0.1,
                   help="JDOT Sinkhorn entropic regularisation.")
    p.add_argument(
        "--dann_weight",
        type=float,
        default=5e-4,
        help="Loss weight for the DANN adversarial term. Separate "
        "from --lambda_uda because the GRL alpha already ramps "
        "the reversal 0->1; keep this small (Wang et al. 2026 "
        "best ~5e-4). A large value collapses the discriminator.",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--subset_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--joint_training", action="store_true", help="Train on concatenated source and target domains")
    p.add_argument(
        "--val_chunks",
        type=int,
        default=88,
        help="Number of time-chunks (200 samples each) held out for "
        "validation, STRIDED across the full record (not a contiguous tail). "
        "At ~877 chunks total 88 is ~10%%; with purge_gap=1 this leaves ~70%% "
        "for training. The old 175 leaves only ~40%% once purging is applied.",
    )
    p.add_argument(
        "--shuffle_buffer_chunks",
        type=int,
        default=8,
        help="Rolling shuffle-buffer size in chunks for the streaming "
        "IterableDataset (train only). Larger = better mixing, "
        "more RAM per worker.",
    )
    p.add_argument(
        "--subset_chunks",
        type=int,
        default=None,
        help="Cap the number of chunks per split (smoke tests). " "Replaces the old --subset_size.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    # Enforce strict IEEE float32 precision for matrix multiplications
    torch.set_float32_matmul_precision("highest")

    hp = _base_hp_path(args.output_dir, args.source_path, args.target_path)
    if hp.exists():
        base = json.loads(hp.read_text())
        print(f"Loaded base HPs: {base}")
        run_training(args, device, lr=base["lr"], batch_size=base["batch_size"], weight_decay=base["weight_decay"])
    else:
        if args.uda_method != "none":
            print(f"WARNING: Base HPs not found at {hp}. Defaulting to argparse values.")
        run_training(args, device)


if __name__ == "__main__":
    main()
