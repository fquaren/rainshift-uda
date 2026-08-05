"""AFM training with UDA. Baseline training and UDA application workflow."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import inverse_transform, load_domain_stats
from data.dataset_zarr import ClimateSRDatasetZarr, JointZarrDataset
from models.afm import AFMModel
from uda import (
    DomainDiscriminator,
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


def _tag(s, t):
    return f"{Path(s).stem}__to__{Path(t).stem}"


def _base_hp_path(o, s, t):
    return Path(o) / "base_hp" / f"{_tag(s, t)}.json"


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
        # Target VALIDATION split: oracle model selection only (see train_unet).
        DataLoader(mk(args.target_path, "validation"), bs, **dl_kw),
    )


def build_uda(method, mmd_levels, device, base_features=64):
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
        # in_dim MUST equal the encoder bottleneck width. DualEncoderUNet's
        # bottleneck is 16 * base_features, so derive it rather than hardcoding
        # (the old literal 1024 silently broke for any base_features != 64).
        d = DomainDiscriminator(16 * base_features, 64).to(device)
        return {"disc": d}, list(d.parameters())
    return {}, []


def build_optimizer(model, base_lr, uda_params):
    encoder_params = list(model.encoder.parameters())
    flow_params = list(model.flow_net.parameters())

    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": base_lr * 0.1, "weight_decay": 1e-3},
            {"params": flow_params, "lr": base_lr, "weight_decay": 1e-5},
            {"params": uda_params, "lr": base_lr * 0.1, "weight_decay": 1e-4},
        ]
    )


def train_one_epoch(
    model, src_ld, tgt_ld, opt, uda_comp, method, lam_eff, beta, device, epoch, total, dann_weight=5e-4,
    jdot_alpha=1.0, jdot_reg=0.1,
):
    model.train()
    sf, se, su, n = 0.0, 0.0, 0.0, 0
    use_f = method in _NEEDS_FEATURES
    alpha = dann_grl_schedule(epoch, total) if method == "dann" else 1.0
    tgt_it = iter(tgt_ld)

    for sb in src_ld:
        try:
            tb = next(tgt_it)
        except StopIteration:
            tgt_it = iter(tgt_ld)
            tb = next(tgt_it)

        xs, ss, ys = (t.to(device, non_blocking=True) for t in sb)
        xt, st, _ = (t.to(device, non_blocking=True) for t in tb)

        # Removed automatic mixed precision to compute everything in float32
        if method == "fda":
            xs = fda_transfer(xs, xt, beta=beta)

        out = model(xs, ss, x_target=ys, extract_features=(use_f or method == "spectral"))
        task = out["total_loss"]
        uda = torch.tensor(0.0, device=device)

        if method == "coral":
            _, tf = model(xt, st, extract_features=True)
            uda = uda_comp["loss_fn"](out["features"]["bottleneck"], tf["bottleneck"])
        elif method == "mmd":
            _, tf = model(xt, st, extract_features=True)
            uda = uda_comp["loss_fn"](out["features"]["bottleneck"], tf["bottleneck"])
        elif method == "mmd_ms":
            _, tf = model(xt, st, extract_features=True)
            uda = uda_comp["loss_fn"](out["features"], tf, levels=uda_comp["levels"])
        elif method == "dann":
            _, tf = model(xt, st, extract_features=True)
            uda = dann_loss(uda_comp["disc"], out["features"]["bottleneck"], tf["bottleneck"], alpha)
        elif method == "joint_ot":
            # Joint alignment needs the target PREDICTION as well as target
            # features: the label-transfer cost couples source labels to the
            # model's output at target inputs.
            tp_pred, tf = model(xt, st, extract_features=True)
            uda = uda_comp["loss_fn"](
                out["features"]["bottleneck"], tf["bottleneck"], ys, tp_pred,
                alpha=jdot_alpha, reg=jdot_reg,
            )
        elif method == "spectral":
            # Reuse the source encoder output already computed in `out` rather
            # than re-running the encoder on xs: a third forward doubles the
            # cost and updates this batch's BatchNorm running stats twice.
            sp = out["mu"] if "mu" in out else model.encoder(xs, ss)
            tp = model.encoder(xt, st)
            uda = uda_comp["loss_fn"](sp, tp)

        # Standard fixed-or-scheduled weighting. lam_eff is computed once per
        # epoch in run_training and passed in as a scalar.
        # DANN uses a separate small weight (GRL alpha already ramps the
        # reversal); 0.1 is ~200x too large and collapses the discriminator.
        if method == "dann":
            loss = task + dann_weight * uda
        else:
            loss = task + lam_eff * uda

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sf += out["flow_loss"].item()
        se += out["encoder_loss"].item()
        su += uda.item()
        n += 1

    return {"flow": sf / n, "enc": se / n, "uda": su / n, "sigma_z": model.sigma_z.item()}


@torch.no_grad()
def evaluate(model, loader, device, stats, var="precipitation", n_ens=0, steps=20):
    model.eval()
    sl, sm, n = 0.0, 0.0, 0
    crit = nn.MSELoss()
    for b in loader:
        x, s, y = (t.to(device, non_blocking=True) for t in b)

        mu = model.deterministic_predict(x, s)
        sl += crit(mu, y).item()

        p = inverse_transform(mu.float().cpu().numpy(), var, stats)
        t = inverse_transform(y.float().cpu().numpy(), var, stats)
        sm += float(np.abs(p - t).mean())
        n += 1
    return {"loss": sl / n, "mae_mm": sm / n}


def run_training(args, device, lr=None, lambda_uda=None, batch_size=None, fda_beta=None, weight_decay=None, enc_w=None):
    _lr = lr or args.lr
    _lam = lambda_uda if lambda_uda is not None else args.lambda_uda
    _bs = batch_size or args.batch_size
    _beta = fda_beta if fda_beta is not None else args.fda_beta
    _wd = weight_decay if weight_decay is not None else args.weight_decay
    _ew = enc_w if enc_w is not None else args.encoder_loss_weight

    ss = load_domain_stats(args.source_path)
    ts = load_domain_stats(args.target_path)
    src_tr, src_val, tgt_tr, tgt_te, tgt_val = build_loaders(args, ss, ts, _bs)

    model = AFMModel(9, 2, 1, args.base_features, encoder_loss_weight=_ew).to(device)

    uc, ep = build_uda(args.uda_method, args.mmd_levels, device, args.base_features)
    opt = build_optimizer(model, _lr, ep)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    tag = f"afm_{_tag(args.source_path, args.target_path)}____{args.uda_method}"
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    best, wait = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # Per-epoch lambda; ignored when the method has no auxiliary loss
        # (none / fda / adabn) because uda stays at zero in those branches.
        lam_eff = lambda_uda_schedule(epoch, args.epochs, _lam, kind=args.lambda_schedule)

        tr = train_one_epoch(
            model,
            src_tr,
            tgt_tr,
            opt,
            uc,
            args.uda_method,
            lam_eff,
            _beta,
            device,
            epoch,
            args.epochs,
            dann_weight=args.dann_weight,
            jdot_alpha=args.jdot_alpha, jdot_reg=args.jdot_reg,
        )
        sched.step()
        val = evaluate(model, src_val, device, ss)
        tgt = evaluate(model, tgt_te, device, ss)

        # Oracle runs select on the JOINT criterion (target validation split,
        # never test); UDA runs select on source validation only so that no
        # target label influences selection. See train_unet.py for the full
        # argument -- selecting an oracle on source risk alone biases the
        # addressable budget downwards.
        if args.joint_training:
            val_tgt = evaluate(model, tgt_val, device, ss)
            select = 0.5 * (val["loss"] + val_tgt["loss"])
            sel_str = f" sel={select:.4f}(src {val['loss']:.4f}/tgt {val_tgt['loss']:.4f})"
        else:
            select = val["loss"]
            sel_str = ""

        print(
            f"[{epoch:3d}/{args.epochs}] {time.time()-t0:5.1f}s  "
            f"flow={tr['flow']:.4f} enc={tr['enc']:.4f} uda={tr['uda']:.4f} "
            f"lam={lam_eff:.4f} sigma_z={tr['sigma_z']:.4f} "
            f"val={val['loss']:.4f} tgt_mae={tgt['mae_mm']:.3f}mm" + sel_str
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
        _adabn_afm(model, tgt_tr, device)
        tgt = evaluate(model, tgt_te, device, ss)
        print(f"  AdaBN -> tgt_mae={tgt['mae_mm']:.3f}mm")
        torch.save(model.state_dict(), out / "best_adabn.pt")

    cfg = {
        **vars(args),
        "lr": _lr,
        "bs": _bs,
        "wd": _wd,
        "lambda": _lam,
        "beta": _beta,
        "enc_w": _ew,
        "best_val_loss": best,
    }

    if args.uda_method == "none":
        hp_path = _base_hp_path(args.output_dir, args.source_path, args.target_path)
        hp_path.parent.mkdir(parents=True, exist_ok=True)
        hp_path.write_text(json.dumps(cfg, indent=2))
    else:
        # Filename must match the Phase 2 skip-guard in run_exp_afm.sh:
        #   best_hp/afm_{src}__to__{tgt}__{method}.json
        # (two underscores before the method, not four).
        combo = (
            Path(args.output_dir)
            / "best_hp"
            / f"afm_{_tag(args.source_path, args.target_path)}__{args.uda_method}.json"
        )
        combo.parent.mkdir(parents=True, exist_ok=True)
        combo.write_text(json.dumps(cfg, indent=2))

    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    return best


@torch.no_grad()
def _adabn_afm(model, loader, device):
    """AdaBN for AFM. Uses encoder prediction (not target labels) for flow net pass."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None
    model.train()
    for x, s, _ in loader:
        x, s = x.to(device), s.to(device)
        mu = model.encoder(x, s)
        t = torch.rand(x.shape[0], device=device)
        z = mu + model.sigma_z * torch.randn_like(mu)
        te = t[:, None, None, None]
        xt = (1 - te) * z + te * mu
        model.flow_net(xt, t, x, s)
    model.eval()


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
        help="Asymptotic UDA loss weight (constant when --lambda_schedule fixed).",
    )
    p.add_argument(
        "--lambda_schedule",
        default="fixed",
        choices=["fixed", "linear", "sigmoid"],
        help="How lambda_uda evolves across epochs. 'fixed' matches "
        "the weighted-sum default of the UDA literature; "
        "'sigmoid' mirrors the DANN GRL ramp.",
    )
    p.add_argument(
        "--mmd_levels",
        nargs="+",
        default=["enc2", "enc3", "enc4", "bottleneck"],
        help="Feature levels for multi-scale MMD (uda_method=mmd_ms). "
        "Must be keys in the encoder's extract_features dict.",
    )
    p.add_argument("--fda_beta", type=float, default=0.01)
    p.add_argument("--jdot_alpha", type=float, default=1.0,
                   help="JDOT weight on the feature term relative to the label-transfer term.")
    p.add_argument("--jdot_reg", type=float, default=0.1,
                   help="JDOT Sinkhorn entropic regularisation.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--subset_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_features", type=int, default=64)
    p.add_argument(
        "--dann_weight",
        type=float,
        default=5e-4,
        help="DANN adversarial loss weight; small because GRL alpha already ramps the reversal (Wang et al. 2026 best ~5e-4).",
    )
    p.add_argument("--encoder_loss_weight", type=float, default=0.1)
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
        b = json.loads(hp.read_text())
        print(f"Loaded base HPs: {b}")
        run_training(
            args,
            device,
            lr=b["lr"],
            batch_size=b["batch_size"],
            weight_decay=b["weight_decay"],
            enc_w=b.get("encoder_loss_weight", 0.1),
        )
    else:
        if args.uda_method != "none":
            print(f"WARNING: Base HPs not found at {hp}. Defaulting to argparse values.")
        run_training(args, device)


if __name__ == "__main__":
    main()
