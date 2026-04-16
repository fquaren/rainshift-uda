"""
UNet training with UDA for climate super-resolution.
GH200-optimised. Single-phase Optuna workflow for baseline hyperparameters.
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

from data.dataset import ClimateSRDatasetNPY, inverse_transform, load_domain_stats
from models.unet import DualEncoderUNet
from uda import (
    DomainDiscriminator,
    apply_adabn,
    coral_loss,
    dann_grl_schedule,
    dann_loss,
    fda_transfer,
    mmd_loss,
    spectral_density_loss,
)

_NEEDS_TARGET_FORWARD = {"coral", "mmd", "spectral", "dann"}
_NEEDS_FEATURES = {"coral", "mmd", "dann"}


def _tag(src, tgt):
    return f"{Path(src).stem}__to__{Path(tgt).stem}"


def _base_hp_path(out_dir, src, tgt):
    return Path(out_dir) / "base_hp" / f"{_tag(src, tgt)}.json"


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

def build_loaders(args, src_stats, tgt_stats, batch_size=None):
    bs = batch_size or args.batch_size
    nw = min(args.num_workers, 2)
    dl_kw = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None)

    mk = lambda p, s, st: ClimateSRDatasetNPY(p, s, stats=st, subset_size=args.subset_size)

    return (
        DataLoader(mk(args.source_path, "train", src_stats), bs, shuffle=True, drop_last=True, **dl_kw),
        DataLoader(mk(args.source_path, "validation", src_stats), bs, shuffle=False, **dl_kw),
        DataLoader(mk(args.target_path, "train", tgt_stats), bs, shuffle=True, drop_last=True, **dl_kw),
        DataLoader(mk(args.target_path, "test", tgt_stats), bs, shuffle=False, **dl_kw),
    )


# ---------------------------------------------------------------------------
#  UDA setup
# ---------------------------------------------------------------------------

def build_uda(method, device):
    if method == "coral":
        return {"loss_fn": coral_loss}, []
    if method == "mmd":
        return {"loss_fn": mmd_loss}, []
    if method == "spectral":
        return {"loss_fn": spectral_density_loss}, []
    if method == "dann":
        disc = DomainDiscriminator(in_dim=1024, hidden=256).to(device)
        return {"disc": disc}, list(disc.parameters())
    return {}, []


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, src_loader, tgt_loader, optimiser, criterion, uda_comp, 
    method, lambda_uda, fda_beta, device, epoch, total_epochs
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

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if method == "fda":
                x_s = fda_transfer(x_s, x_t, beta=fda_beta)

            if use_feat:
                pred_s, feats_s = model(x_s, s_s, extract_features=True)
            else:
                pred_s = model(x_s, s_s)

            task_loss = criterion(pred_s, y_s)
            uda_loss = torch.tensor(0.0, device=device) # Fixed tensor broadcast issue

            if method in ("coral", "mmd"):
                _, feats_t = model(x_t, s_t, extract_features=True)
                uda_loss = uda_comp["loss_fn"](feats_s["bottleneck"], feats_t["bottleneck"])
            elif method == "dann":
                _, feats_t = model(x_t, s_t, extract_features=True)
                uda_loss = dann_loss(uda_comp["disc"], feats_s["bottleneck"], feats_t["bottleneck"], alpha)
            elif method == "spectral":
                pred_t = model(x_t, s_t)
                uda_loss = uda_comp["loss_fn"](pred_s, pred_t)

            loss = task_loss + lambda_uda * uda_loss

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

        # Defensive check
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

def run_training(args, device, lr=None, lambda_uda=None, batch_size=None, fda_beta=None, weight_decay=None, trial=None):
    _lr = lr or args.lr
    _lam = lambda_uda if lambda_uda is not None else args.lambda_uda
    _bs = batch_size or args.batch_size
    _beta = fda_beta if fda_beta is not None else args.fda_beta
    _wd = weight_decay if weight_decay is not None else args.weight_decay

    src_stats = load_domain_stats(args.source_path)
    tgt_stats = load_domain_stats(args.target_path)
    src_tr, src_val, tgt_tr, tgt_te = build_loaders(args, src_stats, tgt_stats, _bs)

    model = DualEncoderUNet(dynamic_channels=9, static_channels=2, out_channels=1, base_features=64).to(device)
    if args.compile:
        model = torch.compile(model, mode="max-autotune")

    uda_comp, extra_params = build_uda(args.uda_method, device)
    opt = torch.optim.AdamW(list(model.parameters()) + extra_params, lr=_lr, weight_decay=_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.MSELoss()

    tag = f"{_tag(args.source_path, args.target_path)}__{args.uda_method}"
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    best, wait = float("inf"), 0
    for epoch in tqdm(range(1, args.epochs + 1), desc=f"Training {tag}", unit="epoch"):
        t0 = time.time()
        tr = train_one_epoch(
            model, src_tr, tgt_tr, opt, criterion, uda_comp, args.uda_method, _lam, _beta, device, epoch, args.epochs
        )
        sched.step()
        val = evaluate(model, src_val, criterion, device, src_stats)
        tgt = evaluate(model, tgt_te, criterion, device, tgt_stats)

        print(
            f"[{epoch:3d}/{args.epochs}] {time.time()-t0:5.1f}s  "
            f"task={tr['task']:.4f}  uda={tr['uda']:.4f}  "
            f"val={val['loss']:.4f}  tgt_mae={tgt['mae_mm']:.3f}mm"
        )

        if val["loss"] < best:
            best = val["loss"]
            wait = 0
            if trial is None:
                torch.save(model.state_dict(), out / "best.pt")
        else:
            wait += 1

        if args.patience > 0 and wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if trial is not None:
            trial.report(val["loss"], epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    if args.uda_method == "adabn" and trial is None:
        if (out / "best.pt").exists():
            model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
        apply_adabn(model, tgt_tr, device)
        tgt = evaluate(model, tgt_te, criterion, device, tgt_stats)
        print(f"  AdaBN -> tgt_mae={tgt['mae_mm']:.3f}mm")
        best = min(best, tgt["loss"])
        torch.save(model.state_dict(), out / "best_adabn.pt")

    if trial is None:
        cfg = {**vars(args), "lr": _lr, "bs": _bs, "wd": _wd, "lambda": _lam, "beta": _beta, "best_val_loss": best}
        
        if args.uda_method != "none":
            combo = Path(args.output_dir) / "best_hp" / f"{tag}__{args.uda_method}.json"
            combo.parent.mkdir(parents=True, exist_ok=True)
            combo.write_text(json.dumps(cfg, indent=2))
            
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        (out / "src_stats.json").write_text(json.dumps(src_stats))
        (out / "tgt_stats.json").write_text(json.dumps(tgt_stats))

    return best


# ---------------------------------------------------------------------------
#  Optuna Phase 1
# ---------------------------------------------------------------------------

def _phase1_objective(trial, args, device):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    bs = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    saved = args.uda_method
    args.uda_method = "none"
    try:
        return run_training(args, device, lr=lr, weight_decay=wd, batch_size=bs, lambda_uda=0.0, trial=trial)
    finally:
        args.uda_method = saved

def run_phase1(args, device):
    import optuna
    from optuna.storages import RDBStorage

    tag = _tag(args.source_path, args.target_path)
    db = Path(args.output_dir) / "optuna" / f"phase1_{tag}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    storage = RDBStorage(url=f"sqlite:///{db}", engine_kwargs={"connect_args": {"timeout": 60}})

    study = optuna.create_study(
        study_name=f"phase1_{tag}",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    study.optimize(lambda t: _phase1_objective(t, args, device), n_trials=args.n_trials, timeout=args.optuna_timeout)

    best = study.best_params
    print(f"\nPhase 1 best: {best} (value={study.best_value:.6f})")

    hp = _base_hp_path(args.output_dir, args.source_path, args.target_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text(json.dumps(best, indent=2))

    args.uda_method = "none"
    run_training(
        args, device, lr=best["lr"], weight_decay=best["weight_decay"], batch_size=best["batch_size"], lambda_uda=0.0
    )
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

    p.add_argument("--uda_method", default="none", choices=["none", "coral", "mmd", "spectral", "fda", "dann", "adabn"])
    p.add_argument("--lambda_uda", type=float, default=0.1)
    p.add_argument("--fda_beta", type=float, default=0.01)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--subset_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_false")

    p.add_argument("--optuna", action="store_false")
    p.add_argument("--optuna_phase", type=int, default=1, choices=[1])
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--optuna_timeout", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if args.optuna and args.optuna_phase == 1:
        run_phase1(args, device)
    else:
        # Automatically load base hyperparameters if present
        hp = _base_hp_path(args.output_dir, args.source_path, args.target_path)
        if hp.exists():
            base = json.loads(hp.read_text())
            print(f"Loaded Phase 1 base HPs: {base}")
            run_training(
                args, 
                device, 
                lr=base["lr"], 
                batch_size=base["batch_size"], 
                weight_decay=base["weight_decay"]
            )
        else:
            if args.uda_method != "none":
                print(f"WARNING: Base HPs not found at {hp}. Defaulting to argparse values.")
            run_training(args, device)


if __name__ == "__main__":
    main()