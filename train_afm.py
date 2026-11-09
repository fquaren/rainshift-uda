"""AFM training with UDA. Baseline training and UDA application workflow."""

import argparse
import json
import math
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from data.dataset import inverse_transform, load_domain_stats
from data.dataset_zarr import ClimateSRDatasetZarr, JointZarrDataset
from models.afm import AFMModel
from prob_metrics import crps_fair
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
# Methods that draw a target batch during training: the target-forward ones
# plus FDA, which borrows the target amplitude. none, adabn and the oracle
# (none --joint_training, whose target data arrive through the joint loader)
# never read it, so their target loader is not iterated at all.
_NEEDS_TARGET_BATCH = _NEEDS_TARGET_FORWARD | {"fda"}
# --uda_grad_ratio, as in train_unet.py: cap on the gradient-matched weight.
_LAM_CAP = 100.0
_GRAD_MATCHED = ("mmd", "mmd_ms", "dann")


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

    # No target-TEST loader: training never looks at the test split. The
    # per-epoch target-test MAE it used to print was never stored or selected
    # on, so dropping it changes the cost and the log line, not the run.
    return (
        train_loader,
        DataLoader(mk(args.source_path, "validation"), bs, **dl_kw),
        DataLoader(mk(args.target_path, "train"), bs, **dl_kw),
        # Target VALIDATION split: oracle model selection only (see train_unet).
        DataLoader(mk(args.target_path, "validation"), bs, **dl_kw),
    )


def build_crps_loader(args, path, src_stats, batch_size):
    """Validation subset for the ensemble CRPS: --val_crps_chunks chunks
    spread evenly over the validation split. The first N chunks of the
    strided split would cover only the start of the record, i.e. one or two
    seasons, so the subset is thinned rather than truncated."""
    ds = ClimateSRDatasetZarr(
        path, "validation", stats=src_stats,
        val_chunks=args.val_chunks, shuffle_buffer_chunks=args.shuffle_buffer_chunks,
    )
    n = args.val_crps_chunks
    if n and n < len(ds.chunk_ids):
        idx = np.linspace(0, len(ds.chunk_ids) - 1, n).round().astype(int)
        ds.chunk_ids = [ds.chunk_ids[i] for i in idx]
    nw = min(args.num_workers, len(ds.chunk_ids))
    return DataLoader(ds, batch_size, num_workers=nw, pin_memory=True, persistent_workers=nw > 0)


def build_uda(method, mmd_levels, device, base_features=64, mmd_pool="gap", mmd_unbiased=False):
    """Every feature-alignment method acts on the ENCODER only: its
    bottleneck (or, for mmd_ms, its levels) and, for spectral, mu. The flow
    network, which turns mu into the ensemble, is never aligned; it adapts
    only through the mu it is conditioned on. That is a property of the AFM
    arm to report alongside its UDA results, not something fixed here."""
    if method == "coral":
        return {"loss_fn": coral_loss}, []
    if method == "mmd":
        return {"loss_fn": partial(mmd_loss, pool=mmd_pool, unbiased=mmd_unbiased)}, []
    if method == "mmd_ms":
        return {"loss_fn": partial(mmd_multiscale_loss, pool=mmd_pool, unbiased=mmd_unbiased),
                "levels": mmd_levels}, []
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


def build_optimizer(model, base_lr, uda_params, encoder_lr_ratio=0.1):
    encoder_params = list(model.encoder.parameters())
    flow_params = list(model.flow_net.parameters())

    # The DANN discriminator gets the encoder's settings. In the UNet arm it
    # sits in the one AdamW group with the network it plays against (model
    # LR, wd 1e-3); here that network is the encoder. It used to run at
    # 0.1 x LR with wd 1e-4, i.e. 10x slower than the encoder at the
    # production encoder_lr_ratio of 1.0. Empty for every other method.
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": base_lr * encoder_lr_ratio, "weight_decay": 1e-3},
            {"params": flow_params, "lr": base_lr, "weight_decay": 1e-5},
            {"params": uda_params, "lr": base_lr * encoder_lr_ratio, "weight_decay": 1e-3},
        ]
    )


def build_ema(model, decay):
    """EMA of the weights AND the buffers (PLAN.md 3.6 step 2). BatchNorm
    running statistics belong to the weights that produced them, so the EMA
    weights get EMA statistics rather than the raw model's last ~10 batches;
    re-estimating them afterwards was measured to cost 0.003-0.015 val CRPS
    (afm_val_diag.py). sigma_z, itself an EMA of the encoder RMSE, is smoothed
    once more. The integer num_batches_tracked is truncated by the update and
    stops advancing, which is harmless: it is read only when momentum is None.
    The first update copies the model, so the random init never enters."""
    if decay <= 0:
        return None
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True)


def train_one_epoch(
    model, src_ld, tgt_ld, opt, uda_comp, method, lam_eff, beta, device, epoch, total, dann_weight=5e-4,
    jdot_alpha=1.0, jdot_reg=0.1, ema=None, grad_ratio=0.0, lam_state=None,
):
    """grad_ratio > 0 (mmd / mmd_ms / dann), as in train_unet.py: instead of
    lam_eff (dann_weight for DANN), weight the alignment term so that its
    gradient on the encoder's dynamic encoder and bottleneck is grad_ratio
    times the task gradient there. The task is flow + enc_w * enc, so a
    fixed lambda would otherwise be relative to that sum, not to the UNet's
    MSE. Measured every 50 steps, smoothed (EMA 0.9), capped at _LAM_CAP and
    carried across epochs in lam_state. For DANN the measured norm is divided
    by the GRL alpha, so the weight matches the unramped reversal and the
    ramp still applies on top (share = grad_ratio * alpha)."""
    model.train()
    sf, se, su, n = 0.0, 0.0, 0.0, 0
    extra = defaultdict(float)
    n_probe = 0
    use_f = method in _NEEDS_FEATURES
    alpha = dann_grl_schedule(epoch, total) if method == "dann" else 1.0
    need_tgt = method in _NEEDS_TARGET_BATCH
    tgt_it = iter(tgt_ld) if need_tgt else None
    matched = grad_ratio > 0 and method in _GRAD_MATCHED
    if matched:
        # model.encoder's own names: model.named_parameters() would also match
        # flow_net.bottleneck, which no alignment loss reaches.
        probe = [p for name, p in model.encoder.named_parameters() if name.startswith(("dyn_enc", "bottleneck"))]
        lam_state = lam_state if lam_state is not None else {}

    for step, sb in enumerate(src_ld):
        xs, ss, ys = (t.to(device, non_blocking=True) for t in sb)
        if need_tgt:
            try:
                tb = next(tgt_it)
            except StopIteration:
                tgt_it = iter(tgt_ld)
                tb = next(tgt_it)
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
        lam = dann_weight if method == "dann" else lam_eff
        if matched:
            if step % 50 == 0:
                def _norm(gs):
                    return math.sqrt(sum(float(g.pow(2).sum()) for g in gs if g is not None))
                n_task = _norm(torch.autograd.grad(task, probe, retain_graph=True, allow_unused=True))
                n_uda = _norm(torch.autograd.grad(uda, probe, retain_graph=True, allow_unused=True))
                # Share actually achieved by the weight in force at this step.
                extra["share"] += lam_state.get("lam", 0.0) * n_uda / max(n_task, 1e-12)
                n_probe += 1
                if n_uda > 0:
                    new = grad_ratio * n_task / (n_uda / alpha)
                    extra["capped"] += float(new > _LAM_CAP)
                    new = min(new, _LAM_CAP)
                    lam_state["lam"] = new if "lam" not in lam_state else 0.9 * lam_state["lam"] + 0.1 * new
            lam = lam_state.get("lam", 0.0)
        loss = task + lam * uda

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:
            ema.update_parameters(model)
        sf += out["flow_loss"].item()
        se += out["encoder_loss"].item()
        su += uda.item()
        extra["lam"] += lam
        n += 1

    res = {"flow": sf / n, "enc": se / n, "uda": su / n, "sigma_z": model.sigma_z.item()}
    if matched:
        res["lam"] = extra["lam"] / n
        res["share"] = extra["share"] / max(n_probe, 1)
        res["capped"] = extra["capped"]
    return res


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


@torch.no_grad()
def evaluate_crps(model, loader, device, n_ens, steps, seed=0):
    """Fair CRPS of an n_ens-member ensemble in standardised space, the
    space of the training loss. The generator is reseeded on every call, so
    each epoch and each run scores the same noise draws and differences in the
    score come from the model, not the sampler."""
    model.eval()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    tot, n = 0.0, 0
    for b in loader:
        x, s, y = (t.to(device, non_blocking=True) for t in b)
        ens = model.sample(x, s, n_samples=n_ens, steps=steps, generator=g)
        c = crps_fair(ens, y)
        tot += c.sum().item()
        n += c.numel()
    return tot / max(n, 1)


def run_training(args, device, lr=None, lambda_uda=None, batch_size=None, fda_beta=None, weight_decay=None, enc_w=None,
                 encoder_lr_ratio=None):
    _lr = lr or args.lr
    _er = encoder_lr_ratio if encoder_lr_ratio is not None else args.encoder_lr_ratio
    _lam = lambda_uda if lambda_uda is not None else args.lambda_uda
    _bs = batch_size or args.batch_size
    _beta = fda_beta if fda_beta is not None else args.fda_beta
    _wd = weight_decay if weight_decay is not None else args.weight_decay
    _ew = enc_w if enc_w is not None else args.encoder_loss_weight

    ss = load_domain_stats(args.source_path)
    ts = load_domain_stats(args.target_path)
    src_tr, src_val, tgt_tr, tgt_val = build_loaders(args, ss, ts, _bs)
    use_crps = args.select_on == "crps" or args.val_crps_every > 0
    if use_crps:
        src_val_crps = build_crps_loader(args, args.source_path, ss, _bs)
        tgt_val_crps = build_crps_loader(args, args.target_path, ss, _bs) if args.joint_training else None

    model = AFMModel(9, 2, 1, args.base_features, encoder_loss_weight=_ew).to(device)
    # With --ema_decay > 0, every validation score, the selection and best.pt
    # use the EMA copy; best.pt stays a plain AFMModel state dict, so the
    # evaluation and AdaBN paths load it unchanged.
    ema = build_ema(model, args.ema_decay)
    ev = ema.module if ema is not None else model

    uc, ep = build_uda(args.uda_method, args.mmd_levels, device, args.base_features,
                       args.mmd_pool, args.mmd_unbiased)
    opt = build_optimizer(model, _lr, ep, _er)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    tag = f"afm_{_tag(args.source_path, args.target_path)}__{args.uda_method}"
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    best, wait, best_epoch = float("inf"), 0, 0
    history, lam_state = [], {}
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
            jdot_alpha=args.jdot_alpha, jdot_reg=args.jdot_reg, ema=ema,
            grad_ratio=args.uda_grad_ratio, lam_state=lam_state,
        )
        sched.step()
        val = evaluate(ev, src_val, device, ss)

        # Validation CRPS of the sampled ensemble. With --select_on mse the
        # checkpoint is chosen on the encoder mean mu alone, and the flow
        # network never enters selection; --select_on crps scores the
        # generative model itself. Computed every epoch when selecting on it.
        crps_src = crps_tgt = float("nan")
        if use_crps and (args.select_on == "crps" or epoch % args.val_crps_every == 0):
            crps_src = evaluate_crps(ev, src_val_crps, device, args.val_crps_members, args.val_crps_steps)
            if args.joint_training:
                crps_tgt = evaluate_crps(ev, tgt_val_crps, device, args.val_crps_members, args.val_crps_steps)

        # Oracle runs select on the JOINT criterion (target validation split,
        # never test); UDA runs select on source validation only so that no
        # target label influences selection. See train_unet.py for the full
        # argument -- selecting an oracle on source risk alone biases the
        # addressable budget downwards.
        val_tgt = None
        if args.joint_training:
            val_tgt = evaluate(ev, tgt_val, device, ss)
        if args.select_on == "crps":
            src_sel, tgt_sel = crps_src, crps_tgt
        else:
            src_sel, tgt_sel = val["loss"], (val_tgt["loss"] if val_tgt else float("nan"))
        if args.joint_training:
            select = 0.5 * (src_sel + tgt_sel)
            sel_str = f" sel={select:.4f}(src {src_sel:.4f}/tgt {tgt_sel:.4f})"
        else:
            select = src_sel
            sel_str = ""
        crps_str = f" val_crps={crps_src:.4f}" if crps_src == crps_src else ""
        # Gradient-matched runs only, so phase 1 keeps its history.csv columns.
        diag_keys = [k for k in ("share", "capped") if k in tr]
        diag_str = "".join(f" {k}={tr[k]:.4g}" for k in diag_keys)

        print(
            f"[{epoch:3d}/{args.epochs}] {time.time()-t0:5.1f}s  "
            f"flow={tr['flow']:.4f} enc={tr['enc']:.4f} uda={tr['uda']:.4f} "
            f"lam={tr.get('lam', lam_eff):.4g} sigma_z={tr['sigma_z']:.4f} "
            f"val={val['loss']:.4f}{crps_str}" + sel_str + diag_str
        )

        if select < best:
            best = select
            best_epoch = epoch
            wait = 0
            torch.save(ev.state_dict(), out / "best.pt")
        else:
            wait += 1

        # Per-epoch curves for the learning-rate sweep. Validation and
        # training quantities only; no target-test number is computed.
        history.append({
            "epoch": epoch, "time_s": round(time.time() - t0, 1),
            "flow": tr["flow"], "enc": tr["enc"], "uda": tr["uda"], "sigma_z": tr["sigma_z"],
            "val_mse": val["loss"], "val_crps": crps_src,
            "tgt_val_mse": val_tgt["loss"] if val_tgt else float("nan"), "tgt_val_crps": crps_tgt,
            "select": select,
            **({"lam": tr["lam"]} if diag_keys else {}), **{k: tr[k] for k in diag_keys},
        })
        with open(out / "history.csv", "w") as f:
            f.write(",".join(history[0]) + "\n")
            for h in history:
                f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in h.values()) + "\n")

        if args.patience > 0 and wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
        # Truncates the run without changing the cosine schedule, which is
        # still set by --epochs: a sweep run sees the first stop_after epochs
        # of exactly the schedule a full run would see.
        if args.stop_after and epoch >= args.stop_after:
            print(f"Stopping after epoch {epoch} (--stop_after; schedule length {args.epochs})")
            break

    # Peak GPU memory of the whole run, for sizing jobs (A100s have 40 GB).
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
          f"(batch {_bs}, method {args.uda_method})")

    if args.uda_method == "adabn":
        if (out / "best.pt").exists():
            model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
        # No target-test score here either; evaluate.py scores best_adabn.pt.
        _adabn_afm(model, tgt_tr, device, steps=args.adabn_steps, max_batches=args.adabn_flow_batches)
        torch.save(model.state_dict(), out / "best_adabn.pt")

    cfg = {
        **vars(args),
        "lr": _lr,
        "bs": _bs,
        "wd": _wd,
        "lambda": _lam,
        "beta": _beta,
        "enc_w": _ew,
        "encoder_lr_ratio": _er,
        "best_val_loss": best,
        "best_epoch": best_epoch,
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


def _reset_bn(module):
    """Zero the BatchNorm running statistics and switch to a cumulative
    average (momentum=None), so every batch that follows counts equally."""
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None


@torch.no_grad()
def _adabn_afm(model, loader, device, steps=20, max_batches=100, seed=0):
    """AdaBN for AFM, label-free, in two passes.

    Encoder: running statistics re-estimated on every target training batch,
    as before.

    Flow net: re-estimated on the inputs it sees when sampling, i.e. on the
    trajectories of AFMModel.sample (midpoint, `steps` steps, both calls of
    every step) started from z = mu + sigma_z * eps, with the adapted encoder
    in eval mode. The old pass fed x_t = (1-t) z + t mu, which is neither the
    training path (towards y) nor the sampler's (towards a sample), so its
    statistics belonged to neither. The trajectory is integrated with the flow
    net in train mode, i.e. normalised by each call's batch statistics, which
    are what the running averages estimate. One trajectory costs 2 * steps
    flow-net calls, so the pass stops after max_batches batches (0 = all);
    the loader is the shuffled training stream, so these span the record.
    Keep `steps` equal to evaluate.py --sample_steps (20)."""
    # Encoder pass (unchanged).
    _reset_bn(model.encoder)
    model.train()
    for x, s, _ in loader:
        x, s = x.to(device), s.to(device)
        # The encoder upsamples x itself, but flow_net concatenates x with the
        # 200x200 fields, so it needs x on the target grid (80x80 crashed).
        x = model._match_dyn(x, s)
        model.encoder(x, s)

    # Flow-net pass on the sampler's own trajectories.
    _reset_bn(model.flow_net)
    model.encoder.eval()
    model.flow_net.train()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    dt = 1.0 / steps
    for b, (x, s, _) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, s = x.to(device), s.to(device)
        x = model._match_dyn(x, s)
        mu = model.encoder(x, s)
        noise = torch.randn(mu.shape, generator=g, device=mu.device, dtype=torch.float32)
        z = mu.float() + model.sigma_z * noise
        for i in range(steps):
            t = torch.full((mu.shape[0],), i * dt, device=mu.device)
            v1 = model.flow_net(z, t, x, s).float()
            v2 = model.flow_net(z + v1 * dt / 2, t + dt / 2, x, s).float()
            z = z + v2 * dt
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
    p.add_argument("--encoder_lr_ratio", type=float, default=0.1,
                   help="Encoder learning rate as a fraction of --lr (the flow net trains at --lr). "
                        "Phase 2 and the oracle read it back from base_hp/.")
    p.add_argument("--select_on", default="mse", choices=["mse", "crps"],
                   help="Checkpoint selection and early stopping: 'mse' of the encoder mean mu on the "
                        "validation split, or the fair CRPS of the sampled ensemble on a validation subset.")
    p.add_argument("--val_crps_members", type=int, default=4)
    p.add_argument("--val_crps_steps", type=int, default=10, help="Midpoint steps (2 network calls each).")
    p.add_argument("--val_crps_chunks", type=int, default=8,
                   help="Validation chunks (200 samples each) for the CRPS, spread evenly over the split.")
    p.add_argument("--val_crps_every", type=int, default=0,
                   help="Also log the validation CRPS every N epochs when selecting on mse (0 = never).")
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="EMA of weights and buffers, updated every step; validation, selection and best.pt "
                        "use the EMA copy (0 = off, the Stage A behaviour). PLAN.md 3.6 sets 0.999.")
    p.add_argument("--mmd_pool", default="gap", choices=["gap", "grid4"],
                   help="Pooling before MMD (uda.pool_features). grid4 is the GAP ablation.")
    p.add_argument("--mmd_unbiased", action="store_true",
                   help="Unbiased MMD^2 (drops the kernel diagonal, a batch-size term with no domain signal).")
    p.add_argument("--uda_grad_ratio", type=float, default=0.0,
                   help="mmd / mmd_ms / dann: weight the alignment term so its gradient on the encoder's "
                        "dynamic encoder and bottleneck is this fraction of the task gradient, replacing "
                        "--lambda_uda (--dann_weight for dann; the GRL ramp still applies). 0 = off.")
    p.add_argument("--adabn_steps", type=int, default=20,
                   help="adabn: midpoint steps of the sampler trajectories used to re-estimate the flow-net "
                        "BatchNorm statistics; keep equal to evaluate.py --sample_steps.")
    p.add_argument("--adabn_flow_batches", type=int, default=100,
                   help="adabn: target training batches for the flow-net pass (0 = all).")
    p.add_argument("--stop_after", type=int, default=0,
                   help="Stop after this many epochs without shortening the cosine schedule (0 = off).")
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
    args = p.parse_args()
    if args.uda_grad_ratio > 0 and args.uda_method not in _GRAD_MATCHED:
        p.error(f"--uda_grad_ratio applies to {', '.join(_GRAD_MATCHED)} only")
    return args


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
            # Phase-1 files written before the flag existed used 0.1.
            encoder_lr_ratio=b.get("encoder_lr_ratio", 0.1),
        )
    else:
        if args.uda_method != "none":
            print(f"WARNING: Base HPs not found at {hp}. Defaulting to argparse values.")
        run_training(args, device)


if __name__ == "__main__":
    main()
