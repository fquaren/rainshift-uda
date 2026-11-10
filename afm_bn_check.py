"""Is the AFM phase-2 failure BatchNorm mixing? (overview 2026-11-07, §9.1)

joint_ot, spectral and MMD gr0.5 diverge on SOURCE validation CRPS in the same
three pairs (horn->europe, melanesia->europe, melanesia->horn), while their
training losses track phase 1 and FDA (no target forward) trains normally.
Every target-forward method runs `encoder(x_target)` in train mode after the
source forward, so the encoder's BatchNorm running statistics -- the ones used
at evaluation -- become a source/target mixture. This script tests that,
without training, on one pair per call. Nothing is written outside --out_dir.

All scores: source validation subset the trainer selected on
(build_crps_loader), fair CRPS in standardised space, the run's own
M / steps, generator seed 0. Five arms:

  p1_saved   phase-1 source model, saved statistics. Must reproduce its
             best_val_loss (checks subset and sampler).
  p1_src     phase-1, every BatchNorm reset and re-estimated (momentum=None,
             cumulative) on --n_batches SOURCE training batches, whole model in
             train mode as in training. The cost of re-estimation itself.
  p1_mix     phase-1, starting from its saved statistics, --n_batches training
             steps' worth of the phase-2 BatchNorm traffic and nothing else: a
             train-mode source forward (full model, x_target given) followed by
             a train-mode target encoder forward, at the layers' own momentum
             (0.1). No gradient, weights unchanged. Does mixing ALONE break a
             model that was fine?
  p2_saved   the phase-2 run, saved statistics. Must reproduce its
             best_val_loss.
  p2_src     the phase-2 run, re-estimated on source only as in p1_src. Does
             removing the target's contribution restore it?

sigma_z is restored after every forward (the training forward updates its
EMA). Re-estimation runs in float32; training used autocast, a small
difference that applies to every arm alike.

Decision rule, fixed before the run (per pair, d = phase-1 damage scale):
  d          = p2_saved - p1_saved
  mixing     : p1_mix - p1_saved  >= 0.5 d   -> mixing alone reproduces the damage
  removal    : p2_src - p1_src    <= 0.5 d   -> source-only statistics undo it
  Confirmed if both hold in every non-diverged pair, and `mixing` holds in the
  diverged pairs. For the diverged pairs best.pt is an epoch-1 model, so d and
  `removal` are not meaningful there; only p1_mix vs p1_saved is read.
"""

import argparse
import copy
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import load_domain_stats
from data.dataset_zarr import ClimateSRDatasetZarr
from models.afm import AFMModel
from prob_metrics import crps_fair
from train_afm import build_crps_loader

TARGET_VAR = "precipitation"


def load_afm(run_dir, device):
    cfg = json.loads((run_dir / "config.json").read_text())
    m = AFMModel(9, 2, 1, cfg["base_features"], encoder_loss_weight=cfg["encoder_loss_weight"]).to(device)
    m.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    m.eval()
    return m, cfg


def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


@torch.no_grad()
def score(model, loader, device, members, steps, seed=0, max_batches=None):
    model.eval()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    per, n_nonfinite, n_values, max_abs = [], 0, 0, 0.0
    t0 = time.time()
    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, s, y = (t.to(device, non_blocking=True) for t in batch)
        ens = model.sample(x, s, n_samples=members, steps=steps, generator=g)
        fin = torch.isfinite(ens)
        n_nonfinite += int((~fin).sum())
        n_values += ens.numel()
        if fin.any():
            max_abs = max(max_abs, float(ens[fin].abs().max()))
        per.append(crps_fair(ens, y).cpu().numpy())
    c = np.concatenate(per)
    return {"crps": float(c.mean()), "crps_median": float(np.median(c)), "n_samples": int(c.size),
            "max_abs_member": max_abs, "nonfinite_frac": n_nonfinite / max(n_values, 1),
            "seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def source_forward(model, x, s, y):
    """Train-mode training forward on a source batch (encoder + flow net)."""
    sigma = model.sigma_z.clone()
    model.train()
    model(x, s, x_target=y)
    model.sigma_z.copy_(sigma)
    model.eval()


@torch.no_grad()
def target_forward(model, x, s):
    """Train-mode encoder-only forward on a target batch, as every phase-2
    target-forward method does (`model(xt, st, extract_features=True)` with no
    x_target returns the encoder output; spectral calls model.encoder)."""
    model.train()
    model(x, s, extract_features=True)
    model.eval()


def train_stream(path, stats, cfg):
    return DataLoader(
        ClimateSRDatasetZarr(path, "train", stats=stats, val_chunks=cfg["val_chunks"],
                             shuffle_buffer_chunks=cfg["shuffle_buffer_chunks"]),
        cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True,
    )


def reestimate_source(model, src_path, stats, cfg, device, n_batches):
    for m in bn_layers(model):
        m.reset_running_stats()
        m.momentum = None
    torch.manual_seed(1000)
    n = 0
    for x, s, y in train_stream(src_path, stats, cfg):
        if n >= n_batches:
            break
        source_forward(model, *(t.to(device, non_blocking=True) for t in (x, s, y)))
        n += 1
    return n


def mix(model, src_path, tgt_path, stats, cfg, device, n_batches):
    torch.manual_seed(1000)
    n = 0
    for (x, s, y), (xt, st, _) in zip(train_stream(src_path, stats, cfg), train_stream(tgt_path, stats, cfg)):
        if n >= n_batches:
            break
        source_forward(model, *(t.to(device, non_blocking=True) for t in (x, s, y)))
        target_forward(model, xt.to(device, non_blocking=True), st.to(device, non_blocking=True))
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--p1_dir", required=True, help="phase-1 run, afm_<src>__to__<src>__none")
    p.add_argument("--p2_dir", required=True, help="phase-2 run, afm_<src>__to__<tgt>__<method>")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_batches", type=int, default=200)
    p.add_argument("--max_batches", type=int, default=None, help="smoke test: validation batches per score")
    args = p.parse_args()

    device = torch.device("cuda")
    out = Path(args.out_dir)
    if out.exists():
        raise SystemExit(f"ERROR: {out} exists; not overwriting.")
    out.mkdir(parents=True)

    p1, cfg1 = load_afm(Path(args.p1_dir), device)
    p2, cfg2 = load_afm(Path(args.p2_dir), device)
    src_path, tgt_path = cfg2["source_path"], cfg2["target_path"]
    if cfg1["source_path"] != src_path:
        raise SystemExit(f"ERROR: phase-1 source {cfg1['source_path']} != phase-2 source {src_path}")
    for k in ("val_crps_members", "val_crps_steps", "val_crps_chunks", "val_chunks", "shuffle_buffer_chunks"):
        if cfg1[k] != cfg2[k]:
            raise SystemExit(f"ERROR: {k} differs between runs ({cfg1[k]} vs {cfg2[k]})")
    stats = load_domain_stats(src_path)   # source stats normalise every dataset

    # Phase 1 trained at batch 64, phase 2 at 32. Each model is scored with its
    # own run's loader (batch size fixes which noise draw meets which sample,
    # so this is what reproduces best_val_loss; the sampling sd is ~1e-4).
    def crps_loader(cfg):
        return build_crps_loader(
            Namespace(val_crps_chunks=cfg["val_crps_chunks"], val_chunks=cfg["val_chunks"],
                      shuffle_buffer_chunks=cfg["shuffle_buffer_chunks"], num_workers=cfg["num_workers"]),
            src_path, stats, cfg["batch_size"])
    loaders = {1: crps_loader(cfg1), 2: crps_loader(cfg2)}
    M, S = cfg2["val_crps_members"], cfg2["val_crps_steps"]

    res = {"p1_dir": args.p1_dir, "p2_dir": args.p2_dir, "source": src_path, "target": tgt_path,
           "method": cfg2["uda_method"], "n_batches": args.n_batches,
           "p1_best": [cfg1["best_epoch"], cfg1["best_val_loss"]],
           "p2_best": [cfg2["best_epoch"], cfg2["best_val_loss"]], "arms": {}}

    def add(name, model, phase, **meta):
        r = score(model, loaders[phase], device, M, S, max_batches=args.max_batches)
        res["arms"][name] = {**r, **meta}
        (out / "bn_check.json").write_text(json.dumps(res, indent=2))
        print(f"  {name:9s} crps={r['crps']:.5f} median={r['crps_median']:.5f} "
              f"max|x|={r['max_abs_member']:.2f} nonfinite={r['nonfinite_frac']:.1e} ({r['seconds']:.0f} s)",
              flush=True)

    print(f"=== {Path(args.p2_dir).name} | p1 best {cfg1['best_val_loss']:.5f} (ep {cfg1['best_epoch']}) "
          f"| p2 best {cfg2['best_val_loss']:.5f} (ep {cfg2['best_epoch']})")
    add("p1_saved", p1, 1)
    add("p2_saved", p2, 2)

    # p1_src and p2_src both re-estimate at phase 2's batch size, so they
    # differ only in the weights; p1_mix replays phase 2's traffic, also at
    # its batch size. All three are scored with the phase-1 loader except p2_src.
    c = copy.deepcopy(p1)
    n = reestimate_source(c, src_path, stats, cfg2, device, args.n_batches)
    add("p1_src", c, 1, batches=n)
    del c
    c = copy.deepcopy(p1)
    n = mix(c, src_path, tgt_path, stats, cfg2, device, args.n_batches)
    add("p1_mix", c, 1, batches=n)
    del c
    c = copy.deepcopy(p2)
    n = reestimate_source(c, src_path, stats, cfg2, device, args.n_batches)
    add("p2_src", c, 2, batches=n)
    del c

    a = {k: v["crps"] for k, v in res["arms"].items()}
    d = a["p2_saved"] - a["p1_saved"]
    res["rule"] = {
        "reproduce_p1": a["p1_saved"] - cfg1["best_val_loss"],
        "reproduce_p2": a["p2_saved"] - cfg2["best_val_loss"],
        "d": d,
        "mix_damage": a["p1_mix"] - a["p1_saved"],
        "mixing_holds": bool(a["p1_mix"] - a["p1_saved"] >= 0.5 * d) if d > 0 else None,
        "removal_residual": a["p2_src"] - a["p1_src"],
        "removal_holds": bool(a["p2_src"] - a["p1_src"] <= 0.5 * d) if d > 0 else None,
        "p2_epoch1": cfg2["best_epoch"] == 1,
    }
    (out / "bn_check.json").write_text(json.dumps(res, indent=2))
    print("  rule: " + json.dumps(res["rule"]))
    print(f"  written: {out / 'bn_check.json'}")


if __name__ == "__main__":
    main()
