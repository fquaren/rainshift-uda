"""Where does the AFM validation-CRPS jitter come from? (PLAN.md 3.6, step 1)

In the Stage A LR sweep the validation CRPS jumped by 0.01-0.05 from epoch to
epoch while the flow loss and the mu MSE fell smoothly. The selection score is
reseeded on every call, so the jitter is in the model, not the noise draws.
This script scores ONE Stage A checkpoint (best.pt) on the same 8-chunk
validation subset the trainer selected on, and splits the score's variability
into its candidate sources. No training; nothing is written outside --out_dir.

  reproduce  seed 0, M=4, 10 steps, saved BN statistics. Must match the run's
             best_val_loss in config.json, or the subset or sampler differ
             from training and nothing below compares with history.csv.
  seeds      generator seeds 0..3 at the selection setting: sampling noise of
             one selection score.
  bn         flow-net BatchNorm running statistics re-estimated with
             momentum=None (cumulative mean) on --bn_sets disjoint sets of
             --bn_set_batches training batches, plus one set of
             --bn_long_batches. The encoder stays in eval mode and its BN is
             untouched; sigma_z is restored after each set because the
             training forward updates its EMA. Seed 0, selection setting.
  steps      5/10/20/40 midpoint steps at M=8, seed 0, saved statistics.
  unet       acceptance (3.5): MAE of the UNet phase-1 checkpoint, a point
             forecast's CRPS, and the MAE of the AFM's own mu.

Every row also reports the per-sample CRPS median, the largest |member| in
standardised space, the fraction of non-finite members, and the CRPS after
mapping members and truth to log(mm + 0.01) (P14).

The epoch jitter J of the run is the sd of its epoch-to-epoch CRPS
differences over epochs 6-12 divided by sqrt(2), from history.csv. The
decision rules of PLAN.md 3.6 step 2 compare sd(bn) and sd(seeds) against J;
the ratios are printed at the end and stored in diag.json.
"""

import argparse
import copy
import csv
import json
import math
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import load_domain_stats
from data.dataset_zarr import ClimateSRDatasetZarr, _LOG_EPS
from evaluate import load_model
from models.afm import AFMModel
from prob_metrics import crps_fair
from train_afm import build_crps_loader

TARGET_VAR = "precipitation"
FLOOR_MM = 0.01


# ---------------------------------------------------------------------------
#  Scoring
# ---------------------------------------------------------------------------


def to_floor(z, mu, sd):
    """Standardised log(mm + 1e-5) -> log(mm + FLOOR_MM), in float64 so that
    exploding members overflow to inf rather than wrapping silently."""
    mm = (torch.exp(z.double() * (sd + 1e-8) + mu) - _LOG_EPS).clamp_min(0.0)
    return torch.log(mm + FLOOR_MM)


def _summ(per_std, per_floor, max_abs, n_nonfinite, n_values, seconds, **meta):
    c = np.concatenate(per_std)
    f = np.concatenate(per_floor)
    return {
        **meta,
        "n_samples": int(c.size),
        "crps": float(c.mean()),
        "crps_median": float(np.median(c)),
        "crps_floor": float(f.mean()),
        "crps_floor_median": float(np.median(f)),
        "crps_floor_finite": float(f[np.isfinite(f)].mean()) if np.isfinite(f).any() else float("nan"),
        "max_abs_member": float(max_abs),
        "nonfinite_frac": n_nonfinite / max(n_values, 1),
        "seconds": round(seconds, 1),
    }


@torch.no_grad()
def score_ensemble(model, loader, device, mu, sd, members, steps, seed, max_batches=None, **meta):
    """Fair CRPS of the sampled ensemble, as train_afm.evaluate_crps: the
    generator is seeded once per call and consumed in loader order."""
    t0 = time.time()
    model.eval()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    per_std, per_floor = [], []
    max_abs, n_nonfinite, n_values = 0.0, 0, 0
    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, s, y = (t.to(device, non_blocking=True) for t in batch)
        ens = model.sample(x, s, n_samples=members, steps=steps, generator=g)
        finite = torch.isfinite(ens)
        n_nonfinite += int((~finite).sum())
        n_values += ens.numel()
        if finite.any():
            max_abs = max(max_abs, float(ens[finite].abs().max()))
        per_std.append(crps_fair(ens, y).cpu().numpy())
        per_floor.append(crps_fair(to_floor(ens, mu, sd), to_floor(y, mu, sd)).cpu().numpy())
    return _summ(per_std, per_floor, max_abs, n_nonfinite, n_values, time.time() - t0,
                 members=members, steps=steps, seed=seed, **meta)


@torch.no_grad()
def score_point(predict, loader, device, mu, sd, max_batches=None, **meta):
    """MAE per sample, the CRPS of a one-member forecast."""
    t0 = time.time()
    per_std, per_floor = [], []
    max_abs, n_nonfinite, n_values = 0.0, 0, 0
    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, s, y = (t.to(device, non_blocking=True) for t in batch)
        p = predict(x, s).float()
        n_nonfinite += int((~torch.isfinite(p)).sum())
        n_values += p.numel()
        max_abs = max(max_abs, float(p[torch.isfinite(p)].abs().max()))
        per_std.append((p - y).abs().flatten(1).mean(1).cpu().numpy())
        per_floor.append((to_floor(p, mu, sd) - to_floor(y, mu, sd)).abs().flatten(1).mean(1).cpu().numpy())
    return _summ(per_std, per_floor, max_abs, n_nonfinite, n_values, time.time() - t0, **meta)


# ---------------------------------------------------------------------------
#  Flow-net BatchNorm re-estimation
# ---------------------------------------------------------------------------


def _flow_bn(model):
    return [m for m in model.flow_net.modules() if isinstance(m, nn.BatchNorm2d)]


def reset_flow_bn(model):
    """Zero the flow net's running statistics and switch them to a cumulative
    mean (momentum=None), as _adabn_afm does for the whole model."""
    for m in _flow_bn(model):
        m.reset_running_stats()
        m.momentum = None


@torch.no_grad()
def accumulate_flow_bn(model, x, s, y):
    """One training forward at random t, with only the flow net in train
    mode: the encoder's BN is left as saved, and sigma_z is restored because
    the training forward updates its EMA."""
    sigma = model.sigma_z.clone()
    model.eval()
    model.flow_net.train()
    model(x, s, x_target=y)
    model.eval()
    model.sigma_z.copy_(sigma)


# ---------------------------------------------------------------------------
#  Epoch jitter from the sweep's history
# ---------------------------------------------------------------------------


def epoch_jitter(history_csv, first=6, last=12):
    """sd of epoch-to-epoch val_crps differences over epochs first..last,
    divided by sqrt(2): the per-epoch noise if successive epochs were
    independent draws around a smooth curve."""
    with open(history_csv) as f:
        rows = {int(r["epoch"]): float(r["val_crps"]) for r in csv.DictReader(f)}
    d = np.diff([rows[e] for e in range(first, last + 1) if e in rows])
    return float(np.std(d) / math.sqrt(2)) if d.size > 1 else float("nan")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="Stage A run, e.g. .../lr1e-4_enc1.0/afm_europe_west__to__europe_west__none")
    p.add_argument("--data_root", required=True)
    p.add_argument("--domain", default="europe_west")
    p.add_argument("--unet_ckpt", default=None, help="UNet phase-1 best.pt on the same domain (acceptance)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--bn_sets", type=int, default=5)
    p.add_argument("--bn_set_batches", type=int, default=10)
    p.add_argument("--bn_long_batches", type=int, default=200)
    p.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20, 40])
    p.add_argument("--steps_members", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=None, help="Smoke test: score this many validation batches only")
    args = p.parse_args()

    device = torch.device("cuda")
    run_dir = Path(args.run_dir)
    out = Path(args.out_dir)
    if out.exists():
        raise SystemExit(f"ERROR: {out} exists; not overwriting.")
    out.mkdir(parents=True)

    cfg = json.loads((run_dir / "config.json").read_text())
    sel_members, sel_steps = cfg["val_crps_members"], cfg["val_crps_steps"]
    bs = cfg["batch_size"]
    domain_path = str(Path(args.data_root) / args.domain)
    stats = load_domain_stats(domain_path)
    mu, sd = stats[TARGET_VAR]

    # The trainer's own subset builder, with the run's settings, including
    # num_workers: chunk-to-worker assignment fixes the sample order, and
    # with it which noise draw meets which sample.
    loader_args = Namespace(
        val_crps_chunks=cfg["val_crps_chunks"], val_chunks=cfg["val_chunks"],
        shuffle_buffer_chunks=cfg["shuffle_buffer_chunks"], num_workers=cfg["num_workers"],
    )
    val_loader = build_crps_loader(loader_args, domain_path, stats, bs)

    model = AFMModel(9, 2, 1, cfg["base_features"], encoder_loss_weight=cfg["encoder_loss_weight"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    model.eval()

    J = epoch_jitter(run_dir / "history.csv")
    result = {
        "run_dir": str(run_dir), "best_epoch": cfg["best_epoch"], "best_val_loss": cfg["best_val_loss"],
        "epoch_jitter_J": J, "floor_mm": FLOOR_MM, "rows": [],
    }
    print(f"=== {run_dir}\n    best epoch {cfg['best_epoch']}, best_val_loss {cfg['best_val_loss']:.6f}, J {J:.4f}")

    def add(row):
        result["rows"].append(row)
        (out / "diag.json").write_text(json.dumps(result, indent=2))
        print(f"  {row['section']:9s} {row.get('label', ''):14s} crps={row['crps']:.5f} "
              f"median={row['crps_median']:.5f} floor={row['crps_floor']:.5f} "
              f"max|x|={row['max_abs_member']:.2f} nonfinite={row['nonfinite_frac']:.2e} "
              f"({row['seconds']:.0f} s)", flush=True)

    kw = dict(max_batches=args.max_batches)

    # 1. Reproduce the trainer's score, then the seed spread around it.
    for sd_i in args.seeds:
        add(score_ensemble(model, val_loader, device, mu, sd, sel_members, sel_steps, sd_i,
                           section="seeds", label=f"seed{sd_i}", bn="saved", **kw))
    rep = result["rows"][0]
    result["reproduce_diff"] = rep["crps"] - cfg["best_val_loss"]
    print(f"  reproduce: seed {args.seeds[0]} {rep['crps']:.6f} vs best_val_loss "
          f"{cfg['best_val_loss']:.6f} (diff {result['reproduce_diff']:+.2e})")

    # 2. Steps, saved statistics.
    for st in args.steps:
        add(score_ensemble(model, val_loader, device, mu, sd, args.steps_members, st, args.seeds[0],
                           section="steps", label=f"steps{st}", bn="saved", **kw))

    # 3. Acceptance: UNet MAE and the AFM's own mu.
    add(score_point(model.deterministic_predict, val_loader, device, mu, sd,
                    section="unet", label="afm_mu", **kw))
    if args.unet_ckpt:
        unet = load_model("unet", args.unet_ckpt, device, base_features=32)
        add(score_point(lambda x, s: unet(x, s), val_loader, device, mu, sd,
                        section="unet", label="unet_mae", **kw))
        del unet
        torch.cuda.empty_cache()

    # 4. Flow-net BN re-estimation. One pass over the training split's stream,
    # one model copy per set: sets 0..bn_sets-1 take consecutive disjoint
    # blocks of bn_set_batches, the long set the bn_long_batches after them.
    n_short = args.bn_sets * args.bn_set_batches
    copies = [copy.deepcopy(model) for _ in range(args.bn_sets + 1)]
    for c in copies:
        reset_flow_bn(c)
    counts = [0] * len(copies)
    train_loader = DataLoader(
        ClimateSRDatasetZarr(domain_path, "train", stats=stats, val_chunks=cfg["val_chunks"],
                             shuffle_buffer_chunks=cfg["shuffle_buffer_chunks"]),
        bs, num_workers=cfg["num_workers"], pin_memory=True,
    )
    torch.manual_seed(1000)   # random t and noise of the training forward
    for b, (x, s, y) in enumerate(train_loader):
        if b >= n_short + args.bn_long_batches:
            break
        i = b // args.bn_set_batches if b < n_short else args.bn_sets
        x, s, y = (t.to(device, non_blocking=True) for t in (x, s, y))
        accumulate_flow_bn(copies[i], x, s, y)
        counts[i] += 1
    del train_loader
    for i, (c, n) in enumerate(zip(copies, counts)):
        label = f"set{i}x{n}" if i < args.bn_sets else f"long{n}"
        add(score_ensemble(c, val_loader, device, mu, sd, sel_members, sel_steps, args.seeds[0],
                           section="bn", label=label, bn=label, **kw))
    del copies

    # 5. Ratios for PLAN.md 3.6 step 2.
    rows = result["rows"]
    seed_c = [r["crps"] for r in rows if r["section"] == "seeds"]
    bn_c = [r["crps"] for r in rows if r["section"] == "bn" and r["label"].startswith("set")]
    step_c = {r["steps"]: r["crps"] for r in rows if r["section"] == "steps"}
    ref = step_c.get(max(step_c)) if step_c else float("nan")
    result["summary"] = {
        "J": J,
        "sd_seeds": float(np.std(seed_c, ddof=1)) if len(seed_c) > 1 else float("nan"),
        "sd_bn_sets": float(np.std(bn_c, ddof=1)) if len(bn_c) > 1 else float("nan"),
        "steps_rel_to_max": {k: v / ref - 1 for k, v in sorted(step_c.items())},
    }
    s = result["summary"]
    s["sd_seeds_over_J"] = s["sd_seeds"] / J
    s["sd_bn_sets_over_J"] = s["sd_bn_sets"] / J
    (out / "diag.json").write_text(json.dumps(result, indent=2))
    print(f"\n  J={J:.4f}  sd(seeds)={s['sd_seeds']:.4f} ({s['sd_seeds_over_J']:.2f} J)  "
          f"sd(bn sets)={s['sd_bn_sets']:.4f} ({s['sd_bn_sets_over_J']:.2f} J)")
    print("  steps vs max: " + ", ".join(f"{k}: {v:+.2%}" for k, v in s["steps_rel_to_max"].items()))
    print(f"  written: {out / 'diag.json'}")


if __name__ == "__main__":
    main()
