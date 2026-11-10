"""
Feature-alignment probe (PLAN.md, "Read 2026-11-10"; MILESTONES M1).

Question: did the UNet feature-alignment methods ever align the bottleneck
features they act on? With source BatchNorm the adapted models score like
source-only, which has two readings:
  (a) the features never aligned: a stronger weight would mean something;
  (b) they aligned, but matching globally pooled bottleneck features does not
      constrain the prediction: no weight will help (the GAP argument).

For one seed and the six ordered pairs, it takes held-out bottleneck features
(512 x 12 x 12, DualEncoderUNet `extract_features=True`) of the source and
target TEST splits, both normalised with the source's stats as in training,
from:
  source-only                   unet_seed<S>/<pair>__none
  dann, mmd, joint_ot, spectral unet_seed<S>/ (as trained) and unet_seed<S>_srcbn/
  dann gr0.5                    unet_seed<S>_danngr05/ and unet_seed<S>_danngr05_srcbn/
under three BatchNorm modes:
  eval_srcbn    eval mode, running stats re-estimated on source (srcbn tree;
                for source-only, its own stats)
  eval_mixed    eval mode, the running stats as trained (source/target mix)
  train_batch   train mode, each domain's batch normalised by its own batch
                statistics, running stats frozen (momentum 0). This is what
                the alignment loss saw during training: train_one_epoch
                forwards the source and the target batch separately.
and reports, per (model, mode):
  mmd_gap, null_gap     unbiased multi-kernel MMD^2 of GAP'd features (what
                        every loss here aligned), median-heuristic bandwidths
                        x 2^{-2..2}, and a source-vs-source null (first vs
                        second half of the source sample, a lower reference)
  mmd_grid4, null_grid4 the same on 4 x 4 grid mean+std pooling (keeps coarse
                        geometry the GAP discards)
  clf_acc, clf_auc      a held-out linear domain classifier (L2 logistic
                        regression on standardised GAP features; fitted on the
                        first half of each domain's sample in time, scored on
                        the second half). 0.5 = indistinguishable.

Samples: every `--stride`-th test step (default 17 -> 1024 per domain after trimming to whole batches),
read once per (domain, stats) and cached on CPU; batches of `--batch_size`
(128, the training batch size) in time order, which span ~3 months, against
the trainer's 8-chunk (1600 h) shuffle buffer.

Writes <out_dir>/probe.csv; --out_dir is refused if it exists.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset_zarr import ClimateSRDatasetZarr
from evaluate import load_model, load_stats
from uda import _median_pairwise_distance, _mmd_from_vectors, pool_features

DOMAINS = ["europe_west", "horn-of-africa", "melanesia"]
METHODS = ["dann", "mmd", "joint_ot", "spectral"]


def _pairs():
    return [(s, t) for s in DOMAINS for t in DOMAINS if s != t]


def read_inputs(domain_dir, stats, stride, batch_size):
    """Every stride-th test sample of a domain, normalised with `stats`."""
    ds = ClimateSRDatasetZarr(str(domain_dir), "test", stats=stats)
    xs, ss, pos = [], [], 0
    for x, s, _ in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1):
        keep = [i for i in range(x.shape[0]) if (pos + i) % stride == 0]
        pos += x.shape[0]
        if keep:
            xs.append(x[keep])
            ss.append(s[keep])
    # Whole batches only, so train-mode BN never normalises a small remainder.
    n = (sum(len(a) for a in xs) // batch_size) * batch_size
    return torch.cat(xs)[:n], torch.cat(ss)[:n]


def set_bn_mode(model, mode):
    model.eval()
    if mode == "train_batch":
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.momentum = 0.0   # batch statistics used, running stats untouched


@torch.no_grad()
def bottleneck(model, x, s, device, batch_size):
    out = []
    for i in range(0, x.shape[0], batch_size):
        _, f = model(x[i:i + batch_size].to(device), s[i:i + batch_size].to(device), extract_features=True)
        out.append(f["bottleneck"].float().cpu())
    return torch.cat(out)


def mmd_and_null(fs, ft, pool, device):
    s = pool_features(fs.to(device), pool)
    t = pool_features(ft.to(device), pool)
    bw = _median_pairwise_distance(s, t)
    bws = [bw * (2.0 ** k) for k in (-2, -1, 0, 1, 2)]
    h = s.size(0) // 2
    mmd = _mmd_from_vectors(s, t, bws, unbiased=True).item()
    null = _mmd_from_vectors(s[:h], s[h:2 * h], bws, unbiased=True).item()
    return mmd, null


def _auc(score, y):
    order = torch.argsort(score)
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=score.dtype, device=score.device)
    n1 = y.sum()
    n0 = len(y) - n1
    return ((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


def domain_classifier(fs, ft, device, l2=1e-2, steps=200):
    """Held-out linear domain classifier on GAP features (time-split halves)."""
    s = pool_features(fs.to(device), "gap").double()
    t = pool_features(ft.to(device), "gap").double()
    hs, ht = s.size(0) // 2, t.size(0) // 2
    xtr = torch.cat([s[:hs], t[:ht]])
    ytr = torch.cat([torch.zeros(hs), torch.ones(ht)]).double().to(device)
    xte = torch.cat([s[hs:], t[ht:]])
    yte = torch.cat([torch.zeros(s.size(0) - hs), torch.ones(t.size(0) - ht)]).double().to(device)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    w = torch.zeros(xtr.size(1), dtype=torch.float64, device=device, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(xtr @ w + b, ytr) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        z = xte @ w + b
        acc = ((z > 0).double() == yte).double().mean().item()
        auc = _auc(z, yte)
    return acc, auc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, help="Zarr RainShift root (read-only)")
    p.add_argument("--exp_root", required=True, help="results_rainshift_uda/")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True, help="refused if it exists")
    p.add_argument("--stride", type=int, default=17)
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; not overwriting.")
    device = torch.device("cuda")
    data_root, root = Path(args.data_root), Path(args.exp_root)
    S = args.seed
    trees = {
        "main": (root / f"unet_seed{S}", root / f"unet_seed{S}_srcbn"),
        "gr05": (root / f"unet_seed{S}_danngr05", root / f"unet_seed{S}_danngr05_srcbn"),
    }

    # Resolve every checkpoint first, so a missing run fails before any GPU work.
    jobs = []
    for src, tgt in _pairs():
        pair = f"{src}__to__{tgt}"
        so = trees["main"][0] / f"{pair}__none"
        jobs.append((src, tgt, "source-only", [("eval_srcbn", so), ("train_batch", so)]))
        for m in METHODS:
            mixed, srcbn = trees["main"][0] / f"{pair}__{m}", trees["main"][1] / f"{pair}__{m}"
            jobs.append((src, tgt, m, [("eval_srcbn", srcbn), ("eval_mixed", mixed), ("train_batch", mixed)]))
        mixed, srcbn = trees["gr05"][0] / f"{pair}__dann", trees["gr05"][1] / f"{pair}__dann"
        jobs.append((src, tgt, "dann_gr0.5", [("eval_srcbn", srcbn), ("eval_mixed", mixed), ("train_batch", mixed)]))
    missing = sorted({str(d) for *_, modes in jobs for _, d in modes
                      if not (d / "best.pt").exists() or not (d / "config.json").exists()})
    if missing:
        raise SystemExit("missing runs:\n  " + "\n  ".join(missing))

    stats = {d: load_stats(str(data_root / d)) for d in DOMAINS}
    cache = {}

    def inputs(domain, stats_domain):
        key = (domain, stats_domain)
        if key not in cache:
            t0 = time.time()
            cache[key] = read_inputs(data_root / domain, stats[stats_domain], args.stride, args.batch_size)
            print(f"  read {domain} (stats {stats_domain}): {cache[key][0].shape[0]} samples, "
                  f"{time.time() - t0:.0f} s", flush=True)
        return cache[key]

    rows = []
    for src, tgt, label, modes in jobs:
        xs, ss = inputs(src, src)
        xt, st = inputs(tgt, src)
        for mode, run in modes:
            cfg = json.loads((run / "config.json").read_text())
            model = load_model("unet", str(run / "best.pt"), device, cfg.get("base_features") or 32)
            set_bn_mode(model, mode)
            fs = bottleneck(model, xs, ss, device, args.batch_size)
            ft = bottleneck(model, xt, st, device, args.batch_size)
            mg, ng = mmd_and_null(fs, ft, "gap", device)
            m4, n4 = mmd_and_null(fs, ft, "grid4", device)
            acc, auc = domain_classifier(fs, ft, device)
            rows.append(dict(seed=S, source=src, target=tgt, model=label, bn_mode=mode,
                             n_src=fs.shape[0], n_tgt=ft.shape[0],
                             mmd_gap=mg, null_gap=ng, mmd_grid4=m4, null_grid4=n4,
                             clf_acc=acc, clf_auc=auc, checkpoint=str(run / "best.pt")))
            print(f"{src}->{tgt} {label:11s} {mode:11s} mmd_gap {mg:.4f} (null {ng:.4f}) "
                  f"grid4 {m4:.4f} (null {n4:.4f}) clf acc {acc:.3f} auc {auc:.3f}", flush=True)
            del model, fs, ft
            torch.cuda.empty_cache()

    out_dir.mkdir(parents=True)
    with open(out_dir / "probe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {out_dir / 'probe.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
