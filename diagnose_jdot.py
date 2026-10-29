"""
Diagnose the JDOT (joint_ot) auxiliary loss on real batches: is the Sinkhorn
coupling converged, how much of the transport cost the feature term carries,
and why the loss may be nearly flat during training.

Runs N_BATCHES training-like batches through the model. No training, no
checkpoint writing; with --output_dir it writes per_batch.csv and
summary.json there and refuses to run if the directory exists.

BATCHES (audit A8)
------------------
The old version used one batch from the first two chunks (one season) with
the model in eval mode, which is where "feature term = 0.3 % of the transport
cost" came from. Now:
  - each batch is drawn from --chunks_per_batch chunks picked at random over
    the whole train split, as the training shuffle buffer does, and the
    batches use disjoint chunks;
  - the target batch holds the same timestamps as the source batch, as in
    training (audit A7), unless --independent_target;
  - features are computed as in training: train-mode BatchNorm (batch
    statistics, source and target in separate forwards) under no_grad, on a
    deepcopy of the model, because a train-mode forward updates the running
    statistics. Eval-mode numbers are reported beside them for comparison.
Every quantity is reported per batch and as mean and standard deviation.

CAUSES AND THEIR SIGNATURES
---------------------------
0. Sinkhorn not converged.
   Training runs a fixed 50 iterations. Each iteration ends on the column
   update, so the column marginals hold and the row marginals only once
   converged. A row-marginal L1 error left over means some source samples
   carry more than 1/n of the mass.
   Signature: marg_err_row at 50 iterations well above 0, falling at 200 and
   1000, with the loss and the feature share moving with it.

1. Entropic regularisation swamps the cost.
   Sinkhorn returns (almost) the uniform coupling, so <gamma, C> collapses to
   mean(C), which barely moves as the model improves.
   Signature: gamma_peakedness ~ 1, effective_support ~ 1, loss/inert ~ 1.

2. One cost term dominates.
   The feature and label terms are both mean-normalised, but GAP-pooled
   bottleneck features and log-precipitation fields need not land on the same
   scale. If one term dominates, JDOT degenerates into single-term alignment.
   Signature: feat_share_transport (the feature term's share of <gamma, C>)
   near 0 or 1; feat_over_label (ratio of the mean cost terms) far from 1.

3. The loss is fine but its weight is too small.
   Signature: peakedness >> 1, terms balanced, but lambda * loss is tiny
   relative to the task loss.

USAGE
  python diagnose_jdot.py --code_root ... --data_root ... \\
      --source europe_west --target horn-of-africa \\
      --exp_root .../unet_seed42 [--output_dir ...]
  alpha and reg default to the loaded run's config.json.
"""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# What every JDOT run used (uda.jdot_loss default); train_unet.py does not
# override it.
TRAIN_ITERS = 50


def _ms(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code_root", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--exp_root", required=True,
                   help="run tree, e.g. unet_seed42/; loads {source}__to__{target}__joint_ot/best.pt "
                        "(falls back to the source-only checkpoint)")
    p.add_argument("--output_dir", default=None,
                   help="write per_batch.csv and summary.json here; must not exist")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_batches", type=int, default=8)
    p.add_argument("--chunks_per_batch", type=int, default=8,
                   help="chunks mixed into each batch (the training shuffle buffer holds 8)")
    p.add_argument("--independent_target", action="store_true",
                   help="draw target chunks independently instead of time-locked to the source")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base_features", type=int, default=32)
    p.add_argument("--jdot_alpha", type=float, default=None, help="default: the run's config.json")
    p.add_argument("--jdot_reg", type=float, default=None, help="default: the run's config.json")
    p.add_argument("--sinkhorn_iters", type=int, nargs="+", default=[TRAIN_ITERS, 200, 1000])
    p.add_argument("--reg_scan", type=float, nargs="*", default=[0.001, 0.01, 0.1, 1.0])
    p.add_argument("--alpha_scan", type=float, nargs="*", default=[0.0, 0.1, 1.0, 10.0])
    a = p.parse_args()

    out_dir = Path(a.output_dir) if a.output_dir else None
    if out_dir is not None and out_dir.exists():
        sys.exit(f"ERROR: {out_dir} exists. Not overwriting.")

    sys.path.insert(0, a.code_root)
    from data.dataset_zarr import ClimateSRDatasetZarr
    from models.unet import DualEncoderUNet
    from uda import jdot_loss

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    droot = Path(a.data_root)
    print(f"device: {dev}")

    # ---- model and its config ---------------------------------------------
    model = DualEncoderUNet(9, 2, 1, base_features=a.base_features).to(dev)
    cfg, run_dir = {}, None
    for tag in (f"{a.source}__to__{a.target}__joint_ot",
                f"{a.source}__to__{a.target}__none",
                f"{a.source}__to__{a.source}__none"):
        ck = Path(a.exp_root) / tag / "best.pt"
        if ck.exists():
            run_dir = ck.parent
            model.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
            if (run_dir / "config.json").exists():
                cfg = json.loads((run_dir / "config.json").read_text())
            print(f"loaded {ck}")
            break
    else:
        print("no checkpoint found; using a freshly initialised model")
    if cfg.get("mask_inputs"):
        sys.exit("ERROR: masked-input run; this diagnostic builds the unmasked model.")
    alpha = a.jdot_alpha if a.jdot_alpha is not None else cfg.get("jdot_alpha", 1.0)
    reg = a.jdot_reg if a.jdot_reg is not None else cfg.get("jdot_reg", 0.001)
    print(f"alpha={alpha}  reg={reg}  (training Sinkhorn iterations: {TRAIN_ITERS})")

    # Source stats normalise both domains, as in training.
    stats_path = run_dir / "src_stats.json" if run_dir and (run_dir / "src_stats.json").exists() \
        else droot / a.source / "normalization_stats.json"
    stats = json.loads(Path(stats_path).read_text())
    val_chunks = cfg.get("val_chunks", 88)
    ds_s = ClimateSRDatasetZarr(str(droot / a.source), "train", stats=stats, val_chunks=val_chunks)
    ds_t = ClimateSRDatasetZarr(str(droot / a.target), "train", stats=stats, val_chunks=val_chunks)

    # ---- batches: random chunks over the whole train split ----------------
    rng = np.random.default_rng(a.seed)
    need = a.n_batches * a.chunks_per_batch
    if need > len(ds_s.chunk_ids):
        sys.exit(f"ERROR: {need} chunks requested, train split has {len(ds_s.chunk_ids)}")
    groups_s = rng.permutation(ds_s.chunk_ids)[:need].reshape(a.n_batches, a.chunks_per_batch)
    locked = not a.independent_target and set(ds_s.chunk_ids) <= set(ds_t.chunk_ids)
    if not a.independent_target and not locked:
        print("WARNING: target train chunks differ from the source's; drawing them independently")
    groups_t = groups_s if locked else \
        rng.permutation(ds_t.chunk_ids)[:need].reshape(a.n_batches, a.chunks_per_batch)
    print(f"{a.n_batches} batches of {a.batch_size} from {a.chunks_per_batch} chunks each "
          f"({'time-locked' if locked else 'independent'} target), "
          f"train split {len(ds_s.chunk_ids)} chunks")

    def read(ds, group):
        xs, ys = zip(*(ds._read_chunk(int(c)) for c in sorted(group)))
        return np.concatenate(xs), np.concatenate(ys)

    # Train-mode copy: BatchNorm uses batch statistics as in training. The
    # forward updates its running statistics, hence the copy.
    model_tr = copy.deepcopy(model).train()
    model.eval()

    rows = []
    for b in range(a.n_batches):
        xs_np, ys_np = read(ds_s, groups_s[b])
        xt_np, _ = read(ds_t, groups_t[b])
        idx_s = rng.choice(len(xs_np), a.batch_size, replace=False)
        idx_t = idx_s if locked else rng.choice(len(xt_np), a.batch_size, replace=False)
        xs = torch.from_numpy(xs_np[idx_s]).float().to(dev)
        ys = torch.from_numpy(ys_np[idx_s]).float().to(dev)
        xt = torch.from_numpy(xt_np[idx_t]).float().to(dev)
        ss = ds_s.static.unsqueeze(0).expand(a.batch_size, -1, -1, -1).to(dev)
        st = ds_t.static.unsqueeze(0).expand(a.batch_size, -1, -1, -1).to(dev)
        if not (torch.isfinite(xs).all() and torch.isfinite(ys).all() and torch.isfinite(xt).all()):
            print(f"WARNING: batch {b} has non-finite inputs or labels")

        with torch.no_grad():
            _, fs = model_tr(xs, ss, extract_features=True)
            pt, ft = model_tr(xt, st, extract_features=True)
            _, fs_ev = model(xs, ss, extract_features=True)
            pt_ev, ft_ev = model(xt, st, extract_features=True)

        def diag(f_s, f_t, pred, al=alpha, r=reg, it=TRAIN_ITERS):
            return jdot_loss(f_s["bottleneck"], f_t["bottleneck"], ys, pred,
                             alpha=al, reg=r, n_iter=it, return_diag=True)[1]

        d = diag(fs, ft, pt)
        d_ev = diag(fs_ev, ft_ev, pt_ev)
        row = {"batch": b, "chunks": " ".join(str(int(c)) for c in sorted(groups_s[b])),
               "loss": d["loss"], "loss_over_inert": d["loss_over_inert"],
               "peakedness": d["gamma_peakedness"], "eff_support": d["gamma_effective_support_frac"],
               "feat_share": d["feat_share_transport"], "feat_over_label": d["feat_over_label"],
               "reg_over_cost_std": d["reg_over_cost_std"],
               "feat_share_evalbn": d_ev["feat_share_transport"],
               "feat_over_label_evalbn": d_ev["feat_over_label"]}
        for it in a.sinkhorn_iters:
            dd = d if it == TRAIN_ITERS else diag(fs, ft, pt, it=it)
            row[f"marg_err_row@{it}"] = dd["marg_err_row"]
            row[f"marg_err_col@{it}"] = dd["marg_err_col"]
            row[f"loss@{it}"] = dd["loss"]
            row[f"feat_share@{it}"] = dd["feat_share_transport"]
        for r in a.reg_scan:
            dd = diag(fs, ft, pt, r=r)
            row[f"reg{r:g}_loss_over_inert"] = dd["loss_over_inert"]
            row[f"reg{r:g}_peakedness"] = dd["gamma_peakedness"]
            row[f"reg{r:g}_marg_err_row"] = dd["marg_err_row"]
        for al in a.alpha_scan:
            dd = diag(fs, ft, pt, al=al)
            row[f"alpha{al:g}_feat_share"] = dd["feat_share_transport"]
            row[f"alpha{al:g}_peakedness"] = dd["gamma_peakedness"]
        rows.append(row)
        print(f"  batch {b}: loss {d['loss']:.4f}  feat share {d['feat_share_transport']:.4f} "
              f"(eval BN {d_ev['feat_share_transport']:.4f})  row err@{TRAIN_ITERS} {d['marg_err_row']:.3e}  "
              f"peakedness {d['gamma_peakedness']:.1f}")

    # ---- summary ------------------------------------------------------------
    keys = [k for k in rows[0] if k not in ("batch", "chunks")]
    summ = {k: dict(zip(("mean", "sd"), _ms([r[k] for r in rows]))) for k in keys}
    for k in keys:
        summ[k]["min"] = float(min(r[k] for r in rows))
        summ[k]["max"] = float(max(r[k] for r in rows))

    def line(label, k, fmt=".4f", note=""):
        s = summ[k]
        print(f"  {label:<34} {s['mean']:{fmt}} +- {s['sd']:{fmt}}   "
              f"[{s['min']:{fmt}}, {s['max']:{fmt}}]{note}")

    print("\n" + "=" * 78)
    print(f"source={a.source} target={a.target}  alpha={alpha} reg={reg}  "
          f"{a.n_batches} batches, mean +- sd [min, max]")
    print(f"at the configured alpha and reg, {TRAIN_ITERS} iterations, train-mode BN:")
    line("loss", "loss")
    line("loss / inert", "loss_over_inert", note="   <- near 1: the OT does nothing")
    line("gamma peakedness max/mean", "peakedness", ".2f", "   <- near 1: uniform coupling")
    line("gamma effective support", "eff_support", ".3f")
    line("feature share of <gamma, C>", "feat_share", note="   <- near 0 or 1: one term dominates")
    line("feature / label mean cost", "feat_over_label", ".3f")
    line("reg / cost std", "reg_over_cost_std", ".3g")
    line("feature share, eval-mode BN", "feat_share_evalbn", note="   <- what the old diagnostic reported")

    print("\nSinkhorn convergence (L1 marginal errors; total mass is 1)")
    print(f"  {'iters':>6} {'row err':>20} {'col err':>10} {'loss':>18} {'feat share':>18}")
    for it in a.sinkhorn_iters:
        r_, c_, l_, f_ = (summ[f"{q}@{it}"] for q in ("marg_err_row", "marg_err_col", "loss", "feat_share"))
        print(f"  {it:6d} {r_['mean']:9.3e} +- {r_['sd']:7.1e} {c_['mean']:10.1e} "
              f"{l_['mean']:8.4f} +- {l_['sd']:6.4f} {f_['mean']:8.4f} +- {f_['sd']:6.4f}")

    print(f"\nreg scan (alpha={alpha}, {TRAIN_ITERS} iterations)")
    print(f"  {'reg':>8} {'loss/inert':>11} {'peakedness':>11} {'row err':>10}")
    for r in a.reg_scan:
        print(f"  {r:8.4f} {summ[f'reg{r:g}_loss_over_inert']['mean']:11.4f} "
              f"{summ[f'reg{r:g}_peakedness']['mean']:11.2f} {summ[f'reg{r:g}_marg_err_row']['mean']:10.2e}")

    print(f"\nalpha scan (reg={reg}, {TRAIN_ITERS} iterations)")
    print(f"  {'alpha':>8} {'feat share':>11} {'peakedness':>11}")
    for al in a.alpha_scan:
        print(f"  {al:8.3f} {summ[f'alpha{al:g}_feat_share']['mean']:11.4f} "
              f"{summ[f'alpha{al:g}_peakedness']['mean']:11.2f}")

    print("\n" + "=" * 78)
    print("VERDICT (on the batch means)")
    err = summ[f"marg_err_row@{TRAIN_ITERS}"]["mean"] if TRAIN_ITERS in a.sinkhorn_iters else None
    if err is not None and err > 0.01:
        last = max(a.sinkhorn_iters)
        print(f"  Cause 0: at {TRAIN_ITERS} iterations {100 * err / 2:.1f} % of the transport mass sits on")
        print(f"  the wrong source rows (row L1 error {err:.3f}; {summ[f'marg_err_row@{last}']['mean']:.3g} "
              f"at {last}). The training coupling is not the OT plan at this reg.")
    if summ["peakedness"]["mean"] < 2.0 or summ["loss_over_inert"]["mean"] > 0.95:
        print("  Cause 1: the coupling is essentially uniform, so <gamma,C> is just")
        print("  mean(C) and the loss cannot respond to the model.")
    fs_m = summ["feat_share"]["mean"]
    if not (0.05 < fs_m < 0.95):
        dom = "label" if fs_m <= 0.05 else "feature"
        print(f"  Cause 2: the {dom} term carries {100 * max(fs_m, 1 - fs_m):.1f} % of the transport")
        print("  cost, so JDOT is effectively single-term.")
    print("  If none of the above fires, the OT is healthy and a flat loss points to")
    print("  Cause 3 (the weight on the term).")

    if out_dir is not None:
        out_dir.mkdir(parents=True)
        with open(out_dir / "per_batch.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        meta = {"source": a.source, "target": a.target, "checkpoint": str(run_dir), "alpha": alpha,
                "reg": reg, "train_iters": TRAIN_ITERS, "time_locked": locked, **vars(a)}
        (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summ}, indent=2))
        print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
