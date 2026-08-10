"""
Diagnose why the JDOT (joint_ot) auxiliary loss is nearly flat during training.

Runs ONE real batch through the model and reports the quantities that separate
the three possible causes. No training, no checkpoint writing; takes seconds.

CAUSES AND THEIR SIGNATURES
---------------------------
1. Entropic regularisation swamps the cost.
   Sinkhorn returns (almost) the uniform coupling, so <gamma, C> collapses to
   mean(C), which barely moves as the model improves.
   Signature: gamma_peakedness ~ 1, effective_support ~ 1, loss/inert ~ 1.
   Fix: lower --jdot_reg (try 0.01, then 0.001).

2. One cost term dominates.
   The feature and label terms are both mean-normalised, but GAP-pooled
   bottleneck features and log-precipitation fields need not land on the same
   scale. If the feature term dominates, JDOT silently degenerates into
   feature-only alignment and loses the joint property that motivates it.
   Signature: feat_over_label far from 1.
   Fix: retune --jdot_alpha so the two terms are comparable.

3. The loss is fine but its weight is too small.
   Signature: peakedness >> 1, terms balanced, but lambda * loss is tiny
   relative to the task loss.
   Fix: raise the JDOT weight (or give it its own, as DANN has).

USAGE
  python diagnose_jdot.py --code_root ... --data_root ... \
      --source europe_west --target horn-of-africa \
      [--exp_root ...]            # optional: load a trained checkpoint
      [--reg_scan 0.001 0.01 0.1] # optional: sweep reg and show the effect
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code_root", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--exp_root", default=None,
                   help="if given, load {source}__to__{target}__joint_ot/best.pt "
                        "(falls back to the source-only checkpoint)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--base_features", type=int, default=32)
    p.add_argument("--jdot_alpha", type=float, default=1.0)
    p.add_argument("--jdot_reg", type=float, default=0.1)
    p.add_argument("--reg_scan", type=float, nargs="*",
                   default=[0.001, 0.01, 0.1, 1.0])
    p.add_argument("--alpha_scan", type=float, nargs="*",
                   default=[0.0, 0.1, 1.0, 10.0])
    a = p.parse_args()

    sys.path.insert(0, a.code_root)
    from models.unet import DualEncoderUNet
    from data.dataset_zarr import ClimateSRDatasetZarr
    from uda import jdot_loss

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    droot = Path(a.data_root)

    import json
    stats = json.loads((droot / a.source / "normalization_stats.json").read_text())

    def loader(dom):
        ds = ClimateSRDatasetZarr(str(droot / dom), "train", stats=stats,
                                  subset_chunks=2)
        return DataLoader(ds, a.batch_size, num_workers=2, pin_memory=True)

    model = DualEncoderUNet(9, 2, 1, base_features=a.base_features).to(dev)
    if a.exp_root:
        for tag in (f"{a.source}__to__{a.target}__joint_ot",
                    f"{a.source}__to__{a.source}__none"):
            ck = Path(a.exp_root) / tag / "best.pt"
            if ck.exists():
                model.load_state_dict(torch.load(ck, map_location=dev,
                                                 weights_only=True))
                print(f"loaded {ck}")
                break
        else:
            print("no checkpoint found; using a freshly initialised model")
    model.eval()

    xs, ss, ys = next(iter(loader(a.source)))
    xt, st, _ = next(iter(loader(a.target)))
    xs, ss, ys = xs.to(dev), ss.to(dev), ys.to(dev)
    xt, st = xt.to(dev), st.to(dev)

    with torch.no_grad():
        _, fs = model(xs, ss, extract_features=True)
        pt, ft = model(xt, st, extract_features=True)

    print(f"\nbatch: source={a.source} target={a.target} B={xs.shape[0]}")
    print("=" * 74)

    _, d = jdot_loss(fs["bottleneck"], ft["bottleneck"], ys, pt,
                     alpha=a.jdot_alpha, reg=a.jdot_reg, return_diag=True)

    print(f"at the configured alpha={a.jdot_alpha}, reg={a.jdot_reg}:")
    print(f"  loss                          {d['loss']:.4f}")
    print(f"  loss if coupling were uniform {d['inert_loss_uniform_coupling']:.4f}"
          f"   (= mean of the cost matrix)")
    print(f"  loss / inert                  {d['loss_over_inert']:.4f}"
          f"   <- near 1.0 means the OT is doing nothing")
    print(f"  gamma peakedness  max/mean    {d['gamma_peakedness']:.2f}"
          f"   <- near 1 = uniform coupling")
    print(f"  gamma effective support       {d['gamma_effective_support_frac']:.3f}"
          f"   <- 1.0 = fully spread")
    print(f"  cost: feature term (x alpha)  {d['cost_feat_mean'] * a.jdot_alpha:.4f}")
    print(f"  cost: label term              {d['cost_label_mean']:.4f}")
    print(f"  feature/label ratio           {d['feat_over_label']:.3f}"
          f"   <- far from 1 = one term dominates")
    print(f"  cost mean / std               {d['cost_mean']:.4f} / {d['cost_std']:.4f}")
    print(f"  reg / cost std                {d['reg_over_cost_std']:.3f}"
          f"   <- >~1 means reg swamps the cost structure")

    print("\n" + "=" * 74)
    print(f"reg scan (alpha={a.jdot_alpha} fixed)")
    print(f"{'reg':>8} {'loss':>10} {'loss/inert':>11} {'peakedness':>11} {'eff.supp':>9}")
    for r in a.reg_scan:
        _, dd = jdot_loss(fs["bottleneck"], ft["bottleneck"], ys, pt,
                          alpha=a.jdot_alpha, reg=r, return_diag=True)
        print(f"{r:8.4f} {dd['loss']:10.4f} {dd['loss_over_inert']:11.4f} "
              f"{dd['gamma_peakedness']:11.2f} {dd['gamma_effective_support_frac']:9.3f}")

    print("\n" + "=" * 74)
    print(f"alpha scan (reg={a.jdot_reg} fixed)")
    print(f"{'alpha':>8} {'loss':>10} {'feat/label':>11} {'peakedness':>11}")
    for al in a.alpha_scan:
        _, dd = jdot_loss(fs["bottleneck"], ft["bottleneck"], ys, pt,
                          alpha=al, reg=a.jdot_reg, return_diag=True)
        print(f"{al:8.3f} {dd['loss']:10.4f} {dd['feat_over_label']:11.3f} "
              f"{dd['gamma_peakedness']:11.2f}")

    print("\n" + "=" * 74)
    print("VERDICT")
    if d["gamma_peakedness"] < 2.0 or d["loss_over_inert"] > 0.95:
        print("  Cause 1: the coupling is essentially uniform, so <gamma,C> is just")
        print("  mean(C) and the loss cannot respond to the model. Pick the largest")
        print("  reg in the scan above whose peakedness is comfortably > 1, and")
        print("  rerun the sweep. Results obtained at the current reg are")
        print("  uninformative about whether JDOT works.")
    elif not (0.1 < d["feat_over_label"] < 10.0):
        dom = "feature" if d["feat_over_label"] > 1 else "label"
        print(f"  Cause 2: the {dom} term dominates by "
              f"{max(d['feat_over_label'], 1 / d['feat_over_label']):.1f}x, so JDOT is")
        print("  effectively single-term. Use the alpha scan to balance them.")
    else:
        print("  Coupling is peaked and the terms are balanced: the OT itself is")
        print("  healthy. If the loss is still flat in training, the remaining")
        print("  candidate is Cause 3 (the weight on the term is too small")
        print("  relative to the task loss).")


if __name__ == "__main__":
    main()
