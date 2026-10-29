"""
Read-out of the GAP ablation (scripts/run_gap_ablation.sh), europe_west ->
melanesia, seeds 42/43/44. Decision rules fixed before any result exists:

1. Manipulation check, per aligned arm (gap, grid4, grid4ms) and seed. The
   aligned discrepancy (unbiased bottleneck MMD; mmd_gap for gap, mmd_grid4
   for the grid arms) at the arm's selected epoch is <= 0.5 x the same
   quantity in the tgtfwd control (same seed, same epoch, or the control's
   last epoch if it stopped earlier). And the mean achieved gradient share
   over training is >= 0.25 (target 0.5). An arm passes if all three seeds
   pass. An arm that fails says nothing about alignment.
2. Effect, per arm: d = Delta_tgt(arm) - Delta_tgt(tgtfwd), paired by seed,
   in log-precipitation MSE. The arm "moves target risk" if d has the same
   sign in all three seeds and |mean d| >= 0.1 x the oracle budget.
3. BatchNorm control: Delta_src(tgtfwd) / Delta_src(reference mmd), both
   against source-only, per seed. If >= 0.8 in all seeds, the source
   inflation of the target-forward methods is BatchNorm running-stat mixing,
   not alignment.

All risks are rescaled to log-precipitation MSE (x s_S^2, see
aggregate_seeds.py, Units); with one pair the factor is common, so only the
budget threshold depends on it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC, TGT = "europe_west", "melanesia"
ARMS = {"tgtfwd": ("mmd", None), "gap": ("mmd", "mmd_gap"), "grid4": ("mmd", "mmd_grid4"),
        "grid4ms": ("mmd_ms", "mmd_grid4")}


def row(csv, method):
    df = pd.read_csv(csv)
    r = df[(df.source == SRC) & (df.target == TGT) & (df.method == method) & (df["transform"] == "none")]
    if len(r) != 1:
        sys.exit(f"{csv}: expected one {SRC}->{TGT} {method} row, found {len(r)}")
    return r.iloc[0]


def msd(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if v.size > 1 else f"{v.mean():.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda")
    p.add_argument("--stats_root", default="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        sys.exit(f"{out} exists; not overwriting.")
    root = Path(args.root)
    s2 = json.loads((Path(args.stats_root) / SRC / "normalization_stats.json").read_text())["precipitation"][1] ** 2

    recs = []
    for seed in args.seeds:
        base = root / "results" / f"seed{seed}" / "unet"
        none, ref = row(base / "none" / "results.csv", "none"), row(base / "none" / "results.csv", "mmd")
        orc = row(base / "oracle" / "results.csv", "none")
        budget = (none.tgt_mse_std - orc.tgt_mse_std) * s2
        arms = {}
        for arm, (method, _) in ARMS.items():
            r = row(root / "results" / "gap_ablation" / f"seed{seed}" / arm / "results.csv", method)
            h = pd.read_csv(root / "gap_ablation" / f"seed{seed}" / arm / f"{SRC}__to__{TGT}__{method}" / "history.csv")
            arms[arm] = (r, h)
        for name, r in [("reference_mmd", ref)] + [(a, arms[a][0]) for a in ARMS]:
            recs.append(dict(seed=seed, arm=name, budget=budget,
                             d_tgt=(r.tgt_mse_std - none.tgt_mse_std) * s2,
                             d_src=(r.src_mse_std - none.src_mse_std) * s2))
        _, hc = arms["tgtfwd"]
        for arm, (method, key) in ARMS.items():
            if key is None:
                continue
            _, h = arms[arm]
            ep = int(h.loc[h.select.idxmin(), "epoch"])
            c = hc[hc.epoch == min(ep, hc.epoch.max())].iloc[0]
            recs.append(dict(seed=seed, arm=arm, check=True, epoch=ep,
                             aligned=float(h.loc[h.epoch == ep, key].iloc[0]), control=float(c[key]),
                             null=float(h.loc[h.epoch == ep, key.replace("mmd", "null")].iloc[0]),
                             share=float(h["share"].mean()), capped=float(h["capped"].sum()),
                             lam=float(h["lam"].mean())))
    df = pd.DataFrame(recs)
    is_chk = df["check"].eq(True) if "check" in df else pd.Series(False, index=df.index)
    eff = df[~is_chk].drop(columns=[c for c in df.columns if c not in
                                                    ("seed", "arm", "budget", "d_tgt", "d_src")])
    chk = df[is_chk][["seed", "arm", "epoch", "aligned", "control", "null", "share", "capped", "lam"]]

    md = [f"# GAP ablation, {SRC} → {TGT}, seeds {args.seeds}", "",
          "Log-precipitation MSE; Δ against source-only of the same seed; mean ± sd over seeds.", "",
          "## Risks", "", "| arm | Δtgt | Δsrc | per seed Δtgt |", "| --- | --- | --- | --- |"]
    for arm in ["reference_mmd"] + list(ARMS):
        e = eff[eff.arm == arm].sort_values("seed")
        md.append(f"| {arm} | {msd(e.d_tgt)} | {msd(e.d_src)} | {', '.join(f'{v:.3f}' for v in e.d_tgt)} |")
    budget = eff.drop_duplicates("seed").budget
    md += ["", f"Oracle budget: {msd(budget)}; effect threshold 0.1 × mean budget = {0.1 * budget.mean():.4f}.", ""]

    md += ["## 1. Manipulation check", "",
           "| arm | seed | sel. epoch | aligned MMD | control MMD | ratio | null | mean share | λ mean | capped |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    passed = {}
    for arm in ("gap", "grid4", "grid4ms"):
        ok = []
        for _, c in chk[chk.arm == arm].sort_values("seed").iterrows():
            ratio = c.aligned / c.control if c.control > 0 else np.nan
            ok.append(bool(ratio <= 0.5 and c.share >= 0.25))
            md.append(f"| {arm} | {int(c.seed)} | {int(c.epoch)} | {c.aligned:.4g} | {c.control:.4g} | {ratio:.3f} | "
                      f"{c.null:.4g} | {c.share:.3f} | {c.lam:.3g} | {int(c.capped)} |")
        passed[arm] = all(ok) and len(ok) == len(args.seeds)
    md += ["", "Pass (all seeds): " + ", ".join(f"{a} {'yes' if v else 'NO'}" for a, v in passed.items()), ""]

    md += ["## 2. Effect against the tgtfwd control", "", "| arm | Δtgt − Δtgt(tgtfwd) | per seed | same sign | |mean| ≥ threshold | moves |",
           "| --- | --- | --- | --- | --- | --- |"]
    c0 = eff[eff.arm == "tgtfwd"].set_index("seed").d_tgt
    thr = 0.1 * budget.mean()
    for arm in ("gap", "grid4", "grid4ms"):
        d = (eff[eff.arm == arm].set_index("seed").d_tgt - c0).sort_index()
        same = bool((d > 0).all() or (d < 0).all())
        big = bool(abs(d.mean()) >= thr)
        md.append(f"| {arm} | {msd(d)} | {', '.join(f'{v:.3f}' for v in d)} | {same} | {big} | "
                  f"{'yes' if same and big else 'no'}{'' if passed[arm] else ' (manipulation check failed)'} |")

    md += ["", "## 3. BatchNorm control", "", "| seed | Δsrc(tgtfwd) | Δsrc(reference mmd) | ratio |", "| --- | --- | --- | --- |"]
    ratios = []
    for seed in args.seeds:
        a = eff[(eff.arm == "tgtfwd") & (eff.seed == seed)].d_src.iloc[0]
        b = eff[(eff.arm == "reference_mmd") & (eff.seed == seed)].d_src.iloc[0]
        ratios.append(a / b if b != 0 else np.nan)
        md.append(f"| {seed} | {a:.4f} | {b:.4f} | {ratios[-1]:.3f} |")
    md += ["", f"Source inflation is BatchNorm mixing (ratio ≥ 0.8 in all seeds): {all(r >= 0.8 for r in ratios)}"]

    out.mkdir(parents=True)
    eff.to_csv(out / "effects.csv", index=False)
    chk.to_csv(out / "manipulation.csv", index=False)
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
