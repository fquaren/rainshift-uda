"""
Mean +- standard deviation over seeds of the transfer matrix, the oracle
decomposition and the per-method recovery table, from the per-seed
results.csv files written by evaluate.py.

Every derived quantity is computed WITHIN a seed and then aggregated, so the
standard deviation of a difference is the spread of the paired differences,
not a combination of two unpaired spreads. All numbers are in one metric,
--metric (mse_std for the UNet arm; crps_std for the AFM ensemble), read as
src_<metric> and tgt_<metric>. Standard deviations use ddof=1.

Units. evaluate.py scores both splits of a run in its SOURCE's standardised
space, z = (log(P + 1e-5) - m_S) / s_S, so a raw mse_std from source S is in
units of s_S^2. Two rows with different sources are not comparable, and the
gap and lambda_tgt below subtract a T-normalised score (h_T) from
S-normalised ones. With --units log (the default) every STANDARDISED score
(a name in STD_POWER, all ending in _std) is multiplied by s_S^p on loading,
p from STD_POWER (2 for squared metrics, 1 for MAE / CRPS / spread, 0 for a
ratio), which gives log-precipitation units, the same for every row. s_S is
the precipitation std in <stats_root>/<S>/normalization_stats.json. The
transform is a log followed by a z-score, so the conversion is exact.
Metrics that evaluate.py already computes in physical units (PHYSICAL:
mse_log_e01 / crps_log_e01 in log(mm + 0.1), the *_mm columns, ...) get
factor 1 under either --units. Any other metric name is refused rather
than guessed. --units std reproduces the raw, per-source numbers. Recovery
is a ratio within one row and is the same in both.

Definitions, per seed, for an off-diagonal pair S -> T and method m:

    E_T(f)       tgt_<metric> of model f, scored on T's test split
    h_T          the in-domain source-only model of T (row T -> T, method none)
    gap          E_T(none) - E_T(h_T)          transfer gap of source-only
    budget       E_T(none) - E_T(oracle)       what joint training recovers
    lambda_tgt   E_T(oracle) - E_T(h_T)        target-side excess of the oracle
                                               (was lambda_hat; one-sided)
    oracle_src   E_S(oracle) - E_S(h_S)        source-side excess of the oracle
    lambda_joint lambda_tgt + oracle_src       two-sided excess: Ben-David's
                                               lambda = min_h eps_S + eps_T,
                                               less the per-domain bests,
                                               with the oracle as the joint h
    *_frac       the same divided by gap (lambda_tgt_frac was lambda_frac)
    delta_tgt    E_T(m) - E_T(none)            < 0 is a target gain
    delta_src    E_S(m) - E_S(none)            > 0 is source inflation
    recovery     -delta_tgt / budget           fraction of the budget won

Inputs (per seed N, under --results_root):
    seed<N>/<subdir>/none/results.csv      source-only and method rows
    seed<N>/<subdir>/oracle/results.csv    oracle rows
    seed<N>/<ngg_dir>/<subdir>_none/summary.csv   optional, NGG per method
                                                  (--ngg_dir, default ngg_v3)

Outputs (--output_dir, refused if it exists):
    transfer_matrix.csv, oracle.csv, methods.csv, ranks.csv, ngg.csv,
    tables.md (the same numbers formatted as mean +- sd), checks.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["source", "target", "method"]

# Metrics evaluate.py computes in the source-standardised space, and the
# power of s_S that turns each into log-precipitation units.
STD_POWER = {"mse_std": 2, "mae_std": 1, "crps_std": 1, "spread_std": 1,
             "ens_var_std": 2, "ensmean_mse_std": 2, "ssr_std": 0}
# Metrics already in physical units (log(mm + floor) or mm): factor 1.
PHYSICAL = [r"mse_log_e\d+", r"crps_log_e\d+", r"ens_var_log_e\d+", r"ensmean_mse_log_e\d+",
            r"ssr_log_e\d+", r"rmse_mm", r"mae_mm", r"bias_mm", r"crps_mm", r"spread_mm"]


def metric_power(metric):
    """Power p of s_S that converts `metric` to physical units: STD_POWER
    for standardised metrics, 0 for PHYSICAL ones. Exits on anything else,
    including an unlisted *_std name, rather than guessing."""
    if metric in STD_POWER:
        return STD_POWER[metric]
    if metric.endswith("_std"):
        sys.exit(f"metric {metric}: standardised (ends in _std) but its power of s_S is not in STD_POWER")
    if any(re.fullmatch(pat, metric) for pat in PHYSICAL):
        return 0
    sys.exit(f"metric {metric}: unknown; neither in STD_POWER nor matching PHYSICAL {PHYSICAL}")


def units_label(metric, units):
    if metric.startswith("ssr_"):
        return "dimensionless spread-skill ratio (not rescaled)"
    if metric_power(metric) == 0:
        m = re.fullmatch(r".*_log_e(\d+)", metric)
        return f"log(mm + 0.{m.group(1)}) (physical, not rescaled)" if m else "physical (not rescaled)"
    if units == "log":
        return "log(P + 1e-5) precipitation, rescaled by s_S^%d" % metric_power(metric)
    return "per-source standardised (rows not comparable)"


def load_seed(root, seed, subdir, metric, scale):
    """scale: {domain: factor} applied to both score columns by the row's
    source (see Units above); all ones for --units std."""
    base = Path(root) / f"seed{seed}" / subdir
    cols = KEY + ["transform", f"src_{metric}", f"tgt_{metric}", "src_n_samples", "tgt_n_samples"]
    frames = {}
    for name in ("none", "oracle"):
        csv = base / name / "results.csv"
        df = pd.read_csv(csv)
        absent = [c for c in cols if c not in df.columns]
        if absent:
            sys.exit(f"{csv}: no column(s) {absent}; re-score this tree with the current evaluate.py")
        frames[name] = df[cols].copy()
    runs, orc = frames["none"], frames["oracle"]
    for name, df in (("none", runs), ("oracle", orc)):
        if (df["transform"] != "none").any():
            sys.exit(f"seed {seed} {name}: rows with an input transform; expected none only")
        if df.duplicated(KEY).any():
            sys.exit(f"seed {seed} {name}: duplicate (source, target, method) rows")
        f = df["source"].map(scale)
        if f.isna().any():
            sys.exit(f"seed {seed} {name}: no normalisation stats for {sorted(set(df.source[f.isna()]))}")
        for c in (f"src_{metric}", f"tgt_{metric}"):
            df[c] = df[c] * f
    runs = runs.assign(seed=seed)
    orc = orc.assign(seed=seed)
    return runs, orc


def precip_scale(stats_root, domains, metric, units):
    """{domain: s_S^p}; all ones for --units std or a physical metric."""
    power = metric_power(metric)
    if units == "std" or power == 0:
        return {d: 1.0 for d in domains}
    out = {}
    for d in domains:
        mean_std = json.loads((Path(stats_root) / d / "normalization_stats.json").read_text())["precipitation"]
        out[d] = float(mean_std[1]) ** power
    return out


def msd(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), (v.std(ddof=1) if v.size > 1 else np.nan), v.size


def summarise(df, by, cols, seeds):
    """Long per-seed frame -> one row per group with per-seed values, mean, sd, n."""
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        for c in cols:
            for s in seeds:
                v = g.loc[g["seed"] == s, c]
                row[f"{c}_s{s}"] = float(v.iloc[0]) if len(v) else np.nan
            row[f"{c}_mean"], row[f"{c}_sd"], row[f"{c}_n"] = msd(g[c].dropna())
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(m, s, d=4):
    if np.isnan(m):
        return "n/a"
    return f"{m:.{d}f} ± {s:.{d}f}" if not np.isnan(s) else f"{m:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--subdir", default="unet", help="unet, afm or afm_prob")
    p.add_argument("--metric", default="mse_std")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--units", default="log", choices=["log", "std"])
    p.add_argument("--stats_root", default="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift",
                   help="Read-only: only normalization_stats.json is read.")
    p.add_argument("--domains", nargs="+", default=["europe_west", "horn-of-africa", "melanesia"])
    p.add_argument("--ngg_dir", default="ngg_v3",
                   help="per-seed NGG tree; ngg_v2 used the own-stats W1 cube (deprecated)")
    args = p.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        sys.exit(f"{out} exists; not overwriting.")
    seeds, m = args.seeds, args.metric
    E_S, E_T = f"src_{m}", f"tgt_{m}"
    checks = []

    scale = precip_scale(args.stats_root, args.domains, m, args.units)
    checks.append(f"units: {args.units}; per-source factors " + ", ".join(f"{d} {v:.4f}" for d, v in scale.items()))
    per_seed = [load_seed(args.results_root, s, args.subdir, m, scale) for s in seeds]
    runs = pd.concat([r for r, _ in per_seed], ignore_index=True)
    orc = pd.concat([o for _, o in per_seed], ignore_index=True)

    # ---- consistency checks ---------------------------------------------
    sets = {s: set(map(tuple, g[KEY].values)) for s, g in runs.groupby("seed")}
    ref = sets[seeds[0]]
    for s in seeds[1:]:
        if sets[s] != ref:
            sys.exit(f"seed {s} has a different set of runs than seed {seeds[0]}: "
                     f"missing {sorted(ref - sets[s])}, extra {sorted(sets[s] - ref)}")
    checks.append(f"runs per seed: {len(ref)} (identical set across seeds {seeds})")
    for c in ("src_n_samples", "tgt_n_samples"):
        n_var = runs.groupby(KEY)[c].nunique()
        if (n_var > 1).any():
            sys.exit(f"{c} differs across seeds for {list(n_var[n_var > 1].index)}")
    checks.append("test-split sample counts identical across seeds")

    diag = runs[(runs.source == runs.target) & (runs.method == "none")].set_index(["seed", "target"])[E_T]
    none = runs[(runs.source != runs.target) & (runs.method == "none")].set_index(["seed", "source", "target"])
    # Off-diagonal source-only checkpoints are byte-identical copies of the
    # diagonal one, so their source score must equal the diagonal score.
    worst = 0.0
    for (s, src, tgt), r in none.iterrows():
        worst = max(worst, abs(r[E_S] - diag[(s, src)]))
    checks.append(f"source-only copies: max |E_S(S->T none) - E_T(S->S none)| = {worst:.2e}"
                  + ("" if worst < 1e-3 else "  <-- NOT the same checkpoint?"))
    if "adabn" in set(runs.method):
        a = runs[runs.method == "adabn"].set_index(["seed", "source", "target"])[E_T]
        same = [(k, v) for k, v in a.items() if abs(v - none.loc[k, E_T]) < 1e-6]
        checks.append(f"AdaBN target rows identical to source-only: {len(same)}"
                      + ("  <-- scored best.pt, not best_adabn.pt" if same else ""))

    # ---- transfer matrix ------------------------------------------------
    tm = runs[runs.method == "none"].rename(columns={E_T: "E_T"})
    tm_s = summarise(tm, ["source", "target"], ["E_T"], seeds)

    # ---- oracle decomposition -------------------------------------------
    o = orc.set_index(["seed", "source", "target"])
    rows = []
    for (s, src, tgt), r in none.iterrows():
        if (s, src, tgt) not in o.index:
            sys.exit(f"seed {s}: no oracle row for {src} -> {tgt}")
        e_none, e_orc, e_hT = r[E_T], o.loc[(s, src, tgt), E_T], diag[(s, tgt)]
        gap = e_none - e_hT
        lam_t = e_orc - e_hT
        # E_S(h_S) is the diagonal S -> S row scored on S's test split.
        lam_s = o.loc[(s, src, tgt), E_S] - diag[(s, src)]
        rows.append(dict(seed=s, source=src, target=tgt,
                         E_T_none=e_none, E_T_oracle=e_orc, E_T_hT=e_hT,
                         gap=gap, budget=e_none - e_orc,
                         lambda_tgt=lam_t, lambda_tgt_frac=lam_t / gap,
                         oracle_src=lam_s,
                         lambda_joint=lam_t + lam_s, lambda_joint_frac=(lam_t + lam_s) / gap))
    od = pd.DataFrame(rows)
    ocols = ["E_T_none", "E_T_oracle", "E_T_hT", "gap", "budget", "lambda_tgt", "lambda_tgt_frac",
             "oracle_src", "lambda_joint", "lambda_joint_frac"]
    od_s = summarise(od, ["source", "target"], ocols, seeds)

    # ---- methods ----------------------------------------------------------
    bud = od.set_index(["seed", "source", "target"])["budget"]
    meth = runs[runs.source != runs.target].copy()
    ref_none = none[[E_S, E_T]]
    meth["E_T"] = meth[E_T]
    meth["E_S"] = meth[E_S]
    idx = list(zip(meth.seed, meth.source, meth.target))
    meth["delta_tgt"] = meth[E_T].values - ref_none.loc[idx, E_T].values
    meth["delta_src"] = meth[E_S].values - ref_none.loc[idx, E_S].values
    meth["recovery"] = -meth["delta_tgt"].values / bud.loc[idx].values
    mcols = ["E_T", "E_S", "delta_tgt", "delta_src", "recovery"]
    me_s = summarise(meth, ["source", "target", "method"], mcols, seeds)
    cnt = meth.groupby(["source", "target", "method"]).agg(
        seeds_tgt_gain=("delta_tgt", lambda v: int((v < 0).sum())),
        seeds_src_inflation=("delta_src", lambda v: int((v > 0).sum())),
    ).reset_index()
    me_s = me_s.merge(cnt, on=["source", "target", "method"])

    # ---- ranking stability -------------------------------------------------
    meth["rank"] = meth.groupby(["seed", "source", "target"])[E_T].rank(method="min")
    rk = summarise(meth, ["source", "target", "method"], ["rank"], seeds)
    order = {}
    for (src, tgt), grp in meth.groupby(["source", "target"], sort=False):
        order[(src, tgt)] = {s: " < ".join(gs.sort_values(E_T)["method"]) for s, gs in grp.groupby("seed")}

    # ---- NGG --------------------------------------------------------------
    ngg_s = None
    ngg_files = [Path(args.results_root) / f"seed{s}" / args.ngg_dir / f"{args.subdir}_none" / "summary.csv"
                 for s in seeds]
    if all(f.exists() for f in ngg_files):
        ngg = pd.concat([pd.read_csv(f).assign(seed=s) for f, s in zip(ngg_files, seeds)])
        ngg_s = summarise(ngg, ["method"], ["mean_ngg", "median_ngg", "mean_delta_src", "mean_delta_tgt"], seeds)
    else:
        checks.append("NGG: summary.csv missing for at least one seed; ngg.csv not written")

    # ---- write --------------------------------------------------------------
    out.mkdir(parents=True)
    tm_s.to_csv(out / "transfer_matrix.csv", index=False)
    od_s.to_csv(out / "oracle.csv", index=False)
    me_s.to_csv(out / "methods.csv", index=False)
    rk.to_csv(out / "ranks.csv", index=False)
    if ngg_s is not None:
        ngg_s.to_csv(out / "ngg.csv", index=False)
    (out / "checks.txt").write_text("\n".join(checks) + "\n")

    doms = list(dict.fromkeys(tm_s.source))
    methods = [x for x in dict.fromkeys(meth.method) if x != "none"]
    pairs = list(dict.fromkeys(zip(od_s.source, od_s.target)))
    g = lambda df, keys, c: df.set_index(keys).loc[c]
    md = [f"# Seed aggregate, {args.subdir}, {m}",
          "", f"Seeds {seeds}; mean ± sd (ddof 1) over seeds; differences are paired within seed.",
          f"Units: {m} in {units_label(m, args.units)}.", "",
          "## Transfer matrix, E_T (rows source, columns target; diagonal in-domain)", "",
          md_table(["source \\ target"] + doms,
                   [[s] + [fmt(*g(tm_s, ["source", "target"], (s, t))[["E_T_mean", "E_T_sd"]]) for t in doms]
                    for s in doms]),
          "", "## Oracle decomposition", "",
          "λ_T = E_T(oracle) − E_T(h_T) is the target-side excess only (formerly λ̂). "
          "λ_joint = λ_T + [E_S(oracle) − E_S(h_S)] is the two-sided Ben-David excess.", "",
          md_table(["pair", "gap", "budget", "λ_T", "λ_T / gap", "oracle source cost", "λ_joint", "λ_joint / gap"],
                   [[f"{s} → {t}"] + [fmt(*g(od_s, ["source", "target"], (s, t))[[f"{c}_mean", f"{c}_sd"]])
                                      for c in ("gap", "budget", "lambda_tgt", "lambda_tgt_frac", "oracle_src",
                                                "lambda_joint", "lambda_joint_frac")] for s, t in pairs])]
    for c, title in (("delta_tgt", "Target change E_T(m) − E_T(none); < 0 is a gain"),
                     ("delta_src", "Source inflation E_S(m) − E_S(none)"),
                     ("recovery", "Recovery, fraction of the oracle budget")):
        md += ["", f"## {title}", "",
               md_table(["pair"] + methods,
                        [[f"{s} → {t}"] + [fmt(*g(me_s, ["source", "target", "method"], (s, t, x))[[f"{c}_mean", f"{c}_sd"]],
                                               d=3 if c == "recovery" else 4) for x in methods] for s, t in pairs])]
    md += ["", "## Seeds with a target gain / with source inflation (out of %d)" % len(seeds), "",
           md_table(["pair"] + methods,
                    [[f"{s} → {t}"] + ["%d / %d" % tuple(g(me_s, ["source", "target", "method"], (s, t, x))
                                                         [["seeds_tgt_gain", "seeds_src_inflation"]])
                                       for x in methods] for s, t in pairs])]
    md += ["", "## Method ordering by E_T, per seed (best first, source-only included)", ""]
    for (s, t), per in order.items():
        stable = "stable" if len(set(per.values())) == 1 else "NOT stable"
        md.append(f"- **{s} → {t}** ({stable})")
        md += [f"    - seed {sd}: {seq}" for sd, seq in per.items()]
    if ngg_s is not None:
        md += ["", f"## NGG (fixed reference, {args.ngg_dir})", "",
               f"Copied from the per-seed {args.ngg_dir} summaries: per-source standardised units whatever "
               "--units says, so the mean over pairs mixes sources.", "",
               md_table(["method", "mean NGG", "median NGG", "mean Δsrc", "mean Δtgt"],
                        [[r.method] + [fmt(r[f"{c}_mean"], r[f"{c}_sd"])
                                       for c in ("mean_ngg", "median_ngg", "mean_delta_src", "mean_delta_tgt")]
                         for _, r in ngg_s.iterrows()])]
    md += ["", "## Checks", ""] + [f"- {c}" for c in checks]
    (out / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
