"""
COSMO C3 figures: the source-only transfer matrix and what does (not) explain it.

Reads only result files, no data store:
  results/cosmo_seed<S>/unet/none/results.csv     (evaluate.py, seeds 42/43/44)
  results/cosmo_references/references.csv        (cosmo_references.py)
  results_cosmo/c2/c2_cape_tcw_block.json, c2_tp_cape_fine.json  (C2)

Every matrix is 5 x 5, rows = source, columns = target, regions ordered by
their test-split mean precipitation (driest first), so a dry -> wet pattern
sits in the upper-right triangle. Every derived quantity is computed per seed
and then averaged (paired within a seed); cells show the mean and, small,
the seed sd.

Colour rule: blue = better, red = worse (diverging, centred on "no change");
absolute quantities use other sequential maps.

Figures (one quantity each, PNG + PDF):
  01_mse_log_e01             absolute target score
  02_mse_log_e01_seed_sd     its seed sd
  03_excess_vs_indomain      E_T(f_S) / E_T(f_T) - 1          transfer cost
  04_excess_vs_bicubic       E_T(f_S) / E_T(bicubic tp) - 1   against no learning
  05_mae_excess_vs_indomain  same as 03 in mm MAE
  06_dryness                 -bias / mean truth (fraction of rain missing)
  07_input_shift_dynamic     W1 per source sd, mean over cape tcw tclw tp u v
  08_input_shift_sp          same for sp (near-static, identifies the region)
  09_cond_excess_cape_tcw    C2 conditional excess over the null (symmetric)
  10_cond_excess_tp_cape     same, (tp, cape), fine target
  11_wetness_contrast        log2(mean_T / mean_S)
  12_scatter_<predictor>     03 against each of 07-11, Spearman rho and an
                             exact Mantel-type p (all 120 relabellings of
                             the five regions on the predictor matrix; the
                             20 cells are not independent, so no other test)

--out_dir is refused if it exists.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm
from scipy.stats import spearmanr

SEEDS = (42, 43, 44)
DYNAMIC = ("cape", "tcw", "tclw", "tp", "u", "v")
LABEL = {"alps": "Alps", "british_isles": "British Isles", "poland": "Poland",
         "scandinavia_south": "S. Scandinavia", "spain_central": "C. Spain"}


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def load(results_root, c2_dir):
    refs = {(r["target"], r["reference"]): r for r in read_csv(results_root / "cosmo_references/references.csv")}
    regions = sorted({t for t, _ in refs}, key=lambda t: -float(refs[(t, "zero")]["bias_mm"]))
    n = len(regions)
    ix = {r: i for i, r in enumerate(regions)}
    E = np.full((len(SEEDS), n, n), np.nan)
    MAE = np.full_like(E, np.nan)
    BIAS = np.full_like(E, np.nan)
    for k, s in enumerate(SEEDS):
        rows = read_csv(results_root / f"cosmo_seed{s}/unet/none/results.csv")
        for r in rows:
            i, j = ix[r["source"]], ix[r["target"]]
            E[k, i, j] = float(r["tgt_mse_log_e01"])
            MAE[k, i, j] = float(r["tgt_mae_mm"])
            BIAS[k, i, j] = float(r["tgt_bias_mm"])
    if np.isnan(E).any():
        raise SystemExit("transfer matrix incomplete")
    ref = {name: np.array([float(refs[(t, name)]["mse_log_e01"]) for t in regions])
           for name in ("zero", "bicubic_tp", "block_tp", "climatology")}
    mean_true = np.array([-float(refs[(t, "zero")]["bias_mm"]) for t in regions])

    c2 = {}
    for tag, fn in (("cape_tcw", "c2_cape_tcw_block.json"), ("tp_cape", "c2_tp_cape_fine.json")):
        c2[tag] = json.loads((c2_dir / fn).read_text())["pairs"]
    cond = {tag: np.full((n, n), np.nan) for tag in c2}
    w1 = {v: np.full((n, n), np.nan) for v in DYNAMIC + ("sp",)}
    for tag, pairs in c2.items():
        for key, d in pairs.items():
            a, b = key.split("__")
            i, j = ix[a], ix[b]
            cond[tag][i, j] = cond[tag][j, i] = d["excess_over_null_mean"]["log_e01"]
            if tag == "cape_tcw":
                for v in w1:
                    s = d["input_shift"][v]
                    # The model lives in SOURCE-normalised space: S -> T is
                    # measured in the source's sd.
                    w1[v][i, j] = s["w1_per_sd_x"]
                    w1[v][j, i] = s["w1_per_sd_y"]
    return regions, E, MAE, BIAS, ref, mean_true, cond, w1


def heat(ax, M, sd=None, cmap="Purples", norm=None, fmt="{:.3f}", sdfmt="{:.3f}", cbar_label="",
         regions=(), diag_text=None):
    n = M.shape[0]
    A = M.copy()
    if diag_text is not None:
        A[np.eye(n, dtype=bool)] = np.nan
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#ececec")
    im = ax.imshow(A, cmap=cm, norm=norm)
    for i in range(n):
        for j in range(n):
            if diag_text is not None and i == j:
                ax.text(j, i, diag_text, ha="center", va="center", fontsize=8, color="#666666")
                continue
            v = M[i, j]
            if np.isnan(v):
                continue
            rgba = cm(im.norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            c = "white" if lum < 0.5 else "#111111"
            if sd is None:
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9, color=c)
            else:
                ax.text(j, i - 0.12, fmt.format(v), ha="center", va="center", fontsize=9, color=c)
                ax.text(j, i + 0.22, "±" + sdfmt.format(sd[i, j]), ha="center", va="center",
                        fontsize=7, color=c)
    names = [LABEL.get(r, r) for r in regions]
    ax.set_xticks(range(n), names, rotation=35, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n), minor=True)
    ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    return im


def save(fig, out_dir, name, title, note):
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=7, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {name}")


def mantel(y, x, off):
    """Spearman over off-diagonal cells; exact p over all relabellings of x."""
    n = y.shape[0]
    rho = spearmanr(y[off], x[off]).statistic
    perms = list(itertools.permutations(range(n)))
    null = []
    for p in perms:
        xp = x[np.ix_(p, p)]
        null.append(spearmanr(y[off], xp[off]).statistic)
    null = np.array(null)
    pval = float(np.mean(np.abs(null) >= abs(rho) - 1e-12))
    return float(rho), pval, len(perms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", required=True, help=".../results_rainshift_uda/results")
    ap.add_argument("--c2_dir", required=True, help=".../results_cosmo/c2")
    ap.add_argument("--out_dir", required=True, help="refused if it exists")
    a = ap.parse_args()
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting.")
    regions, E, MAE, BIAS, ref, mean_true, cond, w1 = load(Path(a.results_root), Path(a.c2_dir))
    out.mkdir(parents=True)
    n = len(regions)
    off = ~np.eye(n, dtype=bool)
    order_note = "Regions ordered by test mean precipitation, driest first. Seeds 42/43/44, mean ± sd."
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

    def fig():
        return plt.subplots(figsize=(5.4, 5.0))

    # 01, 02 absolute
    f, ax = fig()
    heat(ax, E.mean(0), E.std(0, ddof=1), cmap="Purples", norm=LogNorm(E.min(), E.max()),
         cbar_label="mse_log_e01 (log scale)", regions=regions)
    save(f, out, "01_mse_log_e01", "Target score, source-only UNet", order_note)
    f, ax = fig()
    heat(ax, E.std(0, ddof=1), cmap="Greys", fmt="{:.4f}", cbar_label="seed sd of mse_log_e01",
         regions=regions)
    save(f, out, "02_mse_log_e01_seed_sd", "Seed spread of the target score", order_note)

    # 03 transfer cost relative to the target's in-domain model (per seed)
    diag = np.stack([np.diag(E[k]) for k in range(len(SEEDS))])  # (S, n)
    rel = E / diag[:, None, :] - 1.0
    f, ax = fig()
    heat(ax, 100 * rel.mean(0), 100 * rel.std(0, ddof=1), cmap="RdBu_r",
         norm=SymLogNorm(linthresh=10, vmin=-400, vmax=400, base=10), fmt="{:+.0f}%", sdfmt="{:.0f}",
         cbar_label="% above the target's in-domain score", regions=regions, diag_text="in-domain")
    save(f, out, "03_excess_vs_indomain", "Transfer cost: E_T(f_S) / E_T(f_T) − 1",
         order_note + " Blue: better than the target's own model. Red: worse. Symlog colour scale, linear within ±10 %.")

    # 04 against bicubic tp
    vb = E / ref["bicubic_tp"][None, None, :] - 1.0
    f, ax = fig()
    heat(ax, 100 * vb.mean(0), 100 * vb.std(0, ddof=1), cmap="RdBu_r",
         norm=SymLogNorm(linthresh=10, vmin=-400, vmax=400, base=10), fmt="{:+.0f}%", sdfmt="{:.0f}",
         cbar_label="% relative to bicubic tp", regions=regions)
    save(f, out, "04_excess_vs_bicubic", "Against no learning: E_T(f_S) / E_T(bicubic tp) − 1",
         order_note + " Blue: beats bicubic-upsampled coarse tp. Red: worse than it. Diagonal = in-domain.")

    # 05 MAE
    dm = np.stack([np.diag(MAE[k]) for k in range(len(SEEDS))])
    relm = MAE / dm[:, None, :] - 1.0
    f, ax = fig()
    heat(ax, 100 * relm.mean(0), 100 * relm.std(0, ddof=1), cmap="RdBu_r",
         norm=SymLogNorm(linthresh=10, vmin=-400, vmax=400, base=10), fmt="{:+.0f}%", sdfmt="{:.0f}",
         cbar_label="% above the target's in-domain MAE", regions=regions, diag_text="in-domain")
    save(f, out, "05_mae_excess_vs_indomain", "Transfer cost in mm/h: MAE_T(f_S) / MAE_T(f_T) − 1",
         order_note + " Blue: better. Red: worse.")

    # 06 dryness
    dry = -BIAS / mean_true[None, None, :]
    f, ax = fig()
    heat(ax, 100 * dry.mean(0), 100 * dry.std(0, ddof=1), cmap="YlGnBu", fmt="{:.0f}%", sdfmt="{:.0f}",
         cbar_label="% of the target's mean rain missing", regions=regions)
    save(f, out, "06_dryness", "Dry bias: −bias / mean truth",
         order_note + " Every cell, in-domain included, is too dry.")

    # 07, 08 input shift
    dyn = np.mean([w1[v] for v in DYNAMIC], axis=0)
    f, ax = fig()
    heat(ax, dyn, cmap="Greens", fmt="{:.2f}", cbar_label="W1 / source sd", regions=regions, diag_text="—")
    save(f, out, "07_input_shift_dynamic", "Input shift, dynamic inputs (" + ", ".join(DYNAMIC) + ")",
         order_note.split(" Seeds")[0] + " Mean over variables of the per-variable W1 in source sd (C2, test, 600 frames).")
    f, ax = fig()
    heat(ax, w1["sp"], cmap="Greens", fmt="{:.2f}", cbar_label="W1 / source sd", regions=regions, diag_text="—")
    save(f, out, "08_input_shift_sp", "Input shift, surface pressure (near-static, identifies the region)",
         order_note.split(" Seeds")[0])

    # 09, 10 conditional excess
    for k, (tag, title) in enumerate((("cape_tcw", "P(Y | cape, tcw), target block mean"),
                                      ("tp_cape", "P(Y | tp, cape), target 2.2 km pixels")), start=9):
        f, ax = fig()
        heat(ax, cond[tag], cmap="Oranges", fmt="{:.3f}", cbar_label="excess over permutation-null mean",
             regions=regions, diag_text="—")
        save(f, out, f"{k:02d}_cond_excess_{tag}", "Conditional shift: " + title,
             order_note.split(" Seeds")[0] + " C2 statistic (log_e01), symmetric. Not a better/worse quantity.")

    # 11 wetness contrast
    wc = np.log2(mean_true[None, :] / mean_true[:, None])
    f, ax = fig()
    heat(ax, wc, cmap="BrBG", norm=TwoSlopeNorm(0, -np.abs(wc).max(), np.abs(wc).max()), fmt="{:+.2f}",
         cbar_label="log2(mean_T / mean_S)", regions=regions, diag_text="—")
    save(f, out, "11_wetness_contrast", "Wetness contrast: target vs source mean rain",
         order_note.split(" Seeds")[0] + " Green: target wetter than source. Brown: drier.")

    # 12 scatters
    y = rel.mean(0)
    preds = {"input_shift_dynamic": (dyn, "dynamic input W1 / source sd"),
             "input_shift_sp": (w1["sp"], "sp W1 / source sd"),
             "cond_excess_cape_tcw": (cond["cape_tcw"], "conditional excess (cape, tcw)"),
             "cond_excess_tp_cape": (cond["tp_cape"], "conditional excess (tp, cape)"),
             "wetness_contrast": (wc, "log2(mean_T / mean_S)")}
    summary = []
    for name, (x, xl) in preds.items():
        rho, p, nperm = mantel(y, x, off)
        summary.append({"predictor": name, "spearman_rho": rho, "mantel_p": p, "n_perm": nperm})
        f, ax = plt.subplots(figsize=(4.6, 4.2))
        yy = 100 * y[off]
        err = 100 * rel.std(0, ddof=1)[off]
        ax.errorbar(x[off], yy, yerr=err, fmt="o", ms=5, color="#3a3a3a", ecolor="#bbbbbb", capsize=0)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel(xl)
        ax.set_ylabel("% above target's in-domain score")
        ax.text(0.03, 0.97, f"Spearman ρ = {rho:+.2f}\nrelabelling p = {p:.2f} ({nperm})",
                transform=ax.transAxes, va="top", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        save(f, out, f"12_scatter_{name}", "Transfer cost vs " + xl,
             "20 ordered pairs, mean ± seed sd. Pairs share regions: p from all relabellings of the predictor matrix.")
    with open(out / "summary_predictors.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(f"{s['predictor']:24s} rho {s['spearman_rho']:+.2f}  p {s['mantel_p']:.3f}")
    np.savez(out / "matrices.npz", regions=np.array(regions), E=E, rel=rel, vs_bicubic=vb, mae_rel=relm,
             dryness=dry, w1_dynamic=dyn, w1_sp=w1["sp"], cond_cape_tcw=cond["cape_tcw"],
             cond_tp_cape=cond["tp_cape"], wetness_contrast=wc)


if __name__ == "__main__":
    main()
