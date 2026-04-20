"""
Normalised Generalisation Gap (NGG) for deterministic domain transfer on the
RainShift benchmark.

For a model f_S trained on source S and evaluated on target T,

    NGG(S, T) = ( E_T(f_S) - E_S(f_S) ) / ( W1(S, T) + eps )

where E is MSE (the training loss family), W1 is the per-variable 1D
Wasserstein distance computed on domain-normalised pixel distributions, and

    eps = 0.1 * min_{S != T, W1(S, T) > 0} W1(S, T)

as prescribed by the theoretical framework (one order of magnitude below the
minimum non-zero entry of the scalar W1 matrix). A low NGG indicates a source
whose error grows little relative to the underlying distributional shift,
i.e. a resilient transfer in the sense of the Redko et al. (2017) OT bound.

Inputs
------
--w1_matrix, --w1_variables
    The (V, D, D) .npy and its JSON sidecar produced by compute_shift.py in
    --mode normalized (so W1 is measured in the same log-z-scored space the
    model actually trains in).
--results_csv
    results.csv produced by evaluate.py. Rows are filtered by --model,
    --method, and --transform. Multiple matching rows are averaged.
--error_metric
    CSV column used for E. Defaults to rmse_mm, which is squared to obtain
    MSE in mm^2 — matching the training-loss family but in physical units.
    Pass mae_mm or bias_mm to skip the squaring.
--w1_agg
    Collapses the (V, D, D) cube to a (D, D) scalar W1 matrix. Default
    mean_inputs uses only the 9 atmospheric input channels (covariate shift
    in the Redko sense); alternatives are mean_all (inputs + target),
    target (precipitation channel only), max_inputs, or any single variable
    name in the sidecar.

The script needs per-source self-errors E_S(f_S). These come from the results
CSV when rows with source == target are present; otherwise the script warns
and leaves those source rows as NaN. A suggested command to generate missing
self-evaluation rows is printed.

Outputs
-------
{output_dir}/ngg_matrix.npy       (D', D') with NaN on the diagonal
{output_dir}/ngg_meta.json        domain ordering, epsilon, all filter choices
{output_dir}/ngg_heatmap.png      divergent colourmap centred at 0
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Must match data/convert_zarr_to_npy.py
DEFAULT_INPUT_VARS = ["cape", "cp", "sp", "tclw", "tcw", "tisr", "tp", "u", "v"]


# --------------------------------------------------------------------------
#  Aggregation of the (V, D, D) W1 cube
# --------------------------------------------------------------------------

def _aggregate_w1(
    w1: np.ndarray, variables: list, agg: str, input_vars: list,
) -> np.ndarray:
    """Collapse a (V, D, D) W1 cube to a (D, D) scalar matrix."""
    # explicit variable name takes precedence
    if agg in variables:
        return w1[variables.index(agg)]

    input_idx = [i for i, v in enumerate(variables) if v in input_vars]
    target_idx = [i for i, v in enumerate(variables) if v not in input_vars]

    if agg == "mean_inputs":
        if not input_idx:
            raise ValueError(f"No input variables found among {variables}")
        return np.nanmean(w1[input_idx], axis=0)
    if agg == "mean_all":
        return np.nanmean(w1, axis=0)
    if agg == "max_inputs":
        if not input_idx:
            raise ValueError(f"No input variables found among {variables}")
        return np.nanmax(w1[input_idx], axis=0)
    if agg == "target":
        if len(target_idx) != 1:
            raise ValueError(
                "'target' aggregation needs exactly one non-input variable; "
                f"got {[variables[i] for i in target_idx]}"
            )
        return w1[target_idx[0]]
    raise ValueError(
        f"Unknown --w1_agg '{agg}'. Valid: mean_inputs, mean_all, target, "
        f"max_inputs, or any of {variables}"
    )


# --------------------------------------------------------------------------
#  CSV parsing
# --------------------------------------------------------------------------

def _load_error_table(
    results_csv: Path, model: str, method: str, transform: str,
    error_metric: str,
) -> dict:
    """
    Build ``{(source, target): error}`` from the results CSV.
    If error_metric starts with 'rmse', the value is squared to produce MSE.
    Duplicate (source, target) rows are averaged.
    """
    bucket = defaultdict(list)
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if row.get("model") != model:
                continue
            if row.get("method") != method:
                continue
            if row.get("transform", "none") != transform:
                continue
            try:
                val = float(row[error_metric])
            except (KeyError, ValueError, TypeError):
                continue
            if error_metric.startswith("rmse"):
                val = val ** 2
            bucket[(row["source"], row["target"])].append(val)

    return {k: float(np.mean(v)) for k, v in bucket.items()}


# --------------------------------------------------------------------------
#  NGG assembly
# --------------------------------------------------------------------------

def compute_ngg(
    w1: np.ndarray, domains: list, errors: dict,
) -> tuple:
    """
    Build the (D, D) NGG matrix. E_S is taken from (S, S) entries of errors;
    when missing, that source's entire row is NaN.
    """
    n = len(domains)
    positive = w1[w1 > 0]
    if positive.size == 0:
        raise ValueError("No positive entries in the aggregated W1 matrix; "
                         "cannot compute epsilon.")
    eps = 0.1 * float(np.nanmin(positive))

    ngg = np.full((n, n), np.nan, dtype=np.float64)
    for i, s in enumerate(domains):
        e_s = errors.get((s, s))
        if e_s is None:
            continue
        for j, t in enumerate(domains):
            if i == j:
                continue
            e_t = errors.get((s, t))
            if e_t is None:
                continue
            w = w1[i, j]
            if not np.isfinite(w):
                continue
            ngg[i, j] = (e_t - e_s) / (w + eps)

    return ngg, eps


# --------------------------------------------------------------------------
#  Plotting
# --------------------------------------------------------------------------

def plot_heatmap(matrix: np.ndarray, domains: list, title: str, path: Path):
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0.0,
        xticklabels=domains, yticklabels=domains,
        cbar_kws={"label": "NGG"},
    )
    plt.xlabel("Target T")
    plt.ylabel("Source S")
    plt.title(title, fontsize=13)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# --------------------------------------------------------------------------
#  Driver
# --------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    w1_all = np.load(args.w1_matrix)  # (V, D, D)
    meta = json.loads(Path(args.w1_variables).read_text())
    variables, w1_domains = meta["variables"], meta["domains"]

    w1_full = _aggregate_w1(w1_all, variables, args.w1_agg, args.input_vars)

    errors = _load_error_table(
        Path(args.results_csv), args.model, args.method, args.transform,
        args.error_metric,
    )
    if not errors:
        raise RuntimeError(
            f"No rows matched model={args.model}, method={args.method}, "
            f"transform={args.transform} in {args.results_csv}."
        )

    # Restrict to domains present in both the W1 matrix and the CSV.
    csv_domains = {s for s, _ in errors} | {t for _, t in errors}
    domains = [d for d in w1_domains if d in csv_domains]
    if not domains:
        raise RuntimeError(
            f"No overlap between W1 domains {w1_domains} and CSV sources/targets "
            f"{sorted(csv_domains)}."
        )
    idx = [w1_domains.index(d) for d in domains]
    w1 = w1_full[np.ix_(idx, idx)]

    # Source self-errors: warn loudly if missing; NGG rows for those are NaN.
    missing_self = [s for s in domains if (s, s) not in errors]
    if missing_self:
        print(f"\nWarning: missing self-evaluation (source==target) rows for:")
        for s in missing_self:
            print(f"  - {s}")
        print("\nNGG rows for these sources will be NaN.")
        print("Generate them by evaluating each source's vanilla checkpoint")
        print("on its own test set, e.g.:")
        print("  python evaluate.py single --model {unet,afm} \\")
        print("      --checkpoint experiments/<model>/<src>__to__<tgt>__none/best.pt \\")
        print("      --target_path <path to source domain .npy dir>\n")

    ngg, eps = compute_ngg(w1, domains, errors)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ngg_matrix.npy", ngg)
    (out / "ngg_meta.json").write_text(json.dumps({
        "domains": domains,
        "epsilon": eps,
        "w1_agg": args.w1_agg,
        "error_metric": args.error_metric,
        "error_is_mse_like": args.error_metric.startswith("rmse"),
        "model": args.model,
        "method": args.method,
        "transform": args.transform,
        "w1_matrix_source": str(Path(args.w1_matrix).resolve()),
        "results_csv_source": str(Path(args.results_csv).resolve()),
    }, indent=2))

    title = (
        f"NGG  model={args.model}  method={args.method}  "
        f"transform={args.transform}\n"
        f"W1 agg = {args.w1_agg}   error = MSE from {args.error_metric}"
    )
    plot_heatmap(ngg, domains, title, out / "ngg_heatmap.png")

    print(f"\nepsilon = {eps:.4g}")
    print(f"matrix shape = {ngg.shape}")
    print(f"saved to {out}/")

    # Ranking by mean NGG per source (lower = more transferable).
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", category=RuntimeWarning)
        row_mean = np.nanmean(ngg, axis=1)
    print("\nSource transferability ranking (mean NGG across targets, lower = better):")
    order = np.argsort(np.where(np.isnan(row_mean), np.inf, row_mean))
    for r, i in enumerate(order, 1):
        m = row_mean[i]
        tag = "     nan" if np.isnan(m) else f"{m:8.3f}"
        print(f"  {r:2d}. {domains[i]:24s}  mean NGG = {tag}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Normalised Generalisation Gap (Redko et al. 2017 OT bound) for RainShift.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--w1_matrix", required=True,
                   help="Wasserstein_1D_{split}.npy from compute_shift.py (normalized mode)")
    p.add_argument("--w1_variables", required=True,
                   help="JSON sidecar written alongside the W1 matrix")
    p.add_argument("--results_csv", required=True,
                   help="results.csv produced by evaluate.py")
    p.add_argument("--model", required=True, choices=["unet", "afm"])
    p.add_argument("--method", default="none",
                   help="UDA method filter; 'none' is the vanilla baseline")
    p.add_argument("--transform", default="none",
                   help="test-time input transform filter")
    p.add_argument("--error_metric", default="rmse_mm",
                   help="CSV column for E; if it starts with 'rmse' it is "
                        "squared to produce MSE")
    p.add_argument("--w1_agg", default="mean_inputs",
                   help="(V,D,D) -> (D,D) aggregation: mean_inputs (default, "
                        "Redko covariate-shift), mean_all, target, "
                        "max_inputs, or a variable name")
    p.add_argument("--input_vars", nargs="+", default=DEFAULT_INPUT_VARS,
                   help="atmospheric input variable names used by "
                        "mean_inputs and max_inputs")
    p.add_argument("--output_dir", default="./ngg_analysis",
                   help="directory for ngg_matrix.npy, ngg_meta.json, ngg_heatmap.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())