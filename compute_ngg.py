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

CSV schema
----------
The script expects the current ``evaluate.py`` schema where each row carries
both the source-domain and target-domain errors:

    model, source, target, method, transform,
    src_rmse_mm, src_mae_mm, src_bias_mm, src_n_samples, src_mse_std,
    tgt_rmse_mm, tgt_mae_mm, tgt_bias_mm, tgt_n_samples, tgt_mse_std

E_S is read from ``src_<error_metric>`` and E_T from ``tgt_<error_metric>``
on the same row. Self-evaluation rows (source == target) are no longer
required.

For backwards compatibility with the older evaluate.py schema (unprefixed
columns and a (S, S) row giving E_S), pass ``--legacy_schema``.

Inputs
------
--w1_matrix, --w1_variables
    The (V, D, D) .npy and its JSON sidecar produced by compute_shift.py in
    --mode normalized.
--results_csv
    results.csv produced by evaluate.py.
--error_metric
    Base column name for E, without src_/tgt_ prefix. Defaults to mse_std,
    which is the standardized-space MSE matching the training loss family
    exactly. Also accepts rmse_mm (squared to obtain MSE in mm^2),
    mae_mm, or bias_mm.
--w1_agg
    Collapses the (V, D, D) cube to a (D, D) scalar W1 matrix.

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
    w1: np.ndarray,
    variables: list,
    agg: str,
    input_vars: list,
) -> np.ndarray:
    """Collapse a (V, D, D) W1 cube to a (D, D) scalar matrix."""
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
        f"Unknown --w1_agg '{agg}'. Valid: mean_inputs, mean_all, target, " f"max_inputs, or any of {variables}"
    )


# --------------------------------------------------------------------------
#  Error-metric column resolution
# --------------------------------------------------------------------------


def _squared(col: str) -> bool:
    """Return True if the CSV column is an RMSE and should be squared to MSE."""
    return "rmse" in col.lower()


def _pick_columns(fieldnames, metric: str, legacy: bool) -> tuple:
    """
    Resolve the CSV columns that carry E_S and E_T.

    Returns (src_col, tgt_col, squared). `squared` is True if the value
    represents an RMSE and should be squared to produce MSE.
    """
    if legacy:
        if metric not in fieldnames:
            raise KeyError(
                f"--legacy_schema expects an unprefixed '{metric}' column; " f"available: {sorted(fieldnames)}"
            )
        return metric, metric, _squared(metric)

    src_col, tgt_col = f"src_{metric}", f"tgt_{metric}"
    missing = [c for c in (src_col, tgt_col) if c not in fieldnames]
    if missing:
        raise KeyError(
            f"CSV is missing columns {missing}. Found: {sorted(fieldnames)}. "
            f"If the CSV predates the src_/tgt_ schema, pass --legacy_schema."
        )
    return src_col, tgt_col, _squared(metric)


# --------------------------------------------------------------------------
#  CSV parsing
# --------------------------------------------------------------------------


def _load_error_table(
    results_csv: Path,
    model: str,
    method: str,
    transform: str,
    metric: str,
    legacy: bool,
    dedup: str,
) -> dict:
    """
    Build ``{(source, target): (E_S, E_T)}`` from the results CSV under the
    new schema, or ``{(source, target): E}`` under the legacy schema.

    `dedup` controls handling of multiple rows with the same key:
        'mean' - average
        'last' - keep the final row in the file (default, to discard stale
                 rows from prior runs)
        'first' - keep the first row in the file
    """
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        src_col, tgt_col, sq = _pick_columns(fieldnames, metric, legacy)

        bucket_new = defaultdict(list)  # (s, t) -> list[(E_S, E_T, row_idx)]
        bucket_legacy = defaultdict(list)  # (s, t) -> list[(E, row_idx)]

        for row_idx, row in enumerate(reader):
            if row.get("model") != model:
                continue
            if row.get("method") != method:
                continue
            if row.get("transform", "none") != transform:
                continue

            try:
                if legacy:
                    v = float(row[src_col])
                    if sq:
                        v = v**2
                    bucket_legacy[(row["source"], row["target"])].append((v, row_idx))
                else:
                    e_s = float(row[src_col])
                    e_t = float(row[tgt_col])
                    if sq:
                        e_s, e_t = e_s**2, e_t**2
                    bucket_new[(row["source"], row["target"])].append(
                        (e_s, e_t, row_idx),
                    )
            except (ValueError, TypeError):
                # blank or non-numeric cell - skip silently
                continue

    def _pick(entries, n_cols):
        if dedup == "last":
            chosen = max(entries, key=lambda r: r[-1])
            return chosen[:n_cols]
        if dedup == "first":
            chosen = min(entries, key=lambda r: r[-1])
            return chosen[:n_cols]
        arr = np.array([r[:n_cols] for r in entries], dtype=np.float64)
        return tuple(arr.mean(axis=0).tolist())

    if legacy:
        return {k: _pick(v, 1)[0] for k, v in bucket_legacy.items()}
    return {k: _pick(v, 2) for k, v in bucket_new.items()}


# --------------------------------------------------------------------------
#  NGG assembly
# --------------------------------------------------------------------------


def _compute_epsilon(w1: np.ndarray) -> float:
    """0.1 * minimum non-zero, finite W1 in the aggregated (D, D) matrix."""
    positive = w1[np.isfinite(w1) & (w1 > 0)]
    if positive.size == 0:
        raise ValueError("No positive finite entries in the aggregated W1 matrix; " "cannot compute epsilon.")
    return 0.1 * float(positive.min())


def compute_ngg_new(
    w1: np.ndarray,
    domains: list,
    errors: dict,
) -> tuple:
    """New schema: both E_S and E_T come from the same (S, T) row."""
    n = len(domains)
    eps = _compute_epsilon(w1)
    ngg = np.full((n, n), np.nan, dtype=np.float64)
    missing_pairs = []

    for i, s in enumerate(domains):
        for j, t in enumerate(domains):
            if i == j:
                continue
            key = (s, t)
            if key not in errors:
                missing_pairs.append(key)
                continue
            e_s, e_t = errors[key]
            w = w1[i, j]
            if not np.isfinite(w):
                continue
            ngg[i, j] = (e_t - e_s) / (w + eps)

    return ngg, eps, missing_pairs


def compute_ngg_legacy(
    w1: np.ndarray,
    domains: list,
    errors: dict,
) -> tuple:
    """Legacy schema: E_S read from (S, S) rows."""
    n = len(domains)
    eps = _compute_epsilon(w1)
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

    missing_self = [s for s in domains if (s, s) not in errors]
    return ngg, eps, missing_self


# --------------------------------------------------------------------------
#  Plotting
# --------------------------------------------------------------------------


def plot_heatmap(matrix: np.ndarray, domains: list, title: str, path: Path):
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0.0,
        xticklabels=domains,
        yticklabels=domains,
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
    w1_all = np.load(args.w1_matrix)
    meta = json.loads(Path(args.w1_variables).read_text())
    variables, w1_domains = meta["variables"], meta["domains"]

    w1_full = _aggregate_w1(w1_all, variables, args.w1_agg, args.input_vars)

    errors = _load_error_table(
        Path(args.results_csv),
        args.model,
        args.method,
        args.transform,
        args.error_metric,
        args.legacy_schema,
        args.dedup,
    )
    if not errors:
        raise RuntimeError(
            f"No rows matched model={args.model}, method={args.method}, "
            f"transform={args.transform} in {args.results_csv}. Check the "
            f"filter values and that the expected columns exist "
            f"(src_{args.error_metric} / tgt_{args.error_metric} in the new "
            f"schema, or {args.error_metric} with --legacy_schema)."
        )

    csv_domains = {s for s, _ in errors} | {t for _, t in errors}
    domains = [d for d in w1_domains if d in csv_domains]
    if not domains:
        raise RuntimeError(
            f"No overlap between W1 domains {w1_domains} and CSV " f"sources/targets {sorted(csv_domains)}."
        )
    idx = [w1_domains.index(d) for d in domains]
    w1 = w1_full[np.ix_(idx, idx)]

    if args.legacy_schema:
        ngg, eps, missing_self = compute_ngg_legacy(w1, domains, errors)
        if missing_self:
            print("\nWarning: missing self-evaluation (source==target) rows for:")
            for s in missing_self:
                print(f"  - {s}")
            print(
                "\nNGG rows for these sources will be NaN. In the current "
                "evaluate.py schema, each (S, T) row carries E_S directly; "
                "re-export results.csv and drop --legacy_schema."
            )
    else:
        ngg, eps, missing_pairs = compute_ngg_new(w1, domains, errors)
        if missing_pairs:
            n_expected = len(domains) * (len(domains) - 1)
            print(
                f"\nNote: {len(missing_pairs)}/{n_expected} off-diagonal "
                f"(S, T) pairs absent from the CSV; those NGG entries are NaN."
            )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ngg_matrix.npy", ngg)
    (out / "ngg_meta.json").write_text(
        json.dumps(
            {
                "domains": domains,
                "epsilon": eps,
                "w1_agg": args.w1_agg,
                "error_metric": args.error_metric,
                "error_is_mse_like": _squared(args.error_metric),
                "legacy_schema": bool(args.legacy_schema),
                "dedup": args.dedup,
                "model": args.model,
                "method": args.method,
                "transform": args.transform,
                "w1_matrix_source": str(Path(args.w1_matrix).resolve()),
                "results_csv_source": str(Path(args.results_csv).resolve()),
            },
            indent=2,
        )
    )

    title = (
        f"NGG  model={args.model}  method={args.method}  "
        f"transform={args.transform}\n"
        f"W1 agg = {args.w1_agg}   error from {args.error_metric}"
    )
    plot_heatmap(ngg, domains, title, out / "ngg_heatmap.png")

    print(f"\nepsilon = {eps:.4g}")
    print(f"matrix shape = {ngg.shape}")
    print(f"saved to {out}/")

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
    p.add_argument(
        "--w1_matrix", required=True, help="Wasserstein_1D_{split}.npy from compute_shift.py " "(normalized mode)"
    )
    p.add_argument("--w1_variables", required=True, help="JSON sidecar written alongside the W1 matrix")
    p.add_argument("--results_csv", required=True, help="results.csv produced by evaluate.py")
    p.add_argument("--model", required=True, choices=["unet", "afm"])
    p.add_argument("--method", default="none", help="UDA method filter; 'none' is the vanilla baseline")
    p.add_argument("--transform", default="none", help="test-time input transform filter")
    p.add_argument(
        "--error_metric",
        default="mse_std",
        help="Base metric name without src_/tgt_ prefix. "
        "mse_std matches the training loss exactly; "
        "rmse_mm is squared to produce MSE in mm^2",
    )
    p.add_argument("--w1_agg", default="mean_inputs", help="(V,D,D) -> (D,D) aggregation")
    p.add_argument(
        "--input_vars",
        nargs="+",
        default=DEFAULT_INPUT_VARS,
        help="atmospheric input variable names for mean_inputs / " "max_inputs",
    )
    p.add_argument(
        "--legacy_schema",
        action="store_true",
        help="read the pre-src_/tgt_ CSV layout and fetch E_S " "from (S, S) self-evaluation rows",
    )
    p.add_argument(
        "--dedup",
        default="last",
        choices=["mean", "last", "first"],
        help="how to reduce multiple rows with the same "
        "(model, source, target, method, transform) key; "
        "'last' (default) discards stale rows from prior runs",
    )
    p.add_argument(
        "--output_dir", default="./ngg_analysis", help="directory for ngg_matrix.npy, ngg_meta.json, ngg_heatmap.png"
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
