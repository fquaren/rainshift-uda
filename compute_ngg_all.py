"""
Sweep compute_ngg.py across every UDA method present in a results.csv.

Discovers the unique (method, transform) pairs in the CSV for a given model,
calls compute_ngg.py once per pair, and collects a summary table.

Usage:
    python compute_ngg_all.py \\
        --w1_matrix    covariate_shift_analysis/normalized/Wasserstein_1D_test.npy \\
        --w1_variables covariate_shift_analysis/normalized/Wasserstein_1D_test_variables.json \\
        --results_csv  /path/to/results.csv \\
        --model        unet \\
        --output_root  /path/to/ngg

Per-method outputs land in {output_root}/{model}__{method}__{transform}/,
containing ngg_matrix.npy, ngg_meta.json, ngg_heatmap.png (the usual outputs
from compute_ngg.py). A consolidated summary.csv is written at output_root.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


_SCRIPT = Path(__file__).resolve().parent / "compute_ngg.py"


def _discover_combos(results_csv: Path, model: str) -> list:
    """Return sorted unique (method, transform) tuples for the given model."""
    combos = set()
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if row.get("model") != model:
                continue
            combos.add((row.get("method", "none"), row.get("transform", "none")))
    return sorted(combos)


def _summarise(ngg_path: Path, domains: list) -> dict:
    """Return per-run summary statistics from a saved NGG matrix."""
    ngg = np.load(ngg_path)
    off_diag = ngg[~np.eye(ngg.shape[0], dtype=bool)]
    valid = off_diag[np.isfinite(off_diag)]
    if valid.size == 0:
        return {
            "n_finite": 0,
            "mean_ngg": float("nan"),
            "median_ngg": float("nan"),
            "min_ngg": float("nan"),
            "max_ngg": float("nan"),
        }
    return {
        "n_finite": int(valid.size),
        "mean_ngg": float(valid.mean()),
        "median_ngg": float(np.median(valid)),
        "min_ngg": float(valid.min()),
        "max_ngg": float(valid.max()),
    }


def run(args: argparse.Namespace) -> None:
    results_csv = Path(args.results_csv)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    combos = _discover_combos(results_csv, args.model)
    if not combos:
        raise RuntimeError(f"No rows found for model={args.model} in {results_csv}")
    print(f"Found {len(combos)} (method, transform) pairs for {args.model}:")
    for m, t in combos:
        print(f"  - method={m:10s}  transform={t}")

    # Pass-through flags that compute_ngg.py also understands.
    passthrough = []
    if args.error_metric:
        passthrough += ["--error_metric", args.error_metric]
    if args.w1_agg:
        passthrough += ["--w1_agg", args.w1_agg]
    if args.dedup:
        passthrough += ["--dedup", args.dedup]
    if args.legacy_schema:
        passthrough += ["--legacy_schema"]

    summary_rows = []
    for method, transform in combos:
        tag = f"{args.model}__{method}__{transform}"
        run_out = out_root / tag
        cmd = [
            sys.executable,
            str(_SCRIPT),
            "--w1_matrix",
            args.w1_matrix,
            "--w1_variables",
            args.w1_variables,
            "--results_csv",
            str(results_csv),
            "--model",
            args.model,
            "--method",
            method,
            "--transform",
            transform,
            "--output_dir",
            str(run_out),
            *passthrough,
        ]
        print(f"\n=== {tag} ===")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  compute_ngg.py exited with code {rc}; skipping summary.")
            summary_rows.append({"model": args.model, "method": method, "transform": transform, "status": "failed"})
            continue

        meta = json.loads((run_out / "ngg_meta.json").read_text())
        stats = _summarise(run_out / "ngg_matrix.npy", meta["domains"])
        summary_rows.append(
            {
                "model": args.model,
                "method": method,
                "transform": transform,
                "status": "ok",
                "epsilon": f"{meta['epsilon']:.6g}",
                **{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in stats.items()},
            }
        )

    # Write summary.csv
    summary_path = out_root / "summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        # make sure every row has all keys
        for r in summary_rows:
            for k in fieldnames:
                r.setdefault(k, "")
        with open(summary_path, "w") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSummary -> {summary_path}")

        # Console ranking (lower NGG = more transferable).
        print("\nRanking across methods (mean NGG over finite off-diagonal entries):")
        ranked = sorted(
            [r for r in summary_rows if r.get("status") == "ok" and r.get("mean_ngg") not in ("", "nan")],
            key=lambda r: float(r["mean_ngg"]),
        )
        for rank, r in enumerate(ranked, 1):
            print(
                f"  {rank:2d}. method={r['method']:10s} "
                f"transform={r['transform']:10s} "
                f"mean NGG = {r['mean_ngg']}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep compute_ngg.py over every method in a results.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--w1_matrix", required=True)
    p.add_argument("--w1_variables", required=True)
    p.add_argument("--results_csv", required=True)
    p.add_argument("--model", required=True, choices=["unet", "afm"])
    p.add_argument("--output_root", required=True, help="root output directory; per-method subdirs created inside")
    # Pass-through flags. Values here are forwarded to compute_ngg.py;
    # None means 'use compute_ngg.py's own default'.
    p.add_argument("--error_metric", default=None)
    p.add_argument("--w1_agg", default=None)
    p.add_argument("--dedup", default=None, choices=[None, "mean", "last", "first"])
    p.add_argument("--legacy_schema", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
