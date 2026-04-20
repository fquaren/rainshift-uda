"""
Per-variable 1D Wasserstein distributional-shift matrices between RainShift
domains, computed on the .npy datasets produced by
``data/convert_zarr_to_npy.py``.

For every (domain_i, domain_j, variable) triple the script computes the
Wasserstein-1 distance between the two empirical pixel distributions of that
variable, using GPU sort/quantile. Output has shape ``(V, D, D)`` with V the
number of variables and D the number of domains:

    {output_dir}/{mode}/Wasserstein_1D_{split}.npy
    {output_dir}/{mode}/Wasserstein_1D_{split}_{var}.png
    {output_dir}/{mode}/Wasserstein_1D_{split}_variables.json

The JSON sidecar records the channel-to-variable map and the domain ordering,
since both are needed to interpret the matrix and neither is implicit anymore
once ``--domains`` is used to subset.

Variables
---------
The 9 dynamic inputs (cape, cp, sp, tclw, tcw, tisr, tp, u, v) plus the
target variable (precipitation by default) are treated on equal footing. The
target is read from ``{split}_y.npy`` and appended as the last channel of the
output matrix. Disable with ``--target_var none``. The output shape is
therefore ``(len(input_vars) + (target_var != None), D, D)``.

Modes and transformations
-------------------------
raw
    Values straight from ``{domain}/{split}_{x,y}.npy``. ``convert_zarr_to_npy``
    already log-transforms tp, cp, precipitation (and mm-scales tp, cp), so
    "raw" here is pre-z-score, not pre-preprocessing.
normalized
    Each domain is z-scored channel-wise by its own ``stats.json`` before
    comparison. This matches the transformation applied by
    ``ClimateSRDatasetNPY`` at training time exactly: log on precip channels
    is baked into ``.npy`` upstream, and z-score is applied here. The
    precipitation target therefore enters the distance in log-standardised
    space — the same space the training loss is computed in.

Region selection
----------------
Pass ``--domains name1 name2 ...`` to restrict the computation to a subset.
With no ``--domains``, every subdirectory of ``--data_root`` that contains a
``stats.json`` is used.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm


# Must match data/convert_zarr_to_npy.py
DEFAULT_INPUT_VARS = ["cape", "cp", "sp", "tclw", "tcw", "tisr", "tp", "u", "v"]
DEFAULT_TARGET_VAR = "precipitation"


# --------------------------------------------------------------------------
#  Metric
# --------------------------------------------------------------------------


def wasserstein1d_gpu(a: np.ndarray, b: np.ndarray, device: torch.device) -> float:
    """1D Wasserstein-1 via empirical inverse-CDF matching on GPU."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    ta = torch.from_numpy(a).to(device)
    tb = torch.from_numpy(b).to(device)

    # equal-length: pointwise sort is the Monge coupling
    if ta.shape[0] == tb.shape[0]:
        ta, _ = torch.sort(ta)
        tb, _ = torch.sort(tb)
        return torch.mean(torch.abs(ta - tb)).item()

    # unequal length: align on a common quantile grid
    n = min(ta.shape[0], tb.shape[0])
    q = torch.linspace(0.0, 1.0, steps=n, device=device)
    ta = torch.quantile(ta, q)
    tb = torch.quantile(tb, q)
    return torch.mean(torch.abs(ta - tb)).item()


# --------------------------------------------------------------------------
#  Data loading from .npy
# --------------------------------------------------------------------------


def _load_channel(
    data_path: Path,
    filename: str,
    channel: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flat float32 pixel values for one channel from {filename}."""
    fp = data_path / filename
    x = np.load(fp, mmap_mode="r")  # (N, C, H, W)
    n = x.shape[0]
    k = min(n_samples, n)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    return np.asarray(x[idx, channel]).ravel().astype(np.float32)


def _load_domain_stats(data_path: Path) -> dict:
    with open(data_path / "stats.json") as f:
        return json.load(f)


def _zscore(flat: np.ndarray, var: str, stats: dict) -> np.ndarray:
    """
    Z-score with per-domain stats, matching ClimateSRDatasetNPY exactly. The
    log-transform on precip channels (tp, cp, precipitation, z) is baked into
    the .npy files by convert_zarr_to_npy, so no extra transform is applied
    here.
    """
    if var in stats:
        mean, std = stats[var]
    else:
        # fallback: on-the-fly z-score if the var is missing from stats.json
        # (e.g. a user-specified extra variable). Keeps the script robust.
        mean, std = float(flat.mean()), float(flat.std())
    return ((flat - mean) / (std + 1e-8)).astype(np.float32)


# --------------------------------------------------------------------------
#  Plotting
# --------------------------------------------------------------------------


def plot_matrix(matrix, title, labels, save_path):
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title, fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --------------------------------------------------------------------------
#  Device
# --------------------------------------------------------------------------


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("Warning: no hardware accelerator available, using CPU")
    return torch.device("cpu")


# --------------------------------------------------------------------------
#  Driver
# --------------------------------------------------------------------------


def _resolve_domains(data_root: Path, requested) -> list:
    """Return a validated list of domain names."""
    if requested:
        missing = [d for d in requested if not (data_root / d / "stats.json").exists()]
        if missing:
            raise FileNotFoundError(f"Requested domains have no stats.json under {data_root}: {missing}")
        return list(requested)
    discovered = sorted(d.name for d in data_root.iterdir() if d.is_dir() and (d / "stats.json").exists())
    if not discovered:
        raise FileNotFoundError(f"No valid domain subdirs found under {data_root}")
    return discovered


def _build_variables(input_vars, target_var, split):
    """
    Return (name, filename, channel_in_file) triples. Inputs come from
    {split}_x.npy (channels 0..C-1); target comes from {split}_y.npy
    (channel 0) if target_var is not None.
    """
    variables = [(v, f"{split}_x.npy", k) for k, v in enumerate(input_vars)]
    if target_var is not None:
        variables.append((target_var, f"{split}_y.npy", 0))
    return variables


def run(args: argparse.Namespace) -> None:
    device = _pick_device()
    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)

    domains = _resolve_domains(data_root, args.domains)
    print(f"Domains ({len(domains)}): {domains}")

    target_var = None if args.target_var.lower() == "none" else args.target_var
    variables = _build_variables(args.input_vars, target_var, args.split)
    print(f"Variables ({len(variables)}): {[v[0] for v in variables]}")

    n_v, n_d = len(variables), len(domains)

    # Stats are only needed in "normalized" mode.
    stats_cache = {d: (_load_domain_stats(data_root / d) if args.mode == "normalized" else None) for d in domains}

    print(
        f"Computing Wasserstein-1D on {device} " f"(mode={args.mode}, split={args.split}, n_samples={args.n_samples})"
    )
    mat = np.zeros((n_v, n_d, n_d), dtype=np.float64)
    rng = np.random.default_rng(args.seed)

    # Channel-outer loop bounds memory to one channel across all domains at a time.
    for k, (var, fname, ch_in_file) in enumerate(tqdm(variables, desc="Variables")):
        per_domain = {}
        for d in domains:
            try:
                flat = _load_channel(
                    data_root / d,
                    fname,
                    ch_in_file,
                    args.n_samples,
                    rng,
                )
                if args.mode == "normalized":
                    flat = _zscore(flat, var, stats_cache[d])
                per_domain[d] = flat
            except FileNotFoundError:
                print(f"  Missing {fname} for {d}: skipping {var}.")
                per_domain[d] = np.array([], dtype=np.float32)

        for i in range(n_d):
            for j in range(i + 1, n_d):
                w = wasserstein1d_gpu(
                    per_domain[domains[i]],
                    per_domain[domains[j]],
                    device,
                )
                mat[k, i, j] = w
                mat[k, j, i] = w

    out = out_root / args.mode
    out.mkdir(parents=True, exist_ok=True)

    mat_path = out / f"Wasserstein_1D_{args.split}.npy"
    np.save(mat_path, mat)

    meta_path = out / f"Wasserstein_1D_{args.split}_variables.json"
    meta_path.write_text(
        json.dumps(
            {"variables": [v[0] for v in variables], "domains": domains},
            indent=2,
        )
    )

    print(f"Saved matrix {mat.shape} to {mat_path}")
    print(f"Saved variable/domain ordering to {meta_path}")

    if not args.no_plots:
        for k, (var, _, _) in enumerate(variables):
            plot_matrix(
                mat[k],
                f"Wasserstein-1D - {var} ({args.split}, {args.mode})",
                domains,
                out / f"Wasserstein_1D_{args.split}_{var}.png",
            )
        print(f"Saved {n_v} heatmaps to {out}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-variable W1 shift matrices on RainShift .npy data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", required=True, help="directory containing per-domain .npy subdirs")
    p.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="subset of domain names to compare; if omitted, all "
        "subdirs of --data_root containing stats.json are used",
    )
    p.add_argument(
        "--input_vars",
        nargs="+",
        default=DEFAULT_INPUT_VARS,
        help="dynamic input channel names; must match .npy ordering",
    )
    p.add_argument(
        "--target_var",
        default=DEFAULT_TARGET_VAR,
        help="target variable read from {split}_y.npy (channel 0); " "pass 'none' to skip",
    )
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument(
        "--mode",
        default="normalized",
        choices=["raw", "normalized"],
        help="'raw' uses .npy values as-is (log for precip already "
        "baked in); 'normalized' adds per-domain z-score from "
        "stats.json, matching training-time normalisation",
    )
    p.add_argument("--n_samples", type=int, default=1000, help="number of 2D frames per domain (all pixels used)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent),
        help="root output dir; mode subdir is created inside",
    )
    p.add_argument("--no_plots", action="store_true", help="skip saving per-variable heatmap PNGs")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
