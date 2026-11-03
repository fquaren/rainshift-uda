"""
Side-by-side precipitation fields of several finished UNet runs on the SAME
target test timesteps, for one (source, target) pair.

plotting.py draws one run at a time; this puts source-only, the oracle, the
in-domain model and UDA variants next to the observation. Every run reads the
target test split normalised with ITS OWN source domain's stats (the in-domain
model uses the target's stats), one loader per stats domain, iterated in
lockstep: the test split is unshuffled and read with num_workers=1, so the
loaders stay in time order together.

One pass over every STRIDE-th test sample keeps, per run, the predictions in
mm. From that subset it writes:
  fields.png/.pdf   rows = timesteps at the 50/90/99th percentile and the
                    maximum of the observed field mean; columns = ERA5 tp
                    (input), IMERG observation, then each run
  climatology.png   mean precipitation and wet frequency (> 0.1 mm/h) maps
  histogram.png     pixel distribution of log10(P + 0.01)
  summary.csv       per run: field mean, wet fraction, p99, max, mse_log_e01
--output_dir is refused if it exists.

Usage (inside a job, see scripts/run_compare_fields.sh):
  compare_fields.py --data_root ... --output_dir ... --source europe_west \
      --target melanesia --run "source-only=/.../unet_seed42/europe_west__to__melanesia__none" ...
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from torch.utils.data import DataLoader

from data.dataset import inverse_transform
from data.dataset_zarr import ClimateSRDatasetZarr
from evaluate import (_infer_model_type, load_model, load_stats, parse_exp_name, run_base_features,
                      run_finished, target_checkpoint)
from plotting import DEFAULT_INPUT_VARS, _predict

WET_MM = 0.1
FLOOR_MM = 0.1
QUANTILES = (0.5, 0.9, 0.99)


def _to_mm(arr, stats):
    return inverse_transform(arr, "precipitation", stats).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Zarr RainShift root (read-only)")
    p.add_argument("--output_dir", required=True, help="refused if it exists")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--run", action="append", required=True,
                   help="label=run_dir, repeatable; column order follows the flags")
    p.add_argument("--stride", type=int, default=10, help="keep every STRIDE-th test sample")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; not overwriting.")
    data_root = Path(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = []
    for spec in args.run:
        label, run_dir = spec.split("=", 1)
        run_dir = Path(run_dir)
        if not run_finished(run_dir):
            raise SystemExit(f"{run_dir} is not finished.")
        model_type = _infer_model_type(run_dir.name)
        src, tgt, method = parse_exp_name(run_dir.name)
        if tgt != args.target:
            raise SystemExit(f"{run_dir.name}: target {tgt}, expected {args.target}")
        ckpt = target_checkpoint(run_dir / "best.pt", method)
        if ckpt is None:
            raise SystemExit(f"{run_dir.name}: adabn run without best_adabn.pt")
        model = load_model(model_type, str(ckpt), device, run_base_features(run_dir, 32))
        model.eval()
        runs.append(dict(label=label, src=src, model=model, model_type=model_type, ckpt=str(ckpt)))
        print(f"{label}: {ckpt} (stats {src})")

    # One loader per stats domain; the source's comes first and supplies the
    # observation and the ERA5 tp panel.
    stats_domains = [args.source] + sorted({r["src"] for r in runs} - {args.source})
    stats = {d: load_stats(str(data_root / d)) for d in stats_domains}
    loaders = [DataLoader(ClimateSRDatasetZarr(str(data_root / args.target), "test", stats=stats[d]),
                          batch_size=args.batch_size, shuffle=False, num_workers=1)
               for d in stats_domains]

    tp_idx = DEFAULT_INPUT_VARS.index("tp")
    obs, era_tp, idx_kept = [], [], []
    preds = {r["label"]: [] for r in runs}
    pos = 0
    with torch.no_grad():
        for batches in zip(*loaders):
            n = batches[0][0].shape[0]
            keep = [i for i in range(n) if (pos + i) % args.stride == 0]
            pos += n
            if not keep:
                continue
            by_domain = {}
            for d, (x, s, y) in zip(stats_domains, batches):
                by_domain[d] = (x[keep].to(device), s[keep].to(device), y[keep])
            x0, _, y0 = by_domain[args.source]
            y_mm = _to_mm(y0[:, 0].numpy(), stats[args.source])
            # The observation must not depend on which stats normalised it.
            for d in stats_domains[1:]:
                y_alt = _to_mm(by_domain[d][2][:, 0].numpy(), stats[d])
                if not np.allclose(y_alt, y_mm, rtol=1e-3, atol=1e-3):
                    raise SystemExit(f"loaders out of step at sample {pos}: {d} observation differs")
            obs.append(y_mm)
            era_tp.append(inverse_transform(x0[:, tp_idx].cpu().numpy(), "tp", stats[args.source]))
            idx_kept.extend(pos - n + i for i in keep)
            for r in runs:
                x, s, _ = by_domain[r["src"]]
                pr = _predict(r["model"], r["model_type"], x, s)[:, 0].cpu().numpy()
                preds[r["label"]].append(_to_mm(pr, stats[r["src"]]))
    obs = np.concatenate(obs)
    era_tp = np.concatenate(era_tp)
    preds = {k: np.concatenate(v) for k, v in preds.items()}
    print(f"{len(obs)} samples kept of {pos} (stride {args.stride})")

    out_dir.mkdir(parents=True)
    labels = [r["label"] for r in runs]
    fields = {"IMERG obs": obs, **preds}

    # Summary per run on the subset.
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "checkpoint", "mean_mm", "wet_frac_0.1", "p99_mm", "max_mm", "mse_log_e01"])
        for name, a in fields.items():
            ck = next((r["ckpt"] for r in runs if r["label"] == name), "")
            mse = np.mean((np.log(a + FLOOR_MM) - np.log(obs + FLOOR_MM)) ** 2)
            w.writerow([name, ck, f"{a.mean():.4f}", f"{(a > WET_MM).mean():.4f}",
                        f"{np.percentile(a, 99):.3f}", f"{a.max():.2f}", f"{mse:.4f}"])

    # Fields at fixed quantiles of the observed field mean.
    fm = obs.mean(axis=(1, 2))
    order = np.argsort(fm)
    rows = [order[int(q * (len(fm) - 1))] for q in QUANTILES] + [order[-1]]
    row_names = [f"p{int(q * 100)}" for q in QUANTILES] + ["max"]
    ncol = 2 + len(labels)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.6 * ncol, 2.7 * len(rows)), squeeze=False)
    for i, (k, rn) in enumerate(zip(rows, row_names)):
        panels = [("ERA5 tp (input)", era_tp[k]), ("IMERG obs", obs[k])] + [(l, preds[l][k]) for l in labels]
        vmax = max(1.0, max(np.percentile(a, 99.9) for _, a in panels))
        for j, (title, a) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(np.clip(a, FLOOR_MM, None), cmap="Blues", norm=LogNorm(FLOOR_MM, vmax),
                           origin="upper", interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            txt = f"mean {a.mean():.2f}"
            if j >= 2:
                txt += f"\nwet {(a > WET_MM).mean():.0%}"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.7, lw=0, pad=1))
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{rn}  (t={idx_kept[k]})", fontsize=8)
        fig.colorbar(im, ax=axes[i, :].tolist(), fraction=0.012, pad=0.01, label="mm/h")
    fig.suptitle(f"{args.source} → {args.target}, test split, same timesteps (rows: observed field-mean "
                 f"percentile in the 1-in-{args.stride} subset)", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fields.{ext}", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Climatology maps on the subset.
    names = list(fields)
    fig, axes = plt.subplots(2, len(names), figsize=(2.6 * len(names), 5.6), squeeze=False)
    means = {n: fields[n].mean(0) for n in names}
    wets = {n: (fields[n] > WET_MM).mean(0) for n in names}
    vmax_m = np.percentile(means["IMERG obs"], 99.5) * 1.5
    for j, n in enumerate(names):
        im0 = axes[0, j].imshow(means[n], cmap="viridis", vmin=0, vmax=vmax_m)
        im1 = axes[1, j].imshow(wets[n], cmap="magma", vmin=0, vmax=max(0.05, wets["IMERG obs"].max()))
        axes[0, j].set_title(f"{n}\nmean {means[n].mean():.3f}", fontsize=8)
        axes[1, j].set_title(f"wet {wets[n].mean():.1%}", fontsize=8)
        for ax in axes[:, j]:
            ax.set_xticks([]), ax.set_yticks([])
    axes[0, 0].set_ylabel("mean P (mm/h)")
    axes[1, 0].set_ylabel(f"P(P > {WET_MM} mm/h)")
    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.012, pad=0.01)
    fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.012, pad=0.01)
    fig.suptitle(f"{args.source} → {args.target}: time-mean and wet frequency, 1-in-{args.stride} test "
                 f"subset (colour scales clipped to the observation)", fontsize=10)
    fig.savefig(out_dir / "climatology.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Pixel distribution.
    bins = np.linspace(-2, 2.5, 91)
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in names:
        h, _ = np.histogram(np.log10(fields[n] + 0.01), bins=bins, density=True)
        ax.plot(0.5 * (bins[1:] + bins[:-1]), h, lw=2.2 if n == "IMERG obs" else 1.2,
                color="k" if n == "IMERG obs" else None, label=n)
    ax.set_yscale("log")
    ax.axvline(np.log10(WET_MM + 0.01), color="grey", ls=":", lw=1)
    ax.set_xlabel("log10(P + 0.01 mm/h)")
    ax.set_ylabel("density")
    ax.set_title(f"{args.source} → {args.target}: pixel distribution", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "histogram.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
