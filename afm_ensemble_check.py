"""
Is an AFM ensemble's CRPS gain skill, or spread added to a dry forecast?

PLAN.md "Read 2026-11-04": FDA recovers 0.38 / 0.52 of the source-only ->
oracle CRPS budget into europe_west (horn, melanesia), also in mm. But the
source-only ensembles there are near-dry and collapsed (spread 0.012 mm), and
CRPS rewards any spread where it rains: melanesia->europe FDA has 2.5x the
in-domain spread. This scores, on the SAME 1-in-STRIDE target test subset:

  every --run          AFM ensemble, M members, STEPS midpoint steps, the
                       same sampler, autocast and generator seed as
                       evaluate.py (so CRPS is close to, not equal to, the
                       full-split value: the subset differs)
  zero                 the all-zero forecast (CRPS = MAE = field mean)
  climatology          M observed target fields from other subset times at
                       least MIN_GAP hours away: no information about the
                       day, the target's marginal exactly (uses target labels;
                       a reference, not a method)
  ERA5 tp              with --tp_ref: the ERA5 `tp` input channel, decoded to
                       mm/h and bicubic-upsampled 80 -> 200 IN MM SPACE,
                       clipped at 0, as M identical members (CRPS = MAE).
                       The input-conditioned trivial forecast: rain where
                       ERA5 rains, at ERA5 magnitudes. It uses no target
                       labels and no training. Usable as SO_LABEL in --dress
                       under the label "ERA5 tp".
  <so> + clim x k      for each --dress SO_LABEL:LABEL, the SO_LABEL members
                       plus k times the climatology members, k chosen so the
                       subset-mean spread_mm equals LABEL's. Information-free
                       rain added to the source-only forecast at the same
                       spread as the method. If it scores as well as LABEL,
                       LABEL's gain is not evidence of adaptation.

Per forecast (summary.csv): pooled-member mean, wet fraction (> 0.1 mm/h),
p99 and max; wet fraction of the ensemble mean; spread_mm (deprecated ddof-0
definition, as in results.csv); crps_mm, split into pixels observed wet and
dry; crps_log_e01; ssr_log_e01. --rank_npz label=per_sample_loss.npz pools the
target rank histograms evaluate.py already saved (rank_hist.csv).

Usage (inside a job, see scripts/afm_ensemble_check.sh):
  afm_ensemble_check.py --data_root ... --output_dir ... --target europe_west \
      --run "SO horn=/.../afm_seed42/afm_horn-of-africa__to__europe_west__none" ... \
      --dress "SO horn:FDA horn"
--output_dir is refused if it exists.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import inverse_transform
import torch.nn.functional as F

from data.dataset_zarr import DEFAULT_INPUT_VARS, ClimateSRDatasetZarr
from evaluate import (FLOORS_MM, _infer_model_type, load_model, load_stats, parse_exp_name, run_base_features,
                      run_finished, target_checkpoint)

WET_MM = 0.1
FLOOR = FLOORS_MM["e01"]
TP_LABEL = "ERA5 tp"
TP_CHANNEL = DEFAULT_INPUT_VARS.index("tp")


def crps_pixels(ens, y):
    """prob_metrics.crps_fair without the pixel mean: (B, ...) per pixel."""
    M = ens.shape[1]
    skill = (ens - y.unsqueeze(1)).abs().mean(dim=1)
    xs, _ = ens.sort(dim=1)
    k = torch.arange(1, M + 1, dtype=ens.dtype)
    w = (2 * k - M - 1).view(1, M, *([1] * (ens.ndim - 2)))
    return skill - 2 * (w * xs).sum(dim=1) / (2 * M * (M - 1))


def spread(ens, chunk=128):
    """Subset mean of evaluate.py's spread_mm (ddof 0, pixel mean)."""
    return float(torch.cat([ens[i:i + chunk].std(dim=1, unbiased=False).flatten(1).mean(1)
                            for i in range(0, len(ens), chunk)]).mean())


def scores(ens, obs, chunk=128):
    """ens (N, M, H, W), obs (N, H, W), both mm, CPU float32."""
    M = ens.shape[1]
    c_sum = c_wet = c_dry = cl_sum = 0.0
    v_log = e_log = 0.0
    n_wet = int((obs > WET_MM).sum())
    n_pix = obs.numel()
    for i in range(0, len(ens), chunk):
        e, y = ens[i:i + chunk], obs[i:i + chunk]
        c = crps_pixels(e, y)
        wet = y > WET_MM
        c_sum += float(c.sum())
        c_wet += float(c[wet].sum())
        c_dry += float(c[~wet].sum())
        le, ly = torch.log(e + FLOOR), torch.log(y + FLOOR)
        cl_sum += float(crps_pixels(le, ly).sum())
        v_log += float(le.var(dim=1, unbiased=True).sum())
        e_log += float(((le.mean(dim=1) - ly) ** 2).sum())
    flat = ens.flatten()
    # np.percentile on ~10^9 values is slow and memory-hungry; a strided
    # sample of the pooled members is enough for p99.
    p99 = float(np.percentile(flat[:: max(1, flat.numel() // 20_000_000)].numpy(), 99))
    return {
        "mean_mm": float(ens.mean()),
        "wet_frac": float((ens > WET_MM).float().mean()),
        "ensmean_wet_frac": float((ens.mean(dim=1) > WET_MM).float().mean()),
        "p99_mm": p99,
        "max_mm": float(ens.max()),
        "spread_mm": spread(ens),
        "crps_mm": c_sum / n_pix,
        "crps_mm_obs_wet": c_wet / max(n_wet, 1),
        "crps_mm_obs_dry": c_dry / max(n_pix - n_wet, 1),
        "crps_log_e01": cl_sum / n_pix,
        "ssr_log_e01": float(np.sqrt((M + 1) / M * v_log / n_pix) / np.sqrt(e_log / n_pix)),
    }


def climatology(obs, idx, M, min_gap, rng):
    """M fields per sample, drawn from subset times >= min_gap hours away."""
    out = torch.empty(len(obs), M, *obs.shape[1:])
    for t in range(len(obs)):
        pool = np.flatnonzero(np.abs(idx - idx[t]) >= min_gap)
        out[t] = obs[torch.from_numpy(rng.choice(pool, size=M, replace=False))]
    return out


def match_k(base, clim, target_spread, iters=40):
    """k >= 0 with spread(base + k * clim) = target_spread, by bisection
    (spread grows monotonically in k once k * clim dominates base)."""
    lo, hi = 0.0, 1.0
    while spread(base + hi * clim) < target_spread:
        hi *= 2
        if hi > 1e4:
            raise SystemExit(f"cannot reach spread {target_spread:.4f} by dressing")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if spread(base + mid * clim) < target_spread else (lo, mid)
    return 0.5 * (lo + hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Zarr RainShift root (read-only)")
    p.add_argument("--output_dir", required=True, help="refused if it exists")
    p.add_argument("--target", required=True)
    p.add_argument("--run", action="append", required=True, help="label=afm_run_dir, repeatable")
    p.add_argument("--dress", action="append", default=[], help="SO_LABEL:LABEL, repeatable")
    p.add_argument("--rank_npz", action="append", default=[], help="label=per_sample_loss.npz, repeatable")
    p.add_argument("--stride", type=int, default=10, help="keep every STRIDE-th test sample")
    p.add_argument("--n_ensemble", type=int, default=8)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--min_gap", type=int, default=168, help="climatology draws at least this many hours away")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--tp_ref", action="store_true", help="add the upsampled ERA5 tp forecast")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; not overwriting.")
    data_root = Path(args.data_root)
    device = torch.device("cuda")
    M = args.n_ensemble

    runs = []
    for spec in args.run:
        label, run_dir = spec.split("=", 1)
        run_dir = Path(run_dir)
        if not run_finished(run_dir):
            raise SystemExit(f"{run_dir} is not finished.")
        if _infer_model_type(run_dir.name) != "afm":
            raise SystemExit(f"{run_dir.name} is not an AFM run")
        src, tgt, method = parse_exp_name(run_dir.name)
        if tgt != args.target:
            raise SystemExit(f"{run_dir.name}: target {tgt}, expected {args.target}")
        ckpt = target_checkpoint(run_dir / "best.pt", method)
        model = load_model("afm", str(ckpt), device, run_base_features(run_dir, 32))
        model.eval()
        runs.append(dict(label=label, src=src, model=model, ckpt=str(ckpt)))
        print(f"{label}: {ckpt} (stats {src})")
    labels = [r["label"] for r in runs]
    if TP_LABEL in labels:
        raise SystemExit(f"--run label {TP_LABEL!r} is reserved for --tp_ref")
    for d in args.dress:
        a, b = d.split(":", 1)
        if b not in labels or not (a in labels or (a == TP_LABEL and args.tp_ref)):
            raise SystemExit(f"--dress {d}: both labels must be --run labels ({TP_LABEL!r} needs --tp_ref)")

    # One loader per stats domain, iterated in lockstep as in compare_fields.py:
    # unshuffled test split, num_workers=1.
    stats_domains = sorted({r["src"] for r in runs})
    stats = {d: load_stats(str(data_root / d)) for d in stats_domains}
    loaders = [DataLoader(ClimateSRDatasetZarr(str(data_root / args.target), "test", stats=stats[d]),
                          batch_size=args.batch_size, shuffle=False, num_workers=1)
               for d in stats_domains]

    gens = {r["label"]: torch.Generator(device=device).manual_seed(1) for r in runs}
    obs, idx_kept = [], []
    members = {l: [] for l in labels + ([TP_LABEL] if args.tp_ref else [])}
    pos = 0
    with torch.no_grad():
        for batches in zip(*loaders):
            n = batches[0][0].shape[0]
            keep = [i for i in range(n) if (pos + i) % args.stride == 0]
            pos += n
            if not keep:
                continue
            by_domain = {d: (x[keep].to(device), s[keep].to(device), y[keep])
                         for d, (x, s, y) in zip(stats_domains, batches)}
            d0 = stats_domains[0]
            y_mm = inverse_transform(by_domain[d0][2][:, 0].numpy(), "precipitation", stats[d0])
            for d in stats_domains[1:]:
                y_alt = inverse_transform(by_domain[d][2][:, 0].numpy(), "precipitation", stats[d])
                if not np.allclose(y_alt, y_mm, rtol=1e-3, atol=1e-3):
                    raise SystemExit(f"loaders out of step at sample {pos}: {d} observation differs")
            obs.append(torch.from_numpy(y_mm.astype(np.float32)))
            idx_kept.extend(pos - n + i for i in keep)
            if args.tp_ref:
                x0 = by_domain[d0][0]
                if x0.shape[-1] == y_mm.shape[-1]:
                    raise SystemExit("inputs already upsampled by the loader; tp_ref expects 80x80")
                tp_mm = inverse_transform(x0[:, TP_CHANNEL].float().cpu().numpy(), "tp", stats[d0])
                tp_up = F.interpolate(torch.from_numpy(tp_mm.astype(np.float32))[:, None],
                                      size=y_mm.shape[-2:], mode="bicubic", align_corners=False)
                tp_up = tp_up.clamp_min(0.0)  # (k, 1, 200, 200), mm/h
                members[TP_LABEL].append(tp_up.expand(-1, M, -1, -1).contiguous())
            for r in runs:
                x, s, _ = by_domain[r["src"]]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    smp = r["model"].sample(x, s, n_samples=M, steps=args.steps, generator=gens[r["label"]])
                smp = smp[:, :, 0].float().cpu().numpy()
                mm = np.stack([inverse_transform(smp[:, i], "precipitation", stats[r["src"]])
                               for i in range(M)], axis=1)
                members[r["label"]].append(torch.from_numpy(mm.astype(np.float32)))
    obs = torch.cat(obs)
    idx_kept = np.asarray(idx_kept)
    members = {k: torch.cat(v) for k, v in members.items()}
    print(f"{len(obs)} samples kept of {pos} (stride {args.stride})")

    forecasts = dict(members)
    forecasts["zero"] = torch.zeros(len(obs), 2, *obs.shape[1:])
    clim = climatology(obs, idx_kept, M, args.min_gap, np.random.default_rng(0))
    forecasts["climatology"] = clim
    dress_k = {}
    for d in args.dress:
        so, meth = d.split(":", 1)
        if spread(members[meth]) <= spread(members[so]):
            print(f"--dress {d}: {meth} spread <= {so} spread; nothing to dress, skipped")
            continue
        k = match_k(members[so], clim, spread(members[meth]))
        name = f"{so} + clim x k (spread of {meth})"
        forecasts[name] = members[so] + k * clim
        dress_k[name] = k
        print(f"{name}: k = {k:.4f}")

    out_dir.mkdir(parents=True)
    rows = []
    for name, ens in forecasts.items():
        sc = scores(ens, obs)
        ck = next((r["ckpt"] for r in runs if r["label"] == name), "")
        rows.append({"forecast": name, "checkpoint": ck, "dress_k": dress_k.get(name, ""), **sc})
        print(name, {k: round(v, 4) for k, v in sc.items()})
    rows.append({"forecast": "IMERG obs", "checkpoint": "", "dress_k": "",
                 "mean_mm": float(obs.mean()), "wet_frac": float((obs > WET_MM).float().mean())})
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if args.rank_npz:
        with open(out_dir / "rank_hist.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "hist", "n_pixels", "fractions (rank 0 .. M)"])
            for spec in args.rank_npz:
                label, path = spec.split("=", 1)
                with np.load(path) as z:
                    for key in ("tgt_rank_hist_e01", "tgt_rank_hist_wet_e01"):
                        h = z[key].sum(axis=0).astype(np.float64)
                        w.writerow([label, key, int(h.sum()), " ".join(f"{v:.4f}" for v in h / h.sum())])
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
