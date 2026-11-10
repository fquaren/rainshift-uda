"""Phase-2 AFM runs with source-only BatchNorm statistics, as a side tree.

The BN check (afm_bn_check.py, PLAN.md "Read 2026-11-09") showed that
re-estimating every BatchNorm on SOURCE training batches removes 70-100 % of
the phase-2 source degradation of the target-forward methods. The open half
is the target: does the target gain survive once the target's contribution to
the running statistics is removed?

This script builds the model whose target score answers that, and nothing
else. For each phase-2 run directory it:
  1. loads best.pt and re-estimates all BatchNorm statistics exactly as the
     p2_src arm of afm_bn_check.py (reset, momentum=None, --n_batches source
     training batches at the run's batch size, train-mode full forward,
     sigma_z restored, torch seed 1000);
  2. scores source validation CRPS on the trainer's subset, a check against
     the p2_src arm (spectral pairs) -- recorded, not enforced;
  3. writes <out_root>/<tag>/best.pt and then config.json (the run's config
     plus srcbn_* fields), so eval_array.sh / evaluate.py see a finished run.

Only BatchNorm buffers differ from the source checkpoint. Score with
  EXP_SUFFIX=_srcbn OUT_SUFFIX=_srcbn SEED=42 MODEL=afm TREE=none eval_array.sh
Refuses to write into an existing run directory.
"""

import argparse
import json
from pathlib import Path

import torch

from afm_bn_check import load_afm, reestimate_source, score
from argparse import Namespace
from data.dataset import load_domain_stats
from train_afm import build_crps_loader


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="phase-2 run directories afm_<src>__to__<tgt>__<method>")
    p.add_argument("--out_root", required=True, help="side tree, e.g. results_rainshift_uda/afm_seed42_srcbn")
    p.add_argument("--n_batches", type=int, default=200)
    p.add_argument("--max_batches", type=int, default=None, help="smoke: validation batches in the check score")
    args = p.parse_args()

    device = torch.device("cuda")
    out_root = Path(args.out_root)
    runs = [Path(r) for r in args.runs]
    clash = [r.name for r in runs if (out_root / r.name).exists()]
    if clash:
        raise SystemExit(f"ERROR: {clash} already exist in {out_root}; not overwriting.")
    bad = [r.name for r in runs if r.name.split("__")[-1] in ("none", "fda", "adabn")]
    if bad:
        raise SystemExit(f"ERROR: {bad} do not forward the target in training; nothing to remove.")

    for run in runs:
        model, cfg = load_afm(run, device)
        src_path = cfg["source_path"]
        stats = load_domain_stats(src_path)
        n = reestimate_source(model, src_path, stats, cfg, device, args.n_batches)
        loader = build_crps_loader(
            Namespace(val_crps_chunks=cfg["val_crps_chunks"], val_chunks=cfg["val_chunks"],
                      shuffle_buffer_chunks=cfg["shuffle_buffer_chunks"], num_workers=cfg["num_workers"]),
            src_path, stats, cfg["batch_size"])
        r = score(model, loader, device, cfg["val_crps_members"], cfg["val_crps_steps"],
                  max_batches=args.max_batches)

        d = out_root / run.name
        d.mkdir(parents=True)
        torch.save(model.state_dict(), d / "best.pt")
        cfg_out = {**cfg, "srcbn_from": str(run / "best.pt"), "srcbn_n_batches": n,
                   "srcbn_val_crps": r["crps"], "srcbn_val_crps_saved_stats": cfg["best_val_loss"]}
        (d / "config.json").write_text(json.dumps(cfg_out, indent=2))   # last: marks the run finished
        print(f"  {run.name}: {n} batches, source val CRPS {r['crps']:.5f} "
              f"(saved stats {cfg['best_val_loss']:.5f}) -> {d}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
