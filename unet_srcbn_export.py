"""UNet phase-2 runs with source-only BatchNorm statistics, as a side tree.

UNet counterpart of afm_srcbn_export.py (PLAN.md "Read 2026-11-09").
train_unet.train_one_epoch runs the source batch and then the target batch
through the whole model in train mode for coral, mmd, mmd_ms, dann, joint_ot
and spectral, so every BatchNorm running statistic used at evaluation is a
source/target mixture (momentum 0.1, two updates per step). In AFM, removing
the target's share recovered 70-100 % of the phase-2 source damage. This asks
the same of the UNet, and -- once scored with evaluate.py -- whether any target
gain survives.

Default-weight DANN is the cleanest case: audit A2 found its encoder is
barely pushed (the reversed gradient stays ~5e-4 of the task gradient), so
those runs are close to "source-only + BatchNorm mixing".

For each run: load best.pt, reset every BatchNorm (momentum=None, cumulative
average), run --n_batches SOURCE training batches through the full model in
train mode under no_grad at the run's batch size (torch seed 1000), write
<out_root>/<tag>/best.pt, then config.json (the run's config + srcbn_*
fields; written last, so evaluate.py sees a finished run). Only BatchNorm
buffers change. Refuses existing run directories and methods that never
forward the target (none, fda, adabn).

--control (MILESTONES M1, "Source-BN control") inverts the method check: it
accepts only none and fda, which never forward the target, so their trained
BN statistics are already source-only (an EMA, momentum 0.1, over training;
FDA's are over FDA-styled source batches). Re-estimating them the same way
measures what the re-estimation alone does -- the cumulative average over
200 batches of the final weights against the training EMA -- which the
target-forward rows must be read against. adabn is refused either way: its
BN is meant to be the target's. config.json records srcbn_control.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import load_domain_stats
from data.dataset_zarr import ClimateSRDatasetZarr
from evaluate import load_model

TARGET_FORWARD = {"coral", "mmd", "mmd_ms", "dann", "joint_ot", "spectral"}
CONTROL = {"none", "fda"}


@torch.no_grad()
def reestimate_source(model, cfg, device, n_batches):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    stats = load_domain_stats(cfg["source_path"])
    loader = DataLoader(
        ClimateSRDatasetZarr(cfg["source_path"], "train", stats=stats, val_chunks=cfg["val_chunks"],
                             shuffle_buffer_chunks=cfg["shuffle_buffer_chunks"]),
        cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True)
    torch.manual_seed(1000)
    model.train()
    n = 0
    for x, s, _ in loader:
        if n >= n_batches:
            break
        model(x.to(device, non_blocking=True), s.to(device, non_blocking=True))
        n += 1
    model.eval()
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="phase-2 run dirs <src>__to__<tgt>__<method>")
    p.add_argument("--out_root", required=True)
    p.add_argument("--n_batches", type=int, default=200)
    p.add_argument("--control", action="store_true",
                   help="accept only none/fda runs (source-BN control), see the docstring")
    args = p.parse_args()

    device = torch.device("cuda")
    out_root = Path(args.out_root)
    runs = [Path(r) for r in args.runs]
    allowed = CONTROL if args.control else TARGET_FORWARD
    bad = [r.name for r in runs if r.name.split("__")[-1] not in allowed]
    if bad:
        if args.control:
            raise SystemExit(f"ERROR: --control takes only {sorted(CONTROL)} runs; got {bad}.")
        raise SystemExit(f"ERROR: {bad} do not forward the target in training; nothing to remove.")
    if args.control:
        # The source-only tag __none is shared with the oracle (CLAUDE.md):
        # a jointly trained run here would be a different control.
        joint = [r.name for r in runs
                 if json.loads((r / "config.json").read_text()).get("joint_training")]
        if joint:
            raise SystemExit(f"ERROR: {joint} are joint (oracle) runs, not source-only.")
    clash = [r.name for r in runs if (out_root / r.name).exists()]
    if clash:
        raise SystemExit(f"ERROR: {clash} already exist in {out_root}; not overwriting.")

    for run in runs:
        cfg = json.loads((run / "config.json").read_text())
        model = load_model("unet", str(run / "best.pt"), device, cfg.get("base_features") or 32)
        n = reestimate_source(model, cfg, device, args.n_batches)
        d = out_root / run.name
        d.mkdir(parents=True)
        torch.save(model.state_dict(), d / "best.pt")
        (d / "config.json").write_text(json.dumps(
            {**cfg, "srcbn_from": str(run / "best.pt"), "srcbn_n_batches": n,
             "srcbn_control": args.control}, indent=2))
        print(f"  {run.name}: {n} source batches -> {d}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
