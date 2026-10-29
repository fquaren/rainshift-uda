"""Few-shot label curve (PLAN.md P16): fine-tune a source-only UNet on N
labelled target samples and keep the last weights.

The recipe is fixed in advance and selects nothing, because selecting on the
target validation split would make the run an oracle:
- start from the source-only checkpoint unet_seed<S>/{src}__to__{tgt}__none/best.pt
  (the phase-1 model of src, materialised under the pair's name);
- N = n_chunks x 200 samples of the target TRAINING split, the chunks spread
  evenly over the split with a seed-dependent phase (the first N would be one
  season), normalised with SOURCE statistics like every other dataset;
- a fixed budget of --steps optimiser steps at the pair's base LR / --lr_div,
  constant, AdamW with the base weight decay, batch as in phase 1, fp32, MSE
  in the training space;
- the weights after the last step are saved as last.pt and scored by
  `evaluate.py batch --only <tag> --ckpt_name last.pt`.
RainShift's own fine-tuning used 2,000 samples at LR / 5 (Harder et al. 2026).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import load_domain_stats
from data.dataset_zarr import ClimateSRDatasetZarr
from models.unet import DualEncoderUNet


def spread_chunks(chunk_ids, n, seed):
    """n chunks spread evenly over chunk_ids, with a random phase in
    [0, stride) drawn from `seed`, so each seed takes a different draw."""
    if n >= len(chunk_ids):
        return list(chunk_ids)
    stride = len(chunk_ids) / n
    phase = np.random.default_rng(seed).uniform(0, stride)
    idx = np.floor(phase + stride * np.arange(n)).astype(int)
    return [chunk_ids[i] for i in idx]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source_path", required=True)
    p.add_argument("--target_path", required=True)
    p.add_argument("--init_dir", required=True, help="unet_seed<S>/ holding the source-only run and base_hp/")
    p.add_argument("--output_dir", required=True, help="unet_seed<S>_fewshot/")
    p.add_argument("--n_chunks", type=int, required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr_div", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--val_chunks", type=int, default=88)
    p.add_argument("--shuffle_buffer_chunks", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src, tgt = Path(args.source_path).name, Path(args.target_path).name
    tag = f"{src}__to__{tgt}__fs{args.n_chunks}"
    out = Path(args.output_dir) / tag
    if out.exists():
        raise SystemExit(f"{out} exists. Not overwriting.")
    init = Path(args.init_dir) / f"{src}__to__{tgt}__none" / "best.pt"
    base = json.loads((Path(args.init_dir) / "base_hp" / f"{src}__to__{tgt}.json").read_text())
    lr, wd, bs = base["lr"] / args.lr_div, base["weight_decay"], base["batch_size"]

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")

    # Source stats normalise the target too: the model lives in source space.
    src_stats = load_domain_stats(args.source_path)
    ds = ClimateSRDatasetZarr(args.target_path, "train", stats=src_stats, val_chunks=args.val_chunks,
                              shuffle_buffer_chunks=args.shuffle_buffer_chunks)
    n_avail = len(ds.chunk_ids)
    ds.chunk_ids = spread_chunks(ds.chunk_ids, args.n_chunks, args.seed)
    nw = min(args.num_workers, len(ds.chunk_ids))
    loader = DataLoader(ds, bs, num_workers=nw, pin_memory=True, persistent_workers=nw > 0)

    model = DualEncoderUNet(dynamic_channels=9, static_channels=2, out_channels=1, base_features=32).to(device)
    model.load_state_dict(torch.load(init, map_location=device, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    print(f"{tag}: {len(ds.chunk_ids)} of {n_avail} target train chunks {ds.chunk_ids[:5]}..., "
          f"{args.steps} steps, lr {lr:g}, wd {wd:g}, batch {bs}, init {init}")

    out.mkdir(parents=True)
    model.train()
    step, passes, t0, log, run = 0, 0, time.time(), [], 0.0
    while step < args.steps:
        passes += 1
        for x, s, y in loader:
            x, s, y = (t.to(device, non_blocking=True) for t in (x, s, y))
            loss = crit(model(x, s), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            run += loss.item()
            if step % 50 == 0 or step == args.steps:
                log.append({"step": step, "pass": passes, "train_loss": run / (step - 50 * ((step - 1) // 50)),
                            "time_s": round(time.time() - t0, 1)})
                print(log[-1])
                run = 0.0
            if step >= args.steps:
                break
    if not np.isfinite(log[-1]["train_loss"]):
        raise RuntimeError(f"{tag}: non-finite training loss; last.pt not written")

    torch.save(model.state_dict(), out / "last.pt")
    with open(out / "history.csv", "w") as f:
        f.write(",".join(log[0]) + "\n")
        for h in log:
            f.write(",".join(str(v) for v in h.values()) + "\n")
    cfg = {**vars(args), "init": str(init), "lr": lr, "weight_decay": wd, "batch_size": bs,
           "chunk_ids": ds.chunk_ids, "n_samples": 200 * len(ds.chunk_ids), "passes": passes,
           "selection": "none (last step)"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{tag}: done, {passes} passes over the subset, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
