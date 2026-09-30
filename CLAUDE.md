
# rainshift-uda — working on curnagl

## Where you are

A shared HPC login node at UNIL. Compute happens on GPU nodes via SLURM, never here.
Storage: `/work` is shared lab storage that other people depend on, `/scratch` is fast
and periodically purged, `~` is small.

## Hard rules

1. **Nothing that imports torch or touches the data runs outside a job.** The login node
   has the dl-torch env's `python` on PATH; do not run it there. Syntax checks, file
   listings, log tails and text edits are fine. The login node is shared with the whole
   faculty.
2. **Never submit a job without showing me the command first and waiting for a yes.**
   Print the `sbatch` line, the resource request and how many runs the script will loop
   over. One phase-2 run is about 1.5 GPU-hours against a shared allocation.
3. **Never write to `/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift`.**
   That is the lab's shared RainShift copy. Treat it as read-only.
4. **Never overwrite an existing checkpoint, results directory or CSV.** If the output
   path exists, stop and ask. We already lost the source-only baselines once because the
   oracle phase wrote to the same `__none` tag.
5. **Never `scancel` a job this session did not submit.**
6. **Never `git push`.** Commit only when asked, and never commit `*.pt`, anything under
   `results/`, or job logs.
7. **No `rm -rf`, no `rm` on anything under `/work` or `/scratch`.** To get rid of
   something, move it to a `_trash/` folder in the same tree and tell me.
8. **Run python only inside a job, through `scripts/python_env.sh`.** On x86 nodes that
   is the dl-torch mamba env, called by absolute path; on `gpu-gh` it is the GH200
   Singularity container (the env is x86 only). Job scripts `source` it and call
   `run_python`; GPU jobs also call `require_cuda`, which aborts if torch cannot see a
   GPU, because `evaluate.py` silently falls back to CPU.

## Partitions and Python

Experiments run on the `gpu` partition (7 x86 nodes × 2 A100, **40 GB**). The GH200
partition `gpu-gh` is one node with one GPU; use it only when the A100s cannot do the job
(fall back with `sbatch --partition gpu-gh --mem 0 ...`). CPU-only jobs such as
`scripts/ngg.sh` run on `cpu`. `scripts/python_env.sh` picks the interpreter from
`uname -m`:

```
x86_64   (gpu, gpu-h100, gpu-l40, cpu)   /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/mamba_root/envs/dl-torch/bin/python
aarch64  (gpu-gh)                        singularity exec --nv /users/fquareng/singularity/dl_gh200.sif python
```

The two stacks differ, and results trained on one are compared with the other:

```
             dl-torch (x86)       GH200 container
python       3.14                 3.12
torch        2.13.0+cu132         2.9.0a0 (nv25.09)
numpy        2.5.2                2.1.0
xarray       2026.7.0             2025.10.1
zarr         3.3.0                2.18.7
```

The dl-torch env has two numpy dist-infos (2.4.4 and 2.5.2); the installed files are
2.5.2. On `gpu-gh`, `module load singularityce/4.1.0` fails ("unknown module") and
`singularity` is already on PATH; `SINGULARITYENV_LD_PRELOAD` of the hpcx libraries is
set only there. An x86 Singularity image was tried and abandoned on 2026-09-30: the
unpinned recipe resolves xarray 2026.9 with zarr 2.18.7, which cannot open the data.

QOS on `gpu` is chosen by the time limit: ≤ 12 h is `gpu-normal`, 6 GPUs per user at
once; up to 3 days is `gpu-long`, 4 GPUs per user. Phase 2 therefore runs as a job array,
one run per task, 12 h, 24 CPUs, 200 GB (two tasks per node). Phase 1 and the oracle loop
over several runs and are submitted with `--time 72:00:00`.

Phase 1, the oracles and joint_ot for seeds 42/43/44 were trained on GH200 with the
container; the other phase-2 methods on A100 with dl-torch. That hardware-and-software
split lines up with the method comparison. It is checked by one A100/dl-torch re-run of
seed-42 europe→horn joint_ot (`unet_seed42_a100check/`) against the GH200 seed spread;
see PLAN.md for the outcome.

## Paths

```
PYTHON      see "Partitions and Python"
CODE_ROOT   /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda
DATA_ROOT   /work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift   # read-only
OUTPUT_DIR  /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/unet_seed${SEED}
```

Run python as:

```
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda                     # GPU jobs only
run_python "${CODE_ROOT}/<script>.py" ...
```

Domains: `europe_west`, `horn-of-africa`, `melanesia`. `blacksea` is a dummy — ignore it.
Seeds: 42, 43, 44.

## Checking on things cheaply

- `squeue --me`, `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS`
- `tail -n 100 /scratch/fquareng/outputs/<jobid>` — never `cat` a full training log.
  Job logs live under `/scratch/fquareng` (the `--chdir` in the sbatch scripts), not the
  repo; stderr is in `/scratch/fquareng/job_errors/<jobid>`.
- `bash -n script.sh` before every `sbatch`
- Do not poll `squeue` in a loop. Submit, tell me the job id, stop. Chain dependent
  stages with `--dependency=afterok` instead of waiting.

## Traps in this repo, learned the hard way

- **Bash arrays are whitespace-delimited.** `METHODS=("dann" "mmd")`, never commas — a
  trailing comma survives into the string and argparse rejects the choice.
- **Source stats normalise every dataset, target included.** The model lives in
  source-normalised space. Normalising the target by its own stats silently erases the
  covariate shift the whole framework measures.
- **melanesia's IMERG target contains non-finite values.** `nan_to_num` must be applied
  in physical space, before the log, inside `_transform_channel`. This fix has failed to
  reach the cluster twice — check it is present before blaming anything else for a
  `task=nan`.
- **Per-sample losses only align with the sample order at `num_workers=1`.** With more
  workers the ordering permutes and the saved losses are meaningless.
- **DANN memory is two full 200×200 forwards per step, not one.** The network always runs
  at 200×200: `DualEncoderUNet.forward` bicubic-upsamples 80×80 inputs itself, and the
  dataset's `upsample_on_gpu` only decides whether that happens in the worker or on the
  GPU — it changes no activation size. What doubles memory is that `train_one_epoch`
  calls `_, feats_t = model(x_t, s_t, extract_features=True)` on the target batch: a
  full encoder+decoder pass whose prediction is discarded, but whose decoder graph stays
  alive through `backward()` because `_` still holds it. Only `feats_t["bottleneck"]`
  (512×12×12, then GAP'd) is used. coral and mmd do exactly the same, joint_ot uses the
  prediction; none of them is lighter than DANN. `run_training` also clamps DANN to
  `batch_size ≤ 128`.
- **`--seed` does not reach the dataset.** `ClimateSRDatasetZarr` has its own
  `seed=42` default, which sets both the validation offset (`seed % int(stride)` =
  `seed % 9`, `dataset_zarr.py:216`) and the shuffle RNG (`dataset_zarr.py:316`, and
  `:388` for the joint dataset). Neither `train_unet.py` nor `train_afm.py` passes
  `seed=` to it; `--seed` only reaches `torch.manual_seed`. So every run to date,
  seeds 42/43/44 alike, has the same validation split (offset 6) and the same data order,
  and seed error bars measure initialisation variance only. Found 2026-09-30, after the
  seeded runs; left as is so that all seeded runs stay comparable. Passing the seed
  through would give seeds 42/43/44 offsets 6/7/8 (splits one chunk apart) and
  different data orders, and would make new runs incomparable with the existing ones.
  The test split does not depend on the seed either way.
- **The residual UNet lost its A/B and was removed.** Do not reintroduce it.
- **The validation split is strided with a purge gap, `val_chunks=88`.** Do not switch it
  back to a contiguous tail split; the series is autocorrelated.
- **Oracle and source-only checkpoints share the name `{src}__to__{tgt}__none`.** There
  is no `__oracle` tag. The oracle (`PHASE=oracle`, i.e. `--uda_method none
  --joint_training`) is kept apart only by its directory: `unet_seed<N>_oracle/` beside
  `unet_seed<N>/` (afm: `afm_oracle/`, tag `afm_{src}__to__{tgt}____none`). Once the
  baselines are materialised, the source-only tree has the same six off-diagonal names
  as the oracle tree, so anything that globs `*__none` (`conditional_shift.py`,
  `evaluate.py --exp_root`) silently accepts whichever tree it is
  pointed at. Check `--output_dir`/`--exp_root`, not the tag, and never point two
  phases at the same output directory.

## What this project is

An evaluation framework for unsupervised domain adaptation in precipitation
super-resolution on RainShift. It is a diagnosis paper, not a method paper. Four claims:

1. The three domain pairs differ in P(Y|X), not only P(X). Established.
2. The gap is reducible in principle but not identifiable from unlabelled target data —
   the oracle recovers target in-domain risk, so λ̂ ≈ 0. Needs error bars over seeds.
3. Methods fail by trading the source conditional for target risk; feature alignment is
   structurally blocked by global average pooling at the bottleneck.
4. The failure is a property of the task, not of the MSE objective. Needs the
   flow-matching arm.

If a result would make a UDA method look good, check it twice before telling me — the
prior is that it does not.

## How to talk to me

Be a critical colleague, not an assistant. Say when an idea is unsound and why, before
answering what I asked. Do not restate my own results back to me. Do not open with
praise. Prose by default, lists only for genuinely parallel things. No confidence scores.
If you do not know, say so in one sentence and stop.
