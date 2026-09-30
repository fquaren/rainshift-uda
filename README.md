# rainshift-uda

Unsupervised domain adaptation for precipitation super-resolution on the [RainShift](https://arxiv.org/abs/2507.04930) benchmark.

## Models

- **UNet** — Dual-encoder UNet (deterministic baseline). Static topography branch + dynamic atmosphere branch.
- **AFM** — Adaptive Flow Matching (probabilistic). Encoder (same UNet arch) + OT flow matching for stochastic detail generation. Based on [Fotiadis et al., ICML 2025](https://arxiv.org/abs/2410.19814).

## UDA methods

| Method   | Type                                 | Reference                 |
| -------- | ------------------------------------ | ------------------------- |
| CORAL    | Feature-level (2nd-order stats)      | Sun & Saenko, ECCV-W 2016 |
| MMD      | Feature-level (kernel)               | Gretton et al., JMLR 2012 |
| DANN     | Feature-level (adversarial)          | Ganin et al., JMLR 2016   |
| Spectral | Output-level (PSD matching)          | —                         |
| FDA      | Input-level (Fourier amplitude swap) | Yang & Soatto, CVPR 2020  |
| AdaBN    | Test-time (BN stat replacement)      | Li et al., 2018           |

## Repository structure

``` plaintext
data/
  dataset.py              ClimateSRDatasetNPY (fast, recommended)
  convert_zarr_to_npy.py  one-time preprocessing from zarr
models/
  unet.py                 DualEncoderUNet
  afm.py                  AFMModel (encoder + flow UNet)
scripts/
  python_env.sh           sourced by every job: picks the interpreter per node
  run_exp_unet.sh         UNet phases 1 / 2 / oracle (phase 2 as a job array)
  run_exp_afm.sh          AFM phases (not yet seeded or ported, see PLAN.md)
  materialise_baselines.sh  pair-named copies of the source-only models
  evaluate.sh             batch evaluation + NGG, one seed per job
  ngg.sh                  recompute NGG from existing results.csv (CPU)
launch_sbatch.sh          submits the UNet pipeline per seed, stages chained
uda.py                    all UDA methods
train_unet.py             UNet training
train_afm.py              AFM training
evaluate.py               single/batch evaluation
compute_ngg.py            NGG for one (model, method, transform)
compute_ngg_all.py        NGG for every method in a results.csv
```

## Quick start

```bash
# 1. Convert data (one-time)
python data/convert_zarr_to_npy.py \
    --zarr_root /path/to/rainshift \
    --out_root /path/to/rainshift_npy \
    --regions europe_west blacksea horn-of-africa melanesia

# 2. Train UNet (single run)
python train_unet.py \
    --source_path /path/to/rainshift_npy/europe_west \
    --target_path /path/to/rainshift_npy/melanesia \
    --uda_method coral --lambda_uda 0.1

# 3. Train with Optuna HP search (two-phase)
#    Phase 1: base HPs (vanilla)
python train_unet.py --source_path ... --target_path ... \
    --optuna --optuna_phase 1 --n_trials 50
#    Phase 2: UDA weight search
python train_unet.py --source_path ... --target_path ... \
    --uda_method coral --optuna --optuna_phase 2 --n_trials 30

# 4. Evaluate
python evaluate.py single --model unet \
    --checkpoint experiments/europe_west__to__melanesia__coral/best.pt \
    --target_path /path/to/rainshift_npy/melanesia

# 5. Batch evaluate all experiments
python evaluate.py batch \
    --exp_root experiments/unet \
    --data_root /path/to/rainshift_npy \
    --output_dir experiments/results
```

## SLURM (full grid, curnagl)

Jobs run on the `gpu` partition (A100 40 GB) with the dl-torch mamba env; the
GH200 node (`gpu-gh`) is a fallback and uses the Singularity container. See
CLAUDE.md, "Partitions and Python".

```bash
# Print the plan, then submit. Per seed: phase 1 -> materialise -> oracle ->
# phase 2 (array, one run per task) -> evaluate, chained with afterok.
DRY=1 ./launch_sbatch.sh
./launch_sbatch.sh                                   # seeds 42 43 44, all stages
STAGES="4 5" SEEDS="42" METHODS="dann mmd" ./launch_sbatch.sh   # resume partway

# Recompute NGG from existing results (CPU job)
SEEDS="42 43 44" sbatch scripts/ngg.sh
```

Outputs: checkpoints in `results_rainshift_uda/unet_seed<N>/` and
`unet_seed<N>_oracle/`, metrics in `results/seed<N>/unet/{none,oracle}/results.csv`,
NGG in `results/seed<N>/ngg_v2/`.

## NGG

`NGG_m(S, T) = (E_T(f_m) − E_S(f_S^none)) / (W1(S, T) + ε)`, with E the
standardised MSE (`mse_std`). E_S is the source-only in-domain risk for every
method, so a method cannot lower its NGG by degrading its source fit
(`--e_s_ref own` reproduces the earlier definition). `delta_src.npy` and
`delta_tgt.npy` hold the source inflation and target change against
source-only. Report `mse_std`; the mm-space RMSE is dominated by a few
exploding pixels and is not stable across seeds.

## Domain difficulty ranking

Domains are ranked by normalised Wasserstein-1 distance to europe_west (source):

- **blacksea** — easy (W₁ ≈ 0.00)
- **horn-of-africa** — medium (W₁ ≈ 0.05)
- **melanesia** — hard (W₁ ≈ 0.16)

Precomputed W₁ matrices are in `covariate_shift_analysis/`.
