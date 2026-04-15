# WeatherAdaptSR

Unsupervised domain adaptation for precipitation super-resolution on the [RainShift](https://arxiv.org/abs/2507.04930) benchmark.

## Models

- **UNet** — Dual-encoder UNet (deterministic baseline). Static topography branch + dynamic atmosphere branch.
- **AFM** — Adaptive Flow Matching (probabilistic). Encoder (same UNet arch) + OT flow matching for stochastic detail generation. Based on [Fotiadis et al., ICML 2025](https://arxiv.org/abs/2410.19814).

## UDA methods

| Method | Type | Reference |
|--------|------|-----------|
| CORAL | Feature-level (2nd-order stats) | Sun & Saenko, ECCV-W 2016 |
| MMD | Feature-level (kernel) | Gretton et al., JMLR 2012 |
| DANN | Feature-level (adversarial) | Ganin et al., JMLR 2016 |
| Spectral | Output-level (PSD matching) | — |
| FDA | Input-level (Fourier amplitude swap) | Yang & Soatto, CVPR 2020 |
| AdaBN | Test-time (BN stat replacement) | Li et al., 2018 |

## Repository structure

```
data/
  dataset.py              ClimateSRDatasetNPY (fast, recommended)
  convert_zarr_to_npy.py  one-time preprocessing from zarr
models/
  unet.py                 DualEncoderUNet
  afm.py                  AFMModel (encoder + flow UNet)
scripts/
  run_exp_unet.sh         two-phase SLURM launcher (UNet)
  run_exp_afm.sh          two-phase SLURM launcher (AFM)
  evaluate.sh             batch evaluation
uda.py                    all UDA methods
train_unet.py             UNet training + Optuna
train_afm.py              AFM training + Optuna
evaluate.py               single/batch evaluation
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

## SLURM (full grid)

```bash
# Phase 1 then phase 2 with dependency chain
P1=$(PHASE=1 sbatch --parsable scripts/run_exp_unet.sh)
PHASE=2 sbatch --dependency=afterok:${P1} scripts/run_exp_unet.sh

# Same for AFM
P1=$(PHASE=1 sbatch --parsable scripts/run_exp_afm.sh)
PHASE=2 sbatch --dependency=afterok:${P1} scripts/run_exp_afm.sh

# Evaluate everything
sbatch scripts/evaluate.sh
```

## Domain difficulty ranking

Domains are ranked by normalised Wasserstein-1 distance to europe_west (source):
- **blacksea** — easy (W₁ ≈ 0.00)
- **horn-of-africa** — medium (W₁ ≈ 0.05)
- **melanesia** — hard (W₁ ≈ 0.16)

Precomputed W₁ matrices are in `covariate_shift_analysis/`.
