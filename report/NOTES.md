# report/: draft notes (2026-10-05)

`main.tex` and `refs.bib` are the first draft. They have not been compiled,
because curnagl has no TeX. Red `\todo{}` marks text waiting on a result or a
check; blue `\note{}` marks a caveat to keep.

## Where every number comes from

| Draft item | Source |
| --- | --- |
| Table 1 (conditional shift) | `results_rainshift_uda/unet/conditional_shift_binned.json` (Aug 10, `scripts/run_cond_binned_test.sh`: test split, 600 frames, 5 × 5 bins on CAPE × TCW, log precipitation) |
| Tables 2–5 and all numbers in the methods and oracle sections | `results_rainshift_uda/results/seeds_42_43_44_unet_mse_std_log/tables.md` (job 65424641: log-precipitation MSE, seeds 42/43/44, paired within seed) |
| s_S for the unit conversion | `raw_data/rainshift/<domain>/normalization_stats.json`, precipitation std 3.716 / 2.698 / 4.682 |
| Data facts (variables, transforms, split sizes) | `data/dataset_zarr.py`; sample counts in `results.csv` |
| Training settings | `scripts/run_exp_unet.sh`, `train_unet.py`, the runs' `config.json` |
| GAP-ablation design | `PLAN.md` Stage 2 c, Redesign; `scripts/run_gap_ablation.sh` |

Never take numbers from these:
- `results/_trash/seeds_42_43_44_unet_mse_std_mixed_units/`: its gap and λ̂ mix units.
- `results/seed<N>/ngg/`: the old NGG definition.
- Any AdaBN row dated before 2026-10-02.

## What fills the TODOs

1. **GAP ablation and BatchNorm control**, from `results/gap_ablation/summary/summary.md`. This rewrites the mechanism paragraph of the methods section and the abstract's claim-3 sentence. The first attempt ran out of memory in smoke; see PLAN.md.
2. **Field-mean check of the seeded methods into europe_west**, after pointing `field_mean_check.py`'s AdaBN path at `best_adabn.pt`. It decides whether the Europe-as-target gains are a zero-rain artefact.
3. **AFM arm**: LR sweep, training, the CRPS transfer matrix and λ̂_CRPS.
4. **Structure metrics**: FSS, RAPSD, Moran's I, case studies.
5. **Test metrics of `unet_seed42_a100check/`** (Limitations, two software stacks).
6. **Model-based corroboration of claim 1** (`conditional_shift.py`).
7. **RainShift facts**: wind level, time resolution, exact periods.
8. **References**: full entries for Qian et al. (2026) and Glawion et al. (2025), and a check of every entry marked in `refs.bib`.
9. **NGG**: drop it, report it per pair, or rescale each cell before averaging.
10. **Horn → Melanesia**: try a stronger oracle, or report the 18 % as irreducible.
11. **Identifiability**: a short proposition with a proof sketch, or the closest citation.

## Reconciliation with the February specification

`Unsupervised_Domain_Adaptation_for_Km_scale_Super_resolution-5.pdf` (Feb 3,
2026) specifies a different study. The draft keeps what still applies:

| February spec | Now | In the draft |
| --- | --- | --- |
| COSMO 2.2 km, 8 European Köppen clusters, inputs made by 8× pooling of the target | RainShift: ERA5 → IMERG, 3 regions, two different products | RainShift only. Using two products is why conditional shift can arise at all |
| UNet, WGAN-GP, latent diffusion; H1, "physics priors generalise better" | UNet and adaptive flow matching with the same width-32 encoder; claim 4, "the failure is not the objective's" | H1 replaced by claim 4. No GAN, no LDM |
| H2, "explicit adaptation beats architecture" | Diagnosis: UDA does not recover the gap, and why | H2 answered in the negative |
| UDA: CORAL, DANN, spectral. DG: MMD + DSBN, group DRO, multi-source DANN, mixup | UDA only: FDA, MMD, DANN, JDOT, spectral, AdaBN; CORAL and multi-scale MMD audited inert and dropped | No DG arm |
| asinh transform, de-quantisation, pooled multi-source normalisation | log(y + 1e-5), z-scored with the source's statistics for every dataset | Source-statistics normalisation, with the reason |
| Generalisation-gap matrix G(i, j) | Transfer matrix plus oracle decomposition: gap = budget + λ̂ | Kept and extended with the oracle |
| CRPS, FSS, RAPSD, tail W1, Moran's I of residuals, MMD of latents during training | Log-space MSE as headline, fair CRPS for AFM; FSS, RAPSD and Moran's I planned; MMD logged in the GAP ablation | CRPS, FSS, RAPSD and Moran's I kept. Moran's I moves from the residuals to the target, as a difficulty covariate. Tail W1, ECE, NLL, SSIM/PSNR and the operational-cost assessment dropped |
| Chronological split 2011–2018 / 2018–2020 / 2020–2021 | RainShift's test split; validation is 88 strided chunks with a purge gap | Strided, purged validation |
| 96 × 96 patches, overlap-tile inference, GH200 batch 512 | Full 200 × 200 fields, batch 128, A100 and GH200 | Not used |
