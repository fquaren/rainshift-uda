# Lessons-learned paper: outlook and COSMO perfect-model arm

2026-11-07 · @Filippo

Decided with Tom: the paper becomes a lessons-learned paper on the trade-offs of UDA for
precipitation super-resolution. Before it is written, the diagnosis is repeated on a second
dataset: COSMO-CLM 2.2 km, in a perfect-model, ERA5-like 10× setting with block-mean
coarsening (stated as such in the paper), between smaller European regions.

This file holds the project outlook (Part A) and the COSMO plan (Part B). PLAN.md stays the
RainShift record. Every number quoted for RainShift comes from PLAN.md or
`report/overview_2026-11-07.md`.

---

# Part A — Outlook

## A1. What the paper now is

The four-claim diagnosis paper becomes a set of lessons. Each lesson is backed by an experiment
we have already run, or by the COSMO arm. The organising thesis does not change: whether UDA has
anything to do depends on whether P(Y|X) moves between regions. COSMO perfect-model is the
setting where that question is open but every confound of RainShift is removed: one model, one
product, no observation network, same physics everywhere.

Candidate lessons and their evidence:

| # | Lesson | Evidence | State |
|---|---|---|---|
| L1 | Measure the conditional shift before choosing UDA; a perfect model does not make it vanish | RainShift P1b (two pairs yes, horn↔melanesia undetermined); COSMO C2 across 20 pairs | RainShift done; COSMO to run |
| L2 | Bound the problem with labels first: oracle λ̂ and a few-shot curve are cheap and decide what UDA could gain | RainShift oracle (3 seeds) + P16: 2,000 target labels reach the oracle; COSMO C4/C5 | RainShift done |
| L3 | Evaluation traps hide failure and fake success | log floor 1e-5 vs 0.1 mm/h (P3, P14, D1); zero predictor scoring as recovery; climatology beating source-only; units across sources (s_S²); mm blow-ups hidden by log metrics (DANN 10⁶ mm/h) | Done |
| L4 | Implementation traps make methods inert or harmful | FDA DC sign flip (A1), DANN gradient 5e-4 of task (A2), AdaBN scoring the unadapted checkpoint, inert biased MMD, **BatchNorm mixing in every target-forward method** (GAP rule 3 in UNet; confirmed in AFM 2026-11-07) | Done; BN check reads pending (tasks 2–4) |
| L5 | Without target labels, selection decides more than the method | method ordering unstable over seeds in every pair; AFM CRPS selection jitter until EMA | Done |
| L6 | Objective matters less than claimed; generative models change what the metric rewards | AFM gaps have the UNet's shape; CRPS rewards spread against a collapsed baseline; FDA into europe_west is the one partial gain | Mostly done; ERA5-tp reference missing |
| L7 | Feature alignment through a GAP'd bottleneck is inert; spatially resolved alignment trades source for target | GAP ablation (3 seeds, read only at the 1e-5 floor) | Needs re-score at D1 or a scoped sentence |

**BatchNorm check, first results (2026-11-07).** Replaying phase 2's source/target BatchNorm
traffic on an untouched phase-1 model sends its source validation CRPS from 0.217 to 1.99
(horn→europe). For melanesia→europe it goes from 0.304 to 2.63. Re-estimating the phase-2
model's statistics on source only brings horn→europe back from 0.377 to 0.263. The rule holds on
both diverged pairs so far; tasks 2–4 (the pairs that trained) are still running. Unless those
tasks reverse it, L4's BatchNorm sentence rests on a direct test, not on inference.

**Read 2026-11-09: confirmed in all five pairs** (`afm_bn_check_seed42/*/bn_check.json`, job
20205518). Source validation CRPS, spectral, seed 42:

| pair | p1 saved | p1 + mixing | p2 saved | p2 re-est. on source |
|---|---|---|---|---|
| europe→horn | 0.261 | 0.403 | 0.327 | 0.266 |
| europe→melanesia | 0.261 | 0.488 | 0.465 | 0.263 |
| horn→melanesia | 0.217 | 0.290 | 0.287 | 0.215 |
| horn→europe (diverged) | 0.217 | 1.991 | 0.377 | 0.263 |
| melanesia→europe (diverged) | 0.304 | 2.635 | 0.745 | 0.372 |

Both arms of the pre-registered rule hold everywhere. Mixing alone, with weights unchanged,
does at least as much damage as phase 2. In the three pairs that trained, re-estimating on
source leaves a residual of 0.0005–0.004 against a damage of 0.07–0.20. So the phase-2 *weights*
are close to phase 1's on the source, and the whole source degradation is in the running
statistics. Not tested: whether the target-side score also moves with source-only statistics.
That is what a fixed-BN run (A2 optional) would show.

## A2. What is left on RainShift (the minimum for the paper)

Needed:
1. Read BN-check tasks 2–4 (`afm_bn_check_seed42/`).
2. ERA5-tp reference for FDA into europe_west (AFM): add the bicubic-upsampled input `tp` (and
   a dressed version) to `afm_ensemble_check.py`. One short GPU job. Without it, L6's one
   positive cell cannot be stated.
3. Regenerate the tables at the 0.1 mm/h headline, from `results/seed<S>_v3/`,
   `seed<S>_fdafix/`, the oracle trees and the AFM trees, with a mm/wet-fraction column beside
   every log-floor number.
4. P8: test metrics of `unet_seed42_a100check/` (one eval job). It is a limitations sentence,
   but referees ask.
5. GAP ablation: re-score at D1 (A9), or keep only the sentence its `mse_std` reading supports.
6. FSS and RAPSD in `evaluate.py`. Stage 1 never started. A lessons paper on SR without a
   spatial-structure metric will be asked for one, and COSMO at 2.2 km makes it more pressing.
   Implement once, use for both datasets.

Optional, decide now:
- **One AFM run with the BatchNorm fix** (target pass without running-stat updates), on
  horn→europe spectral. It turns L4 from "this breaks" into "this breaks, and here is the
  one-line fix". About 4 A100-h. Recommended.
- Moran's I, P17 (λ of representations), Kendall's W (P12), the NGG appendix: drop.
- More AFM seeds: no, apart from seed 43's europe check, which is done.

## A3. Order and gates (no calendar until a deadline exists)

*2026-11-09: there is now a target, experiments done by March 2027. The calendar is in
MILESTONES.md. The order below still holds.*

1. Now: COSMO C0 feasibility (CPU) in parallel with RainShift items 1–2 above.
2. COSMO preprocessing (C1) while RainShift items 3–6 run.
3. COSMO C2–C3 (diagnosis and the source-only matrix). **Gate:** if every COSMO pair has a
   gap within seed noise *and* no conditional excess, the COSMO arm is a one-paragraph null
   (still a lesson: perfect-model transfer is easy) and C4–C5 shrink to two pairs.
4. COSMO C4–C5 (oracle, methods, few-shot), C7 (PGW).
5. Write. The outline is the lesson list. The draft's current structure (four claims) is
   reused section by section, not rewritten from scratch.

Venue to settle with Tom: AIES or Environmental Data Science suit a lessons paper; a workshop
version can stake it early.

---

# Part B — COSMO perfect-model arm

## B1. Data

Archive: `/reference/FGSE/climate_simulations/` (read-only; treat like `raw_data/rainshift`).
- COSMO v6.0, 2.2 km, 1542 × 1542 rotated-pole grid, roughly 10°W–25°E, 35–60°N.
- ERA5-driven present-climate run 2011–2021, plus a +3K PGW run (files dated 2084–2095).
- Files are HDF5/netCDF4 with a `.nz` suffix: 1h_2D is about 227 MB per hour (98,210 files,
  ~20 TB); 5min_2D is about 8.7 MB per step (1.18 M files).
- Strays to ignore: `1h_2D/lffd20111231000000.nc` (438 MB, uncompressed) and
  `5min_2D/test` (8.3 GB).
- Spin-up runs Oct–Dec 2010; use 2011-01-01 00 to 2021-12-31 23 only.
- ERA5 in `/reference/FGSE/era5/` is daily (1971–2020), so a real-ERA5 hourly arm is not
  possible from local data. Inputs are still chosen among variables ERA5 also has, so that arm
  stays possible later.

## B2. Design

**Regions.** Five boxes of 200 × 200 native grid points (440 km), placed on the native rotated
grid (no reprojection), clear of the lateral relaxation zone. Default margin is 60 points
(~130 km) until C0 reads the actual zone width.

| region | approximate centre | character |
|---|---|---|
| southern Scandinavia | ~58°N, 9–12°E (S Norway / S Sweden / Skagerrak) | frontal, orographic, much sea; limited by 60°N |
| central Spain | ~40°N, 4°W (Meseta) | dry, convective in summer |
| Alps | ~46.5°N, 10°E | orographic |
| British Isles | ~52.5°N, 2°W | maritime |
| continental | ~52°N, 19°E (Poland) or ~47°N, 19.5°E (Pannonian) | continental convective |

The positions are fixed before any data is looked at. Spain↔Scandinavia is the headline pair,
and the other 18 pairs give the gradient.

**Coarsening.** 10× block mean of every input field: 200 × 200 → 20 × 20 (22 km). The paper
states that block means are sharper than ERA5's effective resolution, and that block-mean CAPE
is not the CAPE of a block-mean profile (perfect-model convention).

**Target.** Hourly TOT_PREC in mm/h at 2.2 km, 200 × 200.

**Inputs (9 dynamic, every one available in ERA5).** These are written under RainShift's names,
so that `ClimateSRDatasetZarr`, P1b, `field_mean_check.py` and the 9-channel models run
unchanged. Names that differ in meaning are documented in the store's attributes and in the
paper:

| store name | COSMO source | note |
|---|---|---|
| tp | TOT_PREC (hourly), block mean | the strongest predictor at 10× |
| cape | CAPE_ML | RainShift's cape is ERA5 mixed-layer-like |
| tcw | TQV | precipitable water (ERA5 tcw also includes condensate) |
| tclw | TQC | |
| sp | PS | |
| u, v | U_10M, V_10M | **10 m, not 700 hPa**; components are rotated-grid relative |
| tisr | ASOD_T | TOA downward SW (an hourly average, not an instantaneous value) |
| cp | **TD_2M** in place of cp | COSMO has no convective precip (convection is resolved). The slot keeps the 9-channel models; the name is wrong and must be renamed in the store or the dataset's `input_vars` |

`cp` is the one awkward slot. The clean fix is passing `input_vars` with `d2m` through the
trainers, a small change, instead of mislabelling. Decide in C1.

**Statics:** surface height and land fraction at 2.2 km (HSURF, FR_LAND). They are not in the
README; C0 looks for them. Fallback: an external DEM and land mask remapped to the rotated grid.

**Split.**
- Train/validation 2011–2019 (78,888 h), strided validation with a purge gap, scaled from
  RainShift's 88 of 877 chunks.
- Test 2020–2021 (17,544 h, about RainShift's 17,520).

**Store layout** is identical to RainShift:
- `{region}/train_data_in.zarr`, `test_data_in.zarr`, `train_data_out.zarr`,
  `test_data_out.zarr`;
- `static_variables.nc`, `normalization_stats.json` (from the train split only);
- time chunks of 200.

Data root: `/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_cosmo_pm10/` (new; never
under `raw_data`). Run trees: `unet_cosmo_seed<S>/`, `unet_cosmo_seed<S>_oracle/`, results under
`results/cosmo_seed<S>/`, so no `*__none` glob can mix the datasets.

**Size.** The target is 96k × 200² float32, about 15 GB per region uncompressed and probably
3–6 GB compressed (precipitation is sparse). About 20–40 GB for five regions; check the `/work`
quota before C1.

## B3. Experiments

Each step gates the next. Seed 42 unless stated. Costs assume a phase-1 UNet run of about 1 A100-h:
the network always runs at 200², and the training set is 0.64× RainShift's. C1 measures it.

**C0 — Feasibility (CPU, 1 task, ~30 min, plus a 1-minute `gpu`-partition `ls` of `/reference`).**
- File format and the variable list of one 1h and one 5min file.
- **TOT_PREC semantics.** Is it accumulated since model start (hourly = difference of two
  5-min files at full hours, only 96k reads), or per output step (sum of 12 files per hour)?
  Is it also in the 1h files?
- Constants file present? The rotated-pole parameters and the relaxation-zone width from the
  attributes.
- Read time per variable per file, which sizes C1.
- `/reference` mounted on `cpu` and `gpu` nodes (dnagpu008 lacked it on 2026-10-01).
- Output: a JSON and a log in a side folder; nothing under `/reference` is written.

**C1 — Preprocessing (CPU arrays, one task per month, 132 tasks; intermediates on `/scratch`,
final zarr on `/work` by a merge job).**
- Read the 9 inputs and TOT_PREC for the five boxes, block-mean the inputs, write the stores,
  compute the stats.
- Checks kept as tests:
  - block mean of the target equals the `tp` input to float precision (by construction);
  - no NaN or inf;
  - 96,432 contiguous hours;
  - per-region climatology (mean, wet fraction > 0.1 mm/h, p99) printed against Cui et al.'s
    published maps as a sanity check;
  - the stats JSON has no target leakage (train split only).
- `bash -n`, then a one-month smoke task before the array.

**C2 — Shift diagnosis, no training (CPU or GPU job).** For all 20 ordered pairs:
- input marginal shift (W1 per variable in source units);
- P1b with the conditional-permutation week null and its positive controls, on two
  conditioning sets: (cape, tcw) as on RainShift, and (tp, cape). At 10× perfect model the
  coarse `tp` fixes the block mean, so the conditional that matters is fine structure given
  coarse precip;
- support (fraction of bins shared).

Prediction recorded before running: small but non-zero conditional excess, largest for pairs
with the Alps or Spain, because orography and convective regime are not in (cape, tcw).

*Read 2026-11-16* (job 20502163, `results_cosmo/c2/c2_{cape_tcw_block,tp_cape_fine}.json`;
`conditional_shift_binned.py` with the P1b settings, test split, 600 frames, seed 42; 10
unordered pairs, the statistic is symmetric; support 100 % in every pair and set).
Implementation: `--target_scale fine` (native 2.2 km target, each pixel with its 20×20
cell's covariates, 25 of 100 pixels per cell), because `tp` is the block mean of the target
and a block-mean target makes the (tp, cape) set trivial (checked in the test).

```
log_e01 W1_cond (null mean)     (cape, tcw) block        (tp, cape) fine
alps–british_isles              0.127 (0.060) clear      0.018 (0.008) clear
alps–poland                     0.102 (0.056) clear      0.056 (0.054) within
alps–scandinavia_south          0.088 (0.060) within     0.018 (0.014) weak
alps–spain_central              0.148 (0.045) clear      0.029 (0.027) within
british_isles–poland            0.096 (0.060) clear      0.059 (0.052) weak
british_isles–scandinavia       0.084 (0.066) within     0.014 (0.013) within
british_isles–spain_central     0.125 (0.057) clear      0.029 (0.021) weak
poland–scandinavia_south        0.108 (0.058) clear      0.051 (0.045) weak
poland–spain_central            0.090 (0.054) weak       0.030 (0.025) within
scandinavia–spain_central       0.180 (0.060) clear      0.024 (0.015) weak
```

- **(cape, tcw): 7 clear, 1 weak, 2 within**, in one model with the same physics everywhere.
  Largest with Spain (0.18 with Scandinavia, 0.15 with the Alps); the prediction holds for
  Spain, for the Alps in one pair.
- **(tp, cape), fine: 1 clear (Alps–British Isles), 5 weak, 4 within.** Given the coarse
  precipitation the model sees, the sub-grid conditional is close to shared.
- **Consequence for L1 / RainShift claim 1.** The (cape, tcw) statistic flags conditional
  shift in most pairs of a perfect model, and the excess largely vanishes once coarse
  precipitation is conditioned on. So most of it is the projection (rain depends on more than
  cape and tcw), as the statistic's docstring says: an upper bound. RainShift's "two pairs
  with conditional shift" on (cape, tcw) is therefore weak evidence of a difference in
  P(Y|X) as the model sees it. Claim 1 must be worded as an upper bound on (cape, tcw), with
  the COSMO (tp, cape) contrast beside it. A RainShift (tp, cape) run is the missing control:
  RainShift `tp` is ERA5, not the target's block mean, so block scale is not degenerate there.
- Cautions: p has a floor of 0.0099 at 100 permutations and most excesses sit there across
  10 tests, so "weak" means not established. Pairs share regions and are not independent.
  On (tp, cape) only 10–15 of the 25 bins count, because coarse `tp` ties at zero and the
  quantile bins collapse.
- Input shift (W1 per sd of the first region): `sp`, which is near-static elevation and so
  identifies the region, is 4.4–4.75 sd between Spain and Poland / the British Isles, 2.1
  Scandinavia–Spain; dynamic inputs mostly ≤ 0.3. The P19 issue (statics as domain
  identity) applies to COSMO too.

**C3 — Source-only transfer matrix (5 phase-1 runs × seeds 42/43/44 = 15 runs, ~15 A100-h).**
- Score all 25 cells, plus trivial references in every cell:
  - bicubic-upsampled `tp`;
  - block-constant `tp`;
  - the zero predictor;
  - the target-region climatology.

  The UNet must beat bicubic `tp` in-domain, or the arm is not working (acceptance, like the AFM
  check).
- Headline `mse_log_e01` with mm bias / wet fraction / max beside it. FSS and RAPSD once
  implemented (A2 item 6).
- Key figure: transfer gap against input shift and against P1b across the 20 pairs. The pairs
  share regions and are not independent, so inference is a permutation over region labels, or
  the figure stays descriptive.

*Read 2026-11-17.* Jobs:
- training 20588668–69, 15 runs, `--data_order_seed`, val_chunks 40;
- references 20588686;
- scoring 20588687, `results/cosmo_seed<S>/unet/none/results.csv` and
  `results/cosmo_references/references.csv` under `results_rainshift_uda`.

All figures are `mse_log_e01`, as a mean over seeds 42/43/44, with the seed sd in brackets.

- **Acceptance passes.** Every in-domain UNet beats bicubic `tp`, by 30–38 % in the log floor
  and also in mm MAE:

  | region | UNet | bicubic `tp` | UNet MAE | bicubic MAE |
  |---|---|---|---|---|
  | alps | 0.062 | 0.099 | 0.057 | 0.068 |
  | british_isles | 0.045 | 0.067 | 0.036 | 0.042 |
  | poland | 0.041 | 0.062 | 0.032 | 0.037 |
  | scandinavia_south | 0.049 | 0.073 | 0.041 | 0.048 |
  | spain_central | 0.034 | 0.051 | 0.026 | 0.030 |

  The UNets are dry, as on RainShift. Their bias is −0.020 to −0.033 mm/h, which is 24–36 % of
  the mean (the zero predictor's bias). Block `tp` matches the target's block mean (no warning),
  and the climatology is worse than zero in the log floor (0.37–0.94).
- **Transfer gap** is the cell minus the target's in-domain score.
  - Spain_central and poland as targets: no gap from any source. Into spain_central every
    source scores 0.033–0.035 (in-domain 0.034). Into poland they score 0.041–0.046 (in-domain
    0.041).
  - Alps as target: +0.005 to +0.016, at 1–5 seed sd.
  - British_isles and scandinavia_south as targets: small from alps and from each other
    (≤ +0.014).
  - Four cells, poland and spain_central into british_isles and scandinavia_south, are large:
    0.139 (0.055), 0.148 (0.063), 0.194 (0.052) and 0.204 (0.060), against in-domain 0.045 and
    0.049. **All four are worse than bicubic `tp`.**
  - In those four cells the seed spread is ~40 % of the gap; elsewhere it is ≤ 0.008.
  - RMSE reaches 0.4–6.8 mm/h, against ~0.3 in-domain, while bias stays negative. That is the
    RainShift pattern of dry overall plus a few extreme pixels, but **the maxima are not in
    `results.csv` and have not been checked** (compare_fields on these four cells is the next
    read).
- **Directional.** The reverse cells (british_isles / scandinavia → poland / spain) have no
  gap. The failing direction is from the two driest sources (true wet fraction 0.095 and 0.051)
  into the two wettest non-Alpine targets (0.113 and 0.135). That is a description from five
  regions, not a test.
- **Against C2.** This bullet is corrected by the figures (job 20614078, `figures/cosmo_c3/`).
  - **Sp does not order the gap.** `sp` is 4.4–4.75 sd between Spain and Poland or the British
    Isles, yet British Isles → Spain and Poland → Spain have no gap.
  - **The dynamic inputs do.** The input shift over cape, tcw, tclw, tp, u and v, averaged as
    W1 per source sd, orders the relative excess: Spearman ρ = +0.53. The relabelling p is
    0.008 (over 120 relabellings), which is the floor: no relabelling beats the true one.
    The order is not monotone. Alps → British Isles has the largest shift (0.56) and only
    +31 %. The four large cells sit at 0.28–0.39.
  - **Conditional excess does not order the gap.** (cape, tcw): ρ = +0.09, p = 0.44.
    (tp, cape): ρ = +0.21, p = 0.08.
  - **Wetness contrast.** For log2(mean_T / mean_S), ρ = +0.76, but p = 0.11, because that
    matrix keeps its rank structure under relabelling.
  - **Caveats.** Five regions, so this is little power and one test among five, without
    correction. On COSMO, "the transfer cost does not depend on input shift" is not
    supported for the dynamic inputs; it holds only for `sp`.
- **Hypothesis, support asymmetry (2026-11-17, not tested).** The transfer cost is
  directional: wet→dry is nearly free (−3 % to +11 %, except the two Alps cells), and
  dry→wet costs (+2 % to +333 %).
  - *The idea.* Even the wettest regions are dry 86–89 % of hours, so a wet source's
    training data contain the dry target's conditions. A dry source never sees the wet
    target's moist tail and must extrapolate there, in its own normalised space, through
    the exp of the log output.
  - *Why the metric is not the cause.* The obvious alternative is a floor in the metric:
    every model gets a dry target's dry pixels right, so ratios compress. The references
    argue against it. The headroom over the in-domain model is about the same in the
    driest and a wet target: zero predictor +590 % into Spain and +1000 % into the
    British Isles; bicubic `tp` +48 % and +49 %. So the metric does discriminate in dry
    targets, and the small costs there are real.
  - *Tests.*
    1. Per cell, the fraction of target hours with inputs outside the source's training
       range. This should order the cost better than log2(mean_T / mean_S) does
       (ρ +0.76, relabelling p 0.11).
    2. The log error split into wet-pixel misses, dry-pixel false alarms and extremes.
    3. compare_fields 20613827: the maxima in the four large cells.
  - *What it does not explain.* Alps→British Isles (+31 %) and Alps→Scandinavia (+26 %)
    go wet→dry and still cost, which points to terrain (P(Y|X)), not to support.
- **Gate (A3 item 3).** It is not a null. 4 of 20 pairs have a gap far outside seed noise and
  below a non-learned baseline. 6 are small but clear: +0.005 to +0.016, the four into the
  Alps and alps into british_isles and scandinavia. 10 are within noise. C4's
  pair choice should include at least one of the four. Spain→Scandinavia, already in the C4
  list, is one of them; Scandinavia→Spain is its no-gap reverse.
- **Seed sensitivity (Tom).** It is concentrated where the gap is: sd 0.05–0.06 on gaps of
  0.09–0.16. Any method comparison on those cells needs ≥ 3 seeds.

**C2–C3 synthesis (2026-11-17).** Sources: the C2 JSONs, the three C3 `results.csv`,
`references.csv` and `figures/cosmo_c3/` 01–12. Source-only only. Still pending:
compare_fields 20613827 for the maxima.

1. **In the perfect model, transfer mostly works.**
   - 10 of 20 pairs are within seed noise of the target's own model, in the log floor and in
     mm MAE alike. Every source reaches the in-domain model on C. Spain and on Poland.
   - What one COSMO region learns largely holds in another: same model physics, same
     coarse-to-fine operator. That is the clean contrast with RainShift, where no pair
     transfers for free.
2. **Where it fails, it fails by direction, not by distance.**
   - The cost matrix is strongly asymmetric. The two driest sources into the two wettest
     non-Alpine targets cost +199 % to +333 % in the log floor and +70 % to +110 % in MAE.
     So it is not a log-floor artefact.
   - The reverse cells cost −3 % to +6 %.
   - Pattern: a wet source contains the dry target's conditions, a dry source does not
     contain the wet target's moist tail (support-asymmetry hypothesis above). The
     reference check rules out a metric floor as the cause.
3. **The failing cells are under-determined, not just biased.**
   - Seed sd there is 0.05–0.06 on gaps of 0.09–0.16, and ≤ 0.008 in every other cell.
   - Outside the source's support the training data no longer constrain the model, and
     different initialisations and data orders extrapolate differently.
   - This is Tom's flat-loss point, measured: it sits in the cells where source support
     does not cover the target, and it is absent where it does. The toy problem should
     illustrate this case, a flat source loss over the target's regime. That is cleaner
     than the RainShift oracle-source-cost read, which fitted only horn as a source.
4. **In those cells the learned model is worse than not learning.**
   - All four are worse than bicubic `tp`, which on COSMO is the exact block mean
     upsampled.
   - The UNet throws away the coarse rain the input hands it, so this is extrapolation,
     not a missing sub-grid mapping.
5. **Which diagnostics see it.**
   - *Dynamic-input shift, measured in source units,* orders the cost (ρ +0.53, relabelling
     p 0.008), though not monotonically. It is label-free and directional.
   - *Symmetric statistics cannot predict an asymmetric cost.* The C2 conditional excess on
     (cape, tcw) is largest for Spain–Scandinavia (0.120), yet Spain→Scandinavia costs
     +312 % and Scandinavia→Spain −2 %. On (tp, cape) at 2.2 km the excess is ≤ 0.010:
     given coarse rain, the sub-grid mapping is nearly shared. The (cape, tcw) excess is
     mostly projection (C2 read). The UNet sees all inputs and the statics, and it absorbs
     that excess wherever the source covers the target.
   - *`sp` shift is irrelevant to the cost.* Poland and British Isles → Spain and Alps are
     4.4–5.3 source sd, and cost −3 % to +16 %.
6. **The Alps are the one place where a mapping story fits.**
   - Every source into the Alps costs +8 % to +26 %, including the wet ones.
   - The Alps as source cost +26–31 % going wet→dry into the British Isles and
     Scandinavia.
   - Terrain changes P(Y|X) in a way the matched (cape, tcw) test only partly sees.
7. **Every model is too dry, transferred or not:** 20–38 % of the target's mean rain is
   missing. The dry→wet cells are drier than the target's own model, by 11–17 points of mean
   rain on the British Isles and Scandinavia. Dryness is not what separates them, though:
   Alps → British Isles misses only 12 % and still costs +31 %.
8. **compare_fields read (job 20613827, 2026-11-17).** Poland and Spain into the British Isles
   and Scandinavia, against the same-target control (the other wet region) and the target's
   own model. Seeds 42/43/44, every 10th test hour. Outputs:
   `results/plots/compare_fields_cosmo_c3_seed<S>/<target>/summary.csv`.
   - **Blow-ups exist but do not drive the cost.**
     - Maxima reach 579–56,665 mm/h against a true maximum of 52–71. Above-training-max
       pixels are 0.7–28 per million.
     - Spain seed 44 has no blow-up at all (maxima 19 and 37 mm/h), yet it is the worst
       cell (0.242 and 0.267).
     - The RMSE inflation comes from these few pixels; the log-floor cost does not.
   - **The cost is missing rain occurrence.**
     - The failing cells' wet fraction is 0.050–0.094, against 0.111 and 0.135 true and
       0.098–0.125 for the in-domain and control models.
     - The cell's score follows its wet fraction across seeds. Spain→British Isles: wet
       fraction 0.083 / 0.064 / 0.050 and score 0.141 / 0.188 / 0.242. Mean rain falls with
       it (0.082 / 0.067 / 0.036 against 0.100 true).
     - p99 is also low: 1.0–2.0 against 2.1–2.4 true.
     - So a dry source run on a wet target's moist hours mostly produces no rain where it
       rains, plus a rare extreme. The UNet does not discard the coarse rain wholesale (the
       C3 synthesis point 4 wording is too strong). It under-produces rain occurrence in
       the regime it never saw.
   - **The seed spread is in how dry each seed extrapolates.** Spain→British Isles goes from
     wet 0.083 (seed 42) to 0.050 (seed 44). That is the under-determination of point 3,
     made concrete: the seeds disagree on how often it rains outside the source's support.
   - **Controls behave.** Scandinavia→British Isles and British Isles→Scandinavia stay
     within 0.004 of the in-domain score in every seed, with maxima ≤ 109 mm/h.
   - (The plots label the truth "IMERG obs" and the input "ERA5 tp"; on COSMO they are the
     2.2 km target and the coarse `tp`.)

**What this means for UDA (the hypothesis to take to C4–C5 and to RainShift).**
- Unlabelled target inputs can tell *where* a source will fail: the input shift in source
  units is label-free and orders the cost.
- They cannot supply *what* Y is in the region the source never saw. A feature-alignment term
  makes the target's moist tail look like the source's bulk. That hides the extrapolation
  instead of resolving it.
- Prediction for C5 on Spain→Scandinavia: alignment methods do not recover the gap, and they
  may shrink the seed spread without moving the mean. Few-shot labels do recover it.
  Scandinavia→Spain is a no-gap control: any method that costs there is pure harm.

**C2–C3 caveats.** Five regions, so the relabelling tests have little power. Five predictors
were tested without correction. Source-only only. RainShift seeds do not vary data order and
COSMO seeds do (`--data_order_seed`), so the seed spreads of the two datasets are not
comparable one to one.

**C4 — Oracle on 4 pairs (~8 A100-h; seeds 43/44 only if λ̂ is near noise).** The pairs are
Spain→Scandinavia, Scandinavia→Spain, and two chosen from C2 to span the shift range (lowest and
highest P1b excess). λ̂, gap and budget come from the same aggregation code as RainShift
(`--units log`; the per-region s_S differs, so the units trap applies again).

**C5 — Methods and labels on the same 4 pairs (~25 A100-h).**
- FDA: no target forward, the one method with a partial gain on RainShift.
- AdaBN: the BatchNorm control.
- MMD (GAP, `--uda_grad_ratio 0.5`), **only with the BatchNorm policy decided first** (B4 risk
  9). Otherwise it reproduces a known artefact.
- Few-shot fine-tuning at N = 10 and 100 chunks, BatchNorm frozen (the N = 1 arm dropped, as on
  RainShift).
- Not run: DANN, joint_ot, spectral, CORAL, mmd_ms. The lesson is that they add rows, not
  findings.

**C6 — The model ladder: UNet, AFM, CFM (rewritten 2026-11-17, replaces the optional AFM item).**

*Why a ladder.*
- A deterministic-only study can be dismissed: the UNet regresses the conditional mean.
- AFM alone does not answer that. Its flow starts from z = μ + σ_z·ε, where μ is a
  `DualEncoderUNet` trained with MSE, σ_z is an EMA of μ's RMSE, and the checkpoint is chosen
  on μ's MSE (`models/afm.py`, `train_afm.py`).
- So the arm is three models, each one step from the next, sharing the same
  `DualEncoderUNet`, data, splits, seeds, scoring and BN handling:

| rung | the `DualEncoderUNet` output is | loss | selection |
|---|---|---|---|
| UNet | the prediction | MSE (log space) | val MSE |
| AFM | μ, the start of the flow (z = μ + σ_z ε) | flow + MSE on μ | val MSE of μ (as on RainShift) |
| CFM | a conditioning field for the flow; the flow starts from N(0, I) | flow only | val CRPS |

- In AFM the flow net is conditioned on raw `x_dyn`, `x_stat` and x_t, and the encoder only
  supplies its start. CFM keeps the encoder but feeds its output to the flow net as an extra
  conditioning channel, trained only through the flow loss.
- So all three rungs have the same encoder, and every alignment term acts on the same
  bottleneck. The rungs differ in what the encoder is asked to be: the answer (UNet), the
  centre of the answer (AFM), or a learned description of the input (CFM).
- *Reading.*
  - The four dry→wet cells fail on all three rungs: the failure does not depend on the
    objective.
  - They fail on UNet and AFM but not CFM: the regression mean is the cause.
  - Either reading is a result.

*What the generative rungs add beyond "same result".* Does the ensemble know it is outside
the source's support? Per cell, the spread-skill ratio, rank histograms and CRPS:
- widening spread in the dry→wet cells is a partial rescue (honest uncertainty);
- unchanged, overconfident spread is the UNet's failure in another form.

*Which UDA methods carry over to the generative rungs.*

| method | UNet | AFM | CFM | note |
|---|---|---|---|---|
| FDA | yes | yes | yes | input space, no target forward, model-agnostic |
| AdaBN | yes | yes | yes | flow-net BN sees x_t at every t: re-estimate on the sampler's trajectories over target inputs (already in `train_afm.py --adabn_from`). Re-estimation itself costs up to 0.015 CRPS on source (RainShift 3.6), so it is a labelled control |
| MMD (GAP bottleneck, BN-fixed target pass) | yes | yes | yes | acts on the shared encoder. In AFM the flow net is never aligned (`build_uda` docstring). In CFM the aligned features feed the flow net, so alignment reaches the generative path |
| CORAL, DANN | yes | possible | possible | same placement as MMD. DANN blew up in mm on the UNet; run on generative rungs only if MMD acts |
| joint_ot (DeepJDOT) | yes | no | no | needs a target prediction at every step; for CFM that is an ODE solve (10–20 net calls) or a noisy one-step estimate, and its rationale is void on AFM (D2) |
| spectral | yes | μ only | no | a spectrum loss on the prediction; CFM has no μ, and on samples it costs an ODE solve per step |

The shared method set is FDA, AdaBN and BN-fixed MMD (plus CORAL/DANN only if MMD acts).
joint_ot and spectral stay UNet-only and are reported as such.

*Scope.*
- Same pairs, seeds and methods on every rung wherever a rung runs: a comparison across
  rungs exists only where all three see the same cell.
- **Tier 1, all three rungs (the commitment):**
  1. Phase 1 on all 5 regions × seeds 42/43/44.
  2. The 20-pair source-only matrix.
  3. Oracle, FDA, AdaBN and MMD on a pre-registered core of 6 pairs, three regions both
     ways:
     - Spain ↔ Scandinavia: +312 % / −2 %, failing and control;
     - Poland ↔ British Isles: +209 % / −1 %, a second failing/control pair;
     - Alps ↔ British Isles: +31 % / +13 %, terrain.
- **Tier 2, the goal:** the UNet on all 20 pairs (oracle, FDA, AdaBN, MMD), and AFM and CFM
  extended to all 20 pairs as budget allows, failing cells first.
- **Seeds:** 3 everywhere. In the failing cells the UNet seed spread is ~40 % of the gap, so
  one seed cannot rank anything there.

*Cost, rough. The per-run estimates are scaled, not measured on COSMO.*
- COSMO UNet phase 1 measured 1.2–1.7 A100-h. A target-forward UNet run is about 2×; the
  oracle is about 2×.
- AFM/CFM: about 600 s/epoch, scaled from RainShift's ~1,050 s by train size. About 4 h for
  phase 1 and 6–8 h with a target forward.

| block | runs | A100-h |
|---|---|---|
| UNet, oracle and 3 methods on all 20 pairs | 240 | ~470 |
| AFM + CFM phase 1 | 30 | ~120 |
| AFM + CFM, Tier 1 core (6 pairs × 4 arms × 3 seeds × 2 models) | 144 | ~650 |
| **Tier 1 + UNet on all pairs** | | **~1,250** |
| AFM + CFM on the remaining 14 pairs (Tier 2) | 336 | ~1,550 |
| **Everything on all pairs** | | **~2,800** |

At gpu-normal's 6 GPUs per user, 1,250 A100-h is about 9 days of saturated queue, and
2,800 about 20. That has to be cleared with Tom against the shared allocation before
Tier 2.

*Prerequisites, in order.*
1. Port `--target_bn eval` to `train_afm.py`, with a test like `test_target_bn.py`.
   Without it, AFM/CFM + MMD reproduces the known BN artefact.
2. Implement CFM in `models/afm.py` / `train_afm.py`:
   - a mode flag;
   - start from N(0, I) with σ = 1 in log-normalised space;
   - the encoder output concatenated as conditioning, no encoder loss, no σ_z;
   - `--select_on crps`.

   Smoke test, then a short LR check on one region (AFM's tuned LR is not automatically
   CFM's).
3. Ensemble scoring on COSMO through `evaluate.py`'s CRPS path, checked on one cell
   against `cosmo_references.py`'s deterministic numbers. Add spread-skill and rank
   histograms if missing.
4. The C4 oracle and C5 UDA pairs are replaced by the Tier 1 core above, the same for all
   three rungs.

*Prediction, recorded before any run.*
- The direction asymmetry and the four failing cells reappear on all three rungs: a
  generative objective does not add support.
- FDA and MMD do not close Spain→Scandinavia or Poland→British Isles on any rung; the
  oracle does.
- Whether CFM's spread widens in the failing cells is open; that is the reason to run it.
- Not chosen: a cGAN (Harris et al. 2022) or score-based diffusion (Addison et al.). A
  different family would add architecture as a second variable. Keep in reserve for review.

**C7 — Climate shift (PGW +3K), cheap and high value.**
- Preprocess the PGW run for 2 regions (Spain, Scandinavia), test years only (the 2094–2095
  equivalents), plus train years for an oracle.
- Score the present-climate phase-1 models on PGW (no training).
- One present+PGW oracle per region (~4 A100-h).

This is the shift emulators are deployed under, with the physics held fixed. The lesson it
tests: does the conditional move under warming even in a perfect model?

**C8 — No-statics arm, optional (2 phase-1 runs + 1 oracle, ~4 A100-h).** On the headline pair,
zero HSURF/FR_LAND as in P19. At 10× the statics carry most fine-scale information, so this
says how much of any conditional shift is orography the model can read.

**Total, seed 42 plus phase-1 seeds:** about 60–80 A100-h without C6, plus preprocessing CPU
time that C0 decides (hours if TOT_PREC is accumulated; possibly a day or more of array time if
1.18 M 5-min files must be read).

## B4. Problems anticipated, and what to do about each

1. **No hourly precipitation in the 1h files.** If TOT_PREC exists only every 5 min and resets
   per step, C1 reads 1.18 M files (~10 TB). Mitigation: month-sized array tasks and read only
   the five boxes (hyperslab reads, not whole fields). C0 measures before C1 is planned in
   detail.
2. **TOT_PREC accumulated since model start.** Hourly values are differences, and restarts may
   reset the accumulation. Check for negative differences, and treat any as missing hours to
   log, not to clip silently.
3. **Missing constants.** Fallback to an external DEM/land mask on the rotated grid. If used,
   state it; it is the one input not from COSMO.
4. **Rotated grid.** Boxes are rectangles in rotated coordinates, not in lat/lon. U/V_10M are
   grid-relative and rotate slightly with location; harmless within one domain, but say so.
   No reprojection anywhere, so the block means stay exact.
5. **Southern Scandinavia is half sea and squeezed under 60°N.** It must sit outside the
   relaxation zone, and its land fraction differs strongly from Spain's. That is part of the
   shift, not a flaw, but report the land fraction per region. Over sea COSMO precipitation is
   as valid as over land (an advantage over RainShift's observations).
6. **20 × 20 inputs give little context, and box edges get no information from outside.**
   Default: no halo, consistent with RainShift. If edge errors dominate the FSS or error maps,
   add a 2-pixel input halo cropped in the model (a small code change, applied to every region).
7. **The trivial reference may be hard to beat.** At 10× perfect model, bicubic `tp` already
   gets the block means right. If the UNet barely beats it, gaps will be small for a boring
   reason. C3's acceptance catches this before C4 spends anything.
8. **The 9th input slot (cp).** Mislabelled names are how the `__none` incident happened.
   Prefer passing `input_vars` explicitly, and record the variable list in each run's
   `config.json`.
9. **BatchNorm mixing.** It is now confirmed in AFM and was found in the UNet arm (GAP rule 3).
   Before MMD runs on COSMO, decide one policy and apply it to both datasets:
   - keep it as is (comparable with RainShift, artefact included); or
   - target pass without running-stat updates.

   Recommendation: keep it for comparability, run the fixed variant on one pair per dataset,
   and report both.
10. **Units trap.** s_S differs by region (Spain is much drier), so every cross-row number is in
    `mse_log` units, never `mse_std`. The aggregation code already handles this
    (`--units log`). The COSMO stats JSON must carry the precipitation (mean, std) in the same
    place.
11. **Glob and tree collisions.** All COSMO trees get the `cosmo` prefix, `materialise` and
    `evaluate` take the data root and tree explicitly, and nothing is pointed at an existing
    RainShift tree. Domain lists are hard-coded in 5 launchers (`DOMAINS=(...)`): make them
    environment-overridable rather than copying the scripts.
12. **Hard-coded RainShift assumptions.** `_CHUNK = 200` and `_TARGET_HW = 200` match;
    comments say "80"; the 9-channel AFM and UNet constructors match. Check `blacksea` dummy
    handling and any domain-name string tests with a CPU dataset test on the first COSMO store
    before any GPU job.
13. **Seed variance is initialisation only** (the `--seed` trap). Keep it identical on COSMO for
    comparability, and state it once for both datasets.
14. **Comparability with RainShift.** 10× against 2.5×, 2.2 km against 0.1°, model against
    observations. Cross-dataset statements are qualitative; quantitative claims stay within one
    dataset (across COSMO pairs).
15. **Storage on `/work`.** 20–40 GB of stores plus run trees. Check the quota first.
    Intermediates go to `/scratch`, which is purged, so the merge job must run before purge.
16. **The archive is someone else's.** `/reference` is read-only shared data like
    `raw_data/rainshift`. Proposed for CLAUDE.md: add it to hard rule 3.

## B5. Immediate next steps

1. Read BN-check tasks 2–4 when they finish (RainShift L4).
2. C0 feasibility (`cosmo_c0_feasibility.py`, `scripts/cosmo_c0.sh`). Submitted 2026-11-07:
   smoke 20250765 (`data_cosmo_pm10/_c0_smoke/`), full run 20250766 `afterok` on it
   (`data_cosmo_pm10/_c0/`), gpu-node mount check 20250767.
   **Read 2026-11-09.** All three completed. `/reference` is visible on gpu nodes (dnagpu007),
   so C1 could run on cpu or gpu. netCDF4 1.7.4 opens `.nz`; h5py is absent from dl-torch.
   - TOT_PREC is only in `5min_2D` (`kg m-2`, `cell_methods: time: sum`). The values are
     per-interval amounts, not running totals: the window mean at 00:00 is about 0.004 and
     the maximum is 2.5. The hourly target is the sum of 12 files.
   - The 5-min record is complete: 105,120 files per year, 288 per day. File names snap to
     the 20 s model step (`…120504`, `…121008`, `…121456`). The "missing" probe files and
     the 731 k inventory gaps are an artefact of assuming exact 5-min names. C1 must list
     the directory and bin files by hour, not build names.
   - Settled from `time_bnds` with `ncdump`: a 5-min file stamped t covers (previous stamp, t].
     The hour ending HH:00 is therefore the sum of the 12 files from (HH−1):05:04 through
     HH:00. That matches the 1h files: `time_bnds` [HH−1, HH], with ASOD_T as the hour mean.
     CAPE_ML, TQV, U_10M and the other inputs are instantaneous at HH:00. This is the
     ERA5 `tp` convention.
   - `5min_2D/test` is not a constants file. It is one day (2019-03-19) of 5-min output
     concatenated into 288 time steps.
   - Statics are rebuilt from model output by `cosmo_statics.py`:
     - surface height from inverting PMSL against PS;
     - land mask from where RUNOFF_G is filled.
     The output goes to `data_cosmo_pm10/_statics/`. Submitted 2026-11-09 as job 20343366.
   - The 1h record is complete. Present run: 2010-10-19 to 2022-01-01. PGW run: 2084-10-19
     to 2096-01-01. There are no gaps. One stray file is named `lffd20111231000000.nc`.
   - **No statics:** neither HSURF nor FR_LAND is in any file, and nothing like a constants
     or extpar file exists under `/reference/FGSE`. They are rebuilt from model output, see
     below. If the inverted height's spread over high terrain is large, ask the data owner
     (lraess) for the run's extpar file.
   - Reads are fast but were measured on a warm cache. A 1h variable takes about 0.01–0.03 s
     and a 5-min file about 0.015 s. That is roughly 0.3 s per hour of data for 8 inputs plus
     12 precipitation files, so about 8 h single-process for 2011–2021 and all regions in
     one pass. Parallelise by year.
3. Decide: the 9th input (TD_2M via `input_vars`), the BatchNorm policy (B4.9), whether to run
   the one-pair AFM BatchNorm fix (A2), and the region centres in B2.
   **Settled 2026-11-09 in `cosmo_build.py`:**
   - `cp` = 0. A convection-permitting model has no parameterised convective precipitation,
     so zero is exact. No trainer change and no mislabelled TD_2M.
   - `tcw` = the sum of all six TQ fields, which is ERA5's definition.
   - `tisr` = ASOD_T × 3600, in J m⁻².
   - The continental box is Poland (52°N 19°E), not Pannonian.
   - BatchNorm policy: no target-forward method runs on COSMO until the trainers have a
     target pass that does not update BN statistics.
   - Statics, statics job 20343366: inverted height up to 3659 m. The spread between instants
     has a median of 1.1 m over land and 24 m above 1500 m. Land fraction is 0.49 and the mask
     is identical at all 96 instants.
   - Open on the statics: 943 land points invert below −10 m, and some non-land points invert
     to over 2 km. These are probably lakes and glaciers, which the RUNOFF mask treats as
     non-land. Check per box after C1.
4. **C1 builder ready** (`cosmo_build.py`, `scripts/cosmo_build.sh`):
   - a month array (132 tasks) stages to `/scratch/fquareng/cosmo_pm10_stage/`;
   - a merge array (5 tasks) writes `data_cosmo_pm10/<region>/`.
   Staging is about 77 GB on /scratch; the stores are probably 20–40 GB on /work.
   Submitted 2026-11-09:
   - smoke 20346234 (2015-07, written to `cosmo_pm10_stage_smoke/`);
   - months 20346235 (tasks 0–131, at most 20 at once), `afterok` on the smoke;
   - merge 20346236 (tasks 0–4), `afterok` on the months.
   - 2026-11-09: smoke 20346234 failed the `tp` = block-mean(target)/1000 check on float32 rounding (3.7e-9 m, scandinavia_south); the check is now relative (1e-5 of max `tp`). Months and merge cancelled. Resubmitted: smoke 20348662, months 20348663, merge 20348664 (same resources and dependencies).
4. Update CLAUDE.md's "What this project is" to the lessons-learned framing and add
   `/reference` to the read-only rule.
