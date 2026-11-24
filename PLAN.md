# RainShift UDA — plan to conclusion

Sep 29, 2026 · @Filippo

Four claims have to land, and only the last needs a new model; the rest needs seeding and one ablation run. Lock the numbers that already exist, add the structure-aware metrics to checkpoints you already have, show the mechanism, then show the gap survives a change of model family.

## The four claims

The paper is a diagnosis, not a method. Nothing below needs a UDA method to work; three of the four claims are strengthened when they fail.

| # | Claim | Evidence it needs | Status |
| --- | --- | --- | --- |
| 1 | The three domain pairs differ in P(Y\|X), not only in P(X) | Null-calibrated matched-input W1\_cond, plus the model-based 9D diagnostic as corroboration, plus input QM (P15) recovering little | **Holds for two pairs on (cape, tcw); on (tp, cape) all three pairs show excess, horn↔melanesia included (2026-11-24, upper bound, `tp` mismatch uncalibrated; see "Read 2026-11-24, (tp, cape) control").** Earlier status, horn↔melanesia undetermined (2026-11-02): P1b (conditional-permutation null, 0.1 mm floor, controls pass): europe↔horn clear excess (W1\_cond 0.281 against null 0.032, p = 0.010, full support); europe↔melanesia clear but at 58 % support; horn↔melanesia none (p = 0.71, every statistic). All are upper bounds on (cape, tcw) only. P2 (9-D, model-based) cannot decide: the domains are perfectly separable on their inputs (AUC 1.000, overlap 1–157 of 17,520 samples), and its positive control registers a planted ×1.5 shift as R = 1.02, the size of horn↔melanesia's R = 1.01–1.05. Input QM (P15) recovers at most 0.9 of oracle gaps of 2.7–5.6 in `mse_log`. Likely framing: on RainShift, conditional shift and lack of support cannot be separated for the tropical pair. Superseded: the pixel-null 86.9× / 14.5× and the P1 week-null factors |
| 2 | The gap is reducible in principle but not identifiable from unlabelled target data | Oracle joint training recovering target in-domain risk, giving λ̂ ≈ 0, repeated over seeds; the few-shot label curve (P16) as its constructive counterpart; the input-set ablation (P19), because the statics make the domain readable from X and λ̂ ≈ 0 may only mean the network fits two conditionals | **Holds into the tropics, not into europe\_west (2026-11-02, 3 seeds, `mse_log_e01`).** Into melanesia and horn the oracle reaches in-domain (λ̂ −0.045 to 0.056; into horn there is no source-only gap to explain at this floor). Into europe\_west λ̂ = 0.032–0.056, 22–39 % of the source-only gap, every seed. P16: 2,000 labelled target samples (10 chunks) bring the source-only model to the oracle on every pair; 20,000 do no better; into europe\_west fine-tuning also stops at the oracle, not at in-domain. The 1-chunk arm overfits (14/18 runs with absurd mm bias) and is not a data point. P19, 3 seeds: the seed-42 λ̂ rise under `mask_shift` does not replicate (0.33 / 0.29 / 0.09); robust is a gap shrink of ~40 %, half of it in-domain skill lost. Withdrawn: "λ̂ ≈ 0 holds for the 11-input set, not for the task". Earlier `mse_log` (1e-5) numbers: λ̂ 0.4–6 % of the gap in five pairs, 18 ± 9 % in horn→melanesia |
| 3 | Methods fail by trading the source conditional for target risk, and feature alignment is structurally blocked | Source inflation against budget recovery, spectral shape loss, the JDOT feature-term share, and a GAP ablation | Partial. GAP ablation done (2026-10-06, Stage 2 panel c Result): GAP alignment is inert, grid4 alignment trades source for target in 3 seeds, the effect rule hinges on seed 42, so "structurally blocked" needs rewording. Seeded numbers show a trade only into europe\_west and harm elsewhere, so the claim needs rewording (Stage 0, seed aggregate). The zero-rain check (P3, read 2026-10-27) explains the Europe-target gains as a log-floor artefact. Every foreign model into europe\_west predicts ≤ 3 % of the observed rain. The methods raise the near-zero floor, which pays in log(P + 1e-5). With a 0.01 mm floor the horn→europe gains fall to 0.02–0.10 of a 0.68 gap, and the melanesia→europe ordering changes (P14). The figure and the falsification do not exist. *2026-11-02, at the 0.1 mm headline:* recovery into horn is undefined (gap 0.002–0.011); into europe\_west the melanesia source-only model is the all-zero predictor, so recovery there measures leaving zero (FDA fix: 0.3 % wet pixels against 9.9 %). FDA fix pays more on the source than it gains on the target. DANN (ratio 1 and 0.5) produces models with 10³–10⁶ mm/h pixels that the log floor rewards; ratio 0.5 is dropped (0.38 / 0.38 / −0.32). Every log-floor number needs a mm-space or wet-fraction row beside it |
| 4 | The failure is a property of the task, not of the MSE objective | The same diagnosis on a flow-matching model, with the risk redefined in CRPS | **Half done (2026-11-02, seed 42).** AFM phase 1 (LR 1e-4, ratio 1.0, EMA 0.999) passes acceptance on test: CRPS\_std 0.200 / 0.167 / 0.293 against UNet MAE\_std 0.423 / 0.339 / 0.543 (europe, horn, melanesia); its μ is a worse point forecast than the UNet. Source-only transfer gaps in `crps_log_e01` have the UNet's shape (into melanesia 0.348 / 0.333 against 0.276; into europe\_west 0.197 / 0.195 against 0.121), and tropical sources into europe\_west collapse the ensemble to near-dry members (spread 0.011 mm against 0.102). The oracle is trained; its test scoring (66790277) gives λ̂\_CRPS. Phase 2 waits on D2. Motivation sharpened by the field plots: every UNet, in-domain included, predicts ~13 % of melanesia's rain and a quarter of its wet pixels, so the UNet's numbers compare degrees of dryness |

Claim 4 is what the architecture change buys. It is also the only claim that a referee can currently defeat with one sentence — "you used a loss whose minimiser is the conditional mean, so of course it collapses" — so it is worth the training cost even though it will not close the gap.

## Stage 0 — lock the current results

Everything downstream compares against these numbers, so nothing else starts until the transfer matrix has error bars.

**Where Stage 0 actually stands (checked 2026-09-29).** `launch_sbatch.sh` ran end to end for seeds 42, 43 and 44 on Sep 16–19 (jobs 65068067–81). Phase 1, oracle and phase 2 completed for all three seeds, with `METHODS=("joint_ot")` only. All three evaluate jobs failed, and none of them evaluated a seeded checkpoint:

- **Stage 2 wrote to the wrong tree.** `materialise_baselines.sh` hard-codes `MODEL_DIR=.../unet`, so every seed copied baselines into the old unseeded tree. `unet_seed<N>/` still has only the three diagonal `__none` directories.
- **Stage 5 scored the wrong trees.** `evaluate.sh` hard-codes `unet/` and `unet_oracle/` and runs with `--force`. Each seed re-scored the old tree into `results/unet/`, and the seed-44 job's output is what is there now (every `metrics.json` is dated Sep 19).
- **Stage 5 then failed.** It calls `${CODE_ROOT}/compute_shift.py`, which lives in `covariate_shift_analysis/`. W1 and NGG never ran.

The old `unet/` tree is itself a seed-42 run (Aug 6) with the same config as `unet_seed42/`, but a different best validation loss (0.6086 vs 0.6099). Training is not bitwise reproducible. `unet/` is another seed-42 draw, not a fourth seed, and must not be pooled with the seeded trees. Every single-seed number in this plan comes from `unet/`.

**0.1 Make materialise and evaluate seed-aware.** The source-only baselines are intact. Oracle runs never write diagonal names, the oracle is kept in a separate `_oracle` directory, and every `__none` checkpoint in `unet/` has `joint_training=false`. No `__oracle` rename is needed. What is broken is stages 2 and 5.

- `materialise_baselines.sh`: take `SEED`, target `unet_seed${SEED}/`, and refuse to overwrite an existing directory.
- `evaluate.sh`: take `SEED`, read `unet_seed${SEED}/` and `unet_seed${SEED}_oracle/`, write to `results/seed${SEED}/`, drop `--force`, and fix the `compute_shift.py` path.

Then run stages 2 and 5 for each seed. Done when, for each seed, `results/seed<N>/unet/none/results.csv` holds 9 source-only rows (3 diagonal + 6 transfer) plus the joint_ot rows, `results/seed<N>/unet/oracle/results.csv` holds 6 oracle rows, the byte-identity check in the materialise log passes, and `results/unet/` is unchanged.

**0.1 done (2026-09-30).** Materialise jobs 65212683/85/87 and evaluate jobs 65212684/86/88 completed. Each `results/seed<N>/unet/none/results.csv` has 15 rows (9 source-only + 6 joint_ot), each `oracle/results.csv` has 6, and the byte-identity checks passed. The Jul 2 W1 cube was used as is.

**NGG definition changed.** The old numerator was E_T(f_m) − E_S(f_m), with the method's own source risk, so a method that degrades its source fit lowers its NGG without helping the target. That ranked joint_ot ahead of source-only in all three seeds, while joint_ot inflates source risk in all 18 cells and is worse on target in horn→melanesia and melanesia→horn in every seed. `compute_ngg.py` now uses E_S(f_S^none), the source-only in-domain risk, for every method (`--e_s_ref baseline`, the default). For method none the two definitions coincide. It also saves `delta_src.npy` and `delta_tgt.npy`, the per-cell source inflation and target change against source-only. New outputs go to `results/seed<N>/ngg_v2/`. The old `ngg/` trees keep the old definition and must not be quoted.

**Report `mse_std` only; the mm-space RMSE is not stable.** RMSE in mm is dominated by a few pixels that blow up when predictions are converted back to mm. Source-only europe→melanesia target RMSE is 3.6 / 18.9 / 22.8 mm across seeds 42/43/44, while `mse_std` stays at 1.19. Seed-43 joint_ot europe→melanesia has a target RMSE of 1894 mm and a train RMSE of 11,824 mm, with an ordinary `mse_std` of 1.18. No clipping is applied, by decision. Every table and figure uses `mse_std`, and `*_rmse_mm` stays in the CSVs only as a record. A related open check: every foreign model evaluated on europe_west has bias = −MAE (0.0974) to four decimals, which is the signature of predicting zero rain. Compare against the observed field mean before claiming it.

**0.2 The launcher.** It is `launch_sbatch.sh` (its header still says `launch_pipeline.sh`). It has been syntax-checked and run three times; stages 1, 3 and 4 work. Once 0.1 lands, stages 2 and 5 are fixed with it. Resume a seed with `STAGES="2 5" SEEDS="<N>"`, and always print the plan with `DRY=1` before submitting.

**0.3 Seed the remaining methods.** This is the expensive step and it needs a decision first. Measured wall-clock per run: phase 1 about 0.7 h (early stopping at patience 5), oracle about 1.8 h, phase 2 joint_ot about 1.25 h. Phase 1, the oracles and joint_ot are already done for three seeds.

Decided 2026-09-30: seed five more methods, FDA, spectral, AdaBN, DANN and MMD. FDA and spectral are the two that recover anything, AdaBN is the clean null (withdrawn 2026-10-05, see the seed aggregate), and DANN and MMD are the two methods that align the GAP'd bottleneck features. MMD is seeded, not left single-seed, for three reasons. Claim 3's structural argument needs two alignment losses to fail the same way, so the failure is not put down to adversarial dynamics. The GAP ablation needs a seeded reference, and MMD has no discriminator to confound it. A single draw cannot be ruled out as an unlucky one. That is 5 methods × 6 pairs × 3 seeds = 90 runs, about 112–135 GPU-hours at 1.25–1.5 h per run; FDA and AdaBN do no target forward pass, so they come in lower. `run_exp_unet.sh` uses the same hyper-parameters as the single-seed `unet/` runs (checked in their `config.json`). A SLURM job array over the 24 phase-2 runs per seed, instead of the serial loop, cuts wall-clock time but not GPU-hours.

**Hardware and software check (2026-09-30, job 65226160).** Phase 2 moves to the A100 `gpu` partition with the dl-torch env (torch 2.13, Python 3.14), while everything already trained used GH200 with the container (torch 2.9, Python 3.12). Seed-42 europe→horn joint_ot was re-trained on an A100 into `unet_seed42_a100check/`. Same seed, same split, same hyper-parameters, and the training trajectory reproduces the GH200 run. Task loss agrees to 1e-4 to 4e-4 per epoch (0.6191 vs 0.6190 at epoch 1, 0.5119 vs 0.5117 at epoch 11). Between seeds it differs by about 3e-3. The best validation epoch is the same (6), and so is the early stop (11). Best validation loss is 0.6969 against 0.6949 on GH200. Two GH200 runs of the same seed differed by 0.0013, and the GH200 seed range is 0.671–0.695. At the level of training and validation loss the hardware-and-software change is an order of magnitude below seed variance. The test metrics of this checkpoint were not evaluated. It fits in the A100's 40 GB, and DANN and MMD have the same two-forward memory footprint. It costs 942 s per epoch against 530 s on GH200, 1.8× slower, so a run that stops at epoch 11 takes about 2.9 h, and 25 epochs would take 6.5 h, inside the 12 h limit. Phase 2 on A100 therefore costs about 1.8× the GPU-hours of GH200.

**P8 read 2026-11-24** (job 20811933, `scripts/p8_a100check.sh`: `results/seed42_a100check{,_srcbn}/unet/none/`
against `seed<S>_v3/` and `seed<S>_srcbn/`, europe→horn joint\_ot, `mse_log_e01`). As trained,
A100 − GH200 at seed 42 is −0.0044 on source and −0.0008 on target; the GH200 seed range is
0.387–0.414 (source) and 0.2275–0.2314 (target), so the stack is 0.17× and 0.21× of it. With
source BN, −0.0007 and +0.0022 against seed ranges 0.302–0.315 and 0.209–0.228 (0.05× and
0.12×). Limitations sentence: the GH200/A100 stack difference is a fifth of the seed range or
less. But the source-BN target difference (+0.0022) is the size of the whole into-horn budget
(0.002–0.004), so no method comparison into horn is resolved across stacks — which the
source-BN read already found to be noise.

Done when the transfer matrix and the per-method recovery table are reported as mean ± standard deviation over seeds 42/43/44, and the method ordering is either stable or the instability is stated in the text.

**Seed variance is initialisation variance only.** `--seed` never reaches the dataset (see CLAUDE.md, traps), so all seeded runs share the validation split and the data order. Only the weight initialisation and cuDNN nondeterminism differ. Say so in the methods section. The error bars are narrower than a full re-draw of the training procedure would give. Fixing it now would make the remaining runs incomparable with the finished ones. If it is fixed, re-run everything.

**Status (2026-09-30).** The 90 phase-2 runs (FDA, spectral, AdaBN, DANN, MMD × 6 pairs × seeds 42/43/44) are submitted as A100 job arrays 65230393 / 65230395 / 65230397. Evaluation jobs 65230394 / 65230396 / 65230398 are chained `afterok` on each seed's array.

**Status (2026-10-01).** Seed 42's array 65230393 completed, all 30 tasks with exit 0, a `best.pt` each and a normal early stop between epochs 6 and 14. Its evaluation 65230394 started at 17:15 and found the expected 45 checkpoints. The measured cost is 1.83 A100 GPU-hours per run on average (55 h for the 30, slowest 3–4 h), so the 90 runs come to about 165 GPU-hours, not the 112–135 estimated above. At 18:10 Slurm dispatched all 58 still-pending tasks, seed 43 tasks 2–29 and every seed 44 task, to dnagpu008, where each died in 0 s with signal 53 and wrote no log; NHC drained the node at 18:13 (`/reference` not mounted). Nothing was written to `unet_seed43/` or `unet_seed44/`. Seed 43 task 0 (FDA europe→horn) had completed and task 1 (spectral europe→horn, 65230395_1) kept running on dnagpu006. Its log shows CUDA caching-allocator OOM warnings at 16:48, which it survived. The stale evaluations 65230396 / 65230398 were cancelled and the lost tasks resubmitted with `--exclude=dnagpu008`: seed 43 tasks 2–29 as array 65277220, evaluation 65277221 `afterok` on 65230395_1 and 65277220; seed 44 tasks 0–29 as array 65277222, evaluation 65277223 `afterok` on 65277222. If 65230395_1 fails, resubmit it alone with `--array=1` and re-chain 65277221.

**AdaBN was never evaluated (found 2026-10-02).** Both trainers save the target-adapted model to `best_adabn.pt` and leave the unadapted one in `best.pt`. `evaluate.py` scored `best.pt`, so every AdaBN row is a second source-only run: seed 42, seed 43 (65277221 ran on the old code) and the single-seed `unet/` tree. That is why AdaBN looked like a clean null. `evaluate.py` now scores the target with `best_adabn.pt` (`target_checkpoint()`), and source and train with `best.pt`, since AdaBN keeps per-domain BatchNorm statistics. `scripts/rescore_adabn.sh` moves a seed's six AdaBN result folders to `results/seed<N>/unet/none/_trash/adabn_unadapted/` and reruns `evaluate.sh`, which re-scores them, replaces their rows and regenerates `ngg_v2/`. Jobs: 65294538 (seed 42) and 65294539 (seed 43, `afterok` on 65277221). Seed 44's evaluation 65277223 has not started and will use the fixed code. `field_mean_check.py` (line 147) still scores `best.pt` for AdaBN. Do not quote any AdaBN number dated before these re-scores.

**Status (2026-10-05).** All 90 phase-2 runs and all three evaluations are done. Seed 43's array 65277220 and seed 44's array 65277222 completed (30/30 tasks each, exit 0). Seed 44's evaluation 65277223 ran on 2026-10-02 22:17 → 10-03 02:04, after the AdaBN fix, so its AdaBN rows score `best_adabn.pt` without a re-score. Each `results/seed<N>/unet/none/results.csv` has 45 rows: 9 source-only plus 6 methods × 6 pairs. `aggregate_seeds.py` (job `scripts/aggregate_seeds.sh`, CPU) builds the mean ± sd tables. It writes them to `results/seeds_42_43_44_unet_mse_std/`, with every difference paired within a seed. The tables are the transfer matrix, the oracle decomposition (gap, budget, λ̂, oracle source cost), the per-method Δtgt, Δsrc and recovery, the count of seeds with a gain or inflation, the method ordering per seed with a stability flag, and NGG. It also checks that the run sets and test-sample counts match across seeds, that source-only copies agree with the diagonal, and that no AdaBN row equals source-only. Still open: the test metrics of `unet_seed42_a100check/`. Until they are scored, a joint_ot-vs-rest difference on the test set is confounded with the GH200/A100 stack.

**Seed aggregate (2026-10-05, job 65424641).** The tables are in `results/seeds_42_43_44_unet_mse_std_log/tables.md`. A first run (65422789) subtracted scores in different units and is in `results/_trash/seeds_42_43_44_unet_mse_std_mixed_units/`.

*Units, a trap that applies to every cross-pair number.* `evaluate.py` scores both splits of a run in its source's standardised space, z = (log P − m_S)/s_S, so `mse_std` from source S is in units of s_S². The precipitation s_S is 3.72 for europe\_west, 2.70 for horn and 4.68 for melanesia, so the MSE factors are 13.8, 7.3 and 21.9. Within one row (one source) everything is consistent: Δtgt, Δsrc, budget, recovery, and each NGG cell. Across rows nothing is. Any subtraction that involves h_T, which is scored in T's units, mixes scales; the gap and λ̂ both do. In raw `mse_std` that gave negative λ̂ for three pairs. It also made europe→horn and melanesia→horn look better on horn than horn's own model. The aggregate now multiplies by s_S² (`--units log`), which is exact because the target transform is a log followed by a z-score. The earlier numbers in this plan, "λ̂ within 0.002–0.023" and the transfer asymmetry "0.435 and 0.349 into europe\_west against 0.086 to 0.184", have to be re-derived in these units before they are quoted. The NGG means over pairs weight each source by s_S², so melanesia-source cells count 3× horn-source cells. NGG is reported per pair, or each cell is rescaled before averaging.

*Claim 2.* In log-precipitation MSE the gap is positive in all six pairs. λ̂ is small next to it: 0.004 ± 0.007 of the gap for horn→europe up to 0.056 ± 0.018 for europe→melanesia. horn→melanesia is the exception, with λ̂ = 0.41 ± 0.15 and 18 ± 9 % of the gap. Joint training costs the source 0.05–0.21 log-MSE. Either say horn→melanesia keeps an irreducible part, or show that more oracle capacity or training closes it.

*Claim 3.* The method ordering by target risk is not stable across seeds in any of the six pairs, and that has to be stated. Gains in all three seeds come almost only into europe\_west. horn→europe improves under every method including AdaBN, with recovery 0.11–0.27. melanesia→europe improves under dann, mmd, spectral and joint_ot, with recovery about 0.5. The source inflation that goes with it is 1.2–1.8 (std units) for dann, mmd and spectral. Outside europe\_west, consistent gains are only joint_ot europe→horn (3/3 seeds, recovery 0.32 ± 0.15) and FDA europe→melanesia (3/3, 0.19 ± 0.20). Everything else is worse on both risks or within seed noise. melanesia→horn is worse under every method in every seed. Its budget is small, so its recovery ratios (down to −10.7) mean nothing. So the "trade" exists only for europe\_west as target; elsewhere it is harm. The europe\_west gains also match the predicts-zero-rain signature noted above: any perturbation that lowers predicted rain scores better there. Run `field_mean_check.py` on the seeded methods before calling them recovery, after pointing its AdaBN path at `best_adabn.pt`.

*AdaBN is not a null.* Scored with `best_adabn.pt` it raises target risk in five of six pairs in log-precipitation MSE (four of them well beyond seed noise; melanesia→europe, +0.91 ± 0.94, is within it) and helps only horn→europe, by 1.43 ± 0.08 at source cost 0.01 (the draft's Table 4 and 5; the earlier 'four of six, +0.12 to +0.45' was in std units). Re-normalising BatchNorm on target statistics actively hurts. The "clean null" framing above is withdrawn.

**Phase-2 early stopping is off (found 2026-10-05).** `run_exp_unet.sh` sets `PATIENCE=-1` for phase 2, and `train_unet.py` stops early only if `patience > 0`. Phase 2 therefore runs all 25 epochs and keeps the best-validation checkpoint; phase 1 uses `PATIENCE_P1=5`. The draft said patience 5 for everything and now says this. Check the oracle's `config.json` (P7).

**Data order is the same in every epoch (found 2026-10-05).** `ClimateSRDatasetZarr.__iter__` builds its RNG as `default_rng(self.seed + worker_id)` on every call. Each epoch therefore replays the same chunk order and the same shuffle-buffer permutation; the comment there says otherwise. This holds for every run in both arms, so it does not bias any comparison. It is one more reason the seed error bars understate training-procedure variance. Left as is for comparability.

## Stage 1 — structure-aware metrics, no retraining

This stage costs no GPU training and produces most of the paper's remaining figures. It runs on checkpoints that already exist, hooked into the streaming path in `evaluate.py` so no extra forward passes are needed. It still runs as GPU jobs: nothing that touches the data runs outside a job, and it draws on the same allocation as the seed runs.

**1.1 FSS across spatial scale.** Build the binary exceedance field at each threshold, convolve with a uniform box filter of width n to get fractions P\_f and P\_o, then

```latex
\mathrm{FSS}(n) = 1 - \frac{\langle (P_f - P_o)^2 \rangle}{\langle P_f^2 \rangle + \langle P_o^2 \rangle}
```

The grid is 0.1° over 200 × 200, so scales run from about 10 km to about 2000 km: a factor of 200, about 2.3 decades. Report the skilful scale, where FSS first crosses 0.5 + f₀/2 with f₀ the domain base rate; that compresses each curve to one number per pair and method.

One decision that matters: base rates differ enormously between europe\_west and melanesia, so absolute FSS at a fixed mm/h threshold is not comparable across domains. Use per-domain quantile-matched thresholds for any cross-domain statement, and keep physical thresholds only when comparing methods within one pair. This is convention, not necessity, but the cross-domain comparison is meaningless without it.

**1.2 RAPSD, raw and normalised.** 2D FFT per field, radially averaged (mean, not sum, over each wavenumber annulus), then averaged over samples. Plot both the raw spectrum and the spectrum divided by its integral. The raw panel carries amplitude, the normalised panel carries the shape of the scale distribution, and separating them is the point: a mean-collapsed model loses both, a merely biased model loses only the first. Glawion normalises exactly here to make their generalisation claim, which is worth one sentence in the related work.

**1.3 Moran's I per sample.** Queen contiguity, row-standardised, computed on the high-resolution IMERG target per intensity bin, then averaged over valid bins to one number per sample. Join to the per-sample losses via the `src_time_index` / `tgt_time_index` arrays, which are valid because the test loaders run at `num_workers=1` without shuffling. Do not use `train_time_index`: `evaluate.py` writes it as `arange(n)`, but the train split shuffles chunks, streams through a shuffle buffer and skips validation and purged chunks, so it is not time order. Fix or drop it before Stage 1 starts. Two questions it answers: whether the transfer gap survives conditioning on target morphology, and whether the Moran's I to skill relation holds out of domain as it does in Qian et al. Note it needs target labels, so it is a post-hoc difficulty covariate, never an unlabelled selection signal.

**1.4 Cheap checks before trusting any of it.** Each metric gets a synthetic control with a known answer, run once and kept as a test:

| Metric | Control | Expected |
| --- | --- | --- |
| FSS | Prediction = target | 1.0 at every scale |
| FSS | Independent random field at the same base rate f₀ | ≈ f₀ at scale 1 (FSS\_random), rising towards 1 at domain scale |
| RAPSD | White noise | Flat (radially averaged spectrum) |
| RAPSD | Plane wave of wavelength L | Single spike at k = 1/L |
| RAPSD | Gaussian blob of width σ | Monotone decay: ln P linear in k² with slope −4π²σ² (no peak) |
| Moran's I | Random binary field | ≈ −1/(N−1) ≈ 0 |
| Moran's I | One contiguous blob | Close to 1, approaching 1 as the blob grows |

**1.5 The case-study panel.** Columns: ERA5 input, IMERG target, source-only, best UDA method, oracle, and a difference field. One row per pair. Pick a median-loss and a tail case from the saved test-split per-sample losses rather than by eye — both comparison papers carry such a panel and the draft has none.

## Stage 2 — the mechanism figure

One figure has to carry the claim that methods buy target risk by destroying the source conditional. It is assembled entirely from Stage 1 outputs, except for one small experiment.

**Panel a — the trade.** Change in target risk on x, change in source risk on y, one point per (pair, method), with the source-only baseline at the origin. A trade (source risk up, target risk down) sits upper-left, and claim 3 predicts points along that anti-diagonal. AdaBN sits at the origin on the source axis.

The single-seed numbers already strain this. DANN, MMD and spectral inflated source risk by 61–291% for negative or small target gains. A method with a negative target gain sits upper-right: worse on both risks. That is harm, not a trade, and "trading the source conditional for target risk" does not describe it. Either reword claim 3 to "methods move the model off the source conditional without buying target risk", or show on the seeded numbers that the points really are upper-left. If neither pattern appears, claim 3 is wrong and the paper changes.

**Panel b — what is being destroyed.** Spectral shape loss, measured as the L1 distance between the normalised RAPSD of the prediction and of the target, against the change in standardised MSE. This turns "regression to the mean" from a phrase into a measured quantity: a model that has collapsed has moved its power to low wavenumbers, and the normalised spectrum shows it even where the raw amplitude does not.

**Panel c — why feature alignment cannot work here.** Every feature-level loss (CORAL, MMD, multi-scale MMD, the DANN discriminator, JDOT's feature term; `uda.py`) applies global average pooling first, which collapses the 512 × 12 × 12 bottleneck to 512 numbers and discards the spatial structure that distinguishes the domains. The JDOT diagnostic already measured the consequence: the feature term carried 0.3% of the transport cost, so the coupling was effectively driven by labels alone.

That is an argument, and it is worth one run to make it a measurement. Take one method on a single pair, replace GAP with a spatially-resolved statistic — channel-wise mean and variance over a 4 × 4 grid of 3 × 3 cells, so the feature vector keeps some geometry — and re-run. The reference has to be seeded, or the result cannot be told apart from seed noise. MMD is not in the seeded set, so either run the ablation on DANN, which is seeded, or seed MMD on the chosen pair (three more runs). If it stays MMD, the kernel bandwidth has to be re-derived: the feature goes from 512 to 16,384 dimensions. Two outcomes, both publishable: the MMD term becomes non-trivial and target risk moves, which names a concrete design fault in how these methods are applied to dense prediction; or it does not, which strengthens the identifiability argument by removing the obvious escape. Budget one run, about 1.5 h.

This is the falsification test for claim 3. Without it, a referee can say the methods were crippled by an implementation choice rather than by the task, and there is no answer.

**Redesign (2026-10-05), before running.** Three findings changed the design above.

- *The seeded MMD reference is inert.* Its logged MMD is 0.010–0.012 at λ = 0.1, about 0.2 % of a task loss of ~0.55. Over training it moves only from 0.012 to 0.0099. With the biased estimator, most of that value is the kernel-diagonal term, ≈ 2/B × (1 − mean off-diagonal kernel), which carries no domain signal. A grid-pooled MMD compared against this reference would have been null by construction.
- *BatchNorm mixing.* Every method with a target forward pass (dann, mmd, coral, joint_ot, spectral) runs it in train mode with default BatchNorm momentum. The running statistics used at evaluation are then a source/target mixture, a partial AdaBN. That fits the Δsrc table: FDA, with no target forward, inflates source risk by 0.02–0.08. The target-forward methods inflate it by up to 1.8. AdaBN hurts the target in 4 of 6 pairs. If it holds, much of claim 3's "trade" is a BatchNorm artefact and not alignment.
- *Skip connections.* The decoder takes skips from all four encoder levels, so bottleneck-only alignment leaves most of the fine-scale path untouched. A null from bottleneck alignment alone would not separate "the task" from "alignment bypassed by the skips".

The test is now `scripts/run_gap_ablation.sh`, a 12-task array: europe_west → melanesia, seeds 42/43/44, four arms. Each arm uses that seed's phase-1 HPs and the phase-2 settings of `run_exp_unet.sh`.

| arm | what | answers |
| --- | --- | --- |
| tgtfwd | mmd at λ = 0: the target forward in train mode, no alignment gradient | how much of the methods' effect is BatchNorm mixing |
| gap | unbiased MMD on GAP bottleneck features | does alignment do anything once it is not inert |
| grid4 | unbiased MMD on 4 × 4-grid mean/std bottleneck features (16,384-d) | does keeping geometry matter |
| grid4ms | grid4 MMD at enc2, enc3, enc4 and the bottleneck | does it matter once the skips are aligned too |

The aligned arms set the MMD weight so that its gradient on the dynamic encoder and bottleneck is 0.5 × the task gradient (`--uda_grad_ratio`). It is measured every 50 steps, smoothed, capped at 100, and the cap hits are logged. This is one fixed rule, not a weight search. Every arm logs per epoch the unbiased bottleneck MMD under both poolings, a within-batch null, and the gradient share actually achieved (`history.csv`). The bandwidth is the median heuristic on the pooled vectors. With its defaults `mmd_loss` is unchanged, and `test_mmd_pooling.py` checks that.

Decision rules, fixed in `compare_gap_ablation.py` before any result:
1. *Manipulation check.* At the selected epoch the aligned MMD must be ≤ 0.5 × the tgtfwd control's, and the mean gradient share ≥ 0.25, in all three seeds. An arm that fails says nothing about alignment.
2. *Effect.* Δtgt(arm) − Δtgt(tgtfwd), paired by seed in log-precipitation MSE, must have the same sign in all seeds and |mean| ≥ 0.1 × the oracle budget (≈ 0.27).
3. *BatchNorm.* If Δsrc(tgtfwd) ≥ 0.8 × Δsrc(seeded mmd) in all seeds, the target-forward methods' source inflation is BatchNorm mixing.

Reading: if grid4 or grid4ms moves target risk and gap does not, global pooling is a design fault and claim 3's structural argument falls. If gap moves it too, the methods were under-weighted, not blocked. If nothing moves and the checks pass, the obvious escape is closed. If rule 3 holds, claim 3 is rewritten around BatchNorm mixing, and the AFM arm has to handle the target forward identically to stay comparable. Cost: ~2–3 A100-h per run, ~30 GPU-h, ≤ 12 h per task. Read-out: `scripts/compare_gap_ablation.sh`.

*First attempt (2026-10-05).* The CPU tests (65424881) passed, and so did the tgtfwd smoke (65424882_0). The grid4ms smoke (65424882_9) ran out of memory in epoch 2 at batch 128 on a 40 GB A100. The array (65424883) and read-out (65424884) were then cancelled, since their dependency could never be met. The smoke trees are in `_trash/`. Fix: the feature-only target pass now drops its unused prediction, so its 200 × 200 decoder graph is freed before `backward()`. Loss, gradients and BatchNorm statistics are unchanged; only memory drops. The script also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and `train_unet.py` now prints peak memory. Re-smoke the three aligned arms (tasks 3, 6, 9) before the array. In the control smoke the unbiased bottleneck MMD at initialisation was 0.047 under GAP and 0.21 under grid4, with nulls of about 0.001: both poolings see the domains apart from the start.

**Result (2026-10-06).** The array (65425425) completed and `compare_gap_ablation.sh` (65451818; the first read-out 65425448 failed on `df.transform`, the DataFrame method, instead of `df["transform"]`) wrote `results/gap_ablation/summary/`. Reading it, not the rules' verdicts:
- *Rule 2 verdict is "gap and grid4 move target risk; grid4ms does not", and it rests on seed 42.* The tgtfwd control is bad in that seed (Δtgt +1.55 against −0.28 and −0.12; the seeded mmd reference is also worst there). Without seed 42 the mean differences are −0.21 (gap) and −0.23 (grid4), both under the 0.273 threshold. The history of seed-42 tgtfwd shows nothing odd (selected epoch 3, source val 0.784 against 0.818 and 0.800); `history.csv` has no per-epoch target score, so the cause is not located. Likely BatchNorm mixing on that initialisation, untested.
- *Against source-only, gap does nothing* (Δtgt +0.68, −0.45, −0.37; mean −0.05, sd 0.63). *grid4 gains ≈ 0.4 in all three seeds* (−0.39, −0.40, −0.46) at Δsrc +2.57, about the size of the oracle budget. That is a trade of source for target, not a free gain. The selected epochs are 3, 23, 5; seed 43 was still improving at epoch 23 of 25.
- *Rule 1 passes by sign only.* The aligned unbiased MMD is negative in all nine rows (−0.0007 to −0.005) against nulls of ≈ 0.001, so "ratio ≤ 0.5" is trivial. The criterion needs |MMD| or a null-relative form before it is quoted. What it does show: GAP-feature alignment can be optimised (gradient share 0.64–0.76), so the failure is not that GAP alignment cannot be driven down.
- *Rule 3 holds:* Δsrc(tgtfwd)/Δsrc(mmd) = 0.91, 1.21, 0.96. The source inflation of the target-forward methods is BatchNorm mixing. The aligned arms add 0.5–1.2 on top (Δsrc 2.52, 2.57, 3.17 against 2.00).
- *Consequence for claim 3.* "Structurally blocked by GAP" is not supported as worded. What the data support is: GAP alignment is inert for target risk; spatially resolved alignment trades source for target risk. Three seeds, noisy control, init variance only (`--seed` does not reach the dataset). Do not write a "moves, yes" without the per-seed table and the seed-42 caveat. Optional, not planned: re-score seed-42 tgtfwd `best.pt` with target-recomputed BN statistics.

## Stage 3 — the flow-matching arm

The goal is to show the gap is a property of the task rather than of the MSE objective. Success is the gap persisting. If flow matching closes it, that is a different and better paper, but do not plan for it.

**3.1 Revive AFM rather than building something new.** The encoder plus flow network already takes x\_t and t, which is what a velocity field needs; a UNet that only maps (x\_dyn, x\_static) to a field does not. The known fixes are in place — `_match_dyn` for the `sample()` shape crash, eval-mode side effects removed, μ exposed in the result dict. Smoke-test on one pair with `--subset_chunks` before anything else. `run_exp_afm.sh` passes no `--seed` and writes to a fixed `afm/` tree, so it needs the same per-seed output directories as the UNet launcher before any seeded AFM run. `evaluate.sh`'s AFM section needs the same change.

*Readiness, checked 2026-09-30: not ready to launch.* No AFM run exists anywhere under `results_rainshift_uda/`, and the smoke test has not been run. `run_exp_afm.sh` still hard-codes the GH200 container and `--mem 0` rather than `python_env.sh`. It has no `--seed`, no per-seed tree (`evaluate.sh` already expects `afm_seed<N>/`) and no job array. It uses `PATIENCE=-1` (no early stopping) for phase 2 and the oracle. It lists all six methods, although 3.3 says only the Stage 2 survivors. Unlike the UNet script it passes no `--dann_weight 5e-4` or `--jdot_reg 0.001`, so AFM would fall back to `train_afm.py`'s defaults (`jdot_reg` 0.1) and the two arms would differ in UDA settings. Phase 1 passes no `--lr` or `--weight_decay`, so it trains at `train_afm.py`'s defaults, 1e-5 and 1e-6. That may be right for a flow model, but it is not a tuned choice. Batch size is 512, not tested against the A100's 40 GB. The CRPS-based λ̂ of 3.2 does not exist yet: `evaluate.py` computes a fair-estimator `crps_mm` per run, but `compute_ngg.py` and the oracle decomposition read only `mse_std`.

*Update 2026-10-02.* `run_exp_afm.sh` is ported:
- `python_env.sh`, the `gpu` partition, `--seed` and the `afm_seed<N>/` tree;
- phase 2 as a job array;
- patience 5 for training and 8 for the oracle;
- the UNet arm's DANN and JDOT settings;
- an explicit LR of 1e-5 (still untuned). `--weight_decay` is a no-op in `train_afm.py`, whose optimiser hard-codes per-group weight decay.

AFM tags are now `afm_{src}__to__{tgt}__{method}`, two underscores like UNet. `materialise_baselines.sh` takes `MODEL=afm`. `evaluate.py` reads `base_features` from each run's `config.json` rather than trusting its default. `evaluate.sh` scores the AFM oracle tree, deterministic and with the ensemble, and its ensemble pass no longer scores the training split. Both launchers refuse to retrain over a partial run's `best.pt`. The smoke test `scripts/smoke_afm.sh` is job 65294537. It runs 2 epochs on 8 chunks per split at batch 512 and logs peak GPU memory. Decide the batch size after it. Still open: the method list (after Stage 2) and the CRPS λ̂ (3.2).

**Width 32, decided 2026-10-05.** AFM trains at `base_features=32`, batch 64. At 32 the AFM encoder is exactly the UNet arm's `DualEncoderUNet` (also width 32, `train_unet.py`). The two arms then differ in the objective and not the backbone, which is what claim 4 needs. Width 64 ran out of memory on a 40 GB A100 at batch 512 and at batch 128 (65294537, 65308291). The width-32 smoke test 65311008 passed every phase. Peak memory was 20.5 GiB for none and adabn, 26.0 GiB for dann and 29.6 GiB for joint_ot. MMD was not in the smoke set; it has DANN's two-forward footprint. Width 64 is not to be revisited: a wider AFM that did better would confound capacity with objective. `run_exp_afm.sh` passes `--base_features 32` in every phase.

**What the smoke test exposed (2026-10-05), fixed before any real AFM training:**

- *Selection ignored the flow network.* `train_afm.py` selected checkpoints and early-stopped on the validation MSE of the encoder mean μ. The flow net never entered selection, so the "AFM" checkpoint was whichever epoch had the best regression head. `--select_on crps` now selects on the fair CRPS of the sampled ensemble. It is computed in standardised space on 8 validation chunks spread evenly over the split, with 4 members and 10 midpoint steps. A fixed generator gives the same noise in every epoch and every run. For the oracle the score is the mean of the source and target validation CRPS, as with MSE. The fair estimator is unbiased in M, so the 4-member validation score and the test score estimate the same quantity. `run_exp_afm.sh` defaults to `SELECT_ON=crps`.
- *CRPS existed only in mm.* The smoke run's `crps_mm` was ~1e21 with an infinite spread: one member blowing up under the inverse transform. That is the same failure as `rmse_mm`. `evaluate.py` now also reports `crps_std` and `spread_std`, and these are the AFM headline numbers. `crps_mm` stays in the CSV as a record only, like `rmse_mm`.
- *The evaluator held the whole ensemble in RAM.* At 17,520 samples × 16 members × 200² float32 that is ~45 GB per split. CRPS and spread are now scored per batch (`prob_metrics.py`, shared with the trainer) and saved per sample.
- *The sampler integrated in bf16.* Under `evaluate.py`'s autocast the ODE state became bf16 after the first step. `AFMModel.sample` now keeps the state in float32 and takes an optional generator.
- *The encoder trained at 1e-6.* With `LR=1e-5` and the encoder at LR/10, μ was learning 500× more slowly than the UNet arm's 5e-4. The ratio is now `--encoder_lr_ratio`, read back from `base_hp/` by phase 2 and the oracle, and it is part of the sweep below. `run_exp_afm.sh` refuses a full phase 1 without an explicit `LR`.

**3.1a Learning-rate sweep (prepared 2026-10-05; Stage A ran 2026-10-06, array 65451947; result below).** `scripts/tune_afm_lr.sh` is a job array with one source-only phase-1 run per config `domain|lr|encoder_lr_ratio`. It writes to the side tree `afm_lrtune_seed42/<domain>/lr<lr>_enc<ratio>/`, which no evaluation or materialise script reads, and refuses to overwrite. Each run uses the production schedule truncated: cosine over 25 epochs, stopped after 12 (`--stop_after`), no early stopping. 12 covers where UNet phase 1 stopped (epochs 6–14), and truncating keeps the LR at every epoch identical to a full run's. Every epoch writes validation MSE of μ and validation CRPS to `history.csv`. Target-test numbers are left out of it. `scripts/summarise_lrtune.sh` (awk, safe on the login node) prints one line per run.

- *Stage A*, europe_west, 7 runs: LR 1e-5, 3e-5, 1e-4, 3e-4, 1e-3 at encoder ratio 0.1 (encoder 1e-6 to 1e-4), plus ratio 1.0 at 1e-4 and 3e-4. 1e-5/0.1 is the current default and serves as the reference.
- *Decision rule, fixed now.* Lowest best-epoch validation CRPS wins. If the top two are within 2 % of each other, take the lower LR. A run whose best CRPS is at epoch 12 with CRPS still falling is not converged by the sweep's standard; rank it, but say so. Validation MSE of μ is reported but does not decide.
- *Stage B*, 4 runs: the winner and its better neighbour on horn-of-africa and melanesia (`CONFIGS_ENV=...`). If the winner is best or within 2 % on both, it is used for all three sources. If not, each source takes its own winner. Selection is on that source's validation split only, so no target label enters, as in the UDA protocol.
- *Cost.* Smoke epochs on 1,600 samples took ~18 s. Extrapolated to the ~120k-sample training split, that is ~0.3 h per epoch before the validation CRPS, which adds an estimated 10–20 %. This is an extrapolation, not a measurement; the first Stage A epoch lines will replace it. Stage A ≈ 7 × 12 × 0.35 h ≈ 30 GPU-h, Stage B ≈ 17 GPU-h. Every task is ≤ 12 h, so it runs under gpu-normal.
- *Asymmetry to state in the paper.* The UNet arm's 5e-4 was the trainer default and was never tuned. Tuning only AFM makes AFM the better-tuned arm. That is the conservative direction for claim 4: a gap that persists in the more carefully tuned arm is harder to put down to tuning.

**Stage A result (read 2026-10-27).** All 7 tasks finished (3.5 h each, ~1,050 s per epoch including validation CRPS, so the 0.35 h estimate held). Per-epoch curves are in `afm_lrtune_seed42/europe_west/<config>/afm_europe_west__to__europe_west__none/history.csv`. The sweep does not choose an LR.

- *lr1e-3 is excluded.* The sampler diverged twice: validation CRPS was 1.9e5 at epoch 5 and 1e11 at epoch 10, while the flow loss and μ MSE stayed smooth.
- *The selection score is noisier than the differences it ranks.* Flow loss and validation μ MSE fall monotonically in every run. Validation CRPS does not: at LR ≥ 1e-4 it moves by 0.01–0.055 from one epoch to the next. The sd of its epoch-to-epoch differences over epochs 6–12 is 0.012–0.031 at LR ≥ 1e-4 and 0.005 at 1e-5 and 3e-5, so the jitter grows with LR. The spread between configurations is 6 % (best-epoch CRPS 0.2714 for 1e-4/1.0, 0.2768 for 1e-4/0.1, up to 0.2890). Under the fixed rule, 1e-4/1.0 wins by 1.99 % over 1e-4/0.1, a tie that the "lower LR" tie-break cannot break because both are at 1e-4. The noise is not sampling noise between epochs: the generator is reseeded on every call, so every epoch scores the same noise draws on the same 8 chunks.
- *Averaged over epochs 9–12* the order is 1e-4/1.0 0.283, 3e-5/0.1 0.286, 1e-5/0.1 0.288, 1e-4/0.1 0.291, 3e-4/0.1 0.296, 3e-4/1.0 0.304. The top three lie within 2 % of one another and span the whole LR range.
- *Nothing has converged.* The best epoch is 11 or 12 in every run.
- *μ MSE does separate the low LRs.* At epoch 12 it is 0.754 for 1e-5 and 0.700 for 3e-5, against 0.659–0.671 for the rest. The default 1e-5/0.1 under-trains the encoder within this budget. The UNet europe phase-1 checkpoint (`unet_seed42/`) has a best validation loss of 0.610, so even the best AFM μ is ~8 % worse at epoch 12. Encoder ratio has no consistent effect: at 1e-4, ratio 1.0 is better on μ MSE (0.662 vs 0.671), at 3e-4 ratio 0.1 is (0.659 vs 0.664).
- *Conclusion.* No LR is chosen from Stage A, and Stage B is cancelled: run on this score, it would rank noise. Next steps are in 3.6.

**3.1b Sampler check, after Stage A.** Superseded by 3.6 (1), which folds the step check into the diagnostic job. Original plan: score the Stage A winner's checkpoint on the 8-chunk validation subset at 5, 10, 20 and 40 midpoint steps, with M = 8, and use the smallest step count whose CRPS is within 1 % of the 40-step value for every test-time evaluation.

The CorrDiff-style two-stage variant, regression UNet plus an FM corrector on residuals, goes in future work: under domain shift the residual distribution the corrector trains on is itself shifted, which stacks a second identifiability problem on the first.

**3.2 Redefine the risk. This is the real work.** The oracle decomposition, λ̂ and the whole transfer matrix are currently defined on pointwise MSE, which is the wrong functional for a probabilistic model — scoring the ensemble mean would smuggle the smoothing preference straight back in. Redefine in CRPS:

```latex
\hat{\lambda}_{\mathrm{CRPS}} = \mathrm{CRPS}_T(h_{\mathrm{joint}}) - \mathrm{CRPS}_T(h_T)
```

That needs a target-trained in-domain model h\_T for each domain. As in the UNet arm, h\_T is the phase-1 model trained on T (the diagonal `{T}__to__{T}` run), so the three source-only runs already provide it and no extra training is needed. Fix the ensemble size before any number is reported, since CRPS is biased at small M — use the fair estimator or state M and keep it constant everywhere.

**3.3 What to train.** Three source-only runs (which double as the in-domain models), six joint oracles, then the surviving methods from Stage 2 on six pairs. If the survivors are FDA, spectral and AdaBN, that is 18 method runs, 27 runs in total for one seed. Seed the arm only if Stage 0 shows seed variance is material.

**3.4 Probabilistic evaluation.** CRPS and spread-skill in-domain and under transfer, plus rank histograms per pair. The question no one in this literature has asked is whether ensemble calibration degrades under domain shift the way the point forecast does — a well-calibrated source model that becomes over-confident on the target is the probabilistic signature of the same failure, and it is a cheap extra panel.

**3.5 AFM training and evaluation schedule (planned 2026-10-05).** Seed 42 only. Whether to seed the arm is decided from the seed spread in `results/seeds_42_43_44_unet_mse_std/`. If the UNet seed sd of λ̂ and of the source-only gap is small next to the effects claimed, one AFM seed is enough. All settings come from 3.1a/3.1b: width 32, batch 64, the swept LR and encoder ratio, `SELECT_ON=crps`, patience 5 (oracle 8), cosine over 25 epochs.

Order, each step gated on the previous one:

1. **Sweep** Stage A → `summarise_lrtune.sh` → Stage B → sampler check (3.1a, 3.1b). ~47 GPU-h plus ~1 h. *Superseded 2026-10-27 by 3.6: Stage A ran and did not choose; Stage B is cancelled.*
2. **Launcher changes, before phase 1.** Make phase 1 and the oracle job arrays (one domain or pair per task), as phase 2 already is. The serial loops need `--time 72:00:00`, which puts them on gpu-long (4 GPUs per user), and gpu-long is the QOS that is full whenever long jobs are queued. Phase 1 at ≤ 25 epochs × ~0.35 h fits 12 h per task. An oracle epoch streams both training splits, so ~0.7 h, and up to 25 epochs is ~17 h; the oracle array needs `--time 24:00:00`. This is not yet implemented.
3. **Phase 1**, 3 runs, `LR=<swept> ENC_LR_RATIO=<swept> SEED=42 PHASE=1`. ~3–9 GPU-h each, depending on where early stopping lands. Then `MODEL=afm SEED=42 materialise_baselines.sh`.
4. **Oracle** (6 runs, ~8–17 GPU-h each) and **phase 2** together, both reading `base_hp/`. Phase 2 trains only what Stage 2 keeps. *The method list is reopened as decision D2 (2026-10-27): P3 showed that the "recovery" of FDA and spectral into europe\_west is mostly the log floor.* The default proposal was FDA, spectral and AdaBN, 18 runs. AdaBN is the cheap null, and FDA and spectral are the two that recovered anything in the UNet arm. DANN, MMD and joint_ot are added only if the Stage 2 GAP ablation makes feature alignment worth re-testing under a new objective. Phase 2 at ~0.35–0.5 h per epoch is ~4–8 GPU-h per run. Seed 42 total: roughly 150–250 GPU-h, the oracle and phase 2 dominating. Phase 1 timings replace this estimate.
5. **Evaluation.** The serial `evaluate.sh` does not fit. A 16-member, 20-step ensemble is 640 flow-net calls per sample. Scoring both test splits of one checkpoint is then ~22 M calls, an estimated ~5 GPU-h, and ~33 checkpoints is ~165 GPU-h in one 24 h job. Instead:
   - Ensemble size M = 8 with the fair CRPS (unbiased in M; the variance cost is small next to 17,520 test samples), and the step count from 3.1b. That is ~4× cheaper.
   - A job array with one checkpoint per task. This needs an `--only <tag>` filter in `evaluate.py batch`; not implemented yet.
   - No `ot` input transform for AFM, and no ensemble on the training split. The deterministic pass already gives the training-split risk.
   - Deterministic pass (μ, `mse_std`) on every checkpoint, so AFM's μ can be set beside the UNet in the arm comparison. Probabilistic pass (`crps_std`, `spread_std`) on every checkpoint including the oracle; its rows go to `results/seed42/afm_prob/{none,oracle}/`.
6. **Derived numbers.** λ̂_CRPS, the gap, the budget and the method recovery come from `aggregate_seeds.py --subdir afm_prob --metric crps_std` (with a single seed it reports means without sd). h_T is the diagonal phase-1 AFM run, as in the UNet arm. NGG in CRPS comes from `compute_ngg_all.py --model afm --error_metric crps_std`, which accepts any `src_/tgt_` column. The NGG loop in `evaluate.sh` currently runs only for unet.
7. **3.4 panels** (spread–skill, rank histograms per pair) come from the per-sample CRPS and spread that `evaluate.py` now saves. Rank histograms also need the truth's rank among members, which is not saved yet. Add it to the probabilistic pass before step 5 if the panel is wanted.

**Acceptance.** Source-only AFM should beat source-only UNet on CRPS in-domain, as a sanity check that the arm works at all. The UNet's CRPS is its MAE, a point forecast being a one-member ensemble, and it has to be in standardised space. `evaluate.py` reports only `mae_mm`, so the three diagonal UNet checkpoints need a `mae_std` pass first. Then the transfer gap should have the same sign and roughly the same ordering across pairs as the UNet arm. If the ordering inverts, say so — that is a finding, not a bug.

**3.6 Next steps for AFM (planned 2026-10-27).** This replaces 3.5 step 1. Steps 2–7 of 3.5 still apply, after the changes below. Each step is gated on the previous one. Steps 1–3 are done (2026-10-28); step 5 is implemented (2026-10-29) and phase 1 is next.

1. **Diagnostic job, no training.** Submitted 2026-10-27: smoke test 66641246 (2 batches per score, `_smoke` tree), then the array 66641262 (tasks 0–2), `afterok` on the smoke test. Scripts: `afm_val_diag.py` and `scripts/afm_val_diag.sh`, writing `afm_lrtune_seed42_diag/europe_west/<config>/diag.json`. Its first score (seed 0, saved statistics) must reproduce the run's `best_val_loss`, or the subset or sampler differ from training. It is a 3-task array on the `gpu` partition, ≤ 12 h per task; the first task measures the cost. One task per Stage A checkpoint (`best.pt`): lr1e-4\_enc1.0 (lowest score), lr3e-4\_enc0.1 (largest jitter, sd 0.031) and lr3e-5\_enc0.1 (smoothest curve, sd 0.005). Every score uses the selection subset `build_crps_loader` builds (8 chunks spread over the europe\_west validation split) and the fair CRPS in standardised space, unless stated otherwise. It measures:
   - *Sampling noise.* Generator seeds 0–3 at M = 4 and 10 steps, the selection setting. The sd across seeds is the noise of a single selection score.
   - *Flow-net BatchNorm.* `TimeCondDoubleConv` (`models/afm.py`) uses `BatchNorm2d` with the default momentum 0.1, so the evaluation statistics are dominated by roughly the last 10 training batches. Re-estimate only the flow net's running statistics, with `momentum=None` (cumulative), on 5 disjoint sets of 10 training batches of 64 and on one set of 200. Each re-estimation runs the training forward at random t. The encoder stays in eval mode with its BatchNorm untouched, and `sigma_z` is saved and restored, because the training forward updates its EMA. Score each set at seed 0. The sd across the five sets is the BatchNorm component of the jitter, and the 200-batch score shows what stable statistics would give.
   - *Step count (formerly 3.1b).* 5, 10, 20 and 40 midpoint steps at M = 8, seed 0, saved statistics.
   - *Tails.* Per-sample CRPS mean against median, the largest |member| in standardised space, and the fraction of non-finite members. This tells whether a few exploding pixels, as in lr1e-3, also drive the smaller jumps.
   - *Floor (P14).* The same members and truth mapped back to mm (inverse transform, clipped at 0) and then to log(mm + 0.01), and the CRPS in that space. It tests whether the jitter lives in the sub-0.01 mm values that the 1e-5 floor makes expensive.
   - *Acceptance (3.5).* `unet_seed42/europe_west__to__europe_west__none/best.pt`, MAE in standardised space and in log(mm + 0.01) on the same subset. A point forecast's CRPS is its MAE. If no AFM checkpoint beats it in both spaces, stop: the arm does not work yet, and choosing an LR is premature.

2. **Decide from the diagnostic.** The rules are fixed now. The epoch jitter J of a run is the sd of its epoch-to-epoch CRPS differences over epochs 6–12 divided by √2: 0.013 for 1e-4/1.0, 0.022 for 3e-4/0.1 and 0.0035 for 3e-5/0.1.
   - If the BatchNorm sd is ≥ 0.5 J in at least two of the three checkpoints, switch the flow net to `GroupNorm(8, C)`. The encoder keeps BatchNorm, so it stays identical to the UNet arm's `DualEncoderUNet` and claim 4's backbone match holds. A side benefit is that the flow net becomes immune to the target-forward BatchNorm mixing that Stage 2 rule 3 found in the UNet arm, so only the shared encoder carries it, as in the UNet. The switch needs a re-smoke for memory, and it voids the Stage A checkpoints.
   - Else, if the seed sd is ≥ 0.5 J, sampling noise dominates. Raise `--val_crps_chunks` and `--val_crps_members` until the seed sd is ≤ 0.25 J, measuring the extra validation time.
   - Else the jitter is in the weights. It grows with LR, which points this way. Keep an EMA of the weights (decay 0.999), and sample from it for validation and test, as is standard for flow models. This also needs a code change and retraining.
   - Steps: the smallest count whose CRPS is within 1 % of the 40-step value, used for selection and test alike.
   - If the floored CRPS shows the jitter is mostly in the sub-0.01 mm values, the selection space follows decision D1.

   **Step 1 result (read 2026-10-28).** All four jobs completed, about 70 min per task. Reproduction holds: seed 0 with saved statistics matches `best_val_loss` to within 3.3e-5 in all three runs, below the seed sd. Acceptance passes with a wide margin. The AFM CRPS is 0.271/0.280/0.282 against a UNet MAE of 0.523 in standardised space, and 0.340/0.364/0.354 against 0.476 in log(mm + 0.01). The AFM's own μ is worse than the UNet as a point forecast (MAE 0.60–0.63), which acceptance does not test. In units of each run's J (1e-4/1.0, 3e-4/0.1, 3e-5/0.1):

   | component | 1e-4/1.0 | 3e-4/0.1 | 3e-5/0.1 |
   | --- | --- | --- | --- |
   | sampling (seed sd) | 0.01 J | 0.01 J | 0.07 J |
   | flow-net BatchNorm (sd over 5 sets) | 0.26 J | 0.13 J | 2.26 J |

   BatchNorm reaches 0.5 J only in the smooth run, so the GroupNorm rule does not fire, and neither does the sampling rule. In the two runs that actually jitter, neither component explains it, so **the rule gives EMA of the weights**. Steps: 10 is within 1 % of 40 in all three, while 5 misses in 3e-4/0.1 (−1.6 %), so **10 steps stay**. More steps score slightly *worse* (by 0.1–0.5 %), so a little discretisation error helps. That is harmless for selection, but it means the test-time ensemble at 10 steps is not the converged ODE. The floored CRPS moves in proportion to the standardised one across BatchNorm sets, so there is no sign that the jitter lives below 0.01 mm. D1 is not forced by this.

   *The BatchNorm probe is biased.* Every re-estimate, including the 200-batch one, scores worse than the saved statistics, by 0.011, 0.003 and 0.015. In the smooth run that offset is 4 J, so it is neither sampling noise nor selection luck. The likeliest cause, unverified: training runs the encoder in train mode, so the saved flow-net statistics were collected on batch-statistics μ, while the probe used eval-mode μ. The two biases (this one, and a 10-batch cumulative average being noisier than momentum-0.1 statistics) can only lower the 2.26 J, so the decision stands. Two consequences. (a) `_adabn_afm` re-estimates every BatchNorm layer, and on source validation re-estimation alone costs up to 0.015 CRPS, so an AFM AdaBN result mixes that cost with any adaptation (D2). (b) The EMA averages the BatchNorm buffers too (`use_buffers=True`) rather than re-estimating them for the EMA weights.

   *Implemented 2026-10-28.* `train_afm.py --ema_decay` (default 0 = off, the Stage A behaviour) keeps `torch.optim.swa_utils.AveragedModel` with `get_ema_multi_avg_fn`, updated after every optimiser step. Every validation score, the selection and `best.pt` use the EMA copy, and `best.pt` stays a plain `AFMModel` state dict, so `evaluate.py`, `afm_val_diag.py` and AdaBN load it unchanged. The time constant at 0.999 is 1,000 steps, about half an epoch at batch 64. `run_exp_afm.sh` passes `EMA_DECAY=0.999` to every phase. Not yet run.

3. **Stage A′, 3 runs.** europe\_west, LR 3e-5, 1e-4 and 3e-4 at encoder ratio 1.0, with the fix from step 2, 12 epochs as before. About 11 GPU-h. 1e-3 is out because it diverged, and 1e-5 because it under-trains μ. Ratio 1.0 drops a hyperparameter that Stage A showed has no consistent effect, and it keeps the encoder LR nearer the UNet arm's 5e-4. The decision rule of 3.1a applies, on the fixed score. Script: `EMA=1 scripts/tune_afm_lr.sh`, tasks 0–2, writing `afm_lrtune_ema_seed42/`, preceded by an `EMA=1 SMOKE=1` smoke test. Submitted 2026-10-28: smoke 66669928, then array 66669929 (tasks 0–2), `afterok` on the smoke test. Before trusting the result, check that the EMA history is smoother than Stage A's at the same LR. If J does not fall, the jitter was not in the weights after all, and the step 2 reading was wrong.

   **Stage A′ result (read 2026-10-28).** All three tasks completed (3.5 h each, about 1,050 s per epoch), stopped after epoch 12 as planned. The jitter is gone. J over epochs 6–12 is 3.4e-4 at 3e-5, 2.3e-4 at 1e-4 and 3.4e-4 at 3e-4, against 0.013 for 1e-4/1.0 in Stage A, a 40–60× fall. Validation CRPS falls monotonically from epoch 3 onward in every run except for ±2e-4 wobbles at 3e-5. So the jitter was in the weights, and the step 2 reading holds. EMA also lowers the best CRPS by about 4 %: 0.2603 at 3e-4 (epoch 12), 0.2608 at 1e-4 (epoch 12) and 0.2662 at 3e-5 (epoch 9), against Stage A's best 0.2714. Under the 3.1a rule, 3e-4 and 1e-4 are within 0.23 %, so the lower LR wins: **LR 1e-4, encoder ratio 1.0, EMA 0.999**. Both are at their best at epoch 12 with CRPS still falling, so by the sweep's standard neither has converged. That is expected for a 25-epoch cosine cut at 12, and phase 1 runs the full schedule with patience. Per-epoch curves: `afm_lrtune_ema_seed42/europe_west/<config>/afm_europe_west__to__europe_west__none/history.csv`.

4. **No Stage B.** The three phase-1 runs at the chosen LR are the cross-domain check. If one diverges, or fails acceptance against the UNet diagonal on its own domain, the LR is revisited for that domain only.

5. **Launcher and evaluator work, before phase 1.** *Implemented 2026-10-29.* Submitted 2026-10-29: UNet acceptance pass 66685371 (`eval_array.sh MODEL=unet TREE=diag`, tasks 0–2) with merge 66685372, and AFM phase 1 66685381 (tasks 0–2, LR 1e-4, ratio 1.0, EMA 0.999, seed 42).
   - `run_exp_afm.sh`: every phase is an array, one run per task (`select_indices`). Phase 1 `--array=0-2` (DOMAINS order), oracle `--array=0-5 --time 24:00:00` (src-major pairs). Without `--array` each phase still loops, so `smoke_afm.sh` is unchanged.
   - `evaluate.py batch --only <tag> ...`: scores only the named runs and writes the CSV row to `<out>/rows/<tag>.csv`, because `append_csv` rewrites `results.csv` and concurrent tasks would drop each other's rows. `evaluate.py merge --output_dir <out>` builds `results.csv` from `rows/`, refusing an existing `results.csv` or duplicate keys.
   - New columns, every run: `mae_std` (the point-forecast CRPS, for acceptance), `mse_log_e001`, `mse_log_e01` (D1). Ensembles add `crps_log_e001`, `crps_log_e01` and a per-sample rank histogram (`rank_hist`, (n, M + 1) pixel counts, ties broken at random; `prob_metrics.rank_histogram`) in `per_sample_loss.npz`. Test ensembles now use fixed generators, so a re-score reproduces its row. `append_csv` keeps an existing CSV's header, so these columns appear only in new output directories.
   - `scripts/eval_array.sh`: `MODEL=afm TREE=none|oracle` (deterministic pass with the training split into `afm/<TREE>/`, then M = 8, 10 steps into `afm_prob/<TREE>/`) and `MODEL=unet TREE=diag` (the three diagonal UNet checkpoints into `unet_accept/`, the acceptance pass). `LIST=1` prints the task list without python; tasks check `N_RUNS` against the tree. Each task first runs `test_evaluate_metrics.py` (metrics against direct computation, re-score reproducibility, merge refusals) and the merge is a CPU `--wrap` job `afterok` on the array; see the script header.
   - `test_prob_metrics.py` covers `rank_histogram`.

6. **Then 3.5 steps 3–7**, with the phase-2 method list from D2. Phase 2 is planned in 3.7.

Cost to a decision on the LR: the diagnostic (≤ 36 GPU-h by the time limit, probably a few hours) plus Stage A′ (~11 GPU-h), against the ~18 GPU-h the cancelled Stage B and sampler check would have cost.

**3.7 AFM phase 2: do UDA methods still fail under CRPS? (planned 2026-11-02; submitted 2026-11-02: FDA 66790462, MMD gr0.5 66790463, tasks 0–5 each).** Purpose: claim 3 tested where the in-domain model is not a collapsed conditional mean. It is a falsification test, not a search for a working method; no tuning of λ or of the ratio.

1. *Methods, seed 42, all 6 pairs, 12 runs.* FDA (fixed DC; input space, no target forward, `--fda_beta` default) and MMD with `--uda_grad_ratio 0.5` (GAP'd encoder bottleneck, the GAP-ablation setting; the ratio pins the alignment share of the encoder gradient, which a fixed λ against the flow loss would not). Two submissions, because `--uda_grad_ratio` is refused for FDA: `SEED=42 PHASE=2 METHODS_ENV="fda" sbatch --array=0-5 scripts/run_exp_afm.sh` and `SEED=42 PHASE=2 METHODS_ENV="mmd" UDA_GRAD_RATIO=0.5 sbatch --array=0-5 scripts/run_exp_afm.sh`. ~4–8 GPU-h per run, 50–100 GPU-h. Not included: DANN (discriminator; its UNet runs blew up in mm), joint\_ot (rationale void, D2), spectral (acts on μ only), CORAL/mmd\_ms (out of scope). AdaBN is an optional cheap null (one short run per pair, re-estimates BatchNorm on the sampler's trajectories) but carries the BatchNorm re-estimation cost measured in 3.6 (up to 0.015 CRPS on source validation), so its result mixes that cost with adaptation; add it only as a labelled control.
2. *Mechanism caveat to state with the result.* Feature losses reach only the shared encoder that conditions the flow net; the flow net gets no alignment gradient. A null MMD result is partly expected by construction and extends claim 3's bottleneck argument; FDA is the method that can move what the sampler sees.
3. *Scoring.* `eval_array.sh MODEL=afm TREE=none` lists every `afm_*` run in `afm_seed42/`, so it would re-list the 9 phase-1 runs (they are skipped, metrics.json exists) and `evaluate.py merge` refuses the existing `afm/none/results.csv`. Needs a small launcher change before scoring: a `TAGS_RE` filter (e.g. `__(fda|mmd)$`) and an output sub-directory (`afm_p2/`, `afm_prob_p2/`), so phase 2 merges into its own CSV. *Implemented 2026-11-02* in `eval_array.sh` (`TAGS_RE`, `OUT_SUFFIX`; a filter without a suffix is refused; defaults unchanged, the plain listing still gives the 9 phase-1 runs). Scoring chained 2026-11-02: array 66790506 (tasks 0–11, `afterok` on 66790462 and 66790463), merges 66790507 (`afm_p2/none`) / 66790508 (`afm_prob_p2/none`). 12 tasks at ~2.2 h. *Void 2026-11-03:* MMD failed at launch (no side tree), so this chain never runs; re-planned as FDA scoring into `afm_p2fda/` and MMD gr0.5 in `afm_seed42_mmdgr05/` scored into `afm_p2mmdgr05/` (see "Read 2026-11-03"). Then λ̂\_CRPS, gap and recovery from the phase-1 rows, the oracle rows (66790277) and the phase-2 rows.
4. *Readouts per pair.* Headline `crps_log_e01` on the target; the same on the source (the trade); spread and spread–skill ratio; mm-space bias, wet fraction (> 0.1 mm/h) and maximum (the blow-up check); recovery = (source-only − method)/(source-only − oracle) only where the source-only gap exceeds the noise floor below, so not into horn. Field plots with `compare_fields.py` (needs an AFM path: ensemble member and ensemble mean) for europe→melanesia and melanesia→europe.
5. *Noise floor with one seed.* The UNet's source-only seed range at 0.1 mm is up to 0.064 (europe→melanesia), the same size as FDA's UNet gains (0.03–0.06). So one AFM seed can only show a large effect or a failure. Rule, fixed now: a cell counts as a recovery only if the target gain exceeds the UNet source-only seed range for that pair in the same metric family and the source cost is smaller than the gain; any such cell is re-run at seeds 43 and 44 (method run plus its source-only and oracle) before it is reported. If no cell passes, report seed 42 as is with the rule stated.
6. *Predictions, recorded before the runs.* MMD: inert or harmful on every pair (GAP plus no flow-net gradient). FDA: small target gains into melanesia paid on the source, as in the UNet. Into europe\_west from tropical sources: the near-dry ensemble (spread ≈ 0.01 mm) persists under both methods. Claim 3 survives the change of objective if no cell passes 5; it fails under CRPS, and needs rewriting, if a method passes 5 on two or more pairs after the re-seeding.

## Stage 4 — the write-up

The results section reorders to follow the four claims, which it currently does not.

| Section | Carries | Figure |
| --- | --- | --- |
| Shift diagnosis | Claim 1 | W1\_cond against the null, per pair |
| Transfer matrix and oracle | Claim 2 | Transfer matrix with λ̂, mean ± sd over seeds |
| Structure-aware evaluation | Sets up claim 3 | FSS across scale; RAPSD raw and normalised |
| Why the methods fail | Claim 3 | Mechanism figure, three panels |
| Model-family invariance | Claim 4 | UNet and AFM transfer gaps side by side, in their own risks |
| Case studies | Reader's intuition | Six-column panel, one row per pair |

Two positioning paragraphs are needed in the related work, and they are the difference between the paper surviving review and not.

**Against Qian et al. (2026).** They report good cross-region transfer for a wavelet diffusion model over six US regions. Their low-resolution input is a 10 × 10 block average of the high-resolution target — same product, same timestamp — so P(Y|X) is close to the same map everywhere and only the marginal moves. They state the boundary themselves in their §4.3. The conditional shift you measure cannot arise in their design, which makes their result a useful contrast rather than a contradiction: marginal shift without conditional shift transfers, yours does not.

**Against Glawion et al. (2025).** Harder, because spateGAN-ERA5 is genuinely cross-product, trains on German radar only, and claims robust global applicability including the tropics. Three points, in this order: their generalisation is of structural and distributional realism, not pointwise accuracy, and they say themselves that interpolation and rainFARM win on RMSE and MAE; their raw RAPSD underestimates power at all wavelengths outside Germany and the alignment claim is made on the normalised spectrum, so a marginal-shift amplitude error is divided out before the claim; and their evaluation target changes with geography — gauge-adjusted RADKLIM, non-adjusted MRMS, heterogeneously adjusted AURA — so target-product shift is confounded with geographic shift, where RainShift's single IMERG target is clean.

The honest concession to make in the discussion: your claim is about pointwise risk and about what unlabelled target data can identify. It does not say a generative model cannot produce structurally realistic target-domain fields. Say that before a referee says it for you.

### Literature review (2026-10-28): what the paper is, against what exists

**The gap is real.** Every recent paper on cross-region precipitation downscaling measures how far transfer falls. None runs learned UDA methods or asks why transfer fails:
- RainShift, Sci Rep 2026, doi 10.1038/s41598-025-34557-4: input QM and few-shot fine-tuning only; feature-alignment UDA as future work.
- Harder et al., arXiv:2512.01400: a WGAN over 15 training areas, with drops of 3–46 % and four outright failures, and no adaptation.
- Perfect-model and same-product designs, where transfer works: Singh et al., GMD 19:7545 (2026); CORDEX-ML-Bench, arXiv:2606.29172; Qian 2026.

On the ML side, careful negative studies of UDA are well cited (Musgrave et al. 2021, "UDA: a reality check"; Sagawa et al., ICLR 2022), and feature alignment is known to be weak for regression (DARE-GRAM, CVPR 2023; arXiv:2401.13721). A diagnosis on real scientific data with a continuous target has an audience on both sides. The topic is also timely, with five papers on downscaling transfer in the past year, and Harder/Rolnick list UDA as their next step.

**Not new; cite, do not claim.**
- That alignment fails under conditional shift: Zhao et al. 2019 and Johansson et al. 2019.
- That the joint distribution is not identifiable from unlabelled target data without assumptions: Kong et al. 2023, arXiv:2306.06510.
- Measuring λ with labelled data from both domains: Chen et al., ICML 2019 (BSP), §2.2. They train an MLP head on frozen DANN and ResNet-50 features "over all data with labels from both source and target domains … the target labels are only used for this pilot analysis". They find a much higher joint error on DANN's features, i.e. adversarial alignment raised λ of the representation.

Our oracle differs: it trains the whole network jointly, so it measures λ of the *hypothesis class* (a good joint predictor exists), where Chen et al. measure λ of a *given representation*. Claim 2's contribution is therefore empirical: λ is small *here*, so the failure is one of identifiability, not capacity. Claim 2 is phrased as applying an established measurement. P17 is Chen et al.'s measurement, applied to our methods.

**The organising thesis.** The conflicting transfer reports in the literature sort by one property of the design: whether P(Y|X) is shared across regions by construction.
- Shared, so transfer and UDA work: perfect-model or same-product designs (Qian 2026, Singh 2026, the CORDEX-ML-Bench perfect framework, which itself calls its errors lower bounds, and the COSMO experiments of EGU26-19396).
- Not shared, so transfer fails and unlabelled data cannot fix it: reanalysis-to-observation across regions (RainShift, Harder et al. 2512.01400).

Claim 1's null-calibrated measurement is what makes this more than an assertion. RainShift's introduction already asserts that the "statistical relationships" may not be stationary across space, and that "precipitation depends not only on conditions but also on dynamics, which vary across regions"; we measure it. This is a stronger headline than "six UDA methods fail", which on its own reads as a benchmark table. The introduction and discussion are rewritten around it.

**Our own EGU26-19396 abstract** (Quarenghi and Beucler) reports that simple strategies improve cross-domain consistency on COSMO and calls UDA "a viable pathway". The paper must reconcile that, not ignore it. The thesis above does: COSMO is a shared-conditional design. One sentence in the discussion.

**Lead with the directions that matter.** In practice the data-rich region is the source: europe→horn and europe→melanesia. into-Europe directions are a control, and their gains were mostly the log floor anyway (P3).

**Claim 3 against "property of the task".** If global average pooling makes feature alignment inert, a referee will call that a property of the architecture. The grid4 arm of the GAP ablation (alignment that acts, and still trades source for target) and P17 have to carry this. Otherwise claims 2 and 3 pull against each other.

**Required additions, in priority order:** P15 (QM), P16 (few-shot curve), the claim 3 rewording above, then AFM (Stage 3). QM is the item a RainShift author on the panel would raise first. The few-shot curve turns "unlabelled data cannot identify the conditional" into a recommendation people can act on: how many labels can. AFM still matters, because "does a generative model fix it?" will be asked given RainShift's GAN and diffusion baselines. But it is the most expensive item and the least likely to change the conclusion.

### Paper audit (2026-10-05): fixes and open checks

A read of `report/main.tex` against the code and `tables.md` found the items below. The wording errors are already fixed in the draft (abstract, intro claim 3, Section 4 and 5 text, setup, appendix stub). What remains needs a job or a lookup. Each one has a red `\todo` in the draft, except P3, P5 and P6, whose text is already in the draft. P3 and P8 can change claims, so they come first; none of them is a UDA experiment.

| id | check | blocks | where |
| --- | --- | --- | --- |
| P1 | Recompute the matched-input null with a split by blocks of time (for example by year), not by random pixel. The current null (`conditional_shift_binned.py`, `null_baseline`, `rng.permutation(p.size)`) leaves neighbours in both halves. Restate the 86.9× and 14.5× factors. *Scripted 2026-10-29*: `null_baseline` takes `mode` pixel (the old split; repetition 0 reproduces the Aug 10 value), week (168-h blocks assigned at random, 20 repetitions) or year (test year 1 against year 2, which also carries interannual variability). The verdict uses the larger block-null mean of the two domains. Same sample as Table 1 (600 frames, seed 42, 5 × 5 bins); writes `unet/conditional_shift_binned_p1.json` and leaves the Aug 10 file alone. `scripts/run_cond_binned_test.sh` now uses `python_env.sh`. Submitted 2026-10-29: job 66685373. **Result (read 2026-10-29).** W1\_cond is identical to Aug 10, so it is the same sample. The block nulls are 5–10× the pixel null: week 0.29 ± 0.13 (europe), 0.17 ± 0.05 (horn), 0.17 ± 0.05 (melanesia); year 0.34 / 0.19 / 0.21. Excess over the larger block null of the pair is 6.2× (europe↔horn, week 7.4×), 1.9× (horn↔melanesia, week 2.3×, 1.5× the melanesia week p95), and 6.3× for europe↔melanesia, which is still uninterpretable at 44 % support. The 86.9× and 14.5× factors are withdrawn. Only europe↔horn is a clear excess. horn↔melanesia, the pair with full support, is weak. Two weeks of the same domain differ in P(log P \| CAPE, TCW) by 0.17–0.29, so two binned inputs leave much of the conditional unexplained. That makes P2 (9-D) necessary for claim 1, not just corroboration | claim 1 | CPU or GPU job, `scripts/run_cond_binned_test.sh` |
| P2 | Model-based 9-D diagnostic from `conditional_shift.py`, as corroboration. Needed because the binned test conditions on CAPE and TCW only and is a bound in neither direction. *Scripted 2026-10-29; `conditional_shift.py` rewritten.* The old version never produced a result and could not have been read: it fed model A inputs normalised with B's statistics, compared outputs in two different std spaces, and had no null. The new test runs every source-only model on every domain's test split, each model in its own input space, and scores in log(P + 0.1) and log(P + 0.01) mm. For estimates with independent, centred errors, E d(f\_A, f\_B) = (N\_A + N\_B)/2, so R = D\_AB / mean(N\_A, N\_B) is 1 under a shared conditional, and b2 = D\_AB − mean(N\_A, N\_B) estimates the squared difference of the conditional means. The primary null is split-half models: `train_unet.py --train_half` trains and selects on alternate 365-day blocks, seed 42, six runs (`scripts/run_half_split.sh`). The seed null (42/43/44) is initialisation only and is reported as liberal. Support: a cross-fitted logistic classifier on 4 × 4 block means of the dynamic inputs, with folds by chunk. The overlap is B's samples with P(A\|x) ≥ 0.1 on dyn7 (no sp, tisr); the dyn9 AUC is reported. Chunk bootstrap. The test is asymmetric: R ≈ 1 on the overlap is evidence of a shared conditional; R ≫ 1 is necessary for conditional shift but not sufficient, because approximation error is weighted by each model's own inputs, and statics, sp and tisr are outside the overlap. CPU checks: `test_conditional_shift.py`. Launch: `scripts/run_cond_test.sh`, `afterok` on the half-split array | claim 1 | 6 training runs (~1–2 A100-h each) + one inference job |
| P3 | Field-mean (zero-rain) check of the seeded methods into europe_west, after pointing `field_mean_check.py` line 147 at `best_adabn.pt`. Decides whether the horn→europe and melanesia→europe gains, and above all AdaBN and FDA on horn→europe, are a drier-prediction artefact | claim 3, abstract | done 2026-10-05, read 2026-10-27; see "P3 result" below |
| P14 | Choose the headline log floor. Training and every `mse_std` number use log(P + 1e-5 mm), where values below 0.01 mm carry most of the europe\_west target error. Re-score the transfer matrix, oracle decomposition and method table with a 0.01 mm floor (as in `field_mean_check.py`, `FLOORS_MM`) for every target, not just europe\_west, and decide which floor is the headline (decision D1) | claims 2 and 3, AFM selection | job: generalise `field_mean_check.py` to all targets, or add the floored metric to `evaluate.py` |
| P4 | Time-block null and oracle capacity: horn→melanesia leaves λ̂ = 0.41 ± 0.15 (18 ± 9 % of the gap). Either a stronger oracle closes it, or it is reported as irreducible | claim 2 | one oracle run |
| P5 | Which statistics `evaluate.py` uses for AdaBN source risk. It scores source with `best.pt` (source statistics), which is why AdaBN shows no source inflation. The partial-AdaBN account of the other methods' inflation has to be worded accordingly | claim 3 | read the code |
| P6 | The GAP ablation and BatchNorm control (Stage 2 panel c): re-smoke tasks 3, 6, 9, then the 12-task array and `compare_gap_ablation.sh` | claim 3 | done 2026-10-06; read-out in Stage 2 panel c Result |
| P7 | The oracle's early-stopping setting, from its `config.json` (the draft says "check") | setup text | read |
| P8 | Test metrics of `unet_seed42_a100check/`, scored against the GH200 seed spread. Until then the five A100 methods' Δ carry an unquantified stack difference; only JDOT is compared like with like | all Δ tables | one eval job |
| P9 | Confirm the split sizes 17,520 and 122,520 from `results.csv`; 122,520 is not a multiple of the 200-sample chunk | setup text | read |
| P10 | References: Qian 2026, Glawion 2025 and IMERG are placeholders; verify the Fotiadis title (adaptive flow matching, arXiv 2410.19814) and the ERA5 entry; check the Qian and Glawion summaries in the related work against the papers | submission | by hand |
| P11 | Write the dropped-methods appendix (CORAL, multi-scale MMD): what was run and why each was dropped | appendix | text |
| P12 | Replace "ordering not stable" by a statistic: Kendall's W per pair, or the stable pairwise comparisons. The per-seed order differing is the default for seven models | claim 3 | `aggregate_seeds.py` |
| P13 | Decide whether to keep the NGG appendix; per pair, or rescale by s_S² before averaging | appendix | decision |
| P15 | **Input quantile mapping (QM), RainShift's own adaptation baseline.** It is required, not optional; see "Literature review" below. Map each target input channel's marginal to the source's, using 1,000 quantiles fitted on the source and target *train* splits, pooled over pixels and time and fitted in physical units. Then score the existing source-only checkpoints on the mapped target test set. Two arms: (a) `tp` and `cp` only, which is what RainShift maps, and (b) all nine dynamic channels. The static inputs are never mapped. No target labels, no training: it is an evaluation pass over 6 pairs × 3 seeds × 2 arms, scored in both floors (P14). Reading: if QM recovers little where the oracle recovers almost everything, that is direct evidence for claim 1, since a marginal fix cannot fix a shifted conditional. If QM beats the learned methods, claim 3 and the abstract are reworded before anything else. *Scripted 2026-10-28.* The QM in `quantile_map.py`/`evaluate.py --input_transform qm_*` had never run, and could not have: it read `train_x.npy`, which does not exist (the data are Zarr), from the first 2,000 samples (one season). Its `np.interp` over tied anchors also mapped the target's whole dry mass to the source value at the top of that mass, which is wet when the source is less often dry. Fixed: `fit_qm_from_zarr` reads 40 chunks spread over each training split, in the model's input space with source stats; `transform` uses the mid-rank of tied runs and runs on the GPU; `test_quantile_map.py` checks it. The vehicle is `field_mean_check.py --input_qm tp precip all`. The arms are `tp` (RainShift's), `tp`+`cp`, and all nine. It scores QM rows beside every unmapped model, with floors, bins and wet fractions, so the same pass is P14 for each target. Output: `results/seed<N>/field_mean_check_qm/<target>/`. Submitted 2026-10-28: melanesia seed 42 as 66670277; then, `afterok` on it, melanesia 43/44 (66670279), horn-of-africa (66670280) and europe\_west (66670281). *Result (read 2026-10-28, all 9 tasks, `results/seed<N>/field_mean_check_qm/<target>/summary.csv`).* `test_quantile_map.py` passed: the old `np.interp` transform had mapped 100 % of dry target pixels wet. The unmapped rows reproduce `results.csv` `tgt_mse_std` exactly. Seed means in `mse_log` (gap = source-only − oracle): **no arm helps consistently, and the oracle recovers 2.7–5.6 where QM recovers at most 0.9.** `tp` (RainShift's arm) worsens `mse_log` in 4 of 6 pairs, worst into horn: europe→horn 5.66 → 7.07, more than the whole oracle gap of 1.06. It makes the europe→melanesia models drier still (predicted wet fraction at 0.1 mm 0.015 → 0.007 against an observed 0.20), because Melanesia's tp sits above Europe's in source units and QM maps it down. `all` closes 16 % of the gap at best (horn→europe), 7 % on europe→melanesia (and 21–52 % in the 0.1 mm floor), and wrecks the pairs into horn (melanesia→horn 5.21 → 11.26). The ranking of arms changes with the floor (D1). On europe→melanesia, `all` is not driven by tisr: tisr's medians differ by 0.07 source sd, against 4.9 for tcw. This is the evidence claim 1 asked of P15, and it matches RainShift only in that QM is weak; their finding that QM on tp helps CRPS does not reproduce in this metric | claims 1 and 3, abstract | `QM="tp precip all" TARGET=<t> sbatch --array=42,43,44 scripts/field_mean_check.sh`, 9 tasks of ~15 min |
| P16 | **Few-shot label curve: how many target labels close the gap.** This is the constructive counterpart to claim 2. Fine-tune each source-only checkpoint on N labelled target samples, N ∈ {200, 2,000, 20,000} (1, 10 and 100 chunks, spread evenly over the target train split as in `build_crps_loader`, not the first N, which would be one season). The recipe is fixed with no selection, since selecting on target validation would make it an oracle: a fixed step budget (e.g. 500 steps) at LR/5, as in RainShift's fine-tuning (2,000 samples, LR/5), and the last checkpoint is scored. 6 pairs × 3 N × 3 seeds = 54 short runs. Report the fraction of the oracle budget recovered against N, beside the UDA methods' recovery at N = 0. The chunk draw adds variance; the seed reseeds it. *Scripted 2026-10-29* as a separate `finetune_unet.py` rather than new `train_unet.py` flags, so the main training path is untouched: init from `unet_seed<S>/{src}__to__{tgt}__none/best.pt`, N ∈ {1, 10, 100} chunks spread evenly with a seed-drawn phase, 500 steps at base LR / 5 constant, base weight decay and batch, fp32, `last.pt` and no selection. `scripts/run_fewshot.sh`, 18 tasks per seed, scores each run with `evaluate.py batch --only <tag> --ckpt_name last.pt` into `results/seed<S>/unet_fewshot/`. Headline is `mse_log_e01` (D1); source-only and oracle at that floor come from `field_mean_check_qm`. Submitted 2026-10-29: smoke 66685374 (task 3, europe→melanesia fs1, 20 steps); arrays 66685375 / 66685377 / 66685379 (seeds 42/43/44, tasks 0–17) `afterok` on it; merges 66685376 / 66685378 / 66685380 | claim 2, discussion | new code: `train_unet.py --init_ckpt` and `--train_chunks_spread N`, and a launcher. Short GPU array |
| P17 | **Optional. λ of each method's representation (Chen et al. 2019).** Freeze each method's encoder and retrain only the decoder on labelled source + target, as the joint oracle does. A λ̂ above source-only's means the alignment destroyed conditional information the joint predictor needed. That is the mechanism of claim 3 in Ben-David's own terms, measured rather than argued. Start with seed 42 on europe→horn and europe→melanesia, all six methods (12 runs); extend only if the effect is clear | claim 3 mechanism | new code: `--freeze_encoder --init_ckpt` in `train_unet.py` |
| P18 | References from the 2026-10-28 review into `refs.bib`: RainShift Sci Rep 2026 (replacing or alongside the arXiv entry), Harder et al. arXiv:2512.01400, Singh et al. GMD 2026, CORDEX-ML-Bench arXiv:2606.29172, Prasad et al. arXiv:2407.12517, Musgrave et al. arXiv:2111.15672, Sagawa et al. ICLR 2022, Kong et al. arXiv:2306.06510, Chen et al. ICML 2019, Nejjar et al. CVPR 2023 (DARE-GRAM), arXiv:2401.13721, Beucler et al. Sci Adv 2024, Zhang et al. ICML 2013 | related work | by hand |
| P19 | **Input-set ablation: does λ̂ ≈ 0 rest on the model telling the domains apart?** The static inputs (lsm, z) and the near-static ones (tisr; sp, which within a patch is mostly elevation) let the joint oracle read the domain off its inputs. Then λ̂ ≈ 0 shows that the hypothesis class can hold two conditionals, not that a shared one exists. Omitted inputs push the other way: dropping a cause can create conditional shift while shrinking the P(X) shift. The arms come from OODiag's whole-field Shapley attribution of europe→melanesia (Beucler et al. draft, 2026-10-28, App. F.3 and Fig. 8; Harder et al.'s source-only U-Net, MAE, shift gap 0.246 mm). `mask_geo` = lsm, z, tisr, sp, together 5.8 % of the gap. `mask_shift` = cape, tcw, together 72.4 %. Each arm retrains source-only europe\_west, in-domain melanesia and the oracle at seed 42, with those inputs zeroed in every domain. Architecture and seeded initialisation are identical to the full model (`test_input_mask.py`). Reading `mask_geo`: OODiag predicts the source-only target risk barely moves, so a rise in λ̂ isolates domain identification through the statics. λ̂ still ≈ 0 means the statics were not needed, but not that the conditional is shared, since Europe and Melanesia barely overlap in the dynamic inputs either (OODiag's support audit passes 1 of 4,092 references). Reading `mask_shift`: a fall in source-only target risk means cape and tcw carried shift that hurt more than they informed. That would be the first retraining test of an OODiag attribution. A rise means their omission costs more than their shift. Baseline: the existing GH200 seed-42 runs. The stack difference is about 10× below seed spread in training loss (2026-09-30 check), but λ̂ is a small difference, so an A100 full-input control is the first addition if the effects are small. Caveats: OODiag's attribution is an association under poor analog support, for a different U-Net and loss; it chooses the arms and is not evidence. One seed, one pair. *Scripted 2026-10-28*: `train_unet.py --mask_inputs`, `scripts/run_input_ablation.sh` (6 array tasks), scored by `TREE_SUFFIX=_mask_<arm> TARGET=melanesia scripts/field_mean_check.sh`. Submitted 2026-10-28: smoke 66673419 (task 2); array 66673420\_[0-5] `afterok` on it; scoring 66673421 (`mask_geo`, `afterok` on tasks 0–2) and 66673422 (`mask_shift`, tasks 3–5). *Result (read 2026-10-28, seed 42, `results/seed42/field_mean_check_mask_<arm>/melanesia/`).* All 8 tasks completed; `test_input_mask.py` passed; runs stopped at epochs 7–12. In `mse_log` on the melanesia test split, against the full-input seeds 42–44 (source-only 16.47–16.55, in-domain 13.62–13.64, oracle 13.74–13.84, λ̂ 0.10–0.22, gap source-only − in-domain 2.83–2.93). `mask_geo`: source-only 17.58, in-domain 13.78, oracle 13.81, λ̂ 0.04, gap 3.81. The oracle does not need the statics, so λ̂ does not rise; as anticipated this does not show a shared conditional. Source-only is worse but much wetter (mean ratio 0.57 against 0.05–0.27). `mask_shift`: source-only **15.98**, in-domain 14.26, oracle 14.59, λ̂ **0.33**, gap **1.72**. Source-only transfers better in every metric (0.1 mm floor: 0.96 against 1.03–1.10), while in-domain and oracle lose skill. The gap shrinks 40 % (to 0.02 from 0.14–0.22 in the 0.1 mm floor), and λ̂ rises to 19 % of it (4–7 % with all inputs). In both floored metrics the oracle scores worse than source-only on the target, unexplained. Reading: the bound's trade-off, input shift down and λ up. **λ̂ ≈ 0 is a property of the 11-input set, not of the task; claim 2 must be scoped to it.** Cautions: one seed. The arm was chosen with OODiag, which uses target per-sample errors, so this is not an unsupervised selection. An unlabelled marginal-shift ranking (the QM quantile gaps) would likely pick the same pair, but that is post hoc. For OODiag: the all-zero predictor's MAE on Melanesia is 0.313 mm, the field mean, and the full-input source-only models score 0.315–0.379. OODiag's L\_target of 0.323 mm suggests Harder's U-Net is near-dry there too, so per-sample MAE mostly measures how much it rained (response composition). Retraining does not follow the attribution: `mask_geo` (attributed 0.014 mm) raises MAE by 0.08–0.15 mm, and `mask_shift` (0.18 mm) leaves it unchanged. The all-zero predictor alone has a raw shift gap of 0.313 − 0.097 = 0.215 mm, 88 % of OODiag's 0.246 mm. Write-up and the proposed zero-predictor check: `report/oodiag_note.md` (2026-10-29). Seeds 43 and 44 submitted 2026-10-29: arrays 66685365 (seed 43) and 66685368 (seed 44), tasks 0–5 each; scoring 66685366/66685367 (seed 43, mask_geo/mask_shift) and 66685369/66685370 (seed 44), `afterok` on their arm's tasks. *Read 2026-10-30* (see "Read 2026-10-30" below): the gap shrink under `mask_shift` replicates, the λ̂ rise does not (seed 44: 0.09) | claims 1 and 2 | about 12 A100 GPU-h (oracle about 3 h, single-domain runs about 1.2 h each) |

After P3: if the Europe gains are a zero-rain artefact, claim 3's trade-off sentence and the abstract's Europe sentence are rewritten; if they are not, the AdaBN result needs its own explanation.

**P3 result (jobs of 2026-10-05, read 2026-10-27).** `results/seed<N>/field_mean_check/europe_west/` holds `summary.csv` and `bins.csv` (AdaBN scored with `best_adabn.pt`). Numbers are log-MSE in mm space, the units of the seed aggregate (`mse_log` = `mse_std` × s_S²). The training transform's floor is `log_eps` = 1e-5 mm.

- *Every foreign model into europe\_west predicts essentially no rain.* The field-mean ratio pred/obs is 0.000 for source-only and ≤ 0.004 for every horn→europe method. melanesia→europe gives 0.002–0.07, except AdaBN at 0.11–0.86. The predicted wet fraction at 0.1 mm/h is ≤ 0.005 against an observed 0.10. The in-domain europe model reaches 0.20–0.25 and the oracle 0.05–0.09.
- *The horn→europe gain is a floor lift, not rain.* Source-only predicts ~1e-5 mm everywhere. The methods predict ~1e-4 to 1e-3 mm, still zero for any physical purpose. In log(P + 1e-5) that costs on dry pixels (bin < 0.01 mm: 0.16–0.24 → 0.46–0.67 for FDA) and pays more on wet ones (0.1–1 mm: 6.9–7.1 → 5.9–6.2). Net, the methods sit at 11.2–12.7 against source-only's 12.7–13.0 and a best spatially constant predictor of 12.42 (`reference_predictors.csv`). With a 0.01 mm floor (`mse_log_e001`) the horn→europe gap is 1.80 vs 1.12 in-domain, about 0.68. The methods recover 0.00–0.03 of it, AdaBN 0.08–0.10. With a 0.1 mm floor nothing is left.
- *melanesia→europe keeps a partial gain, and the ordering changes with the floor.* At 1e-5, spectral, dann, mmd and joint_ot are best (9.6–10.7, below the best constant, so the floor carries some wet/dry information) and AdaBN is worst (13.0–14.6). At 0.01 mm, AdaBN (1.44–1.51) and FDA (1.32–1.56) are best, and spectral, dann, mmd and joint_ot give 1.61–1.76 against source-only 1.80. AdaBN on this pair is the only method that predicts a non-trivial amount of rain.
- *Consequence.* The Europe-target "recovery" in Tables 4–5 and claim 3 is mostly the floor. Claim 3's europe\_west sentence, the abstract's Europe sentence and the AdaBN paragraph are rewritten after P14. More generally, a metric that rewards moving 1e-5 mm to 1e-3 mm by more than an oracle's whole budget is a poor headline. P14 decides whether the paper reports log(P + 0.01 mm) and keeps 1e-5 as the training space.

**D1 decided 2026-10-29: the headline floor is 0.1 mm/h, from the literature.** Training stays in log(P + 1e-5 mm), which is RainShift's own transform (Harder et al., Sci. Rep. 16, 3733, 2026, "Data processing": log(x·1000 + 1e-5)). Neither RainShift paper defines a wet threshold; zero is the only one. Evidence for 0.1, all checked in the source: GSDR-I (Pritchard et al. 2023, Sci. Data 10, 393), "A threshold of 0.1 mm is used to distinguish wet hours"; Harris et al. 2022 (arXiv:2204.02028), rainy pixel > 0.1 mm/h; Glawion et al. 2025 (npj CAS 8, 219), lowest FSS threshold 0.1 mm/h, and ERA5 below 0.1 mm/h set to zero; NASA's IMERG–MRMS validation uses 0.1 mm/h, and Tan's IMERG slides use 0.2 mm/h (paper not checked). Against a tiny ε: Santos et al. 2018 (HESS 22, 4583) show log-transformed scores are oversensitive to the added constant. No ML-precipitation paper tests the ε sensitivity, so ours (P3, P14) is new. At 1e-5, values from 0 to 0.01 mm/h span 7 natural-log units that IMERG cannot resolve. Headline: `mse_log_e01` = mean (log(P̂ + 0.1) − log(P + 0.1))², P̂ clipped at 0. 0.01 mm/h and the training space are robustness rows. AFM selection stays in the training space, as the UNet arm's does (both select on standardised validation loss); only reporting moves to the floor. `evaluate.py` now writes `mse_log_e001`/`mse_log_e01` and, for ensembles, `crps_log_e001`/`crps_log_e01`.

**P14 result at 0.1 mm/h (read 2026-10-29, no new job).** `results/seed<N>/field_mean_check_qm/<target>/summary.csv` already holds every method, the oracle and the in-domain model for all three targets and seeds. Seed means of `mse_log_e01` (sd over seeds ≤ 0.05, mostly ≤ 0.01):

| pair | in-domain | source-only | oracle | gap | λ̂ | λ̂ / gap |
| --- | --- | --- | --- | --- | --- | --- |
| horn → europe | 0.3046 | 0.4485 | 0.3498 | 0.144 | 0.045 | 31 % |
| melanesia → europe | 0.3046 | 0.4485 | 0.3507 | 0.144 | 0.046 | 32 % |
| europe → horn | 0.2213 | 0.2236 | 0.2199 | 0.002 | −0.001 | — |
| melanesia → horn | 0.2213 | 0.2277 | 0.2175 | 0.006 | −0.004 | — |
| europe → melanesia | 0.8788 | 1.0686 | 0.8971 | 0.190 | 0.018 | 10 % |
| horn → melanesia | 0.8788 | 0.9750 | 0.8698 | 0.096 | −0.009 | −9 % |

What changes against the 1e-5 headline:
- *Horn as target has no gap.* Every model, foreign or not, scores 0.218–0.230 except AdaBN (0.24–0.27) and melanesia → horn QM-all (0.240). Horn is the driest domain (field mean 0.055 mm/h), and at 0.1 mm/h little is left to predict. Four pairs carry the paper, not six.
- *Europe as target: λ̂ is a third of the gap, not ≈ 0.* Both source-only models score 0.4485 with seed sd ≤ 0.0003, the same as the precip-QM rows; with P3's field-mean ratios ≈ 0 this is the all-zero predictor (inferred, not yet scored). The oracle closes two thirds of the gap. Claim 2 holds for Melanesia as target and must be scoped away from Europe as target, or the Europe oracle strengthened (NOTES item 10 has the same question for horn → melanesia).
- *Two method results now look like recovery; check before believing them.* melanesia → europe: FDA 0.399 (sd 0.025) and AdaBN 0.403, i.e. 51 % and 47 % of the oracle budget, against source-only = all-zero. europe → melanesia: QM-all 0.995 (sd 0.006), 43 %. Everything else is within ~10 % or worse. The prior from P3 is that into-Europe gains come from predicting some rain at all against a dry source-only model, so the comparison to make is with a trivial wet predictor (e.g. the source model rescaled to the target's wet fraction), not only with source-only.
- *P19 at this floor:* full-input gap 0.14–0.22 (seeds 42–44), `mask_shift` 0.02 (seed 42), with its oracle worse than its source-only. The `mask_shift` reading was already floor-dependent.
- *D2* is re-decided from this table.

Not yet scored at 0.1 mm/h: the all-zero and trivial-wet references (`field_mean_check.py` reports only `mse_log`/`mse_std` for them), and seed sd of the paired λ̂.

**Code audit (2026-10-29): `report/audit_2026-10-29.md`.** Results it invalidates or qualifies:
- FDA sign-flips the target mean, so all 18 FDA runs are void (A1).
- DANN never aligns (A2).
- λ̂ is one-sided (A3).
- The P1 binned null misses the within-bin covariate mismatch (A4).
- NGG uses own-domain normalisation (A5).
- Phase-2 patience was 5, not −1 as stated above (A6).
- P16 re-estimates BatchNorm on the few-shot subset (B1).
- `aggregate_seeds` mis-scales the D1 metrics (C).
- P2 needs replicated, decorrelated half models before it is run (D).

**Audit fixes written 2026-10-29.** `scripts/run_tests.sh` (job 66698749) passed all 11 test files; no fix has yet run on real data.

Training and P16:
- FDA uses the target's full DC coefficient and a symmetric window (A1).
- DANN takes `--uda_grad_ratio`; at 0 it is bit-identical to before (A2).
- Sinkhorn reports its marginal errors, and `diagnose_jdot.py` covers 8 batches in train-mode BatchNorm (A8).
- P16 freezes BatchNorm and clips gradients (B1). The P16 jobs 66685374–80 are cancelled and not yet resubmitted.

Aggregation and NGG:
- `aggregate_seeds`:
  - scales only `*_std` metrics;
  - renames λ̂ to `lambda_tgt`;
  - adds `lambda_joint = lambda_tgt + oracle_src` (A3).
- NGG takes W1 from the raw cube ÷ s_S into `ngg_v3/` (A5). The cube predates Zarr: its inputs were upsampled before the log.

Evaluation:
- `evaluate.py` scores only finished runs (config.json newer than every .pt), and `merge` takes `--expect`.
- New metrics:
  - floored and wet-only rank histograms;
  - `ens_var`, `ensmean_mse` and `ssr`.
- `single` and `ot` are removed.

AFM arm:
- Discriminator LR and wd match the UNet arm.
- `--uda_grad_ratio`, `--mmd_pool` and `--mmd_unbiased` are added.
- AdaBN re-estimates on the sampler's trajectories.
- Target batches are drawn only when a method uses them, and the target-test pass is gone. The phase-1 path is unchanged.

Launchers:
- patience 5/8 by default, and abort if a `base_hp` is missing;
- `compute_stats` cannot write into `raw_data`;
- `compute_shift` refuses normalized mode;
- `copy.sh` and `run_conversion.sh` are moved to `_trash/`.

Shift tests:
- P1b: the binned test gets a conditional-permutation null, the D1 floors, wet fraction and wet amount, support counted in weeks, a week bootstrap and a p-value. It writes `_p1b.json`; `_p1.json` is kept.
- P2:
  - 18 half models (seeds 42–44 for half 0, 45–47 for half 1);
  - D from cross-year pairs only, with three Latin quadruples;
  - overlap at 0.1, 0.3 and 0.5;
  - 4-chunk bootstrap blocks;
  - a positive control and a `--from_npz` mode;
  - per-domain npz files written as each domain finishes;
  - an optional masked-statics arm.

AFM phase 1 (66685381) and the acceptance pass (66685371) were released after the tests passed.

Retraining and re-scoring needed:
- FDA: 18 runs.
- DANN with `--uda_grad_ratio`: 18 runs, if kept.
- Re-score the main and GAP-ablation trees for `mse_log_*`, including the oracle source side at 0.1 mm/h.
- Regenerate `ngg_v3/`.

**Submitted 2026-10-29 (after tests passed):**
- FDA retrain (fixed), side tree `unet_seed<S>_fdafix/`: arrays 66708660 / 66708664 / 66708668 (seeds 42/43/44, tasks 0–5); scoring into `results/seed<S>_fdafix/`: 66708661 / 66708665 / 66708669.
- DANN with `--uda_grad_ratio 0.5` (as in the GAP ablation), side tree `unet_seed<S>_danngr05/`: arrays 66708662 / 66708666 / 66708670; scoring into `results/seed<S>_danngr05/`: 66708663 / 66708667 / 66708671.
- Main-tree re-score (all `mse_log_*`, oracle included) into `results/seed<S>_v3/`, with `ngg_v3/` computed at the end: 66708672 / 66708673 / 66708674. This NGG still holds the void FDA rows of the main tree. The GAP-ablation trees are not re-scored yet: `compare_gap_ablation.py` reads fixed paths.
- P1b: 66708675 → `unet/conditional_shift_binned_p1b.json`.
- P16 (BN frozen): smoke 66708676; arrays 66708677 / 66708679 / 66708681 `afterok` on it; merges 66708678 / 66708680 / 66708682.
- P2: half-split smoke 66708729; array 66708730 (0–17%6) `afterok` on it; `run_cond_test.sh` 66708731 `afterok` on the array.

**Read 2026-10-30.** Numbers are target-test `mse_log_e01` (the D1 headline) unless stated; recovery is (source-only − method) / (source-only − oracle), per seed 42/43/44.

- *P19, seeds 43 and 44.* The seed-42 reading does not fully replicate. Robust over three seeds: `mask_shift` lowers europe→melanesia source-only `mse_log` (15.98 / 15.88 / 15.92 against 16.47 / 16.55 / 16.55) and raises in-domain melanesia (14.26 / 14.09 / 14.30 against 13.64 / 13.62 / 13.62), so the gap shrinks to 1.72 / 1.79 / 1.63 from 2.83–2.93; at the 0.1 mm floor it shrinks to 0.019 / 0.044 / 0.058 from 0.216 / 0.213 / 0.141. About half the shrink is the in-domain model losing skill, not transfer improving. Not robust: λ̂ under `mask_shift` is 0.33 / 0.29 / **0.09** in `mse_log` (full inputs 0.11 / 0.22 / 0.17) and 0.040 / 0.012 / 0.035 at 0.1 mm (full 0.056 / 0.007 / −0.008), so the seed-42 rise is one seed's draw and the ranges overlap. The seed-42 oddity of the oracle scoring worse than source-only at 0.1 mm does not recur (seeds 43/44: oracle 0.957 / 0.982 against 0.989 / 1.005). `mask_geo`: λ̂ 0.03 / 0.20 / 0.18 in `mse_log`, inside the full-input range; at 0.1 mm 0.053 / 0.063 / 0.068, slightly above it. Source-only is worse in all three (17.58 / 17.01 / 17.17), and two seeds have mm-space blow-ups (RMSE 79.7 and 16.4 mm). The bold reading in the P19 row ("λ̂ ≈ 0 is a property of the 11-input set") rests on seed 42 and should be withdrawn; what holds is that removing cape and tcw trades in-domain skill for a smaller source-only gap, with λ̂ small in every arm. Still caveated: masked arms ran on A100/dl-torch, the full-input baselines on GH200.
- *P1b* (`unet/conditional_shift_binned_p1b.json`, log `/scratch/fquareng/outputs/66708675`). The positive controls pass: under pure covariate shift the old week null flags 10/10, the conditional permutation 0/10; injected ±0.15 conditional shift gives p = 0.025. europe↔horn: W1\_cond 0.281 against null mean 0.032 (p95 0.043), p = 0.010 (the floor of the permutation count), bootstrap CI [0.246, 0.310], full support; clear excess, an upper bound on conditional shift on (cape, tcw). europe↔melanesia: 0.457 against 0.067, p = 0.010, but 58 % support (15 bins). horn↔melanesia: 0.066 against null mean 0.071, **p = 0.71**, full support; no evidence beyond covariate shift, in every statistic (wet fraction, wet W1, both floors, legacy 1e-5). So with two binned inputs claim 1 holds for the pairs involving europe\_west and fails for the only fully supported tropical pair. Claim 1's "Established" needs P2 (9-D) for horn↔melanesia, or must be scoped to two pairs.
- *v3 main-tree re-score.* At 0.1 mm the source-only gap into horn-of-africa is nil: source-only 0.219–0.228 against in-domain 0.219–0.223, so every recovery ratio into horn has a denominator of 0.002–0.011 and is noise (values from −22 to +2.2). Do not report recovery into horn at this floor. Into europe\_west the oracle does not reach in-domain: λ̂ 0.032–0.056 against a source-only gap of about 0.144, i.e. 22–39 % of the gap (horn→europe) and about 32 % (melanesia→europe), every seed. Into melanesia λ̂ is −0.045 to 0.056. horn→melanesia: every method, every seed, has negative recovery.
- *FDA, fixed (`unet_seed<S>_fdafix/`; seed 44 CSV final at 09:01, the scoring job was still running).* europe→melanesia recovery 0.22 / 0.30 / 0.17 (target 1.046 / 1.033 / 1.005 against 1.080 / 1.095 / 1.031; void FDA was 0.46 / −0.02 / −0.52), with the target bias unchanged (−0.27 to −0.29 mm). melanesia→europe 0.27 / 0.68 / 0.45, close to the void FDA's 0.44 / 0.30 / 0.77, so the DC fix barely moves this pair. Both are paid on the source: source-side 0.1 mm error rises by 0.03–0.05 (europe→melanesia; the oracle pays about as much) and by 0.13–0.27 for melanesia→europe, more than the target gains (0.03–0.07) and more than the oracle's source cost (0.01–0.07). Consistent with claim 3's trade; the 3/3 positive europe→melanesia recovery is small (0.03–0.06 absolute, inside the source-only seed range 1.031–1.095) and is not a win for FDA.
- *DANN `--uda_grad_ratio 0.5`, seeds 42/43* (seed 44 running, 66708670). europe→melanesia recovery 0.38 / 0.38 against −0.27 / 0.30 at ratio 1, but **seed 42's model is broken in mm**: target bias +2.12 mm and MAE 2.69 mm on a field mean of 0.31 mm. The log floor hides it. The main-tree seed-43 DANN has the same failure (+3.19 mm bias). The 0.1 mm floor alone cannot rank DANN; report bias or MAE beside it, and do not read the ratio-0.5 recovery as an improvement.
- *AFM phase 1, seed 42* (66685381). All three runs early-stopped on validation CRPS with patience 5: europe\_west best 0.2610 at epoch 12 (stop 17), horn-of-africa 0.2166 at epoch 18 (stop 23), melanesia 0.3039 at epoch 12 (stop 17). europe\_west reproduces Stage A′ (0.2608 at epoch 12). CRPS plateaus from about epoch 6 (gains after it ≤ 1 %), while validation MSE keeps falling to the stop; in melanesia CRPS rises 0.6 % after epoch 12 as MSE falls, so the ensemble spread, not the mean, sets the stop. No divergence, peak memory 20.5 GiB. Not scored yet: the acceptance check against the UNet diagonal (3.6 step 4) needs the test-set CRPS pass before the oracle and phase 2, and phase 2 still waits on D2.

**Submitted 2026-10-30:** AFM seed 42 materialise 66733908 → scoring array 66733909 (`eval_array.sh MODEL=afm TREE=none`, 9 tasks) → merges 66733910 (`afm/none`) / 66733911 (`afm_prob/none`); AFM oracle 66733912 (tasks 0–5, 24 h). Field comparison 66733974 (`compare_fields.py`, `scripts/run_compare_fields.sh`) → `results/plots/compare_fields_seed42/`.

**Read 2026-11-02.**

- *Field comparison (seed 42, 1-in-10 test subset).* Every model is far too dry, in-domain included. Into melanesia: observed mean 0.31 mm/h, 20 % of pixels > 0.1 mm/h; in-domain 0.041 / 5.9 %, oracle 0.019 / 4.3 %, source-only 0.012 / 1.4 %, `mask_shift` source-only 0.050 / 5.2 %. Into europe\_west the melanesia source-only model is the all-zero predictor (mean 0.0000, max 0.05 mm/h over 1,752 timesteps), so its `mse_log_e01` 0.449 is the cost of predicting no rain, and FDA's "recovery" there moves it to 0.3 % wet against 9.9 % observed. Transfer models carry isolated extreme pixels that the log floor hides: max 1,605 (source-only), 9,161 (FDA fix) and 3.4 × 10⁶ mm/h (DANN gr0.5, whose field mean of 3.2 mm/h is almost all from them), against 128 (in-domain) and 17 (oracle). Recovery into europe\_west must be read against the zero predictor.
- *DANN gr0.5, seed 44.* europe→melanesia recovery −0.32, so 0.38 / 0.38 / −0.32 over seeds, and seed 42's +0.38 is the broken model. No improvement over ratio 1; the variant is dropped.
- *P16 few-shot (BN frozen), all seeds, `results/seed<S>/unet_fewshot/`.* N = 1 chunk (200 samples, 500 steps) is broken: 14 of 18 runs have mm-space bias from 2 mm to 2 × 10¹⁴ mm and most score worse than source-only. It is overfitting one chunk, not a data point on the curve; report it as such or drop it. N = 10 chunks (2,000 samples) reaches the oracle on every pair within the seed spread: europe→melanesia 0.863 / 0.960 / 0.889 against oracle 0.921 / 0.888 / 0.882; horn→europe 0.399 / 0.342 / 0.380 against 0.352 / 0.338 / 0.360; melanesia→europe 0.346 / 0.355 / 0.348 against 0.351 / 0.353 / 0.348. N = 100 is not better than N = 10 (europe→melanesia seed 42 0.951 against 0.863). Into europe\_west, fine-tuning stops at the oracle (0.340–0.352), not at in-domain (0.305), so the residual λ̂ into europe\_west is not closed by target labels on a source-initialised model either. For claim 2: the gap is reducible with ~2,000 labelled target samples, and that is a much cheaper bound than the joint oracle.
- *P2, model-based 9-D test (`unet/conditional_shift_model_p2/`, log 66708731).* **The test has no support and little power, so it cannot settle claim 1.** The domain classifier separates every pair perfectly (AUC 1.000 on dyn9 and dyn7); the 0.1 overlap holds 1–157 of 17,520 samples and ovl0.3/0.5 are mostly empty, so R on "all" is extrapolation. The positive control (europe\_west, ×1.5) passes by its own rule but gives R = 1.02 for a known conditional shift, with b2 at 60 % of the expected value. horn↔melanesia R\_half = 1.01–1.05 is therefore the same size as the control's planted shift and cannot be read as a shared conditional; its per-quadruple R (1.13 / 2.47 / 1.08) also disagree. Pairs into europe\_west: R 3.2–4.4, but with no support that is model disagreement off the training inputs. europe→melanesia 1.15, europe→horn 1.38. With P1b, claim 1 rests on europe↔horn on (cape, tcw) and europe↔melanesia at 58 % support; horn↔melanesia is undetermined by both tests.
- *AFM phase 1, test scores (`results/seed42/afm{,_prob}/none/`, M = 8, 10 steps).* Acceptance passes on the test split: CRPS\_std 0.200 / 0.167 / 0.293 (europe, horn, melanesia) against UNet MAE\_std 0.423 / 0.339 / 0.543. AFM's μ as a point forecast is worse than the UNet in all three (MAE\_std 0.461 / 0.346 / 0.608; `mse_log_e01` 0.380 / 0.230 / 1.015 against 0.305 / 0.223 / 0.865). In `crps_log_e01` the transfer gaps keep the UNet's shape: into melanesia 0.348 (europe) and 0.333 (horn) against 0.276 in-domain; into europe\_west 0.197 (horn) and 0.195 (melanesia) against 0.121; into horn 0.093 / 0.085 against 0.077. Tropical sources into europe\_west collapse the ensemble: spread 0.011–0.012 mm against 0.102 in-domain, i.e. eight near-dry members, the probabilistic version of the UNet's zero predictor.
- *AFM oracle (66733912).* All six runs completed, best validation CRPS at epochs 10–19 of 18–25. Not scored on test yet (`eval_array.sh MODEL=afm TREE=oracle`, 6 tasks); λ̂\_CRPS needs it. Submitted 2026-11-02: array 66790277 (tasks 0–5), merges 66790278 (`afm/oracle`) / 66790279 (`afm_prob/oracle`) `afterok` on it.

**Read 2026-11-03.**

- *AFM oracle, test scores (`results/seed42/afm_prob/oracle/`, M = 8, 10 steps).* `crps_log_e01`, λ̂ = oracle − in-domain, gap = source-only − in-domain, seed 42 only: europe→horn λ̂ −0.0005 (gap 0.016), melanesia→horn −0.0015 (0.008), horn→europe 0.0027 (0.076, 4 %), melanesia→europe 0.0015 (0.074, 2 %), europe→melanesia 0.0035 (0.073, 5 %), horn→melanesia **0.027 (0.057, 47 %)**. CRPS in mm agrees (oracle / in-domain 0.069 / 0.067 and 0.068 / 0.067 into europe, 0.237 / 0.233 europe→melanesia, 0.251 / 0.233 horn→melanesia). Reading: into europe\_west the UNet's λ̂ of 22–39 % of the gap mostly vanishes under CRPS, so much of it was the MSE model, not the task; that supports claim 2 and weakens the europe exception. horn→melanesia reverses the UNet (λ̂ ≈ 0 there at 0.1 mm) but is suspect: that oracle's ensemble is underdispersed (spread 0.198 mm against 0.319 in-domain), the other five are not. Into horn the gap is too small to say anything. Without an AFM seed spread, the 2–5 % values cannot be told from zero.
- *Phase 2 (3.7).* FDA 66790462 trained, all six tasks (3.1–4.6 h). MMD 66790463 failed at launch in all six tasks: `run_exp_afm.sh` refuses `UDA_GRAD_RATIO` without an `OUTPUT_SUFFIX` side tree, and the submission had none. No GPU time lost, but the scoring chain 66790506/07/08 can never start. Fixes: `run_exp_afm.sh` phase 2 now copies the seed's `afm_seed<N>/base_hp/` into a side tree, as `run_exp_unet.sh` does (it would otherwise have failed a second time on missing base HPs); `eval_array.sh` takes `EXP_SUFFIX` for an AFM side tree (refused without `OUT_SUFFIX` or for other trees). MMD gr0.5 goes to `afm_seed42_mmdgr05/`; FDA is scored on its own into `afm_p2fda/`, MMD into `afm_p2mmdgr05/`. Submitted 2026-11-03 (66790506/07/08 cancelled): MMD gr0.5 retraining 66827189 (tasks 0–5, `OUTPUT_SUFFIX=_mmdgr05`); FDA scoring 66827190 (tasks 0–5) → merges 66827192 (`afm_p2fda/none`) / 66827193 (`afm_prob_p2fda/none`); MMD scoring 66827191 (tasks 0–5, `afterok:66827189`) → merges 66827194 (`afm_p2mmdgr05/none`) / 66827195 (`afm_prob_p2mmdgr05/none`).

**Read 2026-11-04.**

- *AFM phase 2, seed 42 (`results/seed42/afm_prob_p2fda/`, `afm_prob_p2mmdgr05/`, M = 8, 10 steps).* `crps_log_e01`; budget = source-only − oracle; recovery = (source-only − method) / budget; Δsrc = method − phase-1 in-domain on the source test split. The `*_bias_mm` columns are μ's, not the ensemble's, so the mm check is `crps_mm` and `spread_mm` against the all-zero predictor (CRPS = MAE = field mean: europe 0.097, horn 0.055, melanesia ~0.31).
  - *MMD gr0.5 is not a valid arm: 5 of 6 runs did not train.* Validation CRPS: europe→horn converges normally (0.343 at epoch 5, FDA 0.336); europe→melanesia sits flat at 0.507 from epoch 2; horn→melanesia converges to 0.301, but on the zero predictor (source `crps_mm` = |bias| = 0.0548, target spread 0.015 mm); horn→europe, melanesia→europe and melanesia→horn reach their best at epoch 1 (0.38 / 0.74 / 0.55) and then diverge (to 4.2 / 2.7 / 2.2), so the scored checkpoint is an epoch-1 model. Three targets are the zero predictor (spread ≤ 0.015 mm, `crps_mm` = field mean). The gradient matching works as designed after epoch 1 (share 0.5–0.77), but the share in epoch 1 is 2.2–5.5, because the EMA'd λ lags a fast-rising MMD gradient; the runs diverge anyway once the share is at the target, so it is the alignment, not only the transient. Recovery −4.3 to 0.39, Δsrc +0.02 to +0.14 in every pair. This is a broken optimisation, not evidence for claim 3's trade; a reviewer would discount it. Retuning the ratio breaks the 3.7 "no tuning" rule; open decision.
  - *FDA trains normally* (europe→horn validation curve 0.52 → 0.336, early stop at 14) *and pays on the source in every pair* (Δsrc +0.002 to +0.027). Recovery: europe→horn 0.36 (budget 0.017, at noise level), europe→melanesia 0.04, horn→melanesia −0.56 (oracle suspect, see above), melanesia→horn −0.42 (budget 0.009). **Into europe\_west: horn 0.38, melanesia 0.52**, also in mm (`crps_mm` recovery 0.35 / 0.39; 0.0959 → 0.0865 and 0.0949 → 0.0846 against oracle 0.069 and zero predictor 0.097). This is the first UDA gain in the project that survives in mm space, and it needs checking before anyone believes it. Source-only into europe\_west is a near-dry, collapsed ensemble (spread 0.012 mm), so any spread lowers CRPS where it rains. melanesia→europe FDA has spread 0.254 mm, 2.5× the in-domain 0.102 and the oracle 0.088 (SSR\_log 1.05 against 0.86), so its gain may be spread rather than skill; horn→europe (spread 0.085) is the cleaner case. Checks before claiming anything: wet fraction and maximum of the FDA ensembles into europe\_west (`compare_fields.py`), a no-information reference (source-only members + target-climatology noise at the same spread), and a second seed. One seed, no AFM seed spread.
  - *Reading for claim 4.* Outside europe\_west, FDA under CRPS behaves like the UNet arm: no recovery, source cost. Into europe\_west the UNet's FDA "recovery" was the log floor leaving zero (0.3 % wet); here it shows in mm CRPS too, but an underdispersed baseline makes CRPS reward spread, so the claim cannot be called either way until the checks above are done.

**Submitted 2026-11-04.** *Ensemble check, seed 42:* 66873813 (`scripts/afm_ensemble_check.sh` → `afm_ensemble_check.py`), the seven europe\_west-target AFM models (source-only, FDA, oracle from horn and melanesia; in-domain) on the 1-in-10 test subset, M = 8, 10 steps, against the zero forecast, a climatology ensemble (observed target fields ≥ 168 h away) and source-only dressed with k × climatology at the FDA run's spread; pooled floored rank histograms. Writes `results/seed42/afm_ensemble_check/europe_west/`. Reading: dressed source-only scoring like FDA means the FDA gain is spread. Subset CRPS should be close to the full-split rows. *Second seed:* AFM phase 1 seed 43 66873948 (tasks 0–2, LR 1e-4, ratio 1.0, EMA 0.999, A100 as seed 42) → materialise 66873949 (cpu); FDA seed 43 66873950 (tasks 2, 4: horn→europe, melanesia→europe) `afterok` on phase 1 → scoring 66873952 (tasks 0–4, `TAGS_RE='__to__europe_west__(none|fda)$'`, `OUT_SUFFIX=_euro`) → merges 66873953 (`afm_euro/none`) / 66873954 (`afm_prob_euro/none`). ~36 A100-h. No seed-43 oracle, so this gives FDA − source-only and the gap to in-domain, not a recovery fraction.

*AFM phase 2, more methods (decided 2026-11-04, since MMD gr0.5 left FDA as the only valid arm):* AdaBN, joint\_ot and spectral, seed 42, all six pairs; `mmd_ms` dropped, CORAL and DANN not run. AdaBN now adapts the seed's phase-1 checkpoint (`train_afm.py --adabn_from`, set by `run_exp_afm.sh` for every phase-2 adabn run) instead of retraining it, with `--adabn_steps 10` to match scoring; `best.pt` is the phase-1 model, so its source score is phase 1's. Submitted: smoke 66874188 (`afm_seed42_smoke2/`, europe→horn, 2 epochs, 2 chunks, all three methods; the 2026-10-02 smoke predates width 32, EMA and CRPS selection) → joint\_ot 66874189, spectral 66874190, AdaBN 66874191 (2 h, 64 GB), tasks 0–5 each, `afterok` on the smoke → scoring 66874192 / 66874195 / 66874198 (`TAGS_RE='__<method>$'`, `OUT_SUFFIX=_p2jointot / _p2spectral / _p2adabn`) → merges 66874193–94 / 66874196–97 / 66874199–200. ~90–95 A100-h. Before reading: AdaBN's flow-net pass blew up in the old smoke (target MAE 71,423 mm), so its rows need a maximum / wet-fraction check (`afm_ensemble_check.py`); spectral acts on μ only.

**Read 2026-11-07** (full reading in `report/overview_2026-11-07.md` §9). joint\_ot, spectral and MMD gr0.5 all diverge on source validation CRPS from epoch 2 in the same three pairs (horn→europe, melanesia→europe, melanesia→horn); their training losses track phase 1, FDA (no target forward) does not diverge, and spectral's alignment term is ~0.007. The likely cause is encoder BatchNorm mixing from the train-mode target forward, not alignment, which corrects the 2026-11-04 MMD reading. AdaBN is worse than source-only in all six pairs, with mm blow-ups in four. FDA into europe\_west replicates at seed 43 (horn→europe 0.37 / 0.34 of the gap to in-domain at Δsrc 0.002) and beats a climatology-dressed source-only reference; an ERA5-tp reference is still missing.

**Submitted 2026-11-07.** BatchNorm check (`afm_bn_check.py`, `scripts/afm_bn_check.sh`, no training): smoke 20205517 (task 0, `afm_bn_check_seed42_smoke/`), then array 20205518 (tasks 0–4, spectral: horn→europe, melanesia→europe, europe→melanesia, horn→melanesia, europe→horn) `afterok` on it, into `afm_bn_check_seed42/`. Arms and decision rule are in the script docstring.

**Read 2026-11-09, BN check (all five pairs; source validation CRPS, AFM spectral, seed 42).**
The rule holds everywhere: `mixing_holds` and `removal_holds` are true in 5 of 5 pairs.

| pair | p1 | p2 | d | p1_mix − p1 | p2_src − p1_src |
|---|---|---|---|---|---|
| eu→horn | 0.261 | 0.327 | 0.066 | +0.142 | 0.004 |
| eu→mel | 0.261 | 0.465 | 0.204 | +0.227 | 0.001 |
| horn→mel | 0.217 | 0.287 | 0.070 | +0.073 | 0.000 |
| horn→eu | 0.217 | 0.377 | 0.160 | +1.77 | 0.049 |
| mel→eu | 0.304 | 0.745 | 0.441 | +2.33 | 0.067 |

Source-only BN re-estimation removes 70–100 % of the phase-2 source degradation. Out of
europe the removal is essentially complete. Into europe, 31 % and 15 % of d remain, and that
part is weight damage. Replaying phase-2 traffic through the phase-1 model is worse than
phase 2 itself into europe. So the phase-2 weights partly adapted to the mixed statistics
rather than causing them.

Consequence: for target-forward AFM methods, most of the "source cost" is a BatchNorm
artefact, not an adaptation trade-off. Claim 3 cannot be stated on these checkpoints. The
open half is target risk of `p2_src`: whether the target gain survives source BN. Until that
is scored, nothing says spectral helps.

**Submitted 2026-11-09, target score with source BN.** `afm_srcbn_export.py` and
`scripts/afm_srcbn_export.sh` build the side tree `afm_seed42_srcbn/`: spectral and joint_ot,
all six pairs, BN statistics re-estimated on source only. The chain:
- smoke 20343367;
- export 20343368, `afterok` on the smoke;
- scoring array 20343369 (`eval_array.sh`, `EXP_SUFFIX=_srcbn OUT_SUFFIX=_srcbn`, tasks 0–11);
- merges 20343381 (`afm_srcbn/none`) and 20343382 (`afm_prob_srcbn/none`).

Compare against `afm_p2spectral/`, `afm_p2jointot/` and the phase-1 rows.

**Read 2026-11-10** (`afm_prob_srcbn/` against `afm_prob_p2{spectral,jointot}/`, `afm_prob/none`,
`afm_prob/oracle`; `crps_log_e01`). With source BN, the source cost is gone. dSrc ranges
+0.02 to +0.16 with mixed BN and −0.001 to +0.006 with source BN for europe and horn sources. For
melanesia sources it is +0.04 to +0.05. That residual is ten times the BN check's, so melanesia
as source keeps a real but small cost. Nothing gains on the target. Recovery is between −0.6 and
+0.12 in 10 of 12 cells. joint_ot europe→horn falls to −2.9 and europe→melanesia stays at −1.7.
spectral europe→horn, the only positive phase-2 cell, goes from 0.40 to −0.33. In mm, joint_ot
widens the spread where the mixed run had collapsed it: horn→europe 0.000 → 4.4 mm and
europe→horn 0.001 → 0.45 mm, with `crps_mm` worse than source-only. So the AFM spectral and
joint_ot gains and losses were both BatchNorm. With source BN they are source-only, or worse.
Seed 42 only.

**Prepared 2026-11-09, UNet BN contamination.** `train_unet.train_one_epoch` runs the target
batch through the whole model in train mode for coral, mmd, mmd_ms, dann, joint_ot and
spectral, the same mixing as AFM. Every UNet claim-3 row is exposed.

`unet_srcbn_export.py` and `scripts/unet_srcbn.sh` re-estimate BN on source and score
source and target test with `evaluate.py`. There are 6 array tasks:
- seeds 42, 43 and 44 of the main tree: dann, mmd, joint_ot and spectral, 24 runs each;
- the same three seeds of the `_danngr05` tree: 6 runs each.

Submitted 2026-11-09: smoke 20346232, then array 20346233 (tasks 0–5) `afterok` on it.

Output goes to `unet_seed<N>[_danngr05]_srcbn/` and `results/seed<N>[_danngr05]_srcbn/unet/none/`.
Compare against `seed<N>_v3/` and `seed<N>_danngr05/`.

Default-weight DANN is the clean case. Audit A2 showed its encoder is barely pushed, so it is
close to "source-only plus BN mixing". Rule, fixed now:
- if the source degradation relative to source-only shrinks by half or more, claim 3's UNet
  evidence is BN artefact;
- target rows decide whether any gain was real.

**Read 2026-11-10.** Inputs: `results/seed<N>[_danngr05]_srcbn/unet/none/` against
`seed<N>_v3/` and `seed<N>_danngr05/`, `mse_log_e01`, with source-only and the oracle from
`seed<N>_v3`. The comparison script is in the session scratchpad and is not kept. The rule is
met: **claim 3's UNet evidence is BatchNorm artefact.**
- *Source.* 72 main-tree cells (4 methods × 6 pairs × 3 seeds). 40 have a source cost
  above 0.01; in 35 of them, re-estimating BN on source halves it or more. Mean dSrc falls from
  0.213 to 0.003. By method, dSrc goes from 0.16, 0.08, 0.16 and 0.44 (MMD, joint_ot, DANN,
  spectral) to below 0.007. The exceptions are small costs: europe→horn DANN, about 0.015 at a
  ratio of 0.7, and joint_ot melanesia→europe at seeds 42 and 44 (ratio 0.98 and 0.52). The
  DANN gr0.5 tree reads the same.
- *Target, into europe\_west and horn.* The phase-2 gains disappear. Into europe\_west, every
  source-BN model returns to source-only (0.448, bias −0.097 = the zero predictor). The
  0.42–0.44 "gains" were the target statistics, i.e. AdaBN by accident, and AdaBN on its own is
  already known to be worse. Into horn the budget is 0.002–0.004 and the recoveries (−4 to +4)
  are noise.
- *Target, into melanesia: check twice before believing.* With source BN, most methods score
  better than source-only. Mean dTgt is −0.006 to −0.023 by method, against +0.014 to +0.046
  with mixed BN. joint_ot europe→melanesia recovers 0.45 / 1.03 / 0.36 over the three seeds.
  DANN gr0.5 recovers 0.99 / 0.63 / 0.09, and default DANN −0.24 / 0.62 / 0.34. These are
  log-floor numbers on models that remain near-dry: bias −0.21 to −0.30 against a melanesia mean
  of ~0.31. 13 of the 72 cells have target RMSE above 5 mm/h, including joint_ot seed 43 at 625
  and seeds 42–44 DANN at 54 / 523 / 258, against 1.3 for source-only. horn→melanesia is
  seed-unstable (joint_ot +0.24 / −0.86 / +0.38). This is the "DANN 10⁶ mm/h scores well in the
  log floor" pattern until wet fraction and maximum say otherwise. Nothing goes in the paper
  before `compare_fields.py` on europe→melanesia (joint_ot, both DANN trees, source BN, three
  seeds).
- *Consequence for the paper.* "Methods trade the source conditional for target risk" does not
  survive on either model. What the alignment terms did to the weights is close to nothing: the
  scores moved through BatchNorm mixing. This moves claim 3 into L3 ("check that the
  adaptation term acts") as its main evidence. The GAP argument becomes secondary. No control
  re-estimates BN on source-only or FDA (cumulative average versus EMA). It would cost little and
  close the "re-estimation itself" objection.

**Read 2026-11-23, source-BN control** (array 20811932, `scripts/unet_srcbn_control.sh`:
`results/seed<N>_srcbnctrl/unet/none/` against `seed<N>_v3/` (none), `seed<N>_fdafix/` (fda)
and `seed<N>_srcbn/`; `mse_log_e01`). The same procedure as `_srcbn` (reset, momentum=None, 200
source train batches, torch seed 1000) applied to source-only and fixed FDA. Comparison script
in the session scratchpad, not kept.
- *Into europe\_west and horn.* Re-estimation moves source-only target scores by ≤ 0.001. The
  objection is closed there: the 11-10 reading (the gains were BN mixing) stands.
- *Source, in-domain.* Re-estimation is not neutral: it costs source-only +0.003 to +0.040
  (melanesia seed 42 +0.040, europe\_west seed 42 +0.016). Against the control instead of
  as-trained source-only, the `_srcbn` UDA dSrc stay within ±0.01, except melanesia→horn DANN and
  MMD, which are 0.04–0.05 *better* than re-estimated source-only.
- *Into melanesia: the 11-10 "methods beat source-only" was the re-estimation.* Re-estimation
  alone improves source-only in all six pair×seed cells, by 0.013–0.044, the size of the 11-10
  mean dTgt (−0.006 to −0.023). Against the control, horn→melanesia: all four methods worse
  than source-only (mean +0.003 to +0.029, ≤ 1/3 seeds better). europe→melanesia: MMD +0.010
  (1/3); DANN −0.021 and spectral −0.022, 2/3 seeds with the third of opposite sign at
  ±0.05–0.11; joint\_ot −0.087, 3/3 (−0.058 / −0.193 / −0.009), the L4 case already flagged
  (625 mm/h RMSE at seed 43). Claim 3 is stronger, but every `_srcbn` delta must be stated
  against the `_srcbnctrl` source-only rows, not the as-trained ones.
- *FDA is not a clean control arm.* Its running statistics are of target-styled source, so
  re-estimating on raw source removes part of what FDA does. Into melanesia, FDA under the
  control is worse than re-estimated source-only in all six cells (+0.03 to +0.13); into
  europe\_west part of the gain survives (horn→europe seed 43 −0.022, melanesia→europe seed 42
  −0.031, the rest ≈ 0). FDA has no target forward, so its as-trained rows stay the scored ones;
  the arm says FDA's UNet effect runs partly through BN statistics of styled inputs.
- *Caveat.* One draw of 200 batches per model. The variance of the re-estimation itself is not
  measured and may account for part of the source-side moves up to 0.04.

**P12 read 2026-11-24, Kendall's W over seeds** (target `mse_log_e01`, per off-diagonal
pair, m = 3 seeds; ties by exact score; comparison awk in the session scratchpad, not kept).
χ² = m(n−1)W with df n−1 is only an approximation at m = 3.
- *As trained* (n = 7: none, fda from `fdafix`, spectral, adabn, dann, mmd, joint\_ot):
  W = 0.75 e→h, 0.61 e→m, 0.84 h→e, 0.94 h→m, 0.92 m→e, 0.67 m→h. Four of six pass the
  χ² approximation at 0.05 (critical W ≈ 0.70); 11–18 of 21 method pairs keep their order in
  all three seeds. The old "ordering not stable" was wrong for as-trained scores.
- *Source BN* (n = 5: `_srcbnctrl` source-only, dann, mmd, joint\_ot, spectral from `_srcbn`):
  W = 0.18–0.48, none passes (critical W ≈ 0.79); 1–4 of 10 pairs stable.
- *Reading.* The seed-stable ordering is the BN-mixing ordering. Once BN is re-estimated on
  source, the methods cannot be told apart across seeds. This backs claim 3 (and L4) better
  than an "unstable ordering" sentence would: the stable ranking was the BN artefact, not the
  alignment terms. Stable vs source-only under source BN: joint\_ot better on e→m (the L4
  case), mmd/dann better on m→h (by ≤ 0.003); mmd worse on h→e, spectral/dann worse on h→m.

**Read 2026-11-24, (tp, cape) control for COSMO C2** (array 20901937,
`scripts/rainshift_c2_tp_cape.sh`; `unet/conditional_shift_binned_p1b_tp_cape_{block,fine}.json`,
logs `/scratch/fquareng/outputs/20901937_{0,1}`). P1b settings, only the conditioning set
changes; the unit tests passed in both tasks. `log_e01`, obs / null mean (null p95):

| pair | block (40×40) | fine (6 px/cell) | bins | P1b (cape, tcw) |
|---|---|---|---|---|
| europe↔horn | 0.101 / 0.045 (0.055) | 0.095 / 0.040 (0.050) | 12 | 0.281, p 0.010 |
| europe↔melanesia | 0.057 / 0.024 (0.037) | 0.056 / 0.020 (0.029) | 20 | 0.457, 58 % support |
| horn↔melanesia | 0.105 / 0.021 (0.032) | 0.103 / 0.018 (0.030) | 20 | 0.066, p 0.71 |

- All six at p = 0.010 (the floor of 100 permutations), support 100 %, bootstrap CIs above the
  null p95, in every statistic (wet fraction, wet W1, both floors). Block and fine agree within
  ~6 %. The one-domain half-size nulls are 0.001–0.026 against cross-half 0.056–0.104.
- *Unlike COSMO.* There, conditioning on coarse precipitation removed most of the (cape, tcw)
  excess. Here the europe pairs shrink (0.28 → 0.10, 0.46 → 0.06) but stay clear, and
  horn↔melanesia goes from none to ~5× its null. The plausible difference: RainShift `tp` is
  ERA5, not the target's block mean, so P(Y | tp, cape) differing includes a regional ERA5→IMERG
  relation. For claim 1 that is a real P(Y|X) difference, the map the model must learn.
- *Caveats.* Still an upper bound on two binned inputs. `tp` is the strongest predictor of y,
  and the fine sub-bins leave a within-bin `tp` mismatch of 0.08–0.14 sd (coarse 0.12–0.21;
  `cape` coarse 0.46 sd on europe↔melanesia, fine 0.15). The permutation null keeps fine
  sub-bin composition, but residual `tp` mismatch inside a sub-bin can produce excess that
  `cape` mismatch would not; how much is not calibrated. Conditioning on (cape, tcw) and on
  (tp, cape) asks different questions: neither implies shift given the full X, and their
  disagreement on horn↔melanesia is not a contradiction. p sits at its floor.
- *Consequence for claim 1.* horn↔melanesia moves from "undetermined" to "excess on (tp, cape),
  upper bound". Word it that way; do not call it established without a calibration of the
  within-sub-bin `tp` mismatch (e.g. a pure-covariate-shift positive control built on `tp`).

**Read 2026-11-10, fields** (job 20376532: `results/plots/compare_fields_srcbn_seed<S>/europe_west__to__melanesia/`,
1 test sample in 10). IMERG has a wet fraction (> 0.1 mm/h) of 0.205, p99 5.8 and max 59 mm/h.
The oracle is itself far too dry: wet 0.043–0.054, p99 0.42–0.63, max 17–41. Source-only is
drier still (wet 0.014–0.023, p99 0.15–0.25), and its maximum is already 1.6×10³–4.7×10⁴ mm/h.
With mixed BN, joint_ot is drier than source-only (wet 0.0003–0.007). With source BN its bulk
moves towards the oracle:

| joint_ot, source BN | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| wet fraction (oracle) | 0.029 (0.043) | 0.109 (0.050) | 0.039 (0.054) |
| p99, mm/h (oracle) | 0.38 (0.42) | 2.5 (0.53) | 0.42 (0.63) |
| max, mm/h | 1.4×10³ | 2.7×10⁶ | 9.3×10⁴ |

Seed 43 overshoots the oracle, and its mean of 1.9 mm/h is outliers. DANN, DANN gr0.5 and
spectral with source BN are seed-unstable: wet fraction 0.009–0.064, maximum 10³–7×10⁵ mm/h.
Reading: the log-floor recovery comes from real wetting of the bulk, mainly by joint_ot.
Every model trained on europe_west, source-only included, also produces pixels two to five
orders above the observed maximum when applied to melanesia. So the maximum does not separate
methods. The pre-stated test ("max in the oracle's range") fails for every model, source-only
included. Paper use: an L4 example, not a gain. A log-floor recovery of 0.4–1.0 coexists with
10⁵–10⁶ mm/h pixels, and mean mm cannot be used. The 2026-11-09 dryness of joint_ot was BN
mixing (L3). Missing: the pixel fraction above, say, 100 mm/h per model, without which
"a few pixels" is unquantified.

**2026-11-10: training-maximum columns and the feature probe (both read 2026-11-11, below).**
- `compare_fields.py` now also writes, per model, the fraction and count of pixels above its
  source's training maximum (from `train_data_out.zarr`, validation chunks included), the
  fraction above 100 mm/h, and the mean and `mse_log_e01` with predictions clamped at that
  maximum. The same rule applies to every model, so this is a robustness row. Re-run: job
  20391536 (`TAG=_trainmax`, `results/plots/compare_fields_srcbn_trainmax_seed<S>/`).
- *Why the extremes exist.* Melanesia's mean tcw is +4.4σ in europe_west units. The network
  extrapolates roughly linearly in standardised log space, and the inverse is exp with
  σ_log = 3.7 (europe_west precipitation), so an output error δ becomes a factor e^{3.7δ}.
  The AFM does it too (774 mm/h in-domain), so it is not determinism.
- `feature_probe.py` / `scripts/feature_probe.sh`: held-out bottleneck features (1024 test
  samples per domain, source stats) of source-only and of dann / mmd / joint_ot / spectral /
  DANN gr0.5. Three BN modes: eval with source BN, eval with mixed BN, and train-mode
  per-domain batch BN with frozen running stats, which is what the loss saw in training. It
  reports unbiased multi-kernel MMD² under GAP and grid4 against a source-vs-source null, plus
  a held-out linear domain classifier on GAP features (time-split halves). Writes
  `results/seed<S>/feature_probe/probe.csv`. Reading, fixed now:
  - **(a) Never aligned.** Method MMD ≈ source-only MMD, well above the null, and classifier
    accuracy ≈ source-only's. A stronger weight would mean something, so the dose ladder goes
    to the M2 gate.
  - **(b) Aligned, prediction unmoved.** Method MMD near the null under eval_srcbn while the
    scores equal source-only's. GAP alignment does not constrain the output; drop the ladder.
  - **(c) Nothing to act on.** Source-only's own train_batch MMD is already near the null.
    Per-domain batch BN made the GAP features near-invariant before any loss: a structural
    (b).

**Read 2026-11-11, training-maximum columns** (job 20394524, `results/plots/compare_fields_srcbn_trainmax_seed<S>/europe_west__to__melanesia/`;
the first submission, 20391536, was OOM-killed in `_train_max` and wrote nothing). Training
maxima: europe_west 83.4 mm/h, melanesia 152.5 mm/h (file maxima, validation steps
included, so the fractions below are conservative).
- Pixels above the source training maximum are rare: 0 to 3.4×10⁻⁴ of pixels for every model,
  source-only included (6×10⁻⁶ to 1×10⁻⁴; maxima 1,605 / 23,194 / 46,557 mm/h for seeds
  42/43/44). Oracle and in-domain stay at or near 0 (in-domain seed 43: 39 pixels).
- Clamping at the training maximum changes `mse_log_e01` by at most 0.005 in every row. The
  log-floor numbers and rankings stand; the extremes do not drive them.
- They drive the mm-space mean entirely: joint_ot src BN seed 43 1.93 → 0.17 mm/h clamped;
  DANN gr0.5 src BN seed 42 0.67 → 0.07. Every clamped mean is below the observed 0.312.
  Mean mm is unusable without the clamp; report the clamped mean next to the fraction.
- Extremes come from extrapolation, not adaptation: source-only has them at every seed.
- Checked twice because it flatters a method: joint_ot src BN seed 43 scores 0.876 vs oracle
  0.887. Clamped it is still 0.876, but its wet fraction is 0.109 against 0.205 observed,
  and seeds 42/44 give 1.009 / 0.978 (oracle 0.920 / 0.881). Seed-unstable, an L4 example,
  not a gain.

**Read 2026-11-11, feature probe** (job 20394525, `results/seed<S>/feature_probe/probe.csv`,
3 seeds × 6 pairs = 18 cases per row). Means over the 18 cases; MMD is GAP, multi-kernel
with a per-model median bandwidth, so ratios between models are approximate; the classifier
is not affected by that.

```
                      deployed (eval, source BN)        what the loss saw (train-mode batch BN)
                      MMD    ratio to SO   clf acc      MMD     ratio to SO   clf acc
source-only (SO)      0.460  1             1.000        0.0135  1             0.908
mmd                   0.248  0.59 (0.14-1.11) 0.999     0.0014  0.12 (0.05-0.21) 0.813
joint_ot              0.370  0.92 (0.39-2.68) 1.000     0.0242  2.09          0.988
dann                  0.464  1.07          1.000        0.0089  0.70          0.894
dann gr0.5            0.441  1.02          0.997        0.0184  1.68          0.883
spectral              0.446  0.99          0.999        0.0118  0.95          0.907
null (src vs src)     ~0.002-0.003                      ~0.000
```
Mixed BN (eval as trained) reads like source BN: 0.25-0.48, classifier ≥ 0.979.

Reading against the rule fixed on 2026-11-10:
- **(a) holds for every method in deployment.** No method brings deployed MMD near the null
  (every case ≥ 29× it, typically ~100×), and the domain classifier is at 0.98-1.00 for all of them, as for
  source-only. DANN, DANN gr0.5 and spectral do not differ from source-only even in the
  space the loss saw; joint_ot increases the distance there.
- **(b) does not hold as stated, but MMD is the partial case.** MMD did act on its own
  objective: 0.12× source-only in train-mode batch BN, i.e. near the null. 41 % of that
  carried into the deployed features (0.59×), and the source-BN predictions still equal
  source-only's. That is weak evidence for "alignment does not reach the output", not a
  (b) reading: the deployed features are far from aligned.
- **(c) partly holds.** Per-domain batch BN alone cuts source-only's GAP MMD by 97 %
  (0.46 → 0.0135), so every loss acted on a small residual. It is not near the null, and the
  classifier still separates the domains (0.85-0.96), so it is not a structural (b).
- **The mechanism.** The loss is computed with per-domain batch statistics; deployment uses
  fixed running statistics. Alignment achieved in the first space does not transfer to the
  second, because the first already removed the per-channel mean and scale shift that
  dominates the second. A larger weight on the current trainer would push on the residual
  and change nothing in deployment. Any ladder therefore has to run on the BN-fixed trainer
  (target pass in eval-mode BN, B4.9), where the loss sees the features the model deploys.

Consequence: by the rule, the dose ladder goes to the M2 gate, on the BN-fixed trainer only.
Proposed arms: DANN (never aligned anywhere) and MMD (aligned in the loss's space; the arm
that can reach deployed alignment and test whether it moves the prediction, i.e. the Zhao
bound). For L3: "the alignment terms did not act" becomes "they acted, if at all, in a
batch-normalised space that deployment never sees".

**2026-11-11: BN-fixed trainer written (B4.9, UNet only), not yet tested or run.**
`train_unet.py --target_bn eval` (default `train`, so every existing tree keeps its meaning)
runs each target forward of coral / mmd / mmd_ms / dann / joint_ot / spectral inside
`_bn_eval`: BatchNorm layers in eval mode, so the target batch is normalised with the
running statistics and never updates them; gradients flow as before. The source pass keeps
train-mode BN. Consequences: the alignment loss sees the target features the deployed model
computes, and `best.pt` holds source-only statistics, so these runs are scored as trained,
with no `_srcbn` re-estimation. Refused for none / fda / adabn. `run_exp_unet.sh` takes
`TARGET_BN=eval` and env overrides `LAMBDA_UDA` / `DANN_WEIGHT`, each of which requires an
`OUTPUT_SUFFIX` side tree. Known asymmetry: source features are normalised by batch
statistics, target features by running statistics; they differ by batch noise and the
running-average lag (momentum 0.1), which matters in the first epoch only.
Test: `test_target_bn.py` (synthetic, in `scripts/run_tests.sh`): after one step, running
stats are independent of the target batch and equal a source-only forward, the weights do
depend on it, and train mode (control) fails the first check. Next, before any ladder: one
default-weight MMD run europe→melanesia, seed 42, `OUTPUT_SUFFIX=_tgtbneval`, read with
`feature_probe.py` (eval_srcbn = as trained here) for whether deployed alignment moves at all.
The AFM trainer is not changed.

**Read 2026-11-12, BN-fixed MMD, seed 42, europe→melanesia** (test job 20438249: all tests
pass, `test_target_bn.py` included; training 20438441, best at epoch 6 on source validation,
stopped at 11; read 20460852: `results/seed42_tgtbneval/feature_probe/probe.csv`,
`results/plots/compare_fields_tgtbneval_seed42/europe_west__to__melanesia/`,
`results/seed42_tgtbneval/unet/none/`). One seed, one pair, default weight (λ 0.1).

- **The term acts in deployment for the first time.** Deployed GAP MMD 0.0073 (null 0.0002),
  against 0.43 for MMD with source BN, 0.24 as trained, 1.00 for source-only; grid4 0.030
  against 0.20–0.71; domain classifier 0.88 against 1.00 for every other model. The ratio to
  the null (~36×) does not separate it from the 2026-11-11 models because the null is tiny;
  the absolute distance is ~60× below the old MMD run.
- **Log floor: recovery ≈ 1.0.** Target `mse_log_e01` 0.921 against source-only 1.080 and
  oracle 0.921 (P16; `compare_fields` gives 0.920), in-domain 0.865. Source 0.310 against
  0.305 (Δsrc +0.006; MMD src BN 0.311). Selected on source validation, no target labels.
- **mm space does not follow.** Target MAE 0.343 mm against source-only 0.315 and in-domain
  0.298 (worse). Bias −0.210 against −0.299 and −0.271: wetter than in-domain. Wet fraction
  (0.1 mm) 0.077 against in-domain 0.059, oracle 0.043, obs 0.205; p99 1.80 against 0.92 mm/h;
  max 10,289 mm/h, pixels above the training maximum 4.3×10⁻⁵ against 6.0×10⁻⁶ for
  source-only (7×). The bulk is wetter, not only the tail. This is the pattern the log floor
  rewards (P3), so the log-floor recovery is not yet adaptation.

**Pre-registered 2026-11-12, before any replication result exists.** Replication: seeds 43
and 44 on europe→melanesia (training 20467868 / 20467869, reads 20467871 / 20467872) and
seed 42 on horn→europe (20467870, read 20467873), all `TARGET_BN=eval`, default weight,
`unet_seed<S>_tgtbneval/`. Every comparison is paired within a seed against that seed's own
source-only and oracle, at `mse_log_e01` on the target test split.

References: europe→melanesia source-only 1.080 / 1.095 / 1.031 and oracle 0.921 / 0.888 /
0.882 (seeds 42/43/44), source-only target MAE 0.315 / 0.379 / 0.358 mm. horn→europe
source-only 0.4486 (the all-zero predictor, MAE 0.0974 = field mean), oracle 0.352,
in-domain 0.305.

Criteria, per run:
- *F (features):* deployed GAP MMD ≤ 0.1× the same seed's source-only, and the domain
  classifier ≤ 0.95. Without F, the run did not reproduce the mechanism, whatever it scores.
- *L (log floor):* recovery (source-only − run) / (source-only − oracle) ≥ 0.5.
- *S (source cost):* source `mse_log_e01` within 0.02 of source-only.
- *M (mm space):* target MAE ≤ source-only's of the same seed, and the fraction of pixels
  above the training maximum ≤ 10× source-only's. Into europe_west, where source-only is the
  zero predictor, M means beating the zero predictor's MAE (0.0974 mm).

Seed 42 europe→melanesia passes F, L and S and fails M.

Readings, decided now:
1. *Confirmed:* F, L, S and M hold on both europe→melanesia replicates and on horn→europe.
   Seed 42 already fails M, so this needs seeds 43 and 44 to pass all four. Then claim 3
   becomes "once the target pass leaves BN alone, a default-weight MMD reaches the oracle",
   the M2 gate reconsiders a full Article, and the dose ladder is reopened as a method result
   (with the prior that it is not one, checked twice).
2. *Log-floor only:* F and L hold in ≥ 2 of 3 europe→melanesia seeds, but M fails in the
   majority. Then the lesson is "an alignment term that acts buys log-floor score with mm
   error": L3/L6, not a gain. Claim 3 keeps "BN, not alignment" for the old trainer and adds
   this sentence for the fixed one. No ladder.
3. *Not reproduced:* L fails at seeds 43 and 44, or F fails. Then seed 42 is noise or an
   unreproduced mechanism, claim 3 stands as rewritten on 2026-11-10, and the BN-fixed MMD
   run is reported as the control that closes it.
horn→europe is read on its own: an L pass there with an M fail is the P3 near-dry artefact
and counts against reading 1, not for it. Anything not covered here is reported as such,
not fitted into a reading afterwards.

**Read 2026-11-13, replication: reading 2 (log-floor only).** Jobs 20467868–70 (training) and
20467871–73 (reads) all completed. Outputs: `results/seed<S>_tgtbneval/feature_probe_<pair>/`,
`results/plots/compare_fields_tgtbneval_seed<S>/<pair>/`,
`results/seed<S>_tgtbneval/unet/none/rows/<pair>__mmd.csv`. Thresholds applied as written.

```
europe→melanesia             seed 42        seed 43        seed 44
L  recovery                  1.00           0.99           1.03
   target mse_log_e01        0.921          0.891          0.877
   (source-only / oracle)    1.080 / 0.921  1.095 / 0.888  1.031 / 0.882
F  deployed GAP MMD / SO     0.007×         0.024×         0.023×
F  domain classifier         0.88           0.65           0.955  (fail)
S  Δsrc                      +0.006         +0.007         +0.024 (fail)
M  target MAE mm (SO)        0.343 (0.315)  3.15 (0.379)   0.325 (0.358)
   M                         fail           fail           pass
   best / stop epoch         6 / 11         10 / 15        3 / 8
```

- **L reproduces in every seed, and clamping does not change it** (seed 43: 0.884 clamped,
  0.889 unclamped, `compare_fields`). On the log floor the BN-fixed MMD lands on the oracle
  three times out of three.
- **M fails in 2 of 3.** Seed 43 is a blow-up: bias +2.6 mm/h, max 8.7×10⁶ mm/h, pixels above
  the training maximum 2.65×10⁻⁴ against 9.96×10⁻⁵ for its source-only (itself 23,194 mm/h max).
  Its training log shows target MAE diverging from epoch 12 (11 → 34 → 198 → 6,327 mm);
  source-validation selection picked epoch 10, already at 3.1 mm. In every seed the run is
  wetter than in-domain in the bulk (wet fraction 0.077 / 0.098 / 0.079 against in-domain
  0.059 / 0.053 / 0.049; p99 0.85–1.8 against 0.71–0.92 mm/h).
- **Seed 44 fails F and S narrowly** (classifier 0.955 against ≤ 0.95; Δsrc +0.024 against
  ≤ 0.02). Recorded as fails.
- **horn→europe, seed 42: no effect.** Target 0.4479 against source-only 0.4486 and oracle
  0.352 (recovery 0.007); MAE 0.0974 mm, the all-zero predictor's, max 0.46 mm/h, wet
  fraction 0. Deployed GAP MMD fell to 0.03× source-only (0.0067 against 0.212), but the
  classifier still separates the domains (0.98). Features moved, the output did not: the
  GAP "(b)" reading that 2026-11-11 could not establish, on one seed and one pair.

Reading 2, as decided in advance: an alignment term that acts on deployed features buys
log-floor score with mm error. For claim 3 / L3: on the old trainer the terms did not act and
BN did; on the BN-fixed trainer MMD acts, reaches the oracle on the log floor into melanesia
by wetting bulk and tail, and does not move the zero predictor into europe_west. No dose
ladder (reading 2). Open, not pre-registered: why L lands on the oracle so exactly
(1.00 / 0.99 / 1.03). A trivial-wetting reference (source-only plus constant drizzle at the
run's wet fraction) would show whether the log floor alone produces it; run before the L3
sentence is written.

**Proposed 2026-11-09, larger RainShift matrix (R-matrix).** All 19 regions are on disk.
Pre-registered additions to cover the regimes: northamerica-east (mid-latitude, a europe_west
analogue), west-africa (monsoon convection), tibetan-plateau (high orography),
australasia-east (southern-hemisphere subtropics). That gives 7 regions and 42 ordered pairs.
- **Scope:** phase-1 source-only × 3 seeds, the full transfer matrix, and P1b/W1 on every
  pair. Oracle, FDA and BN-fixed MMD on 6 pairs only.
- **Cost:** about 60–80 A100-h.
- **Blocker:** none of the 16 new regions has `normalization_stats.json`. Using one as a
  source makes `ClimateSRDatasetZarr` compute the file and write it into `raw_data/rainshift`,
  which violates rule 3. A stats directory outside the shared copy is needed first.
- **What it can and cannot show:** this tests region-specificity, not dataset-specificity.
  Every pair is still ERA5→IMERG, and IMERG's error structure varies by region (gauge-adjusted
  over land, passive microwave over ocean). Only COSMO changes the dataset.

**Revised 2026-11-17: R5, a 5-region RainShift matrix to mirror COSMO C3.** Two regions
are added from the four pre-registered above, so the choice stays inside the list. The
result is 5×5 matrices, the same figures as `figures/cosmo_c3/`, and a relabelling test
with 120 orderings instead of 6.
- **northamerica-east.** A mid-latitude, mostly frontal regime with mixed land and ocean,
  the europe_west analogue. It gives RainShift the pair it lacks: one where little shift
  is expected. COSMO had 10 of 20 pairs within noise and RainShift has none. If
  europe_west ↔ northamerica-east also fails, the RainShift failure is not regime
  distance but something every RainShift pair shares (region identity through statics,
  or IMERG's region-dependent errors).
- **west-africa.** Monsoon convection over land, with a strong dry season and wet season.
  It falls between horn (dry, convective) and melanesia (wet, maritime) in wetness, so it
  adds a middle point to the dry→wet axis behind the COSMO support-asymmetry hypothesis.
  It also gives a land-convective counterpart to maritime melanesia.
- **Not chosen.**
  - tibetan-plateau: IMERG's passive-microwave retrievals are poor over snow and high
    terrain, so target noise would be confounded with orography. RainShift's own ResNet
    already blows up there.
  - australasia-east: its stores have a different layout (`*_opt` variants, including
    `test_data_in_opt`), and the southern-hemisphere season flips `tisr`, another
    region-identity channel. It adds no axis the other two lack.
- **Stats blocker, proposed fix.** Make an overlay root `data_rainshift_r5/<region>/` of
  symlinks to the shared stores. The three existing regions also get a symlink to their
  `normalization_stats.json`. The new regions' stats are computed into the overlay
  directory, which is ours, so nothing is written into `raw_data/rainshift`. Check before
  use that `ClimateSRDatasetZarr` writes the stats file to the root it was given and not
  to a resolved path.
- **Scope.**
  1. Phase-1 source-only on the two new regions × seeds 42/43/44: 6 runs with the existing
     phase-1 settings and no `--data_order_seed`, to stay comparable with the existing
     RainShift seeds. Roughly 2–3 A100-h each, about 15 A100-h.
  2. Score the 16 new cells × 3 seeds.
  3. P1b and per-variable input W1 on the 7 new unordered pairs (CPU).
  4. Trivial references (zero, climatology, bicubic ERA5 `tp`; bicubic is only a weak
     reference here, see report/aies/NOTES.md).
  5. The same figures.
  6. Oracle and BN-fixed methods stay on the existing pairs unless the matrix shows a
     new cell worth it.
- **Open before launching.**
  - NaN and range checks on the two new IMERG targets, inside a job.
  - Whether the new runs go into `unet_seed<N>/` (new names, nothing overwritten) or a
    side tree `unet_seed<N>_r5/`. The side tree is the safer default.
- **Done 2026-11-17.**
  - *Overlay.* `data_rainshift_r5/` holds symlinks to the shared stores. The three existing
    regions also link their stats.
  - *Stats job 20615520 (`rainshift_r5_stats.py`).* It recomputed europe_west in streamed
    chunks and matched the cached file to 1.1e-15. It then wrote the stats for
    northamerica-east and west-africa into the overlay.
  - *Target checks* (`results/r5/target_check.json`, test mean / wet fraction):
    - horn 0.055 / 0.045;
    - west-africa 0.078 / 0.047;
    - europe_west 0.097 / 0.099;
    - northamerica-east 0.149 / 0.100;
    - melanesia 0.313 / 0.205.

    The wetness axis is well spread. Northamerica-east is about 1.5× wetter than europe_west,
    so it is a regime analogue, not a wetness twin.
  - *Non-finite target pixels in train:* west-africa 1,245, the same issue as melanesia
    (2,387). The `_transform_channel` `nan_to_num` fix covers it. Every test split is clean.
  - *Training.* Phase 1 runs as array 20615551 (A100, side tree `unet_seed<S>_r5/`), started
    after the stats job.
  - *Read 2026-11-23* (phase 1 done 2026-11-18; scoring array 20791976,
    `results/seed<S>_r5/unet/none/`, 25 rows per seed, 17,520 test samples each).
    `mse_log_e01`, mean (sd) over seeds 42/43/44, rows source, columns target:

    | src \ tgt | europe\_west | horn | melanesia | NA-east | W-africa |
    |---|---|---|---|---|---|
    | europe\_west | **0.3046** (0.001) | 0.2236 (0.003) | 1.0686 (0.027) | 0.3757 (0.004) | 0.2783 (0.004) |
    | horn | 0.4485 (0.000) | **0.2213** (0.002) | 0.9750 (0.032) | 0.5706 (0.002) | 0.2835 (0.004) |
    | melanesia | 0.4485 (0.000) | 0.2277 (0.000) | **0.8788** (0.011) | 0.5657 (0.002) | 0.2799 (0.001) |
    | NA-east | 0.3646 (0.009) | 0.2188 (0.001) | 0.8483 (0.023) | **0.3946** (0.004) | 0.2717 (0.002) |
    | W-africa | 0.4458 (0.002) | 0.2215 (0.002) | 1.0357 (0.063) | 0.5624 (0.010) | **0.2739** (0.002) |

    - The 3×3 block reproduces `seed<S>_v3` exactly: the three original regions are reused,
      only northamerica-east and west-africa were trained.
    - Much of the matrix is the zero field. Every tropical source predicts zero into
      europe\_west (bias −0.097 = −mean) and northamerica-east (bias −0.148 = −mean). The horn
      and west-africa columns each span ~0.01, and the horn diagonal's bias (−0.054) is the
      zero predictor's too.
    - NA-east as source beats the in-domain model on all three tropical targets: melanesia
      0.848 vs 0.879 (also in mm: RMSE 1.252 vs 1.255, bias −0.253 vs −0.278), horn 0.2188 vs
      0.2213, west-africa 0.2717 vs 0.2739 (the last two within ~1 sd). A negative gap leaves
      λ̂ undefined there. Most likely reading: the tropical in-domain UNets are so dry that a
      wetter source wins. To mirror COSMO C3 the table needs a zero-predictor row per target,
      or it ranks dryness.
    - **West-africa blows up in-domain.** Test RMSE 0.91 / 18.4 / 22.7 mm/h (seeds 42/43/44);
      west-africa→melanesia 107 / 295 / 684 mm/h, bias +1.6 / +3.2 mm/h at seeds 43/44, log
      score also worse (0.95–1.11); europe→west-africa 3.4 / 6.5 / 45.7 mm/h. The `nan_to_num`
      fix is present (`dataset_zarr.py:84`, `:131`, and `rainshift_r5_stats.py`). Cause
      unknown; until then the blow-up was only out of region. Next: `compare_fields.py` with
      training-maximum columns on the west-africa diagonal. No west-africa row is quoted
      before it.
  - *Read 2026-11-24, west-africa fields* (array 20896714, `scripts/compare_fields_r5_wafr.sh`,
    stride 1, `results/plots/compare_fields_r5_seed<S>/west-africa__to__{west-africa,melanesia}/`;
    35–37 GB peak RSS). Training maxima: west-africa 122.4, melanesia 152.6, europe\_west
    83.4 mm/h.
    - *The in-domain west-africa model is the zero field plus a few exploding pixels.* Test
      obs: mean 0.078 mm/h, wet fraction 0.047, p99 2.3, max 61 (below the training maximum).
      In-domain (seeds 42/43/44): mean 0.003 / 0.017 / 0.020, wet fraction 0.002–0.003, p99
      0.007–0.012; 849 / 3,893 / 4,841 pixels above the training maximum (1–7×10⁻⁶), max
      2,228 / 92,629 / 109,731 mm/h. Clamped, the mean is 0.003–0.005 and `mse_log_e01` does not
      move. The 18–23 mm/h RMSE is those pixels; the 0.274 diagonal is a dry predictor's score.
      For comparison the melanesia diagonal is ~10× too dry (mean 0.03–0.04 vs 0.31) but does
      not blow up (≤ 324 pixels, max 384).
    - *west-africa→melanesia is the worst extrapolation seen so far.* 1.4×10⁻⁴ to 1.9×10⁻³ of
      pixels above the training maximum (up to 1.3 M pixels), max 0.36 / 0.99 / 1.39 M mm/h,
      mean 0.30 / 1.90 / 3.55 raw and 0.10–0.68 clamped. 10–20× europe→melanesia's fraction.
      Clamping moves its `mse_log_e01` by up to 0.023, so the log floor does not protect it.
    - *Consequence.* West-africa is no target-side reference: its diagonal is a dry model, and
      every west-africa row carries the clamped numbers beside it.
    - *Not established:* why an in-region model extrapolates. This job measured outputs only.
      Working hypothesis, untested: test-period inputs (tcw or cape most likely) outside
      west-africa's training range. Check: per-channel input ranges, test vs train, CPU job.
      *Submitted 2026-11-24:* array 20900124 (`scripts/rainshift_input_ranges.sh`,
      `rainshift_input_ranges.py`), four stats regions, into `results/r5/input_ranges/<region>/`.
      Also records the smallest positive decoded `tp`/`cp` per store (the int16 packing differs
      per file, so the dry floor after the log can differ between train and test).
    - *Read 2026-11-24, input ranges* (array 20900124, `results/r5/input_ranges/<region>/`
      `ranges.csv`, `exceed.csv`, `<entry>_tmax.npy`). **The hypothesis is weak.** In
      west-africa units, west-africa's own test split leaves the training range only in
      `cape`: 7×10⁻⁸ of pixels (~8 pixels) in ~2 timesteps, up to +2.8σ beyond the training
      maximum. Europe's own test split leaves its range by more (`tclw` +6.2σ, `tcw` +3.8σ) and
      the europe in-domain model does not blow up; horn and melanesia test splits do not leave
      theirs at all. So in-region exceedance neither is large in west-africa nor predicts a
      blow-up elsewhere. The int16 dry floor is identical in every store (smallest positive
      `tp` 9.69×10⁻⁴, `cp` 2.99×10⁻⁴ after ×1000), so packing is ruled out. Out of region the
      exceedance is large, as expected: west-africa→melanesia in west-africa units has `sp`
      below the training minimum in every frame (0.9 % of pixels, to −4.3σ), `cape` to +5.2σ in
      0.4 % of steps, `tisr` above range in 5 % of steps; europe→melanesia in europe units has
      `tcw` above range in 69 % of steps (to +6.5σ) and `tclw` to +15.7σ, yet blows up 10–20×
      less than west-africa→melanesia. *Not established:* whether the ~2 `cape` timesteps are
      the blow-up timesteps; that needs per-timestep output maxima (one GPU pass, not
      submitted). Until then the west-africa in-domain blow-up is unexplained, and exceedance
      in σ is not a usable predictor of blow-up across sources.
  - *Read 2026-11-24, trivial references* (job 20811934, `scripts/rainshift_refs.sh`,
    `results/rainshift_references/references.csv`, test split, 0.1 mm floor).
    `mse_log_e01` of zero / bicubic ERA5 `tp` / train climatology, and the in-domain UNet's
    skill over zero:

    | target | zero | bicubic tp | climatology | in-domain UNet | skill over zero |
    |---|---|---|---|---|---|
    | europe\_west | 0.4488 | 0.4357 | 0.6702 | 0.3046 | 32 % |
    | horn | 0.2320 | 0.3346 | 0.4077 | 0.2213 | 4.6 % |
    | melanesia | 1.1882 | 1.2329 | 1.7180 | 0.8788 | 26 % |
    | NA-east | 0.5819 | 0.5196 | 0.9840 | 0.3946 | 32 % |
    | W-africa | 0.2932 | 0.3817 | 0.5320 | 0.2739 | 6.6 % |

    - Horn and west-africa in-domain UNets are 5–7 % better than predicting zero. The whole
      horn column spans 0.219–0.228 and the west-africa column 0.272–0.284, against zero at
      0.232 and 0.293 (NA-east beats in-domain in both): there is no room for a method to show
      anything into either.
    - Into europe\_west, horn/melanesia/west-africa sources score 0.4458–0.4485 against zero's
      0.4488: they are the zero predictor. Into NA-east the tropical sources (0.553–0.571) beat
      zero by 2–5 %.
    - Climatology is worse than zero everywhere in the log floor; bicubic `tp` beats zero only in
      the extratropics (europe\_west, NA-east).

## Dependencies

- **0.1** (seed-aware materialise + evaluate) blocks everything: it produces the first seeded results.
- **Track A:** 0.3 (seed the remaining methods) → seeded transfer matrix and recovery table.
- **Track B:** fix `train_time_index` → Stage 1 (FSS, RAPSD, Moran's I, case studies) → Stage 2 (mechanism figure, GAP ablation).
- **Stage 3** (AFM), as of 2026-11-02: diagnostic, EMA fix, Stage A′, launcher arrays, phase 1 (trained and test-scored, acceptance passed) and oracle training are done. Running: oracle test scoring (66790277). Next: λ̂\_CRPS and the AFM transfer matrix → phase 2 with the D2 methods → CRPS evaluation of phase 2 (3.5 steps 5–7). This is now the critical path: the UNet arm's numbers compare degrees of dryness under a log loss, and only the AFM arm answers that.
- **Paper audit** (P1–P13 above): P3, P5, P8 and P9 gate the abstract and the methods section; P1, P2 and P4 gate claims 1 and 2; P6 gates claim 3's mechanism.
- **Literature additions** (2026-10-28): P15 (QM) needs only the existing source-only checkpoints and should follow P14, so that it is scored in the chosen floor. P16 (few-shot) and P17 need small `train_unet.py` changes and are independent of Stage 3. P15 and P16 come before the AFM phase 1 in priority.
- **Stage 4** (write-up) joins all tracks.

0.1 is the first job of the week. After that the two tracks run in parallel, but both are GPU jobs on the same allocation: Stage 1 evaluation passes and the GAP ablation queue alongside the 72 seed runs. They are not free, and running them on a login node is not an option. The tracks meet again at the write-up.

## Out of scope

Each of these is safe to drop because the paper's claims do not rest on it.

- **No more UDA methods.** The audit established the implemented ones are numerically doing what they claim and still failing. An eighth method produces another row, not another finding. The exceptions are input QM (P15), which is the benchmark's own published baseline and a direct test of claim 1, and the few-shot curve (P16), which uses labels and so is not a UDA method.
- **No hyperparameter search on the UDA weights.** Tuning λ buys a smaller negative number. The identifiability argument says the ceiling is set by the task, so a better-tuned method landing closer to the ceiling would change nothing about claim 2.
  *Qualified 2026-11-10.* The bullet above rests on "the implemented ones are numerically doing what they claim". The source-BN read broke that premise: the alignment terms did not act, and BN mixing did. One pre-registered **dose ladder** is therefore allowed if the M2 gate approves it. That means MMD and DANN on the BN-fixed trainer, 4 weights, europe→melanesia and horn→europe, seed 42, every weight reported, nothing selected. It tests claim 3 directly, i.e. whether an alignment that acts produces the trade the Zhao et al. (2019) lower bound predicts. It does not search for a better method. It runs only if `feature_probe.py` shows the features never aligned (MILESTONES.md M1).
- **CORAL and mmd\_ms out of the results table entirely.** Not reported as also-rans — removed, with one sentence in the methods and an appendix stub (P11). The reasons recorded in `run_exp_unet.sh`: CORAL subsumed by MMD, mean-blind and weakly scaled; mmd\_ms the same mechanism as MMD and unstable.
- **No 16-member probabilistic sweep across all pairs.** If CRPS becomes a headline metric, run it on the three diagonal pairs and state M.
- **No RainShift regions beyond the three.** Blacksea stays a dummy. Adding a fourth domain multiplies the training cost and the conditional-shift result is already established on three.
- **The two-stage CorrDiff variant.** Future work, one sentence.

The scope risk to watch: if you find yourself trying to make AFM beat the source-only baseline on target risk, you have drifted from a diagnosis paper into a method paper, and that needs a contribution you do not currently have.

## Open decisions

Six answers change what gets built; three are still open.

- [x] **Three seeds or two?** Three: seeds 42/43/44 already exist for phase 1, oracle and JDOT.
- [x] **Which methods get seeded?** FDA, spectral, AdaBN, DANN and MMD on top of the already-seeded JDOT. Decided 2026-09-30 (Stage 0.3); all 90 runs are done.
- [x] **Is there a deadline or target venue?** *Decided 2026-11-09:* no deadline. One paper, AIES; Lessons Learned by default, a full Article only if COSMO C2–C3 give a clean contrast. Experiments done by March 2027 (MILESTONES.md); the skeleton is in `report/aies/`. Was: proposed 2026-10-28: *AIES* (AMS Artificial Intelligence for the Earth Systems), where the downscaling community that needs the result reads. TMLR is the alternative if the identifiability result becomes the headline, since it does not penalise negative results. A short workshop version (e.g. Tackling Climate Change with ML) would stake the claim early. Everything above is ordered by scientific dependency, not by calendar. A submission date reorders Stage 3 against Stage 4 — if the deadline is tight, the write-up starts in parallel and the FM arm becomes a companion note.
- [x] **Include the GAP ablation, and on which method?** Yes, on MMD (seeded, no discriminator), europe_west → melanesia, four arms × three seeds. Decided 2026-10-05; see Stage 2 panel c, Redesign. It runs before the AFM LR sweep.
- [x] ~~Three in-domain AFM runs~~ — not needed; the phase-1 diagonal runs are the in-domain models.
- [x] **D1: the headline log floor (P14, opened 2026-10-27; decided 2026-10-29: 0.1 mm/h, see "D1 decided" under P3 result).** Was: either keep log(P + 1e-5 mm), the training space, as the headline and report a 0.01 mm floor as a robustness check, or the reverse. Under 1e-5, sub-0.01 mm values decide the europe\_west results (P3 result). The choice also sets the AFM selection space (3.6 step 2), so settle it before Stage A′. It needs the P14 re-score of all targets first.
- [ ] **D2: AFM phase-2 methods (reopened 2026-10-27).** The old proposal (FDA, spectral, AdaBN) rested on FDA and spectral being "the two that recovered anything", and P3 shows that was mostly the floor. Re-decide after P14. Candidate: FDA, an input-space method with no target forward and so no BatchNorm mixing, plus joint\_ot, the only method with a 3/3-seed gain outside europe\_west (europe→horn), plus one more chosen from the floored method table. 18 runs either way. *Update 2026-11-02:* the joint\_ot argument is void, because at the 0.1 mm floor there is no source-only gap into horn and the europe→horn gain is noise. Proposed instead: FDA (fixed; input space) and MMD (feature alignment without a discriminator, the GAP-ablation method), 12 runs; DANN only if a discriminator-based method is needed, with the mm-space blow-up check. Note the AFM coupling point: every feature loss acts on the shared encoder's GAP'd bottleneck, which conditions the flow net; the flow net itself receives no alignment gradient (`train_afm.py` probe comment), and spectral acts on μ.

One thing I am not certain about and cannot resolve from here: whether the transfer asymmetry — 0.435 and 0.349 into europe\_west against 0.086 to 0.184 into the tropics — is domain shift or target morphology. Stage 1.3 settles it. If the asymmetry largely survives conditioning on Moran's I, the identifiability framing stands as written. If it does not, the paper gains a second finding and the framing needs a paragraph on what makes a target domain hard independently of the source. Since P3 (2026-10-27) there is a third reading: the asymmetry may be largely the 1e-5 mm log floor. Re-derive it under P14 before Stage 1.3 is used to explain it.

