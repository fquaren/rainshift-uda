# RainShift UDA — plan to conclusion

Sep 29, 2026 · @Filippo

Four claims have to land, and only the last needs a new model; the rest needs seeding and one ablation run. Lock the numbers that already exist, add the structure-aware metrics to checkpoints you already have, show the mechanism, then show the gap survives a change of model family.

## The four claims

The paper is a diagnosis, not a method. Nothing below needs a UDA method to work; three of the four claims are strengthened when they fail.

| # | Claim | Evidence it needs | Status |
| --- | --- | --- | --- |
| 1 | The three domain pairs differ in P(Y\|X), not only in P(X) | Null-calibrated matched-input W1\_cond, plus the model-based 9D diagnostic as corroboration | Done. Excess over null: europe↔horn 86.9×, horn↔melanesia 14.5×, europe↔melanesia support-limited at 43.8% |
| 2 | The gap is reducible in principle but not identifiable from unlabelled target data | Oracle joint training recovering target in-domain risk, giving λ̂ ≈ 0, repeated over seeds | Partial. λ̂ within 0.002–0.023 from the unseeded `unet/` tree. Oracles are trained for seeds 42/43/44; the seeded evaluation has not run |
| 3 | Methods fail by trading the source conditional for target risk, and feature alignment is structurally blocked | Source inflation against budget recovery, spectral shape loss, the JDOT feature-term share, and a GAP ablation | Partial. Numbers exist, the figure and the falsification do not |
| 4 | The failure is a property of the task, not of the MSE objective | The same diagnosis on a flow-matching model, with the risk redefined in CRPS | Not started |

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

**0.2 The launcher.** It is `launch_sbatch.sh` (its header still says `launch_pipeline.sh`). It has been syntax-checked and run three times; stages 1, 3 and 4 work. Once 0.1 lands, stages 2 and 5 are fixed with it. Resume a seed with `STAGES="2 5" SEEDS="<N>"`, and always print the plan with `DRY=1` before submitting.

**0.3 Seed the remaining methods.** This is the expensive step and it needs a decision first. Measured wall-clock per run: phase 1 about 0.7 h (early stopping at patience 5), oracle about 1.8 h, phase 2 joint_ot about 1.25 h. Phase 1, the oracles and joint_ot are already done for three seeds.

Proposal: seed four more methods — FDA, spectral, AdaBN, DANN. FDA and spectral are the two that recover anything, AdaBN is the clean null, and DANN represents the adversarial family. Report MMD from the single-seed `unet/` run, with a sentence saying it is a single draw and strongly negative. That is 4 methods × 6 pairs × 3 seeds = 72 runs, about 90–108 GPU-hours at 1.25–1.5 h per run; FDA and AdaBN do no target forward pass, so they come in lower. A SLURM job array over the 24 phase-2 runs per seed, instead of the serial loop, cuts wall-clock time but not GPU-hours.

Done when the transfer matrix and the per-method recovery table are reported as mean ± standard deviation over seeds 42/43/44, and the method ordering is either stable or the instability is stated in the text.

Seed variance here includes split variance: the validation offset is `seed % 9` (see CLAUDE.md), so the three seeds also have three slightly shifted validation sets. Say so in the methods section.

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

## Stage 3 — the flow-matching arm

The goal is to show the gap is a property of the task rather than of the MSE objective. Success is the gap persisting. If flow matching closes it, that is a different and better paper, but do not plan for it.

**3.1 Revive AFM rather than building something new.** The encoder plus flow network already takes x\_t and t, which is what a velocity field needs; a UNet that only maps (x\_dyn, x\_static) to a field does not. The known fixes are in place — `_match_dyn` for the `sample()` shape crash, eval-mode side effects removed, μ exposed in the result dict. Smoke-test on one pair with `--subset_chunks` before anything else. `run_exp_afm.sh` passes no `--seed` and writes to a fixed `afm/` tree, so it needs the same per-seed output directories as the UNet launcher before any seeded AFM run. `evaluate.sh`'s AFM section needs the same change. The CorrDiff-style two-stage variant, regression UNet plus an FM corrector on residuals, goes in future work: under domain shift the residual distribution the corrector trains on is itself shifted, which stacks a second identifiability problem on the first.

**3.2 Redefine the risk. This is the real work.** The oracle decomposition, λ̂ and the whole transfer matrix are currently defined on pointwise MSE, which is the wrong functional for a probabilistic model — scoring the ensemble mean would smuggle the smoothing preference straight back in. Redefine in CRPS:

```latex
\hat{\lambda}_{\mathrm{CRPS}} = \mathrm{CRPS}_T(h_{\mathrm{joint}}) - \mathrm{CRPS}_T(h_T)
```

That needs a target-trained in-domain model h\_T for each domain. As in the UNet arm, h\_T is the phase-1 model trained on T (the diagonal `{T}__to__{T}` run), so the three source-only runs already provide it and no extra training is needed. Fix the ensemble size before any number is reported, since CRPS is biased at small M — use the fair estimator or state M and keep it constant everywhere.

**3.3 What to train.** Three source-only runs (which double as the in-domain models), six joint oracles, then the surviving methods from Stage 2 on six pairs. If the survivors are FDA, spectral and AdaBN, that is 18 method runs, 27 runs in total for one seed. Seed the arm only if Stage 0 shows seed variance is material.

**3.4 Probabilistic evaluation.** CRPS and spread-skill in-domain and under transfer, plus rank histograms per pair. The question no one in this literature has asked is whether ensemble calibration degrades under domain shift the way the point forecast does — a well-calibrated source model that becomes over-confident on the target is the probabilistic signature of the same failure, and it is a cheap extra panel.

**Acceptance.** Source-only AFM should beat source-only UNet on CRPS in-domain, as a sanity check that the arm works at all. Then the transfer gap should have the same sign and roughly the same ordering across pairs as the UNet arm. If the ordering inverts, say so — that is a finding, not a bug.

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

## Dependencies

- **0.1** (seed-aware materialise + evaluate) blocks everything: it produces the first seeded results.
- **Track A:** 0.3 (seed the remaining methods) → seeded transfer matrix and recovery table.
- **Track B:** fix `train_time_index` → Stage 1 (FSS, RAPSD, Moran's I, case studies) → Stage 2 (mechanism figure, GAP ablation).
- **Stage 3** (AFM) needs `run_exp_afm.sh` seeded, then smoke test → training → CRPS evaluation. It uses Stage 2 to choose which methods survive.
- **Stage 4** (write-up) joins all tracks.

0.1 is the first job of the week. After that the two tracks run in parallel, but both are GPU jobs on the same allocation: Stage 1 evaluation passes and the GAP ablation queue alongside the 72 seed runs. They are not free, and running them on a login node is not an option. The tracks meet again at the write-up.

## Out of scope

Each of these is safe to drop because the paper's claims do not rest on it.

- **No more UDA methods.** The audit established the implemented ones are numerically doing what they claim and still failing. An eighth method produces another row, not another finding.
- **No hyperparameter search on the UDA weights.** Tuning λ buys a smaller negative number. The identifiability argument says the ceiling is set by the task, so a better-tuned method landing closer to the ceiling would change nothing about claim 2.
- **CORAL and mmd\_ms out of the results table entirely.** Not reported as also-rans — removed, with one sentence in the methods saying they were implemented, found numerically inert by audit, and fixed too late to seed.
- **No 16-member probabilistic sweep across all pairs.** If CRPS becomes a headline metric, run it on the three diagonal pairs and state M.
- **No RainShift regions beyond the three.** Blacksea stays a dummy. Adding a fourth domain multiplies the training cost and the conditional-shift result is already established on three.
- **The two-stage CorrDiff variant.** Future work, one sentence.

The scope risk to watch: if you find yourself trying to make AFM beat the source-only baseline on target risk, you have drifted from a diagnosis paper into a method paper, and that needs a contribution you do not currently have.

## Open decisions

Four answers change what gets built. The first blocks 0.3.

- [x] **Three seeds or two?** Three: seeds 42/43/44 already exist for phase 1, oracle and JDOT.
- [ ] **Which methods get seeded?** Proposal: FDA, spectral, AdaBN, DANN on top of the already-seeded JDOT. MMD reported single-seed with the caveat.
- [ ] **Is there a deadline or target venue?** Everything above is ordered by scientific dependency, not by calendar. A submission date reorders Stage 3 against Stage 4 — if the deadline is tight, the write-up starts in parallel and the FM arm becomes a companion note.
- [ ] **Include the GAP ablation, and on which method?** It is the falsification test for claim 3. DANN has a seeded reference; MMD would need three extra seeded runs on the chosen pair.
- [x] ~~Three in-domain AFM runs~~ — not needed; the phase-1 diagonal runs are the in-domain models.

One thing I am not certain about and cannot resolve from here: whether the transfer asymmetry — 0.435 and 0.349 into europe\_west against 0.086 to 0.184 into the tropics — is domain shift or target morphology. Stage 1.3 settles it. If the asymmetry largely survives conditioning on Moran's I, the identifiability framing stands as written. If it does not, the paper gains a second finding and the framing needs a paragraph on what makes a target domain hard independently of the source.

