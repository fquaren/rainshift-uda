# OODiag on RainShift, and what our retraining does and does not say

Internal note, 2026-10-29. To become a paragraph of the discussion (P19) and,
if Tom agrees, a caveat in the OODiag paper's RainShift case.

## What OODiag computes in the RainShift case

OODiag (Beucler, Sirdey, Quarenghi, Asthana, Koch, Engelke; draft 2026-10-28)
does use a model. Section 7.3 and App. F.3: "The RainShift experiment uses the
fixed source-only U-Net and archived outputs from Harder et al. (2026); no
inference or retraining is performed." The diagnostic itself only sees the
inputs (eleven fields pooled to 4 × 4 maps) and one scalar per sample, the
archived per-sample `mae_mm` of that U-Net. It matches reference points to
evaluation analogs in input space and decomposes the shift gap

    Δ_shift = L_target − L_eval←train = 0.32304 − 0.07720 = 0.24584 mm

into Shapley contributions of the input marginals and of their dependence.
Whole-field Shapley: CAPE 0.0968, TCW 0.0811, dependence 0.0234, CP 0.0217,
TP 0.0150, u700 −0.0086, sp 0.0066, lsm 0.0044, z 0.0031, TCLW 0.0023,
v700 −0.0003, tisr 0.0002 mm. Support audit: 1 of 4,092 references passes.

The quantity attributed is defined in their Eq. 53 as "the expected error of
the unchanged model if its covariates followed Q_B … an associational
transport quantity, not the error of a retrained model or a causal
intervention." It tells where, in input space, one fixed function makes its
extra error on the target.

## Point 1: an attribution for a fixed model is not a retraining prediction

P19 asks a different question: if the model never sees CAPE and TCW (or lsm,
z, tisr, sp), does it transfer better? Retraining changes the function. It can
recover the removed information from correlated inputs (TCLW, CP, TP carry
much of TCW and CAPE) or lose information it needed. A large attribution does
not imply that removing the variable helps, and a small one does not imply that
removing it is harmless. OODiag does not claim either, so P19 neither
validates nor refutes it. Its paper proposes comparing with our domain
adaptation (§7.3: "whether successful adaptation reduces the same diagnosed
contributions"), which invites exactly that reading, so it has to be ruled out
in both papers.

In OODiag's own metric, retraining does not follow the attribution
(seed 42, melanesia test split, `results/seed42/field_mean_check_mask_<arm>/`,
full-input seeds from `field_mean_check_qm/`):

| source-only europe_west → melanesia | attributed (mm) | MAE (mm) | mse_log | predicted mean / observed |
| --- | --- | --- | --- | --- |
| full inputs, seeds 42/43/44 | — | 0.315 / 0.379 / 0.358 | 16.47 / 16.55 / 16.55 | 0.04 / 0.27 / 0.21 |
| `mask_geo` (lsm, z, tisr, sp) | 0.014 | 0.460 | 17.58 | 0.57 |
| `mask_shift` (cape, tcw) | 0.178 | 0.313 | 15.98 | 0.16 |

The arm attributed 0.014 mm raises MAE by 0.08–0.15 mm. The arm attributed
0.178 mm leaves it inside the seed range. In mse_log the direction happens to
agree with the attribution (mask_shift helps, mask_geo hurts). Agreement in
one metric and reversal in the other is what two unrelated quantities produce.

## Point 2: on Melanesia, per-sample MAE mostly measures how much it rained

Precipitation is non-negative, so the all-zero predictor's MAE is the field
mean: 0.0974 mm on the europe_west test split and 0.3127 mm on the melanesia
test split (`obs_mean_mm` in the field-mean summaries; the same 17,520-sample
test splits OODiag uses). Then:

- The all-zero predictor alone has a raw shift gap of 0.3127 − 0.0974 =
  0.215 mm, 88 % of OODiag's 0.246 mm. (OODiag's L_eval←train is
  analog-reweighted, so the comparison is approximate, but the reweighting
  changes the europe term only, and only by moving it toward training
  covariates.)
- Harder's U-Net is 21 % better than zero in Europe (0.077 against 0.097) and
  3 % worse than zero on Melanesia (0.323 against 0.313). Our models agree.
  Every melanesia-trained model and every oracle scores 0.297–0.303 mm, within
  5 % of zero. The source-only models score 0.31–0.46, ranked by how much
  rain they predict, since the predicted mean is 0.04–0.57 of the observed and
  every extra millimetre is penalised at misplaced pixels (the double
  penalty).

So on Melanesia per-sample MAE is roughly the per-sample rain amount plus a
penalty that grows with the model's own wetness. Its shift gap is mostly the
target being wetter than Europe, a property of P(Y), not of the model. The
Shapley decomposition then attributes "this sample is wetter than European
samples" to the inputs that predict rain amount, and CAPE and TCW are the
first two of those. Any predictor that misses Melanesian rain would get
roughly the same ranking whatever its failure mechanism. This is response
composition, and the poor support (1 of 4,092 references) means the analogs
cannot separate it from model error.

## The check, and what would change the reading

Run OODiag on the all-zero predictor (per-sample loss = per-sample field mean
of the truth) with the same players and references. For that predictor the
attribution is that of E[y | x] under the shift: a regression of rain amount
on the inputs.

- If the Shapley ranking and the shares come out close to the U-Net's (CAPE
  then TCW, together about 70 %), the RainShift attribution describes the
  target's rain distribution, not the U-Net.
- The U-Net's own contribution is the difference between the two games. It is
  a ~0.03 mm effect, and the support audit suggests it may not be resolvable.
- Alternatives that remove the intensity scaling: a per-sample loss in log
  space (log(P + floor), floor from D1), MAE divided by the sample's mean rain
  (undefined for dry samples, so it needs a wet-sample subset), or conditioning
  the comparison on the sample's observed field mean.

## For our paper

- Say that the P19 arms were chosen from the OODiag attribution, that the
  attribution uses target per-sample errors, and that this makes the selection
  supervised, not a UDA procedure.
- Do not present `mask_shift` as confirming OODiag. Its effect is in mse_log,
  and the relevant claim is about λ̂ (claim 2), not about the attribution.
- MAE on Melanesia is not a usable skill metric (the zero predictor is within
  5 % of the in-domain model). This supports not reporting MAE in mm as a
  headline anywhere, and it bears on the D1 floor choice.
- Seeds 43 and 44 of P19 (read 2026-11-02): `mask_shift` source-only
  `mse_log` 15.98 / 15.88 / 15.92 against 16.47 / 16.55 / 16.55 with all
  inputs, so the lower source-only risk replicates; the λ̂ rise does not
  (0.33 / 0.29 / 0.09). The field plots show `mask_shift` source-only is
  wetter (5.2 % wet pixels against 1.4 %), which is most of what improves.
  Still pending: Tom's run of the zero-predictor game.
- Our own source-only model into europe_west (from melanesia) is exactly the
  zero predictor (field mean 0.0000 mm/h on 1,752 test timesteps), which
  reinforces the point above for any MAE- or log-based attribution there.
