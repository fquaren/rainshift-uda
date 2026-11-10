# Milestones to March 2027

2026-11-09 · @Filippo · realigned 2026-11-10 after the source-BN read

One paper, AIES. No deadline. Target: experiments done by the end of February 2027, so March
is writing. The paper type is decided at the M2 gate. The experiment designs live in PLAN.md
(RainShift) and PLAN_COSMO.md (COSMO); this file is only the calendar, the gates and the cuts.
The skeleton is in `report/aies/main.tex`, and `report/aies/NOTES.md` maps lesson → evidence.

Compute is not the constraint: about 60–80 A100-h for COSMO, plus a few short RainShift jobs
(and 25–30 A100-h if the dose ladder runs). The constraint is that each step waits on the
previous read. Read a job within a day of it finishing, and do not reopen a frozen item.

**What changed on 2026-11-10.** With BatchNorm re-estimated on source, every UNet and AFM
target-forward method behaves like source-only. The source cost and the target gains both
came from BN mixing (PLAN.md, "Read 2026-11-10"). Claim 3 ("methods trade the source
conditional for target risk") has no evidence left. It becomes L3's main evidence: the
adaptation terms did not act. Whether a term that *does* act produces the trade that Zhao et
al. (2019) predict is now an open experiment (the dose ladder below), gated on the feature
probe.

## M1 — by 2026-11-30: RainShift closed, COSMO data built

RainShift, done:
- [x] AFM BatchNorm check read: confirmed in all five pairs (PLAN_COSMO A1, 2026-11-09).
- [x] ERA5-tp reference (job 20357462, read 2026-11-10). ERA5 tp scores worse than zero
      (`crps_mm` 0.130 vs 0.098). FDA into europe_west stays in L5, stated against climatology
      (0.0865 / 0.0849 vs 0.0926), not against source-only, with the melanesia wet bias.
- [x] Source-BN re-scoring, UNet and AFM (read 2026-11-10, PLAN.md): claim 3 is BN artefact.
- [x] `compare_fields.py` on europe→melanesia with source BN (job 20376532, read 2026-11-10): an L4
      example, not a gain (joint_ot wets towards the oracle; 10⁵–10⁶ mm/h pixels; PLAN.md).
- [x] Pipeline against RainShift (arXiv v1, 2026-11-10). Same transform, and source-region
      statistics normalise the target. In-region and single-source CRPS are in their range
      (horn 0.048–0.055 vs 0.053–0.057, melanesia 0.233–0.298 vs 0.272–0.311). Their ResNet
      also blows up out of region (18,113 mm/h, Tibetan Plateau). One difference: they clip
      ERA5 precipitation inputs to IMERG's range and we do not (a setup sentence). Check the
      Sci. Rep. version before citing numbers.

RainShift, open (in this order):
- [ ] Read `compare_fields` with the training-maximum columns (job 20391536,
      `results/plots/compare_fields_srcbn_trainmax_seed<S>/`): the pixel fraction above the
      source training maximum and above 100 mm/h, and scores clamped at that maximum.
- [ ] **Feature probe** (`scripts/feature_probe.sh`, prepared 2026-11-10, not submitted; 3 × 1 A100,
      under 2 h). Did dann / mmd / joint_ot / spectral / DANN gr0.5 ever align the GAP'd
      bottleneck? Measured by held-out MMD (GAP, grid4) and a linear domain classifier under
      source BN, mixed BN and train-mode batch BN. Its reading decides the dose ladder at the
      M2 gate:
  - (a) the features never aligned: a stronger weight means something, so the ladder is worth
    running;
  - (b) they aligned and the prediction did not move: GAP alignment cannot constrain the
    output, the ladder is dropped, and GAP goes back into L3;
  - (c) train-mode BN already makes the GAP features near-indistinguishable: the loss had
    nothing to act on, a structural form of (b).
- [ ] Source-BN control: re-estimate BN on source-only and FDA as well (cumulative average vs
      EMA). It closes the "re-estimation itself did it" objection. `unet_srcbn_export.py`
      refuses non-target-forward runs, so it needs a flag.
- [ ] Rewrite claim 3 everywhere it appears: the CLAUDE.md "What this project is" (yours),
      `report/aies/main.tex` L3 and intro, and NOTES.md. "Trade" becomes "did not act; BN did",
      plus the Zhao lower bound as the reason a working alignment would not help either.
- [ ] Regenerate every table at 0.1 mm/h with mm bias / wet fraction / max / above-training-max
      columns, from `seed<S>_v3/`, `seed<S>_fdafix/`, the `_srcbn` trees, the oracle trees and the
      AFM trees. The `_srcbn` rows sit next to the as-trained ones.
- [ ] P8: test metrics of `unet_seed42_a100check/` (limitations sentence).
- [ ] GAP ablation: re-score at D1 (A9), or keep only the sentence its `mse_std` reading
      supports. Wait for the probe: (b)/(c) gives the ablation a role again.
- [ ] Zero / climatology references at 0.1 mm (L4).
- [ ] P12 (Kendall's W over seeds): one afternoon, backs L4.

COSMO:
- [x] C1 build: months array 20348663 and merge 20348664 completed 2026-11-09. Five stores are in
      `data_cosmo_pm10/<region>`. Statics: job 20343366. `cp` is a constant 0 (std 0; normalises
      to 0 in float64). `z` is bimodal where there is sea (log ε): C2 should know.
- [ ] C2: shift diagnosis on all 20 pairs (no training).

## M2 — by 2026-12-18: C3 and the paper-type gate

- [ ] Trainers: a target pass that does not update BatchNorm statistics (B4.9). **Moved up:**
      write it in the first week of December. The dose ladder, C5's MMD and the optional AFM
      run all wait on it.
- [ ] FSS and RAPSD in `evaluate.py`, once, for both datasets. Needs no `train_time_index` fix.
- [ ] C3: 5 phase-1 runs × 3 seeds, all 25 cells, with trivial references; acceptance against
      bicubic `tp`.
- [ ] **Gate, with Tom, before the break.**
  - Lessons Learned (default).
  - A full AIES Article only if C2–C3 contrast cleanly with RainShift: either no conditional
    excess and small or closable gaps on COSMO, or a clear conditional shift even in the
    perfect model.
  - **The dose ladder: run or drop, from the probe.** If (a), MMD and DANN on the BN-fixed
    trainer, 4 pre-registered weights (spaced up to a visible source cost), europe→melanesia
    and horn→europe, seed 42, about 16 runs and 25–30 A100-h. Every weight is reported with
    feature MMD and domain-classifier accuracy. The best weight in hindsight appears only as
    an upper bound that used target labels. This is not tuning: nothing is selected.
  - Also decide here whether C6 and C8 run (see Cut).
  - Also decide the title (candidates in report/aies/NOTES.md, "Open decisions").

## M3 — January 2027: COSMO labels and methods

- [ ] C4: oracle on 4 pairs (seeds 43/44 only if λ̂ is near noise).
- [ ] C5: FDA, AdaBN, few-shot N = 10 / 100 on the same pairs; MMD only with the BN-fixed pass.
- [ ] If the gate approved it: the dose ladder (RainShift), then its read with `compare_fields`
      (wet fraction, max, above-training-max).
- [ ] C7: PGW +3K on Spain and Scandinavia (preprocess, score, one oracle per region).

## M4 — February 2027: figures, frozen numbers, buffer

- [ ] Optional: one AFM run with the BN-fixed pass (horn→europe spectral). It tests the target
      side that the source-BN re-scoring cannot. Drop it if the ladder ran and read cleanly: the
      UNet ladder already answers "does a working term act".
- [ ] Fix `train_time_index`, then the case-study panel.
- [ ] All paper figures and tables generated from scripts, with numbers frozen.
- [ ] Rest is buffer.

## M5 — March 2027: writing

Fill `report/aies/` lesson by lesson; abstract and significance statement last.

## Cut (decided 2026-11-09; reopen only at the M2 gate)

- Moran's I (Stage 1.3), P17 (λ of representations), the NGG appendix.
- More AFM seeds beyond seed 43's europe check.
- The old draft's multi-source disagreement, physical-prior check and regime-decomposed λ_k.
- C6 (AFM on COSMO), unless C3 shows the dry-collapse or zero-predictor pattern again.
- C8 (no statics), unless C2 points at orography.
- Any new UDA method or weight tuning (PLAN.md "Out of scope"). One exception: the pre-registered
  dose ladder, if the M2 gate approves it. Its premise changed on 2026-11-10: the methods were
  not "doing what they claim".
- Changing the training transform (ε = 1e-5 is RainShift's). The clamp at the training maximum is
  an evaluation robustness row, not a retraining.
