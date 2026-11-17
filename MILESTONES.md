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
- [x] `compare_fields` training-maximum columns (job 20394524, read 2026-11-11, PLAN.md).
      Above-maximum pixels are ≤ 3.4×10⁻⁴ of pixels, source-only included; clamping moves
      `mse_log_e01` by ≤ 0.005, so the log-floor numbers stand. The mm mean is unusable
      unclamped. joint_ot seed 43 beating the oracle is seed-unstable (L4).
- [x] **Feature probe** (job 20394525, read 2026-11-11, PLAN.md). Reading (a) for every method
      in deployment: deployed GAP MMD ≥ 29× the null, domain classifier 0.98-1.00, as for
      source-only. MMD aligned in the space its loss saw (train-mode per-domain batch BN,
      0.12× source-only), but only 41 % of that reached the deployed features and the
      prediction did not move. Per-domain batch BN alone removes 97 % of the GAP distance, so
      every loss acted on a residual. By the rule, the ladder goes to the M2 gate, BN-fixed
      trainer only, arms DANN and MMD.
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
- [x] C2: shift diagnosis (job 20502163, read 2026-11-16, PLAN_COSMO.md). (cape, tcw): clear
      excess in 7 of 10 pairs of the perfect model; (tp, cape) on fine pixels: 1 clear, 5 weak,
      4 within. The (cape, tcw) excess is mostly projection, so claim 1 / L1 becomes an upper
      bound. Open: RainShift (tp, cape) as the control (one CPU job).

## M2 — by 2026-12-18: C3 and the paper-type gate

- [ ] Trainers: a target pass that does not update BatchNorm statistics (B4.9). UNet
      **written 2026-11-11** (`train_unet.py --target_bn eval`, `TARGET_BN=eval` in
      `run_exp_unet.sh`); `test_target_bn.py` passed (job 20438249). Default-weight MMD run
      (europe→melanesia, seed 42, `_tgtbneval`) read 2026-11-12: features align in deployment,
      log-floor recovery ≈ 1.0, mm MAE worse (PLAN.md). Replication read 2026-11-13 (jobs
      20467868–73): reading 2, log-floor only. Recovery 1.00 / 0.99 / 1.03 into melanesia,
      mm fails in 2 of 3 seeds (seed 43 blows up), horn→europe unchanged (zero predictor).
      The dose ladder is dropped. Open: a trivial-wetting reference before the L3 sentence.
      AFM trainer not changed (only needed for the optional M4 run). The dose ladder and C5's
      MMD wait on this.
- [ ] FSS and RAPSD in `evaluate.py`, once, for both datasets. Needs no `train_time_index` fix.
- [ ] Before C3: let `--seed` reach the COSMO data order. *Flag written 2026-11-13:*
      `train_unet.py --data_order_seed` (off by default; train splits only, recorded in
      config.json). Not yet run; the COSMO C3 launcher must pass it. Generalise the `train_half` branch
      of `build_loaders` (`ds.seed = args.seed` after construction, so shuffle follows the seed
      and the validation offset stays seed 42's) into a flag, and use it for every COSMO run.
      COSMO has no earlier runs to stay comparable with. RainShift stays as is; the paper states
      that its seed spread is initialisation (plus GPU nondeterminism) only, not data order
      (Tom, 2026-11-12: "weights initialization, stochastic optimization").
- [ ] Read the oracle source-cost column in `seeds_42_43_44_unet_mse_std` against seed sd. If
      it is within noise, the target-good model lies in the source-optimal set: under-
      determination measured on real data, and the toy problem (gate item below) illustrates
      it. If it is clearly non-zero, the flat-loss framing does not fit RainShift.
      *Read 2026-11-13* (`mse_log_e01`, oracle source − source-only in-domain, paired per seed
      42/43/44; oracle rows from `results/seed<S>/unet/oracle/`, source-only from `seed<S>_v3/`;
      source-only seed sd in brackets). Europe source: e→horn +0.024 / −0.001 / +0.017,
      e→melanesia +0.053 / +0.044 / +0.025 (sd 0.0009), clearly non-zero. Horn source:
      h→europe −0.004 / −0.000 / −0.003, h→melanesia −0.006 / −0.004 / −0.004 (sd 0.002),
      zero or better. Melanesia source: m→europe +0.072 / −0.007 / +0.006, m→horn +0.028 /
      −0.039 / +0.018 (sd 0.013), within noise. So the framing fits by source: with horn
      (and, noisily, melanesia) as source, the target-good model is in the source-optimal
      set; with europe_west as source it costs source risk (e→melanesia ≈ 25 % of its target
      budget). The toy problem illustrates the first case only, and the paper says so.
- [x] C3: 5 phase-1 runs × 3 seeds, all 25 cells, with trivial references; acceptance against
      bicubic `tp`. *Read 2026-11-17* (jobs 20588668–69, 20588686, 20588687; PLAN_COSMO.md C3).
      Acceptance passes: in-domain is 30–38 % below bicubic `tp`, and the UNets are still dry
      (bias −24 to −36 % of the mean). Gaps by pair:
      - 10 of 20 within noise;
      - 6 small (+0.005 to +0.016);
      - 4 large: poland or spain_central into british_isles or scandinavia_south, 0.14–0.20
        against 0.045–0.049 in-domain, seed sd 0.05–0.06, worse than bicubic `tp`.

      Not a null for the A3 gate. Open: check the maxima and wet fraction on the four large
      cells (compare_fields), because RMSE up to 6.8 mm/h suggests blow-up pixels.
- [ ] **Gate, with Tom, before the break.**
  - Lessons Learned (default).
  - A full AIES Article only if C2–C3 contrast cleanly with RainShift: either no conditional
    excess and small or closable gaps on COSMO, or a clear conditional shift even in the
    perfect model.
  - **The dose ladder: run or drop.** The probe read (a) (2026-11-11), so it is on the table:
    MMD and DANN on the BN-fixed trainer, 4 pre-registered weights (spaced up to a visible source cost), europe→melanesia
    and horn→europe, seed 42, about 16 runs and 25–30 A100-h. Every weight is reported with
    feature MMD and domain-classifier accuracy. The best weight in hindsight appears only as
    an upper bound that used target labels. This is not tuning: nothing is selected.
    *Against it (Tom, 2026-11-12: the trade-off is sensitive to the pair and the seed).* One
    draw per weight cannot show a curve when the method ordering already flips across seeds
    (L5). A robust version is ≥ 3 seeds × 2 pairs, about 48 runs and 75 A100-h, for a curve
    Zhao et al. (2019) already predict. Recommendation: drop it, and carry the trade-off with
    the toy problem below.
  - **Toy problem (Tom, 2026-11-12).** Two-feature linear regression: collinear in the
    source, decorrelated in the target, so the source loss is flat along one direction while
    the target loss is not, and no marginal alignment picks that direction without labels.
    It illustrates claim 2 (underspecification, D'Amour et al. 2020). Caveat to state in the
    paper: source-only seeds barely spread on target (melanesia `mse_log` 16.47–16.55), so the
    set is large but SGD does not explore it. The claim is that unlabelled data cannot move
    within the set, not that runs scatter across it. Gated on the oracle source-cost read above.
  - Transfer pair as a factor (Tom, 2026-11-12): analyse sensitivity to the pair over all 20
    C2 pairs, not over the 4 labelled C4/C5 pairs.
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
