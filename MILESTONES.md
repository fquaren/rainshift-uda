# Milestones to March 2027

2026-11-09 · @Filippo

One paper, AIES. No deadline. Target: experiments done by the end of February 2027, so March
is writing. The paper type is decided at the M2 gate. The experiment designs live in PLAN.md
(RainShift) and PLAN_COSMO.md (COSMO); this file is only the calendar, the gates and the cuts.
The skeleton is in `report/aies/main.tex`, and `report/aies/NOTES.md` maps lesson → evidence.

Compute is not the constraint: about 60–80 A100-h for COSMO, plus a few short RainShift jobs.
The constraint is that each step waits on the previous read. Read a job within a day of it
finishing, and do not reopen a frozen item.

## M1 — by 2026-11-30: RainShift closed, COSMO data built

RainShift (after this, its numbers are frozen):
- [x] AFM BatchNorm check read: confirmed in all five pairs (PLAN_COSMO A1, 2026-11-09).
- [ ] ERA5-tp (bicubic-upsampled input `tp`) reference in `afm_ensemble_check.py`, then the
      FDA-into-europe_west cell is either stated or dropped (L5).
- [ ] P8: test metrics of `unet_seed42_a100check/` (limitations sentence).
- [ ] Regenerate every table at 0.1 mm/h with mm bias / wet fraction / max columns, from
      `seed<S>_v3/`, `seed<S>_fdafix/`, the oracle trees and the AFM trees.
- [ ] GAP ablation: re-score at D1 (A9), or keep only the sentence its `mse_std` reading supports.
- [ ] Zero / climatology references at 0.1 mm (L4).
- [ ] P12 (Kendall's W over seeds): one afternoon, backs L4.

COSMO:
- [ ] C1 build: months array 20348663 → merge 20348664 (running 2026-11-09). Check the store
      tests and the per-box statics (lakes and glaciers).
- [ ] C2: shift diagnosis on all 20 pairs (no training).

## M2 — by 2026-12-18: C3 and the paper-type gate

- [ ] FSS and RAPSD in `evaluate.py`, once, for both datasets. Needs no `train_time_index` fix.
- [ ] Trainers: a target pass that does not update BatchNorm statistics (B4.9). Write it now so
      January does not wait on it.
- [ ] C3: 5 phase-1 runs × 3 seeds, all 25 cells, with trivial references; acceptance against
      bicubic `tp`.
- [ ] **Gate, with Tom, before the break.**
  - Lessons Learned (default).
  - A full AIES Article only if C2–C3 contrast cleanly with RainShift: either no conditional
    excess and small or closable gaps on COSMO, or a clear conditional shift even in the
    perfect model.
  - Also decide here whether C6 and C8 run (see Cut).

## M3 — January 2027: COSMO labels and methods

- [ ] C4: oracle on 4 pairs (seeds 43/44 only if λ̂ is near noise).
- [ ] C5: FDA, AdaBN, few-shot N = 10 / 100 on the same pairs; MMD only with the BN-fixed pass.
- [ ] C7: PGW +3K on Spain and Scandinavia (preprocess, score, one oracle per region).

## M4 — February 2027: figures, frozen numbers, buffer

- [ ] Optional: one AFM run with the BN-fixed pass (horn→europe spectral). It makes L3
      "this breaks, and here is the fix", and tests the target side the BN check did not.
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
- Any new UDA method or weight tuning (PLAN.md "Out of scope").
