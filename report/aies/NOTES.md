# report/aies/: lessons-learned draft, notes (2026-11-09)

`main.tex` restructures the Overleaf project "Unsupervised domain adaptation for super
resolution" (the arxiv.sty project description) into an AIES Lessons Learned paper. It is a
skeleton: prose where the argument is settled, `\todo{}` where a number or paragraph is
missing, `\ev{}` pointing at the source of each number. Numbers come from
`report/overview_2026-11-07.md`, PLAN.md and PLAN_COSMO.md; nothing was recomputed.

## Lesson → evidence → state

| Lesson | RainShift evidence | COSMO | Missing before writing |
|---|---|---|---|
| L1 conditional shift first, calibrated null | P1b (2 pairs yes, horn↔mel undetermined), P1 (pixel null withdrawn), P2 (no power) | C2 | COSMO C2; figure |
| L2 labels first: oracle + few-shot | oracle 3 seeds (UNet), AFM oracle seed 42, P16 | C4–C5 | tables at 0.1 mm; paired-λ̂ seed sd; P4 (stronger oracle into europe) |
| L3 the adaptation term must act | audit A1 (FDA), A2 (DANN ratio), MMD magnitude, GAP ablation, BN mixing (UNet rule 3, AFM BN check), AdaBN | — | BN-check tasks 2–4; GAP ablation re-score at D1 or scoped sentence (A9) |
| L4 evaluation manufactures recovery | P3, P14, D1, zero predictor, DANN 10⁶ mm/h, climatology > source-only, seed-unstable ranking | — | trivial references at 0.1 mm; P12 (Kendall's W) |
| L5 objective does not change diagnosis | AFM phase 1/oracle/phase 2, FDA into europe (2 seeds), ensemble check | — | ERA5-tp reference in `afm_ensemble_check.py` |

PLAN_COSMO.md A1 had seven lessons. Merged: L5 (selection/seeds) into L4, and L7 (GAP) into
L3. A short paper cannot carry seven lessons; five is already generous.

## What was cut from the Overleaf draft and why

- **Redko/HΔH bound commentary, four identifiability assumptions, method-to-bound table.**
  Condensed to one intro paragraph and Table 1 (method → assumption). A lessons paper needs
  the bound as motivation, not as content.
- **Regime-decomposed λ_k and Proposition 1.** Never estimated, and the proposition is wrong
  as stated. Since min_h Σ_k w_k g_k(h) ≥ Σ_k w_k min_h g_k(h), the weighted sum of
  per-regime λ_k is a *lower* bound on λ, not an upper bound. The weights are also wrong:
  R_S weights regimes by μ_S and R_T by μ_T, so a single μ_T(X_k) cannot factor both. Do not
  bring it back without a correct statement.
- **"Expected contributions", "Future directions" (PAC-Bayes, causal mechanisms).** These were
  project-proposal sections.
- **Pre-training diagnostics: joint-marginal alignment residual, multi-source disagreement,
  physical-prior check.** Only the matched-input conditional test was run in a form that
  survived; it is L1.
- **CorrDiff rejection paragraph, quantile mapping as a method.** QM moves to Table 1 as
  RainShift's baseline.

## Graveyard: numbers in the Overleaf draft that are withdrawn

From overview §6. Do not reuse them:
- λ̂ "0.002–0.023 standardised MSE" and the Table "transfer gaps and the addressable budget"
  (single-seed `unet/` tree, 1e-5 floor, `mse_std`).
- Asymmetry "0.435 / 0.349", "four- to fivefold".
- Matched-input "86.9×", "14.5×", "74.5×", "137×" (pixel null 5–10× too small, P1).
- "All three pairs exhibit conditional shift" (now two pairs; horn↔mel undetermined).
- "Oracle recovers 34 % (earlier 72 %)" for europe→melanesia.
- Model-based disagreement "0.22–0.85", concentration "1.9–3.6" (pre-P2, superseded).
- σ ratios 0.729/0.723, 1.256/1.279 (old cross-normalisation conversion; use `mse_log`).

## AIES format

AMS lists "Lessons Learned" as an AIES submission type: short papers on insights into how well
AI methods work for a whole class of Earth-system applications
(https://www.ametsoc.org/ams/publications/author-information/choosing-a-journal-and-submission-type/).
The AMS site refuses automated fetches from the cluster, so the length, abstract and
significance-statement limits are **not verified**. Check them in a browser. The draft
already has the AMS components (significance statement, data availability, acknowledgments,
appendices). Moving to the AMS LaTeX class is left for submission.

An example of the type in review: arXiv:2607.18416 (hard conservation correctors in
autoregressive emulators). Worth reading for length and tone.

## Open decisions

- Title (three candidates in `main.tex`).
- Whether COSMO evidence sits inside L1/L2 (current) or in its own section. If the COSMO gate
  in PLAN_COSMO.md A3 comes out null, it becomes one paragraph in L1.
- Whether L5 stays a full lesson or becomes a paragraph of L4, if the ERA5-tp reference kills
  the FDA cell.
