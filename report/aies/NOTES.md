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
| L3 the adaptation term must act | **main evidence since 2026-11-10: source-BN re-scoring.** Every target-forward method, UNet (3 seeds) and AFM (seed 42), behaves like source-only once BN is re-estimated on source. Source cost 0.213 → 0.003 (UNet), and the target gains vanish with it. Also audit A1 (FDA), A2 (DANN ratio), MMD magnitude, GAP ablation, AdaBN | — | feature probe read 2026-11-11: never aligned in deployment; MMD aligned only under per-domain batch BN, which alone removes 97 % of the GAP distance. Missing: source-BN control on source-only and FDA; dose ladder (M2 gate, BN-fixed trainer) |
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
**Template (2026-11-09):** AMS LaTeX Package 6.2 (`ametsocV6.2.cls`, `ametsocV6.bst`,
supplied by the user). It is the same package for every AMS journal: there is no journal
option and no keywords command. From the template:
- abstract at most 250 words, in `\abstract{}` before `\begin{document}`;
- significance statement at most 120 words, `\sigstatement`;
- `\acknowledgments` and `\availstatement`, after a `\clearpage`;
- appendices with `\appendix[A]` + `\appendixtitle{}`, numbered from `\subsection`;
- table captions above the tabular, AMS rules `\topline`/`\midline`/`\botline` (no booktabs),
  no colour or shading in tables; cite only with `\citet`/`\citep`;
- submit in the default 1.5-spaced format; `[twocol]` is for our own reading only.

The total length limit for Lessons Learned is still unverified.

The arxiv.sty version is in `_trash/` (`main_arxiv_2026-11-09.tex`, `arxiv.sty`).

An example of the type in review: arXiv:2607.18416 (hard conservation correctors in
autoregressive emulators). Worth reading for length and tone.

## Open decisions

- Title: decided at the M2 gate (2026-11-09). Constraints: "unsupervised domain adaptation"
  in the title; "downscaling" over "super-resolution" (AIES audience); no "generative" (one
  model, one lesson, mostly seed 42: it would promise a generative-model study); American
  spelling ("unlabeled"). Candidates:
  1. Unsupervised domain adaptation for precipitation downscaling in data-sparse regions:
     what unlabeled data can and cannot do
  2. When does unsupervised domain adaptation help precipitation downscaling across
     regions? Lessons learned
  3. Lessons from unsupervised domain adaptation for cross-region precipitation
     downscaling (working title)
  4. Shared physics, shared mapping? Lessons from unsupervised domain adaptation for
     precipitation downscaling (if COSMO contrasts cleanly with RainShift)
  The current "What unlabelled target data cannot fix ..." commits to the negative before
  COSMO; drop it unless COSMO also fails.
- Whether COSMO evidence sits inside L1/L2 (current) or in its own section. If the COSMO gate
  in PLAN_COSMO.md A3 comes out null, it becomes one paragraph in L1.
- L5 stays a full lesson (2026-11-10): ERA5 tp scores worse than zero, so it does not kill the
  FDA cell. State FDA against climatology, not source-only.
- **Claim 3 / L3 rewrite (2026-11-10).** "Methods trade the source conditional for target risk"
  is withdrawn: the trade was BN mixing. New wording: the alignment terms did not act, and BN
  did. Add the theory for why a working alignment would not help either. Under label-marginal
  shift, invariant features force ε_S + ε_T up (Zhao et al. 2019, the JS lower bound). The
  intuition: if Z is invariant then Ŷ = g(Z) is invariant, but P(Y) differs (europe 0.097 vs
  melanesia 0.31 mm/h). Add `zhao2019invariant` there (already in references.bib). The dose
  ladder, if it runs, is the empirical test.
- Europe→melanesia with source BN goes into L4: a log-floor recovery of 0.4–1.0 next to
  10⁵–10⁶ mm/h pixels. The extremes come from extrapolation (tcw +4.4σ) through exp with
  σ_log = 3.7; RainShift's own ResNet does the same (18,113 mm/h CRPS, Tibetan Plateau).
- Setup sentence: RainShift clips ERA5 precipitation inputs to IMERG's range; we do not.
- **Candidate section: "complex topography is hard" (2026-11-17, from COSMO C3).**
  - *Evidence.* Every source into the Alps costs +8 % to +26 %, wet sources included. The
    Alps as source cost +26–31 % going wet→dry, the only wet→dry cells with a clear cost.
    The Alps also have the worst in-domain score (0.062).
  - *Caveat, which decides whether it can be a section.* It is one region. Topography is
    confounded with everything else about the Alps: the wettest region, summer convection,
    and the largest `sp` spread. A claim about topography needs a second complex-terrain
    region against a flat one with similar wetness.
  - *Where that region could come from.* Within COSMO, a further C1 build (e.g. the
    Pyrenees, the Carpathians or western Norway, if inside the COSMO domain). On RainShift,
    tibetan-plateau, which IMERG's poor high-terrain retrievals confound. Until then it is
    an observation in the COSMO paragraph, not a section.
- **The bicubic-`tp` reference means different things in the two datasets (2026-11-17).**
  Methods text and any figure that puts both side by side must say so.
  - *COSMO (perfect model).* The coarse input `tp` is the exact 10×10 block mean of the
    2.2 km target (checked: block error 0, bias 0). Bicubic `tp` is therefore a strong
    reference. It gets most of the log-floor score (0.051–0.099 against 0.24–0.59 for zero),
    and the UNet beats it by only 30–38 %. "Worse than bicubic `tp`", as in the four dry→wet
    C3 cells, means a learned model that has lost information the input hands it directly.
  - *RainShift.* `tp` is ERA5's own precipitation, a different product from IMERG, in m/h
    and not clipped to IMERG's range. ERA5 `tp` scores worse than zero (L5 note above). So
    bicubic `tp` is a weak reference there, and beating it says little.
  - *Consequence.* Do not compare "skill over bicubic" across the two datasets. Use
    in-domain-relative cost (E_T(f_S) / E_T(f_T) − 1) for cross-dataset statements, and
    bicubic only within COSMO.
