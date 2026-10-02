#!/bin/bash -l
# ===========================================================================
#  Re-score the AdaBN runs of one seed with the adapted model.
#
#  Until 2026-10-02 evaluate.py scored best.pt for AdaBN runs, i.e. the model
#  BEFORE BatchNorm adaptation (the adapted one is best_adabn.pt), so every
#  AdaBN row was a second source-only run. This job moves the seed's AdaBN
#  result folders to results/seed<N>/unet/none/_trash/adabn_unadapted/ and
#  then runs evaluate.sh, which skips everything already scored, re-scores
#  the six AdaBN runs (target with best_adabn.pt, source and train with
#  best.pt), replaces their rows in results.csv and regenerates ngg_v2/.
#
#  Usage:  SEED=42 sbatch scripts/rescore_adabn.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name rescore_adabn
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 200G
#SBATCH --time 06:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
SEED="${SEED:?SEED must be set, e.g. SEED=42}"
RES="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/results/seed${SEED}/unet/none"
TRASH="${RES}/_trash/adabn_unadapted"

shopt -s nullglob
dirs=("${RES}"/*__adabn)
moved=("${TRASH}"/*__adabn)
echo "SEED ${SEED}: ${#dirs[@]} AdaBN result folders in ${RES}, ${#moved[@]} already in ${TRASH}"

if (( ${#dirs[@]} == 6 && ${#moved[@]} == 0 )); then
    mkdir -p "${TRASH}"
    mv "${dirs[@]}" "${TRASH}/"
    echo "moved to ${TRASH}:"
    ls "${TRASH}"
elif (( ${#moved[@]} == 6 )); then
    # A previous attempt moved them and then failed; evaluate.sh skips what
    # it has already re-scored, so rerunning it resumes.
    echo "resuming: the unadapted results were moved by an earlier attempt"
else
    echo "ERROR: expected 6 AdaBN folders in ${RES} (or 6 already in ${TRASH})."
    exit 1
fi

SEED="${SEED}" bash "${CODE_ROOT}/scripts/evaluate.sh"
