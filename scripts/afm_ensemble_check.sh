#!/bin/bash -l
# ===========================================================================
#  AFM ensembles into europe_west against information-free references
#  (afm_ensemble_check.py; PLAN.md "Read 2026-11-04"): is FDA's CRPS gain
#  skill, or spread added to a near-dry source-only forecast? Seed 42 by
#  default, 1-in-10 test subset. Every run is read, never written.
#
#  Writes results/seed${SEED}/afm_ensemble_check/europe_west/ and refuses to
#  start if that directory exists.
#
#  Usage:  SEED=42 SUFFIX=_tpref sbatch scripts/afm_ensemble_check.sh
#  (2026-11-09: adds the bicubic ERA5-tp reference, --tp_ref, and dresses it
#  to each FDA spread.)
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name afm_enscheck
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 03:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"   # read-only
SEED="${SEED:?SEED must be set, e.g. SEED=42}"
STRIDE="${STRIDE:-10}"
# SUFFIX: a new output directory beside the first run (which must not be
# overwritten), e.g. SUFFIX=_tpref for the run with the ERA5-tp reference.
OUT="${OUTPUT_ROOT}/results/seed${SEED}/afm_ensemble_check/europe_west${SUFFIX:-}"
A="${OUTPUT_ROOT}/afm_seed${SEED}"
O="${OUTPUT_ROOT}/afm_seed${SEED}_oracle"
P="${OUTPUT_ROOT}/results/seed${SEED}"

if [[ -e "${OUT}" ]]; then
    echo "ERROR: ${OUT} exists; not overwriting."
    exit 1
fi

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

HE="horn-of-africa__to__europe_west"
ME="melanesia__to__europe_west"
EE="europe_west__to__europe_west"
run_python "${CODE_ROOT}/afm_ensemble_check.py" \
    --data_root "${DATA_ROOT}" --output_dir "${OUT}" \
    --target europe_west --stride "${STRIDE}" --n_ensemble 8 --steps 10 \
    --run "SO horn=${A}/afm_${HE}__none" \
    --run "FDA horn=${A}/afm_${HE}__fda" \
    --run "oracle horn=${O}/afm_${HE}__none" \
    --run "SO melanesia=${A}/afm_${ME}__none" \
    --run "FDA melanesia=${A}/afm_${ME}__fda" \
    --run "oracle melanesia=${O}/afm_${ME}__none" \
    --run "in-domain=${A}/afm_${EE}__none" \
    --dress "SO horn:FDA horn" \
    --dress "SO melanesia:FDA melanesia" \
    --tp_ref \
    --dress "ERA5 tp:FDA horn" \
    --dress "ERA5 tp:FDA melanesia" \
    --rank_npz "SO horn=${P}/afm_prob/none/afm_${HE}__none/per_sample_loss.npz" \
    --rank_npz "FDA horn=${P}/afm_prob_p2fda/none/afm_${HE}__fda/per_sample_loss.npz" \
    --rank_npz "oracle horn=${P}/afm_prob/oracle/afm_${HE}__none/per_sample_loss.npz" \
    --rank_npz "SO melanesia=${P}/afm_prob/none/afm_${ME}__none/per_sample_loss.npz" \
    --rank_npz "FDA melanesia=${P}/afm_prob_p2fda/none/afm_${ME}__fda/per_sample_loss.npz" \
    --rank_npz "oracle melanesia=${P}/afm_prob/oracle/afm_${ME}__none/per_sample_loss.npz" \
    --rank_npz "in-domain=${P}/afm_prob/none/afm_${EE}__none/per_sample_loss.npz"

echo "=== -> ${OUT} ==="
