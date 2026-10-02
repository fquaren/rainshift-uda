#!/bin/bash -l
# ===========================================================================
#  Field-mean check: where in the intensity distribution do the models scored
#  on one target domain make their error? See field_mean_check.py.
#
#  Scores every checkpoint with target TARGET in unet_seed<SEED>/ (phase 2 and
#  the in-domain model) and unet_seed<SEED>_oracle/, on the TEST split only.
#  Writes results/seed<SEED>/field_mean_check/<TARGET>/ and refuses to run if
#  it exists. Runs a 2-batch smoke pass into /scratch first.
#
#  Usage:  SEED=42 TARGET=europe_west sbatch scripts/field_mean_check.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name fieldmean
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 64G
#SBATCH --time 01:30:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda
RESULTS_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"

SEED="${SEED:?SEED must be set, e.g. SEED=42 sbatch scripts/field_mean_check.sh}"
TARGET="${TARGET:-europe_west}"
OUT="${RESULTS_ROOT}/results/seed${SEED}/field_mean_check/${TARGET}"
SMOKE_OUT="/scratch/fquareng/field_mean_check_smoke/${SLURM_JOB_ID}"

[[ -e "${OUT}" ]] && { echo "ERROR: ${OUT} exists. Not overwriting."; exit 1; }
echo "SEED: ${SEED}   TARGET: ${TARGET}   output: ${OUT}"

echo "=== smoke: 2 batches -> ${SMOKE_OUT} ==="
run_python "${CODE_ROOT}/field_mean_check.py" \
    --seed "${SEED}" --target "${TARGET}" \
    --results_root "${RESULTS_ROOT}" --data_root "${DATA_ROOT}" \
    --output_dir "${SMOKE_OUT}" --max_batches 2

echo "=== full test split ==="
run_python "${CODE_ROOT}/field_mean_check.py" \
    --seed "${SEED}" --target "${TARGET}" \
    --results_root "${RESULTS_ROOT}" --data_root "${DATA_ROOT}" \
    --output_dir "${OUT}"

echo "=== field-mean check complete ==="
