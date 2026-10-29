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
#  QM="tp precip all" (PLAN.md P15) also scores each off-diagonal source-only
#  model on quantile-mapped target inputs, one row per arm, and writes
#  results/seed<SEED>/field_mean_check_qm/<TARGET>/ instead. It runs
#  test_quantile_map.py first. The unmapped rows make it P14 for that target.
#
#  TREE_SUFFIX=_mask_geo scores unet_seed<SEED>_mask_geo/ and its _oracle/
#  (input-set ablation, PLAN.md P19) into field_mean_check_mask_geo/<TARGET>/.
#
#  Usage:  sbatch --array=42,43,44 scripts/field_mean_check.sh      # one task per seed
#          SEED=42 TARGET=europe_west sbatch scripts/field_mean_check.sh
#          QM="tp precip all" TARGET=melanesia sbatch --array=42,43,44 scripts/field_mean_check.sh
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
#SBATCH --time 03:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda
RESULTS_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"

SEED="${SEED:-${SLURM_ARRAY_TASK_ID:?SEED or --array=42,43,44 must be set}}"
TARGET="${TARGET:-europe_west}"
SUB="field_mean_check"
# Bash arrays are whitespace-delimited: QM="tp precip all", NO commas.
read -ra QM_ARMS <<< "${QM:-}"
QM_ARGS=()
SMOKE_QM_ARGS=()
if (( ${#QM_ARMS[@]} > 0 )); then
    SUB="field_mean_check_qm"
    QM_ARGS=(--input_qm "${QM_ARMS[@]}")
    SMOKE_QM_ARGS=(--qm_fit_chunks 2)
fi
# TREE_SUFFIX=_mask_geo scores the input-set ablation tree (PLAN.md P19).
TREE_SUFFIX="${TREE_SUFFIX:-}"
SUB="${SUB}${TREE_SUFFIX}"
QM_ARGS+=(--tree_suffix "${TREE_SUFFIX}")
OUT="${RESULTS_ROOT}/results/seed${SEED}/${SUB}/${TARGET}"
SMOKE_OUT="/scratch/fquareng/${SUB}_smoke/${SLURM_JOB_ID}"

[[ -e "${OUT}" ]] && { echo "ERROR: ${OUT} exists. Not overwriting."; exit 1; }
echo "SEED: ${SEED}   TARGET: ${TARGET}   QM: ${QM_ARMS[*]:-none}   output: ${OUT}"

if (( ${#QM_ARMS[@]} > 0 )); then
    echo "=== test_quantile_map.py ==="
    run_python "${CODE_ROOT}/test_quantile_map.py"
fi

echo "=== smoke: 2 batches -> ${SMOKE_OUT} ==="
run_python "${CODE_ROOT}/field_mean_check.py" \
    --seed "${SEED}" --target "${TARGET}" \
    --results_root "${RESULTS_ROOT}" --data_root "${DATA_ROOT}" \
    --output_dir "${SMOKE_OUT}" --max_batches 2 \
    "${QM_ARGS[@]}" "${SMOKE_QM_ARGS[@]}"

echo "=== full test split ==="
run_python "${CODE_ROOT}/field_mean_check.py" \
    --seed "${SEED}" --target "${TARGET}" \
    --results_root "${RESULTS_ROOT}" --data_root "${DATA_ROOT}" \
    --output_dir "${OUT}" "${QM_ARGS[@]}"

echo "=== field-mean check complete ==="
