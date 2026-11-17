#!/bin/bash -l
# ===========================================================================
#  AFM phase-2 BatchNorm-mixing check (overview 2026-11-07, §9.1). No training.
#
#  One spectral phase-2 run per task, against its source's phase-1 model.
#  Scores source validation CRPS with saved statistics, with source-only
#  re-estimated statistics, and with the phase-2 source/target BatchNorm
#  traffic replayed on the phase-1 model. See afm_bn_check.py for the arms
#  and the decision rule.
#
#  spectral is used because its alignment term is negligible (~0.007), so its
#  damage is the target forward and little else.
#
#  Output: results_rainshift_uda/afm_bn_check_seed42/<tag>/bn_check.json, a
#  side tree no evaluation or materialise script reads. Refuses to overwrite.
#
#  Usage:  sbatch --array=0-4 scripts/afm_bn_check.sh
#  Smoke:  SMOKE=1 sbatch --array=0 --time 00:30:00 scripts/afm_bn_check.sh
#          (2 validation batches per score, 4 re-estimation batches, _smoke tree)
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name afm_bn_check
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

# One A100, half a node, <= 12 h: gpu-normal QOS.
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --exclude=dnagpu008
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --mem 200G
#SBATCH --time 03:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
SEED=42
METHOD="spectral"
RUNS="${OUTPUT_ROOT}/afm_seed${SEED}"
TREE="${OUTPUT_ROOT}/afm_bn_check_seed${SEED}"

EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
    TREE="${OUTPUT_ROOT}/afm_bn_check_seed${SEED}_smoke"
    EXTRA+=(--max_batches 2 --n_batches 4)
fi

# Bash arrays are whitespace-delimited: NO commas. "source:target".
# 0-1 diverged (best.pt is epoch 1: read p1_mix only); 2-4 trained.
PAIRS=(
    "horn-of-africa:europe_west"
    "melanesia:europe_west"
    "europe_west:melanesia"
    "horn-of-africa:melanesia"
    "europe_west:horn-of-africa"
)

TASK="${SLURM_ARRAY_TASK_ID:?run as an array job, e.g. sbatch --array=0-$((${#PAIRS[@]} - 1))}"
(( TASK < ${#PAIRS[@]} )) || { echo "ERROR: task ${TASK} >= ${#PAIRS[@]} pairs"; exit 1; }
SRC="${PAIRS[$TASK]%%:*}"
TGT="${PAIRS[$TASK]##*:}"

P1_DIR="${RUNS}/afm_${SRC}__to__${SRC}__none"
P2_DIR="${RUNS}/afm_${SRC}__to__${TGT}__${METHOD}"
OUT_DIR="${TREE}/afm_${SRC}__to__${TGT}__${METHOD}"
for d in "${P1_DIR}" "${P2_DIR}"; do
    [[ -f "${d}/best.pt" && -f "${d}/config.json" ]] || { echo "ERROR: ${d} incomplete"; exit 1; }
done
[[ -e "${OUT_DIR}" ]] && { echo "ERROR: ${OUT_DIR} exists; not overwriting."; exit 1; }
mkdir -p "${TREE}"

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== AFM BN check | task ${TASK}/${#PAIRS[@]} | ${SRC} -> ${TGT} (${METHOD}) ==="
echo "Phase 1: ${P1_DIR}"
echo "Phase 2: ${P2_DIR}"
echo "Output:  ${OUT_DIR}"

run_python "${CODE_ROOT}/afm_bn_check.py" \
    --p1_dir  "${P1_DIR}" \
    --p2_dir  "${P2_DIR}" \
    --out_dir "${OUT_DIR}" \
    "${EXTRA[@]}"

echo "=== done: ${OUT_DIR} ==="
