#!/bin/bash -l
# ===========================================================================
#  Source-BN control (MILESTONES M1). The _srcbn re-scoring re-estimated BN
#  on source for every target-forward run and found they all score like
#  source-only. This asks whether the re-estimation itself moves a model
#  whose BN statistics were never contaminated: the same procedure
#  (unet_srcbn_export.py --control: reset, momentum=None, 200 source train
#  batches, torch seed 1000) on
#    - source-only  unet_seed<N>/*__none        (9 runs: 3 diagonals and
#                   their 6 byte-identical pair copies; source-only, checked)
#    - FDA (fixed)  unet_seed<N>_fdafix/*__fda  (6 runs)
#  into unet_seed<N>_srcbnctrl/, scored with evaluate.py (source and target
#  test, no training split) into results/seed<N>_srcbnctrl/unet/none/.
#  Compare with the as-trained rows: results/seed<N>_v3/unet/none (none) and
#  results/seed<N>_fdafix/unet/none (fda).
#
#  Tasks (3): seeds 42/43/44.
#  Usage:  sbatch --array=0-2 scripts/unet_srcbn_control.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name srcbn_ctrl
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --mem 200G
#SBATCH --time 03:00:00

set -euo pipefail
export LC_ALL=C

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"

# Bash arrays are whitespace-delimited: NO commas.
SEEDS=(42 43 44)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job: --array=0-2}"
(( TASK < ${#SEEDS[@]} )) || { echo "ERROR: task ${TASK} >= ${#SEEDS[@]}"; exit 1; }
SEED="${SEEDS[$TASK]}"

SIDE="${OUTPUT_ROOT}/unet_seed${SEED}_srcbnctrl"
RES="${OUTPUT_ROOT}/results/seed${SEED}_srcbnctrl/unet/none"
[[ -e "${SIDE}" ]] && { echo "ERROR: ${SIDE} exists; not overwriting."; exit 1; }
[[ -e "${RES}" ]] && { echo "ERROR: ${RES} exists; not overwriting."; exit 1; }

RUNS=()
for d in "${OUTPUT_ROOT}/unet_seed${SEED}"/*__to__*__none "${OUTPUT_ROOT}/unet_seed${SEED}_fdafix"/*__to__*__fda; do
    [[ -d "${d}" ]] || continue
    [[ -f "${d}/best.pt" && -f "${d}/config.json" ]] || { echo "ERROR: ${d} incomplete"; exit 1; }
    RUNS+=("${d}")
done
(( ${#RUNS[@]} == 15 )) || { echo "ERROR: ${#RUNS[@]} runs, expected 15 (9 none + 6 fda)"; exit 1; }

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== source-BN control | seed ${SEED} | ${#RUNS[@]} runs -> ${SIDE} ==="
run_python "${CODE_ROOT}/unet_srcbn_export.py" --control --runs "${RUNS[@]}" --out_root "${SIDE}"

echo "=== scoring ${SIDE} -> ${RES} ==="
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${SIDE}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RES}" \
    --input_transform none \
    --transform_cache_dir "${OUTPUT_ROOT}/results/transforms" \
    --save_samples 0
echo "=== done: ${RES} ==="
