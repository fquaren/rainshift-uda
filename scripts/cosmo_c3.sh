#!/bin/bash -l
# ===========================================================================
#  COSMO C3, training: source-only UNet per region and seed (PLAN_COSMO.md C3).
#  5 regions x seeds 42/43/44 = 15 runs, one per array task (seed-major).
#  RainShift phase-1 settings (25 epochs, batch 128, lr 5e-4, wd 1e-3,
#  patience 5, 12 workers), except:
#    --val_chunks 40       10 % of COSMO's 395 train chunks, as 88 of 877 on
#                          RainShift (strided, purge gap, offset 6 as there).
#    --data_order_seed     --seed also sets the training-data order
#                          (MILESTONES M2: COSMO seeds vary data order, not only
#                          initialisation; RainShift seeds do not).
#  Writes unet_cosmo_seed<S>/<r>__to__<r>__none/ and base_hp/; never a
#  RainShift tree. Refuses to start if the run directory exists.
#  Scoring of the 25 cells, the trivial references and the bicubic-tp
#  acceptance are a separate job, after this array.
#
#  Usage:  sbatch --array=0-14 scripts/cosmo_c3.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cosmo_c3
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --mem 200G
#SBATCH --time 12:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_cosmo_pm10"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"

# Bash arrays are whitespace-delimited: NO commas.
SEEDS=(42 43 44)
REGIONS=(alps british_isles poland scandinavia_south spain_central)
N=$(( ${#SEEDS[@]} * ${#REGIONS[@]} ))
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job: --array=0-$((N - 1))}"
(( TASK < N )) || { echo "ERROR: task ${TASK} >= ${N}"; exit 1; }
SEED="${SEEDS[$(( TASK / ${#REGIONS[@]} ))]}"
R="${REGIONS[$(( TASK % ${#REGIONS[@]} ))]}"

OUTPUT_DIR="${OUTPUT_ROOT}/unet_cosmo_seed${SEED}"
RUN="${OUTPUT_DIR}/${R}__to__${R}__none"
[[ -e "${RUN}" ]] && { echo "ERROR: ${RUN} exists; not overwriting."; exit 1; }
[[ -e "${OUTPUT_DIR}/base_hp/${R}__to__${R}.json" ]] && {
    echo "ERROR: base_hp for ${R} exists in ${OUTPUT_DIR}; not overwriting."; exit 1; }
[[ -d "${DATA_ROOT}/${R}/train_data_in.zarr" ]] || { echo "ERROR: no ${DATA_ROOT}/${R}"; exit 1; }

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

mkdir -p "${OUTPUT_DIR}/base_hp" "${OUTPUT_DIR}/best_hp"
echo "=== C3 | seed ${SEED} | ${R} -> ${RUN} ==="
run_python "${CODE_ROOT}/train_unet.py" \
    --source_path "${DATA_ROOT}/${R}" \
    --target_path "${DATA_ROOT}/${R}" \
    --output_dir  "${OUTPUT_DIR}" \
    --data_format npy \
    --uda_method  none \
    --epochs      25 \
    --batch_size  128 \
    --patience    5 \
    --num_workers 12 \
    --val_chunks  40 \
    --seed        "${SEED}" \
    --data_order_seed \
    2>&1 | tee "${OUTPUT_DIR}/phase1_${R}.log"
echo "=== done: ${RUN} ==="
