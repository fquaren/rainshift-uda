#!/bin/bash -l
# ===========================================================================
#  R5 (PLAN.md 2026-11-17): phase-1 source-only UNet on the two new RainShift
#  regions, northamerica-east and west-africa, seeds 42/43/44 = 6 runs, one
#  per array task (seed-major). Settings as run_exp_unet.sh PHASE=1 (25
#  epochs, batch 128, lr 5e-4, wd 1e-3, patience 5, 12 workers,
#  val_chunks 88). No --data_order_seed, so the seeds mean what they mean for
#  the existing RainShift runs (initialisation only).
#
#  Data from the overlay root data_rainshift_r5/ (stats computed there by
#  rainshift_r5_stats.sh; nothing written to raw_data/rainshift).
#  Side tree unet_seed<S>_r5/, never unet_seed<S>/. Refuses if the run
#  directory or its base_hp file exists, or if the region's stats are missing.
#
#  Stack: A100 + dl-torch. The existing phase-1 models were trained on GH200
#  with the container; the 2026-09-30 check put that difference an order of
#  magnitude below seed variance in training/validation loss (test metrics:
#  P8, open).
#
#  Usage:  sbatch --array=0-5 --dependency=afterok:<stats job> scripts/rainshift_r5_phase1.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name r5_p1
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
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_rainshift_r5"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"

# Bash arrays are whitespace-delimited: NO commas.
SEEDS=(42 43 44)
REGIONS=(northamerica-east west-africa)
N=$(( ${#SEEDS[@]} * ${#REGIONS[@]} ))
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job: --array=0-$((N - 1))}"
(( TASK < N )) || { echo "ERROR: task ${TASK} >= ${N}"; exit 1; }
SEED="${SEEDS[$(( TASK / ${#REGIONS[@]} ))]}"
R="${REGIONS[$(( TASK % ${#REGIONS[@]} ))]}"

OUTPUT_DIR="${OUTPUT_ROOT}/unet_seed${SEED}_r5"
RUN="${OUTPUT_DIR}/${R}__to__${R}__none"
[[ -e "${RUN}" ]] && { echo "ERROR: ${RUN} exists; not overwriting."; exit 1; }
[[ -e "${OUTPUT_DIR}/base_hp/${R}__to__${R}.json" ]] && {
    echo "ERROR: base_hp for ${R} exists in ${OUTPUT_DIR}; not overwriting."; exit 1; }
[[ -f "${DATA_ROOT}/${R}/normalization_stats.json" && ! -L "${DATA_ROOT}/${R}/normalization_stats.json" ]] || {
    echo "ERROR: ${DATA_ROOT}/${R}/normalization_stats.json missing (run rainshift_r5_stats.sh)"; exit 1; }

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

mkdir -p "${OUTPUT_DIR}/base_hp" "${OUTPUT_DIR}/best_hp"
echo "=== R5 phase 1 | seed ${SEED} | ${R} -> ${RUN} ==="
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
    --val_chunks  88 \
    --seed        "${SEED}" \
    2>&1 | tee "${OUTPUT_DIR}/phase1_${R}.log"
echo "=== done: ${RUN} ==="
