#!/bin/bash -l
# ===========================================================================
#  COSMO C3, scoring: materialise the pair-named source-only checkpoints of one
#  seed's tree (unet_cosmo_seed<S>/, 5 diagonal -> 25 cells, byte-identical
#  copies), then evaluate.py on all 25 cells (source and target test).
#  One array task per seed (42/43/44). Chain afterok on the training array.
#  Trivial references are scripts/cosmo_refs.sh (target-only, no training).
#
#  Writes results/cosmo_seed<S>/unet/none/; refuses if it exists. The
#  materialise step never overwrites a differing pair directory.
#
#  Usage:  sbatch --array=0-2 --dependency=afterok:<train jobs> scripts/cosmo_c3_score.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cosmo_c3s
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 12:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_cosmo_pm10"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"

# Bash arrays are whitespace-delimited: NO commas.
SEEDS=(42 43 44)
REGIONS=(alps british_isles poland scandinavia_south spain_central)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job: --array=0-2}"
(( TASK < ${#SEEDS[@]} )) || { echo "ERROR: task ${TASK} >= ${#SEEDS[@]}"; exit 1; }
SEED="${SEEDS[$TASK]}"
TREE="${OUTPUT_ROOT}/unet_cosmo_seed${SEED}"
RES="${OUTPUT_ROOT}/results/cosmo_seed${SEED}/unet/none"
[[ -e "${RES}" ]] && { echo "ERROR: ${RES} exists; not overwriting."; exit 1; }

echo "=== materialise ${TREE} ==="
SEED="${SEED}" MODEL=unet_cosmo DOMAINS_ENV="${REGIONS[*]}" OUTPUT_DIR="${OUTPUT_ROOT}" \
    bash "${CODE_ROOT}/scripts/materialise_baselines.sh"
n=$(ls -d "${TREE}"/*__to__*__none | wc -l)
(( n == ${#REGIONS[@]} * ${#REGIONS[@]} )) || { echo "ERROR: ${n} run dirs, expected 25"; exit 1; }

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== evaluate ${TREE} -> ${RES} ==="
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${TREE}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RES}" \
    --input_transform none \
    --transform_cache_dir "${OUTPUT_ROOT}/results/transforms_cosmo" \
    --save_samples 0
echo "=== done: ${RES} ==="
