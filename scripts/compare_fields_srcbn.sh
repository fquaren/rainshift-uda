#!/bin/bash -l
# ===========================================================================
#  europe_west -> melanesia with source-only BatchNorm (PLAN.md, "Read 2026-11-10"):
#  are the log-floor target gains of the source-BN models real, or the
#  near-dry-plus-outliers pattern? compare_fields.py writes field mean, wet
#  fraction, p99, max and mse_log_e01 per run (summary.csv) plus field panels,
#  climatology maps and the pixel histogram. Every run is read, never written.
#
#  One array task per seed (42/43/44). Columns: source-only, joint_ot with mixed
#  BN (as trained), joint_ot / DANN / DANN gr0.5 / spectral with source BN,
#  oracle, in-domain.
#
#  Writes results/plots/compare_fields_srcbn${TAG}_seed<S>/europe_west__to__melanesia/
#  and refuses to start if it exists.
#
#  Usage:  sbatch --array=0-2 scripts/compare_fields_srcbn.sh
#  Re-run (2026-11-10, adds the training-maximum columns; the first run, job
#  20376532, wrote compare_fields_srcbn_seed<S>/ and is kept):
#          TAG=_trainmax sbatch --array=0-2 scripts/compare_fields_srcbn.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cmpf_srcbn
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 01:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"   # read-only

# Bash arrays are whitespace-delimited: NO commas.
SEEDS=(42 43 44)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job: --array=0-2}"
(( TASK < ${#SEEDS[@]} )) || { echo "ERROR: task ${TASK} >= ${#SEEDS[@]}"; exit 1; }
SEED="${SEEDS[$TASK]}"
STRIDE="${STRIDE:-10}"

U="${OUTPUT_ROOT}/unet_seed${SEED}"
EM="europe_west__to__melanesia"
OUT="${OUTPUT_ROOT}/results/plots/compare_fields_srcbn${TAG:-}_seed${SEED}/${EM}"

if [[ -e "${OUT}" ]]; then
    echo "ERROR: ${OUT} exists; not overwriting."
    exit 1
fi

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== seed ${SEED}: ${EM}, source BN -> ${OUT} ==="
run_python "${CODE_ROOT}/compare_fields.py" \
    --data_root "${DATA_ROOT}" --output_dir "${OUT}" \
    --source europe_west --target melanesia --stride "${STRIDE}" \
    --run "source-only=${U}/${EM}__none" \
    --run "joint_ot mixed BN=${U}/${EM}__joint_ot" \
    --run "joint_ot src BN=${U}_srcbn/${EM}__joint_ot" \
    --run "DANN src BN=${U}_srcbn/${EM}__dann" \
    --run "DANN gr0.5 src BN=${U}_danngr05_srcbn/${EM}__dann" \
    --run "spectral src BN=${U}_srcbn/${EM}__spectral" \
    --run "oracle=${U}_oracle/${EM}__none" \
    --run "in-domain=${U}/melanesia__to__melanesia__none"
echo "=== done: ${OUT} ==="
