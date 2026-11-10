#!/bin/bash -l
# ===========================================================================
#  Feature-alignment probe (feature_probe.py; PLAN.md "Read 2026-11-10").
#  Did dann / mmd / joint_ot / spectral (and DANN gr0.5) ever align the GAP'd
#  bottleneck features they act on? Held-out MMD (GAP and grid4) and a linear
#  domain classifier, under source BN, the as-trained mixed BN and train-mode
#  per-domain batch BN, for the six ordered pairs. Inference only; every run
#  is read, never written.
#
#  One array task per seed (42/43/44); each loads 102 checkpoints.
#  Writes results/seed<S>/feature_probe/probe.csv and refuses to start if
#  that directory exists.
#
#  Usage:  sbatch --array=0-2 scripts/feature_probe.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name featprobe
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 48G
#SBATCH --time 02:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"   # read-only

# Bash arrays are whitespace-delimited: NO commas.
SEEDS=(42 43 44)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job: --array=0-2}"
(( TASK < ${#SEEDS[@]} )) || { echo "ERROR: task ${TASK} >= ${#SEEDS[@]}"; exit 1; }
SEED="${SEEDS[$TASK]}"
OUT="${OUTPUT_ROOT}/results/seed${SEED}/feature_probe"

if [[ -e "${OUT}" ]]; then
    echo "ERROR: ${OUT} exists; not overwriting."
    exit 1
fi

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== feature probe, seed ${SEED} -> ${OUT} ==="
run_python "${CODE_ROOT}/feature_probe.py" \
    --data_root "${DATA_ROOT}" --exp_root "${OUTPUT_ROOT}" --seed "${SEED}" --out_dir "${OUT}"
echo "=== done: ${OUT} ==="
