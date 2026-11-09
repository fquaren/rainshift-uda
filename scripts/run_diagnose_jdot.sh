#!/bin/bash -l
# ===========================================================================
#  JDOT diagnostic (audit A8): Sinkhorn marginal errors at the training
#  iteration count and beyond, and the feature term's share of the transport
#  cost, over N_BATCHES training-like batches in train-mode BatchNorm. See
#  diagnose_jdot.py. No training, no checkpoint writing.
#
#  Loads <SRC>__to__<TGT>__joint_ot/best.pt from unet_seed<SEED>/ (alpha and
#  reg from its config.json). Writes
#  results/seed<SEED>/diagnose_jdot/<SRC>__to__<TGT>/ and refuses to run if it
#  exists. Runs a 1-batch smoke pass into /scratch first.
#
#  Usage:  sbatch scripts/run_diagnose_jdot.sh
#          SEED=42 SRC=europe_west TGT=horn-of-africa N_BATCHES=8 sbatch scripts/run_diagnose_jdot.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name diagjdot
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 01:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda
RESULTS_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"

SEED="${SEED:-42}"
SRC="${SRC:-europe_west}"
TGT="${TGT:-horn-of-africa}"
N_BATCHES="${N_BATCHES:-8}"
EXP_ROOT="${RESULTS_ROOT}/unet_seed${SEED}"
OUT="${RESULTS_ROOT}/results/seed${SEED}/diagnose_jdot/${SRC}__to__${TGT}"
SMOKE_OUT="/scratch/fquareng/diagnose_jdot_smoke/${SLURM_JOB_ID}"

[[ -e "${OUT}" ]] && { echo "ERROR: ${OUT} exists. Not overwriting."; exit 1; }
CKPT="${EXP_ROOT}/${SRC}__to__${TGT}__joint_ot/best.pt"
[[ -f "${CKPT}" ]] || { echo "ERROR: ${CKPT} not found."; exit 1; }
echo "SEED: ${SEED}   ${SRC} -> ${TGT}   batches: ${N_BATCHES}   output: ${OUT}"

echo "=== smoke: 1 batch -> ${SMOKE_OUT} ==="
run_python "${CODE_ROOT}/diagnose_jdot.py" \
    --code_root "${CODE_ROOT}" --data_root "${DATA_ROOT}" \
    --source "${SRC}" --target "${TGT}" --exp_root "${EXP_ROOT}" \
    --n_batches 1 --output_dir "${SMOKE_OUT}"

echo "=== ${N_BATCHES} batches ==="
run_python "${CODE_ROOT}/diagnose_jdot.py" \
    --code_root "${CODE_ROOT}" --data_root "${DATA_ROOT}" \
    --source "${SRC}" --target "${TGT}" --exp_root "${EXP_ROOT}" \
    --n_batches "${N_BATCHES}" --output_dir "${OUT}"

echo "=== JDOT diagnostic complete ==="
