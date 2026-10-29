#!/bin/bash -l
# ===========================================================================
#  Matched-input conditional-shift test (conditional_shift_binned.py), with
#  the time-block nulls of PLAN.md P1. Same sample as the Aug 10 run behind
#  report Table 1 (test split, 600 frames, seed 42, 5 x 5 bins on CAPE x TCW,
#  log precipitation), so only the null changes. The Aug 10 output,
#  conditional_shift_binned.json, is left as it is; this writes OUT and
#  refuses to run if it exists.
#
#  Usage:  sbatch scripts/run_cond_binned_test.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cond
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 02:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
source "${CODE_ROOT}/scripts/python_env.sh"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
OUT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/unet/conditional_shift_binned_p1.json"

[[ -e "${OUT}" ]] && { echo "ERROR: ${OUT} exists. Not overwriting."; exit 1; }

run_python "${CODE_ROOT}/conditional_shift_binned.py" \
    --data_root "${DATA_ROOT}" \
    --split test \
    --n_frames 600 \
    --n_bins 5 \
    --seed 42 \
    --null_modes pixel week year \
    --null_reps 20 \
    --out "${OUT}"

echo "=== conditional shift test (P1) complete ==="
