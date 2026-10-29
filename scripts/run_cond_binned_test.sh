#!/bin/bash -l
# ===========================================================================
#  Matched-input conditional-shift test (conditional_shift_binned.py), P1b.
#  Same sample as the Aug 10 run behind report Table 1 and as P1 (test split,
#  600 frames, seed 42, 5 x 5 bins on CAPE x TCW). New in P1b (audit A4):
#  conditional-permutation null (4 x 4 fine sub-bins, whole (domain, week)
#  units, 100 draws) and empirical p-value; statistics at the D1 floors
#  (log(P + 0.1) headline, + 0.01 check), wet fraction and wet amounts
#  separately, 1e-5 as a legacy column; support in distinct weeks (>= 10);
#  week-block bootstrap CI; the P1 one-domain nulls kept as diagnostics next
#  to a size-matched cross-domain half comparison.
#  conditional_shift_binned.json (Aug 10) and conditional_shift_binned_p1.json
#  (P1) are left as they are; this writes OUT and refuses to run if it exists.
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
#SBATCH --time 03:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
source "${CODE_ROOT}/scripts/python_env.sh"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
OUT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/unet/conditional_shift_binned_p1b.json"

[[ -e "${OUT}" ]] && { echo "ERROR: ${OUT} exists. Not overwriting."; exit 1; }

cd "${CODE_ROOT}"
run_python "${CODE_ROOT}/test_conditional_shift_binned.py"

run_python "${CODE_ROOT}/conditional_shift_binned.py" \
    --data_root "${DATA_ROOT}" \
    --split test \
    --n_frames 600 \
    --n_bins 5 \
    --n_fine 4 \
    --min_weeks 10 \
    --seed 42 \
    --n_perm 100 \
    --n_boot 200 \
    --null_modes pixel week year \
    --null_reps 20 \
    --out "${OUT}"

echo "=== conditional shift test (P1b) complete ==="
