#!/bin/bash -l
# ===========================================================================
#  RainShift (tp, cape) control for COSMO C2 (PLAN_COSMO.md C2 "Consequence
#  for L1"; MILESTONES M1 COSMO). On COSMO the (cape, tcw) excess largely
#  vanished once coarse precipitation was conditioned on, so the RainShift
#  P1b (cape, tcw) excess is an upper bound. This asks whether the same
#  happens on RainShift, where `tp` is ERA5, not the target's block mean.
#
#  Same sample and settings as P1b (scripts/run_cond_binned_test.sh: three
#  domains, test split, 600 frames, seed 42, 5 x 5 bins, 4 x 4 fine sub-bins,
#  100 permutations, 200 bootstrap draws, >= 10 weeks per bin); only the
#  conditioning set changes. CPU only, read only.
#
#  Two array tasks:
#    0  (tp, cape), target block-averaged to the 40 x 40 common grid: the P1b
#       geometry. Not degenerate on RainShift (tp is ERA5).
#    1  (tp, cape), target at its native 200 x 200 grid (--target_scale
#       fine), 6 of the 25 fine pixels per cell: 9,600 pixels per frame,
#       the COSMO C2 fine budget (10,000), for a like-for-like contrast.
#
#  Runs test_conditional_shift_binned.py first and stops if it fails.
#  Writes results_rainshift_uda/unet/conditional_shift_binned_p1b_<set>.json;
#  refuses to run if it exists. P1b's own JSON is not touched.
#
#  Usage:  sbatch --array=0-1 scripts/rainshift_c2_tp_cape.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name rs_c2_tpcape
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 12:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"   # read-only
OUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/unet"

# Bash arrays are whitespace-delimited: NO commas.
DOMAINS=(europe_west horn-of-africa melanesia)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job: --array=0-1}"
case "${TASK}" in
    0) SET="tp_cape_block"; EXTRA=(--cond tp cape --target_scale block) ;;
    1) SET="tp_cape_fine";  EXTRA=(--cond tp cape --target_scale fine --fine_per_cell 6) ;;
    *) echo "ERROR: task ${TASK} not in 0-1"; exit 1 ;;
esac
OUT="${OUT_DIR}/conditional_shift_binned_p1b_${SET}.json"
[[ -e "${OUT}" ]] && { echo "ERROR: ${OUT} exists. Not overwriting."; exit 1; }
[[ -d "${OUT_DIR}" ]] || { echo "ERROR: ${OUT_DIR} missing"; exit 1; }

source "${CODE_ROOT}/scripts/python_env.sh"
cd "${CODE_ROOT}"

echo "=== test_conditional_shift_binned.py ==="
run_python "${CODE_ROOT}/test_conditional_shift_binned.py"

echo "=== RainShift ${SET}: ${DOMAINS[*]} -> ${OUT} ==="
run_python "${CODE_ROOT}/conditional_shift_binned.py" \
    --data_root "${DATA_ROOT}" \
    --domains "${DOMAINS[@]}" \
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
    "${EXTRA[@]}" \
    --out "${OUT}"
echo "=== RainShift ${SET} done ==="
