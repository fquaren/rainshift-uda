#!/bin/bash -l
# ===========================================================================
#  COSMO C2: shift diagnosis on all region pairs, no training (PLAN_COSMO.md
#  C2). CPU only. conditional_shift_binned.py on the five COSMO regions, test
#  split, 600 frames, seed 42, with the P1b settings (5 x 5 bins, 4 x 4 fine
#  sub-bins, 100 permutations, 200 bootstrap draws, >= 10 weeks per bin).
#  The statistic is symmetric, so 10 unordered pairs cover the 20 ordered
#  ones; the per-variable input W1 is reported per each region's sd, so both
#  directions are in the JSON.
#
#  Two array tasks, two conditioning sets:
#    0  (cape, tcw), target block-averaged to the 20 x 20 input grid, plus the
#       per-variable input W1 (--input_w1). The RainShift P1b statistic.
#    1  (tp, cape), target at its native 2.2 km grid (--target_scale fine),
#       25 of the 100 fine pixels per cell (same positions in every frame).
#       tp is the block mean of the target by construction, so a block-mean
#       target would make this set trivial; the fine pixels test sub-grid
#       structure given the coarse inputs.
#  Support is in the JSON (support_min) for both.
#
#  Prediction recorded before running (PLAN_COSMO.md C2): small but non-zero
#  conditional excess, largest for pairs with the Alps or Spain.
#
#  Runs test_conditional_shift_binned.py first and stops if it fails.
#  Writes results_cosmo/c2/c2_<set>.json; refuses to run if it exists.
#
#  Usage:  sbatch --array=0-1 scripts/cosmo_c2.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cosmo_c2
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 64G
#SBATCH --time 08:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_cosmo_pm10"
OUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_cosmo/c2"

# Bash arrays are whitespace-delimited: NO commas.
REGIONS=(alps british_isles poland scandinavia_south spain_central)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job: --array=0-1}"
case "${TASK}" in
    0) SET="cape_tcw_block"; EXTRA=(--cond cape tcw --target_scale block --input_w1) ;;
    1) SET="tp_cape_fine";   EXTRA=(--cond tp cape --target_scale fine --fine_per_cell 25) ;;
    *) echo "ERROR: task ${TASK} not in 0-1"; exit 1 ;;
esac
OUT="${OUT_DIR}/c2_${SET}.json"
[[ -e "${OUT}" ]] && { echo "ERROR: ${OUT} exists. Not overwriting."; exit 1; }
for r in "${REGIONS[@]}"; do
    [[ -d "${DATA_ROOT}/${r}/test_data_in.zarr" && -d "${DATA_ROOT}/${r}/test_data_out.zarr" ]] \
        || { echo "ERROR: ${DATA_ROOT}/${r} incomplete"; exit 1; }
done
mkdir -p "${OUT_DIR}"

source "${CODE_ROOT}/scripts/python_env.sh"
cd "${CODE_ROOT}"

echo "=== test_conditional_shift_binned.py ==="
run_python "${CODE_ROOT}/test_conditional_shift_binned.py"

echo "=== C2 ${SET}: ${#REGIONS[@]} regions -> ${OUT} ==="
run_python "${CODE_ROOT}/conditional_shift_binned.py" \
    --data_root "${DATA_ROOT}" \
    --domains "${REGIONS[@]}" \
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
echo "=== C2 ${SET} done ==="
