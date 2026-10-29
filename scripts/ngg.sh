#!/bin/bash -l
# ===========================================================================
#  Recompute NGG from existing results.csv files, no model evaluation.
#
#  Uses the fixed-reference definition (--e_s_ref baseline): E_S is the
#  source-only in-domain risk for every method. W1 is in SOURCE-normalised
#  space: the raw cube of covariate_shift_analysis/raw/ divided by s_S per
#  channel (compute_ngg.py --w1_space source_norm). Writes to
#  results/seed<N>/ngg_v3/ and refuses to run for a seed whose ngg_v3/
#  already exists. results/seed<N>/ngg_v2/ (W1 from the own-stats
#  normalized/ cube, deprecated, audit 2026-10-29 A5) and ngg/ are not
#  touched.
#
#  Usage:  SEEDS="42 43 44" sbatch scripts/ngg.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name ngg
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

# No GPU work: numpy and matplotlib only, so a CPU node with the x86 env.
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 00:30:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
# dl-torch mamba env on x86 nodes.
source "${CODE_ROOT}/scripts/python_env.sh"
OUTPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"   # read-only: stats only
W1_MATRIX="${CODE_ROOT}/covariate_shift_analysis/raw/Wasserstein_1D_test.npy"
W1_META="${CODE_ROOT}/covariate_shift_analysis/raw/Wasserstein_1D_test_variables.json"
NGG_NAME="ngg_v3"
[[ -f "${W1_MATRIX}" && -f "${W1_META}" ]] || {
    echo "ERROR: raw W1 cube ${W1_MATRIX} missing; NGG refuses the own-stats normalized/ cube."; exit 1; }

SEEDS="${SEEDS:?SEEDS must be set, e.g. SEEDS=\"42 43 44\" sbatch scripts/ngg.sh}"
TRANSFORMS=("none")

# Check every seed before writing anything, so a clash aborts the whole job.
for SEED in ${SEEDS}; do
    RESULTS_DIR="${OUTPUT_DIR}/results/seed${SEED}"
    [[ -f "${RESULTS_DIR}/unet/none/results.csv" ]] || { echo "ERROR: no results.csv for seed ${SEED}"; exit 1; }
    [[ -e "${RESULTS_DIR}/${NGG_NAME}" ]] && { echo "ERROR: ${RESULTS_DIR}/${NGG_NAME} exists; not overwriting."; exit 1; }
done

for SEED in ${SEEDS}; do
    RESULTS_DIR="${OUTPUT_DIR}/results/seed${SEED}"
    for tf in "${TRANSFORMS[@]}"; do
        echo "=== NGG | seed ${SEED} | unet | transform: ${tf} ==="
        run_python "${CODE_ROOT}/compute_ngg_all.py" \
            --w1_matrix "${W1_MATRIX}" \
            --w1_variables "${W1_META}" \
            --w1_space source_norm \
            --stats_root "${DATA_ROOT}" \
            --results_csv "${RESULTS_DIR}/unet/${tf}/results.csv" \
            --model unet \
            --error_metric "mse_std" \
            --w1_agg "mean_inputs" \
            --e_s_ref "baseline" \
            --output_root "${RESULTS_DIR}/${NGG_NAME}/unet_${tf}"
        echo ""
    done
done

echo "=== NGG done ==="
