#!/bin/bash -l
# ===========================================================================
#  RainShift trivial references at the 0.1 mm/h floor (MILESTONES M1 "Zero /
#  climatology references at 0.1 mm", L4; PLAN.md R5 step 4): zero,
#  train-split climatology and bicubic ERA5 `tp`, per target region, on the
#  test split, with mse_log_e01 / mse_log_e001, mm MAE/RMSE/bias, wet
#  fraction and maximum (cosmo_references.py; block_tp is left out because
#  200 is not a multiple of 80). All five R5 regions, read through the
#  overlay data_rainshift_r5/ (read-only symlinks to the shared stores).
#  CPU only.
#
#  Writes results/rainshift_references/references.csv; refuses if it exists.
#  Usage:  sbatch scripts/rainshift_refs.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name rs_refs
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 06:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_rainshift_r5"
OUT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/results/rainshift_references/references.csv"
# Bash arrays are whitespace-delimited: NO commas.
REGIONS=(europe_west horn-of-africa melanesia northamerica-east west-africa)

[[ -e "${OUT}" ]] && { echo "ERROR: ${OUT} exists; not overwriting."; exit 1; }

source "${CODE_ROOT}/scripts/python_env.sh"
cd "${CODE_ROOT}"
run_python "${CODE_ROOT}/cosmo_references.py" \
    --data_root "${DATA_ROOT}" --regions "${REGIONS[@]}" --out "${OUT}"
