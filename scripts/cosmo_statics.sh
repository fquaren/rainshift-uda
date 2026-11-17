#!/bin/bash -l
# ===========================================================================
#  COSMO perfect-model arm: surface height and land mask from the model's own
#  hourly output (cosmo_statics.py; PLAN_COSMO.md B2 "Statics", B5). CPU.
#
#  Usage:  sbatch scripts/cosmo_statics.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cosmo_statics
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 01:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_cosmo_pm10/_statics"

[[ -e "${OUT_DIR}" ]] && { echo "ERROR: ${OUT_DIR} exists; not overwriting."; exit 1; }

source "${CODE_ROOT}/scripts/python_env.sh"

echo "=== COSMO statics -> ${OUT_DIR} ==="
run_python "${CODE_ROOT}/cosmo_statics.py" --out_dir "${OUT_DIR}"
echo "=== done ==="
