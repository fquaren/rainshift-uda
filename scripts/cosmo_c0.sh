#!/bin/bash -l
# ===========================================================================
#  COSMO perfect-model arm, C0 feasibility (PLAN_COSMO.md B3). CPU, no GPU.
#
#  Reads a few files of /reference/FGSE/climate_simulations (read-only) and
#  writes c0.json + grid.npz to a new side folder. See cosmo_c0_feasibility.py.
#
#  Usage:   sbatch scripts/cosmo_c0.sh
#  Smoke:   SMOKE=1 sbatch --time 00:20:00 scripts/cosmo_c0.sh
#           (skips the 1.18 M-entry directory inventory; _smoke folder)
#  Mount:   MODE=mount sbatch --partition gpu --gres gpu:1 --cpus-per-task 1 \
#               --mem 2G --time 00:05:00 scripts/cosmo_c0.sh
#           (no python: is /reference visible on a gpu node? prints and exits)
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cosmo_c0
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16G
#SBATCH --time 01:30:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
ARCHIVE="/reference/FGSE/climate_simulations"
OUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_cosmo_pm10"

echo "node: $(hostname)  partition: ${SLURM_JOB_PARTITION:-?}"
if [[ ! -d "${ARCHIVE}/cosmo6_2km_climate/1h_2D" ]]; then
    echo "ERROR: ${ARCHIVE} not visible on $(hostname)"; exit 1
fi
echo "/reference visible: $(ls "${ARCHIVE}" | tr '\n' ' ')"

if [[ "${MODE:-}" == "mount" ]]; then
    ls -la "${ARCHIVE}/cosmo6_2km_climate/1h_2D/lffd20150701120000.nz"
    echo "=== mount check done on $(hostname) ==="
    exit 0
fi

OUT_DIR="${OUT_ROOT}/_c0"
EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
    OUT_DIR="${OUT_ROOT}/_c0_smoke"
    EXTRA+=(--skip_inventory)
fi
[[ -e "${OUT_DIR}" ]] && { echo "ERROR: ${OUT_DIR} exists; not overwriting."; exit 1; }
mkdir -p "${OUT_ROOT}"

source "${CODE_ROOT}/scripts/python_env.sh"

echo "=== COSMO C0 feasibility -> ${OUT_DIR} ==="
run_python "${CODE_ROOT}/cosmo_c0_feasibility.py" --out_dir "${OUT_DIR}" "${EXTRA[@]}"
echo "=== done: ${OUT_DIR} ==="
