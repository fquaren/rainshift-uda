#!/bin/bash -l
# ===========================================================================
#  COSMO perfect-model arm, C1 (cosmo_build.py; PLAN_COSMO.md B2/B3). CPU.
#
#  I/O-bound: 13 netCDF reads per hour off the archive. With 4 cores and 4 workers the
#  2026-11-09 build ran at 8-15 % CPU efficiency (DCSR warning, 136 jobs; peak RSS 1.5 GB
#  of 24 GB). Now 1 core, 4 G, and WORKERS=4 processes oversubscribed on it: they mostly
#  wait on reads. Throughput at 1 core is untested.
#
#  Months (132 tasks, 2011-01 .. 2021-12; staging on /scratch, resumable):
#      MODE=month sbatch --array=0-131%20 scripts/cosmo_build.sh
#  Smoke (2015-07 only, separate staging dir):
#      MODE=month SMOKE=1 sbatch --array=0 scripts/cosmo_build.sh
#  Merge (5 tasks, one per region; zarr stores on /work):
#      MODE=merge sbatch --array=0-4 --mem 128G --time 06:00:00 \
#          --dependency=afterok:<months> scripts/cosmo_build.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cosmo_build
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --time 03:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
MODE="${MODE:?MODE=month or merge}"
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
STAGE="/scratch/fquareng/cosmo_pm10_stage"
[[ -n "${SMOKE:-}" ]] && STAGE="${STAGE}_smoke"

# Bash arrays are whitespace-delimited: NO commas.
REGIONS=("scandinavia_south" "spain_central" "alps" "british_isles" "poland")

source "${CODE_ROOT}/scripts/python_env.sh"

case "${MODE}" in
  month)
    if [[ -n "${SMOKE:-}" ]]; then
        MONTH="201507"
    else
        (( TASK < 132 )) || { echo "ERROR: task ${TASK} >= 132"; exit 1; }
        MONTH="$(( 2011 + TASK / 12 ))$(printf %02d $(( TASK % 12 + 1 )))"
    fi
    echo "=== COSMO C1 month ${MONTH} -> ${STAGE} ==="
    run_python "${CODE_ROOT}/cosmo_build.py" month --month "${MONTH}" --stage_dir "${STAGE}" \
        --workers "${WORKERS:-4}"
    ;;
  merge)
    [[ -n "${SMOKE:-}" ]] && { echo "ERROR: no merge smoke (needs every month)"; exit 1; }
    (( TASK < ${#REGIONS[@]} )) || { echo "ERROR: task ${TASK} >= ${#REGIONS[@]}"; exit 1; }
    R="${REGIONS[$TASK]}"
    echo "=== COSMO C1 merge ${R} ==="
    run_python "${CODE_ROOT}/cosmo_build.py" merge --region "${R}" --stage_dir "${STAGE}"
    ;;
  *) echo "ERROR: MODE=${MODE}"; exit 1 ;;
esac
echo "=== done ==="
