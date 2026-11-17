#!/bin/bash -l
# ===========================================================================
#  COSMO C3 trivial references (cosmo_references.py): zero, train
#  climatology, block-constant tp and bicubic tp, per target region, test
#  split, scored like evaluate.py. CPU, no training, independent of C3
#  training. Writes results/cosmo_references/references.csv; refuses if it
#  exists.
#
#  Usage:  sbatch scripts/cosmo_refs.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cosmo_refs
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
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_cosmo_pm10"
OUT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/results/cosmo_references/references.csv"
# Bash arrays are whitespace-delimited: NO commas.
REGIONS=(alps british_isles poland scandinavia_south spain_central)
[[ -e "${OUT}" ]] && { echo "ERROR: ${OUT} exists; not overwriting."; exit 1; }

source "${CODE_ROOT}/scripts/python_env.sh"
cd "${CODE_ROOT}"
run_python "${CODE_ROOT}/cosmo_references.py" \
    --data_root "${DATA_ROOT}" --regions "${REGIONS[@]}" --out "${OUT}"
