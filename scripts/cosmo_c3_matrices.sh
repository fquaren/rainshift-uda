#!/bin/bash -l
# ===========================================================================
#  COSMO C3 figures: transfer matrices and their candidate explanations
#  (cosmo_c3_matrices.py). Reads result CSVs and the C2 JSONs only; no data
#  store, no GPU. Writes figures/cosmo_c3/ in the repo (PNG, PDF,
#  summary_predictors.csv, matrices.npz); refuses if it exists.
#
#  Usage:  sbatch scripts/cosmo_c3_matrices.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cosmo_c3m
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --time 00:30:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
RES="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/results"
C2="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_cosmo/c2"
OUT="${CODE_ROOT}/figures/cosmo_c3"

[[ -e "${OUT}" ]] && { echo "ERROR: ${OUT} exists; not overwriting."; exit 1; }
for s in 42 43 44; do
    [[ -f "${RES}/cosmo_seed${s}/unet/none/results.csv" ]] || { echo "ERROR: no results for seed ${s}"; exit 1; }
done
[[ -f "${RES}/cosmo_references/references.csv" ]] || { echo "ERROR: no references.csv"; exit 1; }

source "${CODE_ROOT}/scripts/python_env.sh"
run_python "${CODE_ROOT}/cosmo_c3_matrices.py" --results_root "${RES}" --c2_dir "${C2}" --out_dir "${OUT}"
