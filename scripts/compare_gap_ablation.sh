#!/bin/bash -l
# ===========================================================================
#  Read-out of the GAP ablation (compare_gap_ablation.py). CSV arithmetic,
#  CPU. Writes results/gap_ablation/summary/, refused if it exists.
#
#  Usage:  sbatch --dependency=afterok:<ablation array id> scripts/compare_gap_ablation.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name gap_ablation_cmp
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --time 00:15:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
source "${CODE_ROOT}/scripts/python_env.sh"

run_python "${CODE_ROOT}/compare_gap_ablation.py" \
    --output_dir "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/results/gap_ablation/summary"
