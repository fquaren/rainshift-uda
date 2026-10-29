#!/bin/bash -l
# ===========================================================================
#  Seed aggregate (mean +- sd over seeds) of the per-seed evaluation results.
#  Reads results/seed<N>/<SUBDIR>/{none,oracle}/results.csv and
#  results/seed<N>/ngg_v2/<SUBDIR>_none/summary.csv; writes
#  results/seeds_<N1>_<N2>_..._<SUBDIR>_<METRIC>_<UNITS>/. aggregate_seeds.py refuses
#  to run if that directory exists.
#
#  Usage:
#    sbatch scripts/aggregate_seeds.sh                       # UNet, mse_std
#    SUBDIR=afm_prob METRIC=crps_std sbatch scripts/aggregate_seeds.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name agg_seeds
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

# CSV arithmetic only: a CPU node with the x86 env.
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --time 00:15:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
source "${CODE_ROOT}/scripts/python_env.sh"
RESULTS_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/results"

SEEDS="${SEEDS:-42 43 44}"
SUBDIR="${SUBDIR:-unet}"
METRIC="${METRIC:-mse_std}"
# log: every score in log-precipitation units, comparable across sources
# (see aggregate_seeds.py, Units). std: raw per-source standardised scores.
UNITS="${UNITS:-log}"
OUT="${RESULTS_ROOT}/seeds_${SEEDS// /_}_${SUBDIR}_${METRIC}_${UNITS}"

run_python "${CODE_ROOT}/aggregate_seeds.py" \
    --results_root "${RESULTS_ROOT}" \
    --seeds ${SEEDS} \
    --subdir "${SUBDIR}" \
    --metric "${METRIC}" \
    --units "${UNITS}" \
    --output_dir "${OUT}"
