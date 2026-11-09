#!/bin/bash -l
# ===========================================================================
#  Run every test_*.py in the repo (audit fixes, 2026-10-29). One GPU job,
#  because test_evaluate_metrics.py and test_quantile_map.py use CUDA when it
#  is there. Each test runs even if an earlier one failed; the job exits
#  non-zero if any failed, so dependent jobs can chain with afterok.
#
#  Usage:  sbatch scripts/run_tests.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name tests
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00

set -uo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda
cd "${CODE_ROOT}"

failed=()
for t in test_*.py; do
    echo "=================== ${t} ==================="
    if run_python "${CODE_ROOT}/${t}"; then
        echo "--- PASS ${t}"
    else
        echo "--- FAIL ${t}"
        failed+=("${t}")
    fi
done

echo "=================== summary ==================="
if (( ${#failed[@]} )); then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
echo "all tests passed"
