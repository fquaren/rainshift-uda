#!/bin/bash -l
# ===========================================================================
#  Batch evaluation of all trained models
#  Scans experiment directories for best.pt and evaluates on target test sets.
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name eval
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 450G
#SBATCH --time 12:00:00

set -euo pipefail

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0"

CONTAINER="/users/fquareng/singularity/dl_gh200.sif"
CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_DIR="/scratch/fquareng/rainshift_uda"
RESULTS_DIR="/scratch/fquareng/rainshift_uda/results"

DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/rainshift_npy"

run_python() {
    singularity exec --nv "${CONTAINER}" python "$@"
}

echo "=== Evaluating UNet ==="
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${OUTPUT_DIR}/unet" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RESULTS_DIR}/unet" \
    --save_samples 10

echo "=== Evaluating AFM (deterministic) ==="
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${OUTPUT_DIR}/afm" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RESULTS_DIR}/afm" \
    --save_samples 10

echo "=== Evaluating AFM (probabilistic, 16 members) ==="
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${OUTPUT_DIR}/afm" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RESULTS_DIR}/afm_prob" \
    --n_ensemble 16 --sample_steps 20 \
    --save_samples 5

echo "=== Evaluation complete ==="
echo "Results: ${RESULTS_DIR}/*/results.csv"
