#!/bin/bash -l
# ===========================================================================
#  Batch evaluation of all trained models × all input transforms.
#
#  For each model (unet, afm), evaluates every checkpoint under:
#    none, qm_tp, qm_precip, qm_all, ot
#
#  Results land in {RESULTS_DIR}/{model}/{transform}/results.csv
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
#SBATCH --mem 90G
#SBATCH --time 24:00:00

set -euo pipefail

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0"

CONTAINER="/users/fquareng/singularity/dl_gh200.sif"
CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_DIR="/scratch/fquareng/rainshift_uda"
RESULTS_DIR="/scratch/fquareng/rainshift_uda/results"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/rainshift_npy"
CACHE_DIR="${RESULTS_DIR}/transforms"

TRANSFORMS=("none" "qm_tp" "qm_precip" "qm_all" "ot")

# AFM probabilistic settings
N_ENSEMBLE=16
SAMPLE_STEPS=20

run_python() {
    singularity exec --nv "${CONTAINER}" python "$@"
}

# -------------------------------------------------------------------
#  UNet: deterministic evaluation across all transforms
# -------------------------------------------------------------------
for tf in "${TRANSFORMS[@]}"; do
    echo "=== UNet | transform: ${tf} ==="
    run_python "${CODE_ROOT}/evaluate.py" batch \
        --exp_root "${OUTPUT_DIR}/unet" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${RESULTS_DIR}/unet/${tf}" \
        --input_transform "${tf}" \
        --transform_cache_dir "${CACHE_DIR}" \
        --save_samples 10
    echo ""
done

# -------------------------------------------------------------------
#  AFM deterministic: across all transforms
# -------------------------------------------------------------------
for tf in "${TRANSFORMS[@]}"; do
    echo "=== AFM (deterministic) | transform: ${tf} ==="
    run_python "${CODE_ROOT}/evaluate.py" batch \
        --exp_root "${OUTPUT_DIR}/afm" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${RESULTS_DIR}/afm/${tf}" \
        --input_transform "${tf}" \
        --transform_cache_dir "${CACHE_DIR}" \
        --save_samples 10
    echo ""
done

# -------------------------------------------------------------------
#  AFM probabilistic: only none + best transform (avoid 5×16-member cost)
#  Run the full set on "none" and "ot" to compare; add others if needed.
# -------------------------------------------------------------------
for tf in "none" "ot"; do
    echo "=== AFM (probabilistic, ${N_ENSEMBLE} members) | transform: ${tf} ==="
    run_python "${CODE_ROOT}/evaluate.py" batch \
        --exp_root "${OUTPUT_DIR}/afm" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${RESULTS_DIR}/afm_prob/${tf}" \
        --input_transform "${tf}" \
        --transform_cache_dir "${CACHE_DIR}" \
        --n_ensemble "${N_ENSEMBLE}" \
        --sample_steps "${SAMPLE_STEPS}" \
        --save_samples 5
    echo ""
done

echo "=== Evaluation complete ==="
echo "Results:"
echo "  UNet:          ${RESULTS_DIR}/unet/*/results.csv"
echo "  AFM (det):     ${RESULTS_DIR}/afm/*/results.csv"
echo "  AFM (prob):    ${RESULTS_DIR}/afm_prob/*/results.csv"