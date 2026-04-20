#!/bin/bash -l
# ===========================================================================
#  Covariate shift analysis — pairwise Wasserstein-1D matrices.
#
#  Runs both `raw` and `normalized` modes on the test split and dumps outputs
#  to covariate_shift_analysis/{mode}/ alongside per-channel heatmap PNGs.
#
#  Usage: sbatch scripts/run_compute_shift.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cov_shift
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 100G
#SBATCH --time 12:00:00

set -euo pipefail

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0"

CONTAINER="/users/fquareng/singularity/dl_gh200.sif"
CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/rainshift_npy"
OUT_ROOT="${CODE_ROOT}/covariate_shift_analysis"

SPLIT="test"
N_SAMPLES=10000
SEED=42

run_python() {
    singularity exec --nv "${CONTAINER}" python "$@"
}

for MODE in raw normalized; do
    echo "=== covariate shift | split=${SPLIT} | mode=${MODE} ==="
    run_python "${CODE_ROOT}/covariate_shift_analysis/compute_shift.py" \
        --data_root  "${DATA_ROOT}" \
        --split      "${SPLIT}" \
        --mode       "${MODE}" \
        --n_samples  "${N_SAMPLES}" \
        --seed       "${SEED}" \
        --output_dir "${OUT_ROOT}" \
        2>&1 | tee "${OUT_ROOT}/compute_shift_${SPLIT}_${MODE}.log"
done

#--domains "europe_west" "melanesia" \

echo "=== Done ==="