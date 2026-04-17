#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name plot
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
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/rainshift_npy"
CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
RESULTS_DIR="/scratch/fquareng/rainshift_uda/results"

run_python() {
    singularity exec --nv "${CONTAINER}" python "$@"
}

# Iterate through evaluated directories and plot
for dir in "${RESULTS_DIR}/unet"/*; do
    if [ -d "${dir}" ]; then
        run_python "${CODE_ROOT}/plotting.py" --result_dir "${dir}" --data_root "${DATA_ROOT}" --n_samples 5
    fi
done

for dir in "${RESULTS_DIR}/afm"/*; do
    if [ -d "${dir}" ]; then
        run_python "${CODE_ROOT}/plotting.py" --result_dir "${dir}" --data_root "${DATA_ROOT}" --n_samples 5
    fi
done