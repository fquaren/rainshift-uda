#!/bin/bash -l
# ===========================================================================
#  Example-field PDFs (plotting.py) for every finished run of one tree.
#  Zarr data, SOURCE stats for both splits, best.pt for the source panels
#  and evaluate.target_checkpoint() for the target panels (best_adabn.pt for
#  AdaBN). Unfinished runs (no config.json newer than every checkpoint) are
#  skipped with a warning.
#
#  Writes results/plots/<TREE>/<run>/ and refuses to start if
#  results/plots/<TREE>/ exists.
#
#  Usage:  TREE=unet_seed42 sbatch scripts/run_plotting.sh
#          TREE=unet_seed42_oracle N_SAMPLES=10 sbatch scripts/run_plotting.sh
#  (TREE is a directory under results_rainshift_uda/.)
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name plot
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

# Default: one A100 on the gpu partition. To fall back to the GH200 node:
# sbatch --partition gpu-gh --mem 0 ...
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 04:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"   # read-only
TREE="${TREE:?TREE must be set, e.g. TREE=unet_seed42}"
N_SAMPLES="${N_SAMPLES:-5}"
EXP_ROOT="${OUTPUT_ROOT}/${TREE}"
OUT="${OUTPUT_ROOT}/results/plots/${TREE}"

[[ -d "${EXP_ROOT}" ]] || { echo "ERROR: ${EXP_ROOT} does not exist."; exit 1; }
if [[ -e "${OUT}" ]]; then
    echo "ERROR: ${OUT} exists; not overwriting."
    exit 1
fi

# Same rule as evaluate.py run_finished.
run_finished() {
    local d="$1" p
    [[ -e "${d}/config.json" ]] || return 1
    for p in "${d}"/*.pt; do
        [[ "${p}" -nt "${d}/config.json" ]] && return 1
    done
    return 0
}

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

for ckpt in "${EXP_ROOT}"/*/best.pt; do
    [[ -e "${ckpt}" ]] || continue
    run="$(dirname "${ckpt}")"
    if ! run_finished "${run}"; then
        echo "WARNING: $(basename "${run}") is unfinished; not plotted."
        continue
    fi
    run_python "${CODE_ROOT}/plotting.py" \
        --run_dir "${run}" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${OUT}/$(basename "${run}")" \
        --n_samples "${N_SAMPLES}"
done
echo "=== plots -> ${OUT} ==="
