#!/bin/bash -l
# ===========================================================================
#  Covariate shift analysis — pairwise Wasserstein-1D matrices.
#
#  Raw mode only, on the test split; outputs go to ${OUT_ROOT}/raw/ with
#  per-channel heatmap PNGs. The `normalized` mode (each domain z-scored by
#  its own stats) is refused by compute_shift.py: it is the CLAUDE.md
#  normalisation trap (audit 2026-10-29, A5). For W1 in source-normalised
#  space divide the raw W1(S, T) by the source std of the channel.
#
#  Inputs are the old .npy conversion (DATA_ROOT below, data/convert_zarr_
#  to_npy.py), not the Zarr stores training reads. They may have been
#  deleted; the job checks before starting. They also predate the Zarr
#  pipeline's fixes, so compare with care.
#
#  Refuses to run if the W1 outputs for this split already exist in
#  ${OUT_ROOT}/raw/ (CLAUDE.md rule 4): choose a fresh OUT_ROOT instead.
#
#  Usage: OUT_ROOT=/path/in/your/tree sbatch scripts/run_analysis.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cov_shift
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

# compute_shift.py sorts on the GPU. One A100, x86 env.
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 100G
#SBATCH --time 12:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
# dl-torch mamba env on x86 nodes, the GH200 container on gpu-gh.
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/rainshift_npy"
OUT_ROOT="${OUT_ROOT:-${CODE_ROOT}/covariate_shift_analysis}"

SPLIT="test"
MODE="raw"
N_SAMPLES=10000
SEED=42
DOMAINS=("europe_west" "horn-of-africa" "melanesia")

for d in "${DOMAINS[@]}"; do
    for f in "${SPLIT}_x.npy" "${SPLIT}_y.npy"; do
        [[ -f "${DATA_ROOT}/${d}/${f}" ]] || { echo "ERROR: ${DATA_ROOT}/${d}/${f} is gone."; exit 1; }
    done
done

OUT="${OUT_ROOT}/${MODE}"
shopt -s nullglob
existing=("${OUT}/Wasserstein_1D_${SPLIT}"*)
shopt -u nullglob
if (( ${#existing[@]} )); then
    echo "ERROR: ${OUT} already holds ${#existing[@]} Wasserstein_1D_${SPLIT}* file(s); not overwriting."
    echo "       Set OUT_ROOT to a fresh directory, or move them to a _trash/ folder."
    exit 1
fi
mkdir -p "${OUT}"

echo "=== covariate shift | split=${SPLIT} | mode=${MODE} | out=${OUT} ==="
run_python "${CODE_ROOT}/covariate_shift_analysis/compute_shift.py" \
    --data_root  "${DATA_ROOT}" \
    --split      "${SPLIT}" \
    --mode       "${MODE}" \
    --n_samples  "${N_SAMPLES}" \
    --seed       "${SEED}" \
    --output_dir "${OUT_ROOT}" \
    --domains    "${DOMAINS[@]}" \
    --emit_aggregates \
    2>&1 | tee "${OUT}/compute_shift_${SPLIT}_${MODE}.log"

echo "=== Done ==="
