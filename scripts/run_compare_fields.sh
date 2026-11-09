#!/bin/bash -l
# ===========================================================================
#  Side-by-side precipitation fields (compare_fields.py), seed 42, for the
#  two pairs with the latest results: europe_west -> melanesia (FDA fix,
#  DANN grad ratio 0.5, P19 mask_shift) and melanesia -> europe_west (FDA
#  fix, DANN grad ratio 0.5). Every run is read, never written.
#
#  Writes results/plots/compare_fields_seed${SEED}/<src>__to__<tgt>/ and
#  refuses to start if that directory exists.
#
#  Usage:  SEED=42 sbatch scripts/run_compare_fields.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cmpfields
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

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
SEED="${SEED:?SEED must be set, e.g. SEED=42}"
STRIDE="${STRIDE:-10}"
OUT="${OUTPUT_ROOT}/results/plots/compare_fields_seed${SEED}"
U="${OUTPUT_ROOT}/unet_seed${SEED}"

if [[ -e "${OUT}" ]]; then
    echo "ERROR: ${OUT} exists; not overwriting."
    exit 1
fi

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

EM="europe_west__to__melanesia"
run_python "${CODE_ROOT}/compare_fields.py" \
    --data_root "${DATA_ROOT}" --output_dir "${OUT}/${EM}" \
    --source europe_west --target melanesia --stride "${STRIDE}" \
    --run "source-only=${U}/${EM}__none" \
    --run "FDA fix=${U}_fdafix/${EM}__fda" \
    --run "DANN gr0.5=${U}_danngr05/${EM}__dann" \
    --run "src-only mask_shift=${U}_mask_shift/${EM}__none" \
    --run "oracle=${U}_oracle/${EM}__none" \
    --run "in-domain=${U}/melanesia__to__melanesia__none"

ME="melanesia__to__europe_west"
run_python "${CODE_ROOT}/compare_fields.py" \
    --data_root "${DATA_ROOT}" --output_dir "${OUT}/${ME}" \
    --source melanesia --target europe_west --stride "${STRIDE}" \
    --run "source-only=${U}/${ME}__none" \
    --run "FDA fix=${U}_fdafix/${ME}__fda" \
    --run "DANN gr0.5=${U}_danngr05/${ME}__dann" \
    --run "oracle=${U}_oracle/${ME}__none" \
    --run "in-domain=${U}/europe_west__to__europe_west__none"

echo "=== plots -> ${OUT} ==="
