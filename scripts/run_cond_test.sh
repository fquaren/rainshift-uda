#!/bin/bash -l
# ===========================================================================
#  Model-based conditional-shift test (PLAN.md P2, conditional_shift.py).
#  Runs the seeded source-only models (unet_seed42/43/44) and the 18 split-half
#  models (unet_seed{42,43,44}_half0, unet_seed{45,46,47}_half1, from
#  scripts/run_half_split.sh) of all three domains on every domain's test
#  split, then the cross-fitted support classifier (overlap at P(A|x) >= 0.1,
#  0.3, 0.5), the cross-year split-half decomposition per Latin quadruple and
#  pooled, and the positive control (europe_west, wet x 1.5). Inference only.
#
#  Output: results_rainshift_uda/unet/conditional_shift_model_p2/
#          (summary.json, per_sample_<B>.npz written as each domain finishes,
#          propensities; refuses to overwrite)
#
#  Usage:  sbatch --dependency=afterok:<half-split array> scripts/run_cond_test.sh
#          MASK_GEO=1 ...      also the masked-statics half models, into ..._maskgeo
#          SKIP_HALVES=1 ...   seed null only, into ..._seedonly
#          FROM_NPZ=<dir> ...  analysis only from <dir>/per_sample_*.npz, into ..._from_npz
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name condmodel
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 100G
#SBATCH --time 08:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"

OUT="${OUTPUT_ROOT}/unet/conditional_shift_model_p2"; EXTRA=()
if [[ "${SKIP_HALVES:-0}" == "1" ]]; then
    OUT="${OUT}_seedonly"; EXTRA+=(--skip_halves)
elif [[ "${MASK_GEO:-0}" == "1" ]]; then
    OUT="${OUT}_maskgeo"; EXTRA+=(--masked_suffix _mask_geo)
fi
if [[ -n "${FROM_NPZ:-}" ]]; then
    OUT="${OUT}_from_npz"; EXTRA+=(--from_npz "${FROM_NPZ}")
fi
[[ -e "${OUT}" ]] && { echo "ERROR: ${OUT} exists. Not overwriting."; exit 1; }

cd "${CODE_ROOT}"
run_python "${CODE_ROOT}/test_conditional_shift.py"

run_python "${CODE_ROOT}/conditional_shift.py" \
    --data_root      "${DATA_ROOT}" \
    --results_root   "${OUTPUT_ROOT}" \
    --out_dir        "${OUT}" \
    --seeds 42 43 44 \
    --half_base_seed 42 \
    --half_reps      3 \
    --n_chunks       88 \
    --boot_block     4 \
    --overlaps 0.1 0.3 0.5 \
    --control_domain europe_west \
    --control_factor 1.5 \
    "${EXTRA[@]}"

echo "=== model-based conditional shift test (P2) complete ==="
