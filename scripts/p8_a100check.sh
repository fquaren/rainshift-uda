#!/bin/bash -l
# ===========================================================================
#  P8 (PLAN.md paper audit): test metrics of the A100/dl-torch re-run of
#  seed-42 europe_west -> horn-of-africa joint_ot (unet_seed42_a100check/),
#  to set against the GH200 seed spread of the same cell. Scored both ways
#  the GH200 runs were:
#    as trained     -> results/seed42_a100check/unet/none/
#                      vs results/seed<S>_v3/unet/none/      (S = 42/43/44)
#    source BN      -> unet_seed42_a100check_srcbn/ (unet_srcbn_export.py,
#                      same settings as unet_srcbn.sh), scored into
#                      results/seed42_a100check_srcbn/unet/none/
#                      vs results/seed<S>_srcbn/unet/none/
#  One run, so one task. Refuses if any output exists.
#
#  Usage:  sbatch scripts/p8_a100check.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name p8_a100
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --mem 200G
#SBATCH --time 02:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"

TREE="${OUTPUT_ROOT}/unet_seed42_a100check"
RUN="${TREE}/europe_west__to__horn-of-africa__joint_ot"
SIDE="${OUTPUT_ROOT}/unet_seed42_a100check_srcbn"
RES="${OUTPUT_ROOT}/results/seed42_a100check/unet/none"
RES_SRCBN="${OUTPUT_ROOT}/results/seed42_a100check_srcbn/unet/none"
[[ -f "${RUN}/best.pt" && -f "${RUN}/config.json" ]] || { echo "ERROR: ${RUN} incomplete"; exit 1; }
for p in "${SIDE}" "${RES}" "${RES_SRCBN}"; do
    [[ -e "${p}" ]] && { echo "ERROR: ${p} exists; not overwriting."; exit 1; }
done

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== as trained: ${TREE} -> ${RES} ==="
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${TREE}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RES}" \
    --input_transform none \
    --transform_cache_dir "${OUTPUT_ROOT}/results/transforms" \
    --save_samples 0

echo "=== source BN: ${RUN} -> ${SIDE} -> ${RES_SRCBN} ==="
run_python "${CODE_ROOT}/unet_srcbn_export.py" --runs "${RUN}" --out_root "${SIDE}"
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${SIDE}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RES_SRCBN}" \
    --input_transform none \
    --transform_cache_dir "${OUTPUT_ROOT}/results/transforms" \
    --save_samples 0
echo "=== done ==="
