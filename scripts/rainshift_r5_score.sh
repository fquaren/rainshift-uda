#!/bin/bash -l
# ===========================================================================
#  R5 (PLAN.md 2026-11-17), scoring: the 5x5 source-only matrix of one seed.
#  One array task per seed (42/43/44).
#
#  1. Bring the three existing phase-1 diagonals (europe_west, horn-of-africa,
#     melanesia) from unet_seed<S>/ into the side tree unet_seed<S>_r5/ as
#     byte-identical copies, after checking each is source-only
#     ("joint_training": false: unet_seed<S>/, never the _oracle tree).
#     An existing copy that is identical is skipped; one that differs aborts.
#  2. materialise_baselines.sh on the side tree with all 5 regions:
#     5 diagonals -> 25 pair-named cells.
#  3. evaluate.py batch on the 25 cells (target and source test), data from
#     the overlay data_rainshift_r5/. Stats are the source's
#     normalization_stats.json in the overlay (symlinks for the old three,
#     computed by rainshift_r5_stats.sh for the new two).
#
#  The 9 old cells are re-scored on purpose: they must reproduce
#  results/seed<S>/unet/none/ (same weights, same test split), which checks
#  the overlay and the A100/dl-torch scoring before anyone reads the 16 new
#  cells. No --eval_train: the source train split does not depend on the
#  target, so it would score each source 5 times for nothing.
#
#  Writes results/seed<S>_r5/unet/none/; refuses if it exists. Never writes
#  to unet_seed<S>/ or raw_data/rainshift.
#
#  Usage:  sbatch --array=0-2 scripts/rainshift_r5_score.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name r5_score
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 100G
#SBATCH --time 06:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_rainshift_r5"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"

# Bash arrays are whitespace-delimited: NO commas.
SEEDS=(42 43 44)
OLD_REGIONS=(europe_west horn-of-africa melanesia)
NEW_REGIONS=(northamerica-east west-africa)
REGIONS=("${OLD_REGIONS[@]}" "${NEW_REGIONS[@]}")
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job: --array=0-2}"
(( TASK < ${#SEEDS[@]} )) || { echo "ERROR: task ${TASK} >= ${#SEEDS[@]}"; exit 1; }
SEED="${SEEDS[$TASK]}"

OLD_TREE="${OUTPUT_ROOT}/unet_seed${SEED}"
TREE="${OUTPUT_ROOT}/unet_seed${SEED}_r5"
RES="${OUTPUT_ROOT}/results/seed${SEED}_r5/unet/none"
[[ -e "${RES}" ]] && { echo "ERROR: ${RES} exists; not overwriting."; exit 1; }
[[ -d "${TREE}" ]] || { echo "ERROR: ${TREE} missing (R5 phase 1 not run)."; exit 1; }
for R in "${REGIONS[@]}"; do
    [[ -f "${DATA_ROOT}/${R}/normalization_stats.json" ]] || {
        echo "ERROR: ${DATA_ROOT}/${R}/normalization_stats.json missing"; exit 1; }
done
for R in "${NEW_REGIONS[@]}"; do
    [[ -f "${TREE}/${R}__to__${R}__none/best.pt" ]] || {
        echo "ERROR: ${TREE}/${R}__to__${R}__none/best.pt missing"; exit 1; }
done

echo "=== copy old diagonals ${OLD_TREE} -> ${TREE} ==="
for R in "${OLD_REGIONS[@]}"; do
    src="${OLD_TREE}/${R}__to__${R}__none"
    dst="${TREE}/${R}__to__${R}__none"
    [[ -f "${src}/best.pt" && -f "${src}/config.json" ]] || { echo "ERROR: ${src} incomplete"; exit 1; }
    grep -q '"joint_training": false' "${src}/config.json" || {
        echo "ERROR: ${src}/config.json is not a source-only run."; exit 1; }
    if [[ -e "${dst}" ]]; then
        cmp -s "${src}/best.pt" "${dst}/best.pt" || {
            echo "ERROR: ${dst} exists and differs from ${src}. Not overwriting."; exit 1; }
        echo "  ${R} (already present, identical, skipped)"
        continue
    fi
    mkdir "${dst}"
    # best.pt before config.json, as the trainers do (run_finished rule).
    cp -p "${src}/best.pt" "${dst}/best.pt"
    for f in src_stats.json tgt_stats.json config.json; do
        [[ -f "${src}/${f}" ]] && cp -p "${src}/${f}" "${dst}/${f}"
    done
    echo "  ${R}"
done

echo "=== materialise ${TREE} ==="
SEED="${SEED}" MODEL=unet OUTPUT_SUFFIX=_r5 DOMAINS_ENV="${REGIONS[*]}" OUTPUT_DIR="${OUTPUT_ROOT}" \
    bash "${CODE_ROOT}/scripts/materialise_baselines.sh"
n=$(ls -d "${TREE}"/*__to__*__none | wc -l)
(( n == ${#REGIONS[@]} * ${#REGIONS[@]} )) || { echo "ERROR: ${n} run dirs, expected 25"; exit 1; }

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== evaluate ${TREE} -> ${RES} ==="
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${TREE}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RES}" \
    --input_transform none \
    --transform_cache_dir "${OUTPUT_ROOT}/results/transforms" \
    --save_samples 10
echo "=== done: ${RES} ==="
