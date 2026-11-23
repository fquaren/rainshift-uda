#!/bin/bash -l
# ===========================================================================
#  West-africa blow-up in R5 (PLAN.md, R5 "Read 2026-11-23"): the in-domain
#  west-africa UNet has test RMSE 0.91 / 18.4 / 22.7 mm/h over seeds 42/43/44,
#  and west-africa -> melanesia 107 / 295 / 684 mm/h. compare_fields.py gives
#  per run the field mean, wet fraction, p99, max, the fraction of pixels above
#  the source training maximum and above 100 mm/h, and the clamped scores.
#
#  STRIDE defaults to 1 (every test step): the in-domain RMSE may come from a
#  few timesteps that a stride of 10 would skip.
#
#  One array task per seed (42/43/44), two passes each:
#    target west-africa (obs stats: west-africa)
#      in-domain W-africa, europe_west -> W-africa (also blows up), horn ->
#      W-africa (a well-behaved reference)
#    target melanesia (obs stats: west-africa)
#      W-africa -> melanesia, europe_west -> melanesia, in-domain melanesia
#  Data from the overlay data_rainshift_r5/ (symlinks to the shared stores
#  plus the R5 stats), checkpoints from unet_seed<S>_r5/. Read only.
#
#  Writes results/plots/compare_fields_r5${TAG}_seed<S>/<src>__to__<tgt>/ and
#  refuses to start if either output exists.
#
#  Usage:  sbatch --array=0-2 scripts/compare_fields_r5_wafr.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cmpf_r5_wafr
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 64G
#SBATCH --time 02:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_rainshift_r5"   # overlay, read only

# Bash arrays are whitespace-delimited: NO commas.
SEEDS=(42 43 44)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job: --array=0-2}"
(( TASK < ${#SEEDS[@]} )) || { echo "ERROR: task ${TASK} >= ${#SEEDS[@]}"; exit 1; }
SEED="${SEEDS[$TASK]}"
STRIDE="${STRIDE:-1}"

U="${OUTPUT_ROOT}/unet_seed${SEED}_r5"
PLOTS="${OUTPUT_ROOT}/results/plots/compare_fields_r5${TAG:-}_seed${SEED}"
OUT_W="${PLOTS}/west-africa__to__west-africa"
OUT_M="${PLOTS}/west-africa__to__melanesia"

for o in "${OUT_W}" "${OUT_M}"; do
    if [[ -e "${o}" ]]; then
        echo "ERROR: ${o} exists; not overwriting."
        exit 1
    fi
done
for r in west-africa europe_west horn-of-africa melanesia; do
    [[ -f "${DATA_ROOT}/${r}/normalization_stats.json" ]] || {
        echo "ERROR: ${DATA_ROOT}/${r}/normalization_stats.json missing"; exit 1; }
done

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== seed ${SEED}: target west-africa -> ${OUT_W} (stride ${STRIDE}) ==="
run_python "${CODE_ROOT}/compare_fields.py" \
    --data_root "${DATA_ROOT}" --output_dir "${OUT_W}" \
    --source west-africa --target west-africa --stride "${STRIDE}" \
    --run "in-domain=${U}/west-africa__to__west-africa__none" \
    --run "from europe_west=${U}/europe_west__to__west-africa__none" \
    --run "from horn=${U}/horn-of-africa__to__west-africa__none"

echo "=== seed ${SEED}: target melanesia -> ${OUT_M} (stride ${STRIDE}) ==="
run_python "${CODE_ROOT}/compare_fields.py" \
    --data_root "${DATA_ROOT}" --output_dir "${OUT_M}" \
    --source west-africa --target melanesia --stride "${STRIDE}" \
    --run "from W-africa=${U}/west-africa__to__melanesia__none" \
    --run "from europe_west=${U}/europe_west__to__melanesia__none" \
    --run "in-domain=${U}/melanesia__to__melanesia__none"
echo "=== done: ${PLOTS} ==="
