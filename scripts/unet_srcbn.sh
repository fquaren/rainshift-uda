#!/bin/bash -l
# ===========================================================================
#  UNet BatchNorm-contamination check (PLAN.md "Read 2026-11-09").
#  Per task, one (seed, tree): re-estimate BN on source for every
#  target-forward phase-2 run (unet_srcbn_export.py) into
#  unet_seed<N><TREE>_srcbn/, then score that tree with evaluate.py (source
#  and target test, no training split) into results/seed<N><TREE>_srcbn/unet/none/.
#  Compare with results/seed<N>_v3/unet/none (or seed<N>_danngr05/).
#
#  Tasks (6):  0-2 seeds 42/43/44 main tree (dann, mmd, joint_ot, spectral;
#              24 runs each), 3-5 the same seeds' _danngr05 tree (6 DANN runs).
#  Usage:  sbatch --array=0-5 scripts/unet_srcbn.sh
#  Smoke:  SMOKE=1 sbatch --array=0 --time 00:30:00 scripts/unet_srcbn.sh
#          (first run only, 4 BN batches, *_srcbn_smoke trees, no scoring)
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name unet_srcbn
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --mem 200G
#SBATCH --time 12:00:00

set -euo pipefail
export LC_ALL=C

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"

# Bash arrays are whitespace-delimited: NO commas. "seed:tree_suffix".
TASKS=("42:" "43:" "44:" "42:_danngr05" "43:_danngr05" "44:_danngr05")
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job, --array=0-$((${#TASKS[@]} - 1))}"
(( TASK < ${#TASKS[@]} )) || { echo "ERROR: task ${TASK} >= ${#TASKS[@]}"; exit 1; }
SEED="${TASKS[$TASK]%%:*}"
TREE="${TASKS[$TASK]#*:}"

SRC_TREE="${OUTPUT_ROOT}/unet_seed${SEED}${TREE}"
SIDE="${OUTPUT_ROOT}/unet_seed${SEED}${TREE}_srcbn"
RES="${OUTPUT_ROOT}/results/seed${SEED}${TREE}_srcbn/unet/none"
EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
    SIDE="${SIDE}_smoke"
    EXTRA+=(--n_batches 4)
fi

RUNS=()
for d in "${SRC_TREE}"/*__to__*__{dann,mmd,joint_ot,spectral}; do
    [[ -d "${d}" ]] || continue
    [[ -f "${d}/best.pt" && -f "${d}/config.json" ]] || { echo "ERROR: ${d} incomplete"; exit 1; }
    RUNS+=("${d}")
done
(( ${#RUNS[@]} > 0 )) || { echo "ERROR: no target-forward runs in ${SRC_TREE}"; exit 1; }
[[ -n "${SMOKE:-}" ]] && RUNS=("${RUNS[0]}")
[[ -e "${SIDE}" ]] && { echo "ERROR: ${SIDE} exists; not overwriting."; exit 1; }
[[ -z "${SMOKE:-}" && -e "${RES}" ]] && { echo "ERROR: ${RES} exists; not overwriting."; exit 1; }

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== UNet source-BN | seed ${SEED} tree '${TREE}' | ${#RUNS[@]} runs -> ${SIDE} ==="
run_python "${CODE_ROOT}/unet_srcbn_export.py" --runs "${RUNS[@]}" --out_root "${SIDE}" "${EXTRA[@]}"

if [[ -n "${SMOKE:-}" ]]; then
    echo "=== smoke: export only, done ==="; exit 0
fi

echo "=== scoring ${SIDE} -> ${RES} ==="
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${SIDE}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RES}" \
    --input_transform none \
    --transform_cache_dir "${OUTPUT_ROOT}/results/transforms" \
    --save_samples 0
echo "=== done: ${RES} ==="
