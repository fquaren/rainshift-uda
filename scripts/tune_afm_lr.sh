#!/bin/bash -l
# ===========================================================================
#  AFM learning-rate sweep, phase-1 (source-only) runs, one config per task.
#
#  Each config is domain|lr|encoder_lr_ratio. The flow network trains at lr,
#  the encoder at lr * encoder_lr_ratio. Every run uses the production
#  schedule (cosine over EPOCHS=25) truncated after STOP_AFTER=12 epochs, no
#  early stopping, and selects its checkpoint on the validation CRPS
#  (standardised space, 4 members, 10 midpoint steps, 8 chunks spread over
#  the validation split). Selection uses source validation only, so the LR
#  chosen for a source domain has seen no other domain's labels.
#
#  Stage A (default): the grid on europe_west.
#  Stage B: the winning config and a neighbour on the other two domains,
#  passed explicitly, e.g.
#    CONFIGS_ENV="horn-of-africa|3e-4|0.1 melanesia|3e-4|0.1" sbatch --array=0-1 ...
#
#  Output: afm_lrtune_seed42/<domain>/lr<lr>_enc<ratio>/, a side tree that no
#  evaluation or materialise script reads. Refuses to run if the config's
#  directory exists. Curves: <run>/afm_*__none/history.csv; compare them
#  with scripts/summarise_lrtune.sh.
#
#  Stage A' (EMA=1): ratio 1.0 at 3e-5, 1e-4 and 3e-4 with an EMA of the
#  weights, in afm_lrtune_ema_seed42/.
#
#  Usage:  sbatch --array=0-6 scripts/tune_afm_lr.sh
#          EMA=1 sbatch --array=0-2 scripts/tune_afm_lr.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name afm_lrtune
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

# One A100, half a node, <= 12 h: gpu-normal QOS.
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --exclude=dnagpu008
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --mem 200G
#SBATCH --time 12:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
SEED=42
TREE="${OUTPUT_ROOT}/afm_lrtune_seed${SEED}"

# Stage A' (EMA=1, PLAN.md 3.6 step 3): EMA of weights and buffers at 0.999,
# in its own tree, so Stage A stays as it was.
EMA_DECAY=0
if [[ -n "${EMA:-}" ]]; then
    EMA_DECAY=0.999
    TREE="${OUTPUT_ROOT}/afm_lrtune_ema_seed${SEED}"
fi

# SMOKE=1: 2 epochs on 8 chunks per split in a separate tree, to exercise the
# CRPS selection path before the real array, e.g.
#   SMOKE=1 sbatch --array=0 --time 01:00:00 scripts/tune_afm_lr.sh
EXTRA=()
STOP_AFTER=12
if [[ -n "${SMOKE:-}" ]]; then
    TREE="${TREE}_smoke"
    EXTRA+=(--subset_chunks 8)
    STOP_AFTER=2
fi

# Bash arrays are whitespace-delimited: NO commas.
# Stage A: half-decade grid at the current ratio 0.1 (encoder 1e-6 .. 1e-4),
# plus ratio 1.0 at the two middle LRs, which brings the encoder up to the
# UNet arm's 5e-4 range. 1e-5 is the untuned default, kept as the reference.
CONFIGS=(
    "europe_west|1e-5|0.1"
    "europe_west|3e-5|0.1"
    "europe_west|1e-4|0.1"
    "europe_west|3e-4|0.1"
    "europe_west|1e-3|0.1"
    "europe_west|1e-4|1.0"
    "europe_west|3e-4|1.0"
)
# Stage A': ratio 1.0 only, 1e-3 (diverged) and 1e-5 (under-trains mu) dropped.
if [[ -n "${EMA:-}" ]]; then
    CONFIGS=(
        "europe_west|3e-5|1.0"
        "europe_west|1e-4|1.0"
        "europe_west|3e-4|1.0"
    )
fi
if [[ -n "${CONFIGS_ENV:-}" ]]; then
    read -ra CONFIGS <<< "${CONFIGS_ENV}"
fi

TASK="${SLURM_ARRAY_TASK_ID:?run as an array job, e.g. sbatch --array=0-$((${#CONFIGS[@]} - 1))}"
(( TASK < ${#CONFIGS[@]} )) || { echo "ERROR: task ${TASK} >= ${#CONFIGS[@]} configs"; exit 1; }
IFS='|' read -r DOMAIN LR RATIO <<< "${CONFIGS[$TASK]}"

RUN_DIR="${TREE}/${DOMAIN}/lr${LR}_enc${RATIO}"
[[ -e "${RUN_DIR}" ]] && { echo "ERROR: ${RUN_DIR} exists; not overwriting."; exit 1; }
mkdir -p "${RUN_DIR}"

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== AFM LR sweep | task ${TASK}/${#CONFIGS[@]} | ${DOMAIN} | lr ${LR} | encoder ratio ${RATIO} ==="
echo "Output: ${RUN_DIR}"

run_python "${CODE_ROOT}/train_afm.py" \
    --source_path "${DATA_ROOT}/${DOMAIN}" \
    --target_path "${DATA_ROOT}/${DOMAIN}" \
    --output_dir  "${RUN_DIR}" \
    --data_format npy \
    --uda_method  none \
    --base_features 32 \
    --batch_size  64 \
    --lr          "${LR}" \
    --encoder_lr_ratio "${RATIO}" \
    --epochs      25 \
    --stop_after  "${STOP_AFTER}" \
    --patience    -1 \
    --select_on   crps \
    --val_crps_members 4 \
    --val_crps_steps   10 \
    --val_crps_chunks  8 \
    --ema_decay   "${EMA_DECAY}" \
    --num_workers 12 \
    --seed        "${SEED}" \
    "${EXTRA[@]}" \
    2>&1 | tee "${RUN_DIR}/train.log"

echo "=== done: ${RUN_DIR} ==="
