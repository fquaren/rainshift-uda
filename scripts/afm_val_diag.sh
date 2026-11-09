#!/bin/bash -l
# ===========================================================================
#  AFM validation-CRPS diagnostic (PLAN.md 3.6, step 1). No training.
#
#  One Stage A checkpoint per task. Scores its best.pt on the 8-chunk
#  validation subset the trainer selected on and splits the epoch-to-epoch
#  jitter of that score into sampling noise, flow-net BatchNorm statistics
#  and the step count; also the 0.01 mm floored CRPS and the UNet acceptance
#  check. See afm_val_diag.py for what each section does.
#
#  Output: afm_lrtune_seed42_diag/<domain>/<config>/diag.json, a side tree
#  that no evaluation or materialise script reads. Refuses to run if the
#  config's directory exists.
#
#  Usage:  sbatch --array=0-2 scripts/afm_val_diag.sh
#  Smoke:  SMOKE=1 sbatch --array=0 --time 01:00:00 scripts/afm_val_diag.sh
#          (2 validation batches per score, separate _smoke tree)
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name afm_val_diag
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
#SBATCH --time 08:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DOMAIN="europe_west"
SWEEP="${OUTPUT_ROOT}/afm_lrtune_seed42/${DOMAIN}"
TREE="${OUTPUT_ROOT}/afm_lrtune_seed42_diag/${DOMAIN}"
UNET_CKPT="${OUTPUT_ROOT}/unet_seed42/${DOMAIN}__to__${DOMAIN}__none/best.pt"

EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
    TREE="${OUTPUT_ROOT}/afm_lrtune_seed42_diag_smoke/${DOMAIN}"
    EXTRA+=(--max_batches 2 --bn_set_batches 2 --bn_long_batches 4)
fi

# Bash arrays are whitespace-delimited: NO commas.
# Lowest score, largest jitter, smoothest curve (PLAN.md 3.6).
CONFIGS=(
    "lr1e-4_enc1.0"
    "lr3e-4_enc0.1"
    "lr3e-5_enc0.1"
)

TASK="${SLURM_ARRAY_TASK_ID:?run as an array job, e.g. sbatch --array=0-$((${#CONFIGS[@]} - 1))}"
(( TASK < ${#CONFIGS[@]} )) || { echo "ERROR: task ${TASK} >= ${#CONFIGS[@]} configs"; exit 1; }
CONFIG="${CONFIGS[$TASK]}"

RUN_DIR="${SWEEP}/${CONFIG}/afm_${DOMAIN}__to__${DOMAIN}__none"
OUT_DIR="${TREE}/${CONFIG}"
[[ -f "${RUN_DIR}/best.pt" ]] || { echo "ERROR: ${RUN_DIR}/best.pt missing"; exit 1; }
[[ -f "${UNET_CKPT}" ]] || { echo "ERROR: ${UNET_CKPT} missing"; exit 1; }
[[ -e "${OUT_DIR}" ]] && { echo "ERROR: ${OUT_DIR} exists; not overwriting."; exit 1; }
mkdir -p "$(dirname "${OUT_DIR}")"

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== AFM val diagnostic | task ${TASK}/${#CONFIGS[@]} | ${CONFIG} ==="
echo "Checkpoint: ${RUN_DIR}/best.pt"
echo "Output:     ${OUT_DIR}"

run_python "${CODE_ROOT}/afm_val_diag.py" \
    --run_dir   "${RUN_DIR}" \
    --data_root "${DATA_ROOT}" \
    --domain    "${DOMAIN}" \
    --unet_ckpt "${UNET_CKPT}" \
    --out_dir   "${OUT_DIR}" \
    "${EXTRA[@]}"

echo "=== done: ${OUT_DIR} ==="
