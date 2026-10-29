#!/bin/bash -l
# ===========================================================================
#  Input-set ablation (PLAN.md P19), europe_west -> melanesia, one seed.
#
#  Retrains the three models behind lambda-hat with some inputs zeroed in
#  every domain (train_unet.py --mask_inputs; the architecture and the seeded
#  initialisation are those of the full model). Arms, chosen from the
#  OODiag whole-field Shapley attribution of the europe -> melanesia shift gap
#  (Beucler et al., OODiag draft 2026-10-28, App. F.3, Fig. 8):
#
#    mask_geo    lsm z tisr sp   the location identifiers: 5.8 % of the gap
#                                together (sp 2.7, lsm 1.8, z 1.3, tisr 0.1)
#    mask_shift  cape tcw        the two leading shift associations: 72.4 %
#                                (cape 39.4, tcw 33.0)
#
#  Runs per arm, one array task each, hyper-parameters as in the seed-42
#  baselines (lr 5e-4, wd 1e-3, batch 128, 25 epochs; patience 5 for the
#  single-domain runs, 8 for the oracle):
#    src        europe_west only    unet_seed<N>_<arm>/europe_west__to__melanesia__none
#    in_domain  melanesia only      unet_seed<N>_<arm>/melanesia__to__melanesia__none
#    oracle     joint               unet_seed<N>_<arm>_oracle/europe_west__to__melanesia__none
#  The source-only run gets melanesia as --target_path so its directory has
#  the off-diagonal name field_mean_check.py looks for; the target loader is
#  never drawn from with --uda_method none, so the weights are those of a
#  europe_west-only run.
#
#  Task id: 0-2 mask_geo (src, in_domain, oracle), 3-5 mask_shift.
#  Scoring: scripts/field_mean_check.sh with TREE_SUFFIX=_<arm>, TARGET=melanesia.
#
#  Usage:  SMOKE=1 sbatch --array=2 --time 01:00:00 scripts/run_input_ablation.sh
#          sbatch --array=0-5 --dependency=afterok:<smoke> scripts/run_input_ablation.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name inputabl
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

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
SEED="${SEED:-42}"
SRC="europe_west"
TGT="melanesia"

# Bash arrays are whitespace-delimited: NO commas.
ARMS=("mask_geo" "mask_shift")
declare -A MASK=(
    [mask_geo]="lsm z tisr sp"
    [mask_shift]="cape tcw"
)
RUNS=("src" "in_domain" "oracle")

SMOKE="${SMOKE:-0}"
if (( SMOKE )); then
    echo "=== test_input_mask.py ==="
    run_python "${CODE_ROOT}/test_input_mask.py"
    TASK="${SLURM_ARRAY_TASK_ID:-2}"   # 2: oracle of mask_geo, joint loader, masked statics
    ROOT="/scratch/fquareng/input_ablation_smoke/${SLURM_JOB_ID}"
    EXTRA=(--subset_chunks 2 --epochs 1)
else
    TASK="${SLURM_ARRAY_TASK_ID:?submit with --array=0-5}"
    ROOT="${OUTPUT_ROOT}"
    EXTRA=(--epochs 25)
fi

ARM="${ARMS[$(( TASK / 3 ))]}"
RUN="${RUNS[$(( TASK % 3 ))]}"
read -ra MASK_VARS <<< "${MASK[${ARM}]}"
OUT_DIR="${ROOT}/unet_seed${SEED}_${ARM}"

case "${RUN}" in
    src)       S="${SRC}"; T="${TGT}"; PATIENCE=5; JOINT=() ;;
    in_domain) S="${TGT}"; T="${TGT}"; PATIENCE=5; JOINT=() ;;
    oracle)    S="${SRC}"; T="${TGT}"; PATIENCE=8; JOINT=(--joint_training); OUT_DIR="${OUT_DIR}_oracle" ;;
esac
RUN_DIR="${OUT_DIR}/${S}__to__${T}__none"

echo "SEED ${SEED}  task ${TASK}  arm ${ARM} (masked: ${MASK_VARS[*]})  run ${RUN}: ${S} -> ${T}"
echo "Output: ${RUN_DIR}"
# Never overwrite a checkpoint or a partial run (CLAUDE.md rule 4).
[[ -e "${RUN_DIR}" ]] && { echo "ERROR: ${RUN_DIR} exists. Not overwriting."; exit 1; }
mkdir -p "${OUT_DIR}"

run_python "${CODE_ROOT}/train_unet.py" \
    --source_path "${DATA_ROOT}/${S}" \
    --target_path "${DATA_ROOT}/${T}" \
    --output_dir  "${OUT_DIR}" \
    --data_format npy \
    --uda_method  none \
    "${JOINT[@]}" \
    --batch_size  128 \
    --patience    "${PATIENCE}" \
    --num_workers 12 \
    --seed        "${SEED}" \
    --mask_inputs "${MASK_VARS[@]}" \
    "${EXTRA[@]}"

echo "=== input ablation task ${TASK} complete ==="
