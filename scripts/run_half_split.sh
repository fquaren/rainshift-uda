#!/bin/bash -l
# ===========================================================================
#  Split-half source-only models, the null of the model-based conditional-
#  shift test (PLAN.md P2, conditional_shift.py). One array task per
#  (domain, half, replicate): train_unet.py --train_half trains AND selects on
#  alternate 365-day blocks of the training years (chunks touching a block
#  boundary dropped), so the two halves of a domain share no data. Half 0 uses
#  seeds 42/43/44, half 1 seeds 45/46/47; --seed sets init and, for half runs,
#  the shuffle order, so no two models compared by the test share either.
#  Hyper-parameters as in phase 1 (lr 5e-4, wd 1e-3, batch 128, 25 epochs,
#  patience 5; identical for all domains and seeds).
#
#  Tasks 0-17: TASK = 6 * domain + 3 * half + rep, SEED = 42 + 3 * half + rep
#    0-5   europe_west     h0 r0-2 (42 43 44), h1 r0-2 (45 46 47)
#    6-11  horn-of-africa  same
#    12-17 melanesia       same
#  Output:  unet_seed<SEED>_half<h>${SUFFIX}/{d}__to__{d}__none (refuses to overwrite)
#  MASK_GEO=1: masked-statics arm, --mask_inputs lsm z sp tisr, SUFFIX=_mask_geo.
#
#  Usage:  SMOKE=1 sbatch --array=0 --time 01:00:00 scripts/run_half_split.sh
#          sbatch --array=0-17%6 --dependency=afterok:<smoke> scripts/run_half_split.sh
#          MASK_GEO=1 sbatch --array=0-17%6 scripts/run_half_split.sh   # optional arm
#  SMOKE=1: test_conditional_shift.py, then 1 epoch on 2 chunks into
#  /scratch/fquareng/half_split_smoke/<jobid>/ (task 0 only: half 1 has no
#  chunk among the first two).
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name halfsplit
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
BASE_SEED=42
N_REP=3

# Bash arrays are whitespace-delimited: NO commas.
DOMAINS=("europe_west" "horn-of-africa" "melanesia")
N_TASKS=$(( ${#DOMAINS[@]} * 2 * N_REP ))
TASK="${SLURM_ARRAY_TASK_ID:?submit with --array}"
(( TASK < N_TASKS )) || { echo "ERROR: task ${TASK} out of range 0-$(( N_TASKS - 1 ))"; exit 1; }

if [[ "${MASK_GEO:-0}" == "1" ]]; then
    SUFFIX="_mask_geo"; MASK_ARGS=(--mask_inputs lsm z sp tisr)
else
    SUFFIX=""; MASK_ARGS=()
fi

echo "task mapping (TASK -> domain half rep seed):"
for (( t = 0; t < N_TASKS; t++ )); do
    h=$(( (t % (2 * N_REP)) / N_REP )); r=$(( t % N_REP ))
    echo "  ${t} -> ${DOMAINS[$(( t / (2 * N_REP) ))]} h${h} r${r} seed $(( BASE_SEED + N_REP * h + r ))"
done

D="${DOMAINS[$(( TASK / (2 * N_REP) ))]}"
HALF=$(( (TASK % (2 * N_REP)) / N_REP ))
REP=$(( TASK % N_REP ))
SEED=$(( BASE_SEED + N_REP * HALF + REP ))

if [[ "${SMOKE:-0}" == "1" ]]; then
    (( TASK == 0 )) || { echo "ERROR: smoke runs task 0 only"; exit 1; }
    cd "${CODE_ROOT}"
    run_python "${CODE_ROOT}/test_conditional_shift.py"
    OUT_DIR="/scratch/fquareng/half_split_smoke/${SLURM_JOB_ID}/unet_seed${SEED}_half${HALF}${SUFFIX}"
    EXTRA=(--subset_chunks 2 --epochs 1)
else
    OUT_DIR="${OUTPUT_ROOT}/unet_seed${SEED}_half${HALF}${SUFFIX}"
    EXTRA=(--epochs 25)
fi
RUN_DIR="${OUT_DIR}/${D}__to__${D}__none"

echo "task ${TASK}: ${D} half ${HALF} rep ${REP} seed ${SEED} ${MASK_ARGS[*]:-} -> ${RUN_DIR}"
# Never overwrite a checkpoint or a partial run (CLAUDE.md rule 4).
[[ -e "${RUN_DIR}" ]] && { echo "ERROR: ${RUN_DIR} exists. Not overwriting."; exit 1; }
mkdir -p "${OUT_DIR}"

run_python "${CODE_ROOT}/train_unet.py" \
    --source_path "${DATA_ROOT}/${D}" \
    --target_path "${DATA_ROOT}/${D}" \
    --output_dir  "${OUT_DIR}" \
    --data_format npy \
    --uda_method  none \
    --batch_size  128 \
    --patience    5 \
    --num_workers 12 \
    --seed        "${SEED}" \
    --train_half  "${HALF}" \
    "${MASK_ARGS[@]}" \
    "${EXTRA[@]}"

echo "=== half-split task ${TASK} (${D} half ${HALF} rep ${REP} seed ${SEED}) complete ==="
