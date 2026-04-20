#!/bin/bash -l
# ===========================================================================
#  RainShift UDA — Two-phase experiment launcher
#
#  PHASE 1: Optuna search for base HPs (lr, batch_size, wd) training on
#           BOTH source and target domains simultaneously (joint training).
#           Results saved to {OUTPUT_DIR}/base_hp/{src}__and__${tgt}.json
#
#  PHASE 2: Standard training for UDA methods. 
#           Loads base HPs from phase 1 automatically.
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name uda_exp
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 400G
#SBATCH --time 72:00:00

set -euo pipefail

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0"

# --- Configuration --------------------------------------------------------
CONTAINER="/users/fquareng/singularity/dl_gh200.sif"
CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/rainshift_npy"
OUTPUT_DIR="/scratch/fquareng/rainshift_uda/unet_joint"
DATA_FORMAT="npy"

PHASE="${PHASE:-1}"
echo "Selected PHASE: ${PHASE}"

mkdir -p "${OUTPUT_DIR}/base_hp"
mkdir -p "${OUTPUT_DIR}/best_hp"

SOURCE_REGIONS=(
    "europe_west"
)

TARGET_REGIONS=(
    "melanesia"
)

METHODS=("fda" "mmd" "adabn")

EPOCHS=25
PATIENCE=-1
NUM_WORKERS=8
BATCH_SIZE=256

OPTUNA_TIMEOUT=172800

FDA_BETA=0.01
LAMBDA_UDA=0.1

COMPILE="--compile" 
# --------------------------------------------------------------------------

run_python() {
    singularity exec --nv "${CONTAINER}" python "$@"
}

# ===========================================================================
#  PHASE 1: Base HP search (joint training on src + tgt)
# ===========================================================================
if [[ "${PHASE}" == "1" ]]; then
    PAIRS=()
    for src in "${SOURCE_REGIONS[@]}"; do
        for tgt in "${TARGET_REGIONS[@]}"; do
            [[ "$src" == "$tgt" ]] && continue
            PAIRS+=("${src}|${tgt}")
        done
    done

    echo "=== PHASE 1: Base HP search (joint training) ==="
    
    for i in "${!PAIRS[@]}"; do
        IFS='|' read -r src tgt <<< "${PAIRS[$i]}"
        echo "--- [$((i+1))/${#PAIRS[@]}] ${src} + ${tgt} | joint HP search ---"

        HP_FILE="${OUTPUT_DIR}/base_hp/${src}__and__${tgt}.json"
        if [[ -f "${HP_FILE}" ]]; then
            echo "  Base HPs already exist: ${HP_FILE}, skipping."
            continue
        fi

        # Added --joint_training flag assuming train_unet.py handles dataset concatenation
        run_python "${CODE_ROOT}/train_unet.py" \
            --source_path "${DATA_ROOT}/${src}" \
            --target_path "${DATA_ROOT}/${tgt}" \
            --output_dir  "${OUTPUT_DIR}" \
            --data_format "${DATA_FORMAT}" \
            --joint_training \
            --uda_method  none \
            --epochs      "${EPOCHS}" \
            --batch_size  "${BATCH_SIZE}" \
            --patience    "${PATIENCE}" \
            --num_workers "${NUM_WORKERS}" \
            2>&1 | tee "${OUTPUT_DIR}/phase1_joint_${src}__and__${tgt}.log"
    done