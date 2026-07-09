#!/bin/bash -l
# ===========================================================================
#  RainShift UDA — AFM two-phase experiments
#
#  PHASE 1: Baseline training on vanilla model.
#  PHASE 2: Standard training for UDA methods.
#
#  Usage: PHASE=1 sbatch scripts/run_exp_afm.sh
#         PHASE=2 sbatch scripts/run_exp_afm.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name afm_exp
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 0
#SBATCH --time 72:00:00

set -euo pipefail

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0"

CONTAINER="/users/fquareng/singularity/dl_gh200.sif"
CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
OUTPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/afm"
DATA_FORMAT="npy"

PHASE="${PHASE:-1}"
echo "Selected PHASE: ${PHASE}"

mkdir -p "${OUTPUT_DIR}/base_hp"
mkdir -p "${OUTPUT_DIR}/best_hp"


SOURCE_REGIONS=(
    "europe_west"
    # "blacksea"
    "horn-of-africa"
    "melanesia"
)

TARGET_REGIONS=(
    "europe_west"
    # "blacksea"
    "horn-of-africa"
    "melanesia"
)

# Bash arrays are whitespace-delimited: NO commas, or each element keeps a
# trailing comma and argparse --uda_method choices rejects it.
METHODS=("dann" "mmd" "mmd_ms" "coral" "spectral" "fda" "adabn")

EPOCHS=25
PATIENCE=-1
NUM_WORKERS=4
BATCH_SIZE=512

FDA_BETA=0.01
LAMBDA_UDA=0.1

run_python() {
    singularity exec --nv "${CONTAINER}" python "$@"
}

if [[ "${PHASE}" == "1" ]]; then
    PAIRS=()
    for src in "${SOURCE_REGIONS[@]}"; do
        for tgt in "${TARGET_REGIONS[@]}"; do
            [[ "$src" == "$tgt" ]] && continue
            PAIRS+=("${src}|${tgt}")
        done
    done

    echo "=== AFM PHASE 1: Baseline (no UDA) ==="
    echo "Domain pairs: ${#PAIRS[@]}"

    for i in "${!PAIRS[@]}"; do
        IFS='|' read -r src tgt <<< "${PAIRS[$i]}"
        echo "--- [$((i+1))/${#PAIRS[@]}] ${src} -> ${tgt} ---"

        HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
        if [[ -f "${HP_FILE}" ]]; then
            echo "  Already done, skipping."
            continue
        fi

        run_python "${CODE_ROOT}/train_afm.py" \
            --source_path "${DATA_ROOT}/${src}" \
            --target_path "${DATA_ROOT}/${tgt}" \
            --output_dir  "${OUTPUT_DIR}" \
            --data_format "${DATA_FORMAT}" \
            --uda_method  none \
            --epochs      "${EPOCHS}" \
            --batch_size  "${BATCH_SIZE}" \
            --patience    "${PATIENCE}" \
            --num_workers "${NUM_WORKERS}" \
            2>&1 | tee "${OUTPUT_DIR}/afm_phase1_${src}__to__${tgt}.log"
        echo ""
    done
    echo "=== AFM PHASE 1 complete ==="

elif [[ "${PHASE}" == "2" ]]; then
    RUNS=()
    for src in "${SOURCE_REGIONS[@]}"; do
        for tgt in "${TARGET_REGIONS[@]}"; do
            [[ "$src" == "$tgt" ]] && continue
            HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
            if [[ ! -f "${HP_FILE}" ]]; then
                echo "WARNING: Missing base HPs for ${src} -> ${tgt}, using default."
                continue
            fi
            for method in "${METHODS[@]}"; do
                RUNS+=("${src}|${tgt}|${method}")
            done
        done
    done

    echo "=== AFM PHASE 2: UDA training (Fixed Hyperparameters) ==="
    echo "Total runs: ${#RUNS[@]}"

    for i in "${!RUNS[@]}"; do
        IFS='|' read -r src tgt method <<< "${RUNS[$i]}"
        echo "--- [$((i+1))/${#RUNS[@]}] ${src} -> ${tgt} | ${method} ---"

        BEST_FILE="${OUTPUT_DIR}/best_hp/afm_${src}__to__${tgt}__${method}.json"
        if [[ -f "${BEST_FILE}" ]]; then
            echo "  Already done, skipping."
            continue
        fi

        run_python "${CODE_ROOT}/train_afm.py" \
            --source_path "${DATA_ROOT}/${src}" \
            --target_path "${DATA_ROOT}/${tgt}" \
            --output_dir  "${OUTPUT_DIR}" \
            --data_format "${DATA_FORMAT}" \
            --uda_method  "${method}" \
            --lambda_uda  "${LAMBDA_UDA}" \
            --fda_beta    "${FDA_BETA}" \
            --epochs      "${EPOCHS}" \
            --patience    "${PATIENCE}" \
            --num_workers "${NUM_WORKERS}" \
            2>&1 | tee "${OUTPUT_DIR}/afm_phase2_${src}__to__${tgt}__${method}.log"
        echo ""
    done
    echo "=== AFM PHASE 2 complete ==="

# ===========================================================================
#  PHASE oracle: joint source+target training (upper bound on transfer).
#  Produces E_T(h_joint) for the addressable-budget denominator. Written to a
#  separate output dir to avoid colliding with Phase 1 source-only 'none'
#  checkpoints. Evaluate on both source and target test sets afterwards.
# ===========================================================================
elif [[ "${PHASE}" == "oracle" ]]; then
    ORACLE_DIR="${OUTPUT_DIR}_oracle"
    mkdir -p "${ORACLE_DIR}/base_hp"

    PAIRS=()
    for src in "${SOURCE_REGIONS[@]}"; do
        for tgt in "${TARGET_REGIONS[@]}"; do
            [[ "$src" == "$tgt" ]] && continue
            PAIRS+=("${src}|${tgt}")
        done
    done

    echo "=== AFM PHASE oracle: joint source+target training ==="
    echo "Domain pairs: ${#PAIRS[@]}   output: ${ORACLE_DIR}"

    for i in "${!PAIRS[@]}"; do
        IFS='|' read -r src tgt <<< "${PAIRS[$i]}"
        echo "--- [$((i+1))/${#PAIRS[@]}] ${src} + ${tgt} (joint) ---"

        DONE_MARKER="${ORACLE_DIR}/afm_${src}__to__${tgt}____none/best.pt"
        if [[ -f "${DONE_MARKER}" ]]; then
            echo "  Oracle checkpoint exists, skipping."
            continue
        fi

        HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
        if [[ -f "${HP_FILE}" ]]; then
            cp "${HP_FILE}" "${ORACLE_DIR}/base_hp/${src}__to__${tgt}.json"
        else
            echo "  WARNING: no base HPs for ${src}->${tgt}; oracle uses argparse defaults."
        fi

        run_python "${CODE_ROOT}/train_afm.py" \
            --source_path "${DATA_ROOT}/${src}" \
            --target_path "${DATA_ROOT}/${tgt}" \
            --output_dir  "${ORACLE_DIR}" \
            --data_format "${DATA_FORMAT}" \
            --uda_method  none \
            --joint_training \
            --epochs      "${EPOCHS}" \
            --batch_size  "${BATCH_SIZE}" \
            --patience    "${PATIENCE}" \
            --num_workers "${NUM_WORKERS}" \
            2>&1 | tee "${ORACLE_DIR}/afm_oracle_${src}__to__${tgt}.log"
    done
    echo "=== AFM PHASE oracle complete ==="

else
    echo "ERROR: PHASE must be 1, 2, or oracle"; exit 1
fi