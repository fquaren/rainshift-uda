#!/bin/bash -l
# ===========================================================================
#  RainShift UDA — Two-phase experiment launcher
#
#  PHASE 1: Optuna search for base HPs (lr, batch_size, wd) on vanilla model.
#           One job per (source, target) pair.
#           Results saved to {OUTPUT_DIR}/base_hp/{src}__to__{tgt}.json
#
#  PHASE 2: Standard training for UDA methods. (Method A: No UDA HP tuning).
#           One job per (source, target, method) triple.
#           Loads base HPs from phase 1 automatically.
#
#  Usage:
#    PHASE=1 sbatch run_experiments.sh
#    # After completion:
#    PHASE=2 sbatch run_experiments.sh
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
#SBATCH --cpus-per-task 24
#SBATCH --mem 0
#SBATCH --time 72:00:00

set -euo pipefail

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0"

# --- Configuration --------------------------------------------------------
CONTAINER="/users/fquareng/singularity/dl_gh200.sif"
CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
OUTPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/unet"
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
NUM_WORKERS=12
BATCH_SIZE=128

FDA_BETA=0.01
LAMBDA_UDA=0.1
# DANN uses a separate, small adversarial weight (the GRL alpha already
# ramps the reversal 0->1). ~5e-4 per Wang et al. 2026; a large value
# collapses the discriminator to ln(2). Ignored by non-DANN methods.
DANN_WEIGHT=5e-4

# --------------------------------------------------------------------------

run_python() {
    singularity exec --nv "${CONTAINER}" python "$@"
}

# ===========================================================================
#  PHASE 1: Base HP search (vanilla, one per domain pair)
# ===========================================================================
if [[ "${PHASE}" == "1" ]]; then
    PAIRS=()
    for src in "${SOURCE_REGIONS[@]}"; do
        for tgt in "${TARGET_REGIONS[@]}"; do
            [[ "$src" == "$tgt" ]] && continue
            PAIRS+=("${src}|${tgt}")
        done
    done

    echo "=== PHASE 1: Vanilla UNet (no UDA) ==="
    
    for i in "${!PAIRS[@]}"; do
        IFS='|' read -r src tgt <<< "${PAIRS[$i]}"
        echo "--- [$((i+1))/${#PAIRS[@]}] ${src} -> ${tgt} ---"

        HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
        if [[ -f "${HP_FILE}" ]]; then
            echo "  Base HPs already exist: ${HP_FILE}, skipping."
            continue
        fi

        run_python "${CODE_ROOT}/train_unet.py" \
            --source_path "${DATA_ROOT}/${src}" \
            --target_path "${DATA_ROOT}/${tgt}" \
            --output_dir  "${OUTPUT_DIR}" \
            --data_format "${DATA_FORMAT}" \
            --uda_method  none \
            --epochs      "${EPOCHS}" \
            --batch_size  "${BATCH_SIZE}" \
            --patience    "${PATIENCE}" \
            --num_workers "${NUM_WORKERS}" \
            2>&1 | tee "${OUTPUT_DIR}/phase1_${src}__to__${tgt}.log"
    done

# ===========================================================================
#  PHASE 2: UDA application (Fixed HPs, no Optuna)
# ===========================================================================
elif [[ "${PHASE}" == "2" ]]; then
    RUNS=()
    for src in "${SOURCE_REGIONS[@]}"; do
        for tgt in "${TARGET_REGIONS[@]}"; do
            [[ "$src" == "$tgt" ]] && continue

            HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
            if [[ ! -f "${HP_FILE}" ]]; then
                echo "WARNING: Missing base HPs for ${src} -> ${tgt}, skipping."
                continue
            fi

            for method in "${METHODS[@]}"; do
                RUNS+=("${src}|${tgt}|${method}")
            done
        done
    done

    echo "=== PHASE 2: Fixed UDA training ==="
    
    for i in "${!RUNS[@]}"; do
        IFS='|' read -r src tgt method <<< "${RUNS[$i]}"
        echo "--- [$((i+1))/${#RUNS[@]}] ${src} -> ${tgt} | ${method} ---"

        BEST_FILE="${OUTPUT_DIR}/best_hp/${src}__to__${tgt}__${method}.json"
        if [[ -f "${BEST_FILE}" ]]; then
            echo "  Run completed previously, skipping."
            continue
        fi

        run_python "${CODE_ROOT}/train_unet.py" \
            --source_path "${DATA_ROOT}/${src}" \
            --target_path "${DATA_ROOT}/${tgt}" \
            --output_dir  "${OUTPUT_DIR}" \
            --data_format "${DATA_FORMAT}" \
            --uda_method  "${method}" \
            --lambda_uda  "${LAMBDA_UDA}" \
            --fda_beta    "${FDA_BETA}" \
            --dann_weight "${DANN_WEIGHT}" \
            --epochs      "${EPOCHS}" \
            --batch_size  "${BATCH_SIZE}" \
            --patience    "${PATIENCE}" \
            --num_workers "${NUM_WORKERS}" \
            2>&1 | tee "${OUTPUT_DIR}/phase2_${src}__to__${tgt}__${method}.log"
    done

# ===========================================================================
#  PHASE oracle: joint source+target training (upper bound on transfer).
#  Produces the addressable-budget denominator E_T(h_joint). Runs
#  --uda_method none --joint_training. Written to a SEPARATE output dir so it
#  does not collide with the Phase 1 source-only 'none' checkpoints (which
#  also use uda_method=none). Evaluate this checkpoint on both source and
#  target test sets to obtain E_S(h_joint) and E_T(h_joint).
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

    echo "=== PHASE oracle: joint source+target training ==="
    echo "Domain pairs: ${#PAIRS[@]}   output: ${ORACLE_DIR}"

    for i in "${!PAIRS[@]}"; do
        IFS='|' read -r src tgt <<< "${PAIRS[$i]}"
        echo "--- [$((i+1))/${#PAIRS[@]}] ${src} + ${tgt} (joint) ---"

        # Oracle completion marker: the source-only base HPs are reused as the
        # base-HP source, but the oracle checkpoint lives under ORACLE_DIR.
        DONE_MARKER="${ORACLE_DIR}/${src}__to__${tgt}__none/best.pt"
        if [[ -f "${DONE_MARKER}" ]]; then
            echo "  Oracle checkpoint exists, skipping."
            continue
        fi

        # Reuse the Phase 1 base HPs for this pair if present, so lr/bs/wd
        # match the source-only baseline (fair oracle comparison).
        HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
        if [[ -f "${HP_FILE}" ]]; then
            cp "${HP_FILE}" "${ORACLE_DIR}/base_hp/${src}__to__${tgt}.json"
        else
            echo "  WARNING: no base HPs for ${src}->${tgt}; oracle uses argparse defaults."
        fi

        run_python "${CODE_ROOT}/train_unet.py" \
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
            2>&1 | tee "${ORACLE_DIR}/oracle_${src}__to__${tgt}.log"
    done
    echo "=== PHASE oracle complete ==="

else
    echo "ERROR: PHASE must be 1, 2, or oracle (got: ${PHASE})"
    exit 1
fi