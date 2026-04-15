#!/bin/bash -l
# ===========================================================================
#  RainShift UDA — Two-phase experiment launcher
#
#  PHASE 1: Optuna search for base HPs (lr, batch_size, wd) on vanilla model.
#           One job per (source, target) pair.
#           Results saved to {OUTPUT_DIR}/base_hp/{src}__to__{tgt}.json
#
#  PHASE 2: Optuna search for UDA-specific HPs (lambda_uda, fda_beta).
#           One job per (source, target, method) triple.
#           Loads base HPs from phase 1 automatically.
#
#  Usage:
#    # Run phase 1 first
#    PHASE=1 sbatch run_experiments.sh
#
#    # After all phase 1 jobs complete, run phase 2
#    PHASE=2 sbatch run_experiments.sh
#
#    # Or submit phase 2 with SLURM dependency:
#    P1_JOB=$(PHASE=1 sbatch --parsable run_experiments.sh)
#    PHASE=2 sbatch --dependency=afterok:${P1_JOB} run_experiments.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name uda_exp
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 435G
#SBATCH --time 72:00:00

set -euo pipefail

# export SINGULARITY_BINDPATH="/work,/scratch,/users"
# export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0"

# --- Configuration --------------------------------------------------------
CONTAINER="/users/fquareng/singularity/dl_gh200.sif"
CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/rainshift_npy"
OUTPUT_DIR="/scratch/fquareng/rainshift_uda/unet"
DATA_FORMAT="npy"

# Phase selection: set via environment variable before sbatch
# PHASE=1 for base HP search, PHASE=2 for UDA weight search
PHASE="${PHASE:-1}"

REGIONS=(
    "europe_west"
    # "blacksea"
    # "horn-of-africa"
    "melanesia"
)

# UDA methods (phase 2 only — phase 1 always uses "none")
METHODS=("coral" "mmd" "spectral" "fda" "dann" "adabn")

# Shared training settings
EPOCHS=100
PATIENCE=10
NUM_WORKERS=8
SUBSET_SIZE=1000

# Optuna settings
N_TRIALS_P1=10      
N_TRIALS_P2=10      # phase 2: fewer trials, 1-2D search space
OPTUNA_TIMEOUT=172800

# Defaults for non-Optuna params
FDA_BETA=0.01
LAMBDA_UDA=0.1

# GH200
COMPILE="--compile"
# --------------------------------------------------------------------------

run_python() {
    singularity exec --nv "${CONTAINER}" python "$@"
}

# ===========================================================================
#  PHASE 1: Base HP search (vanilla, one per domain pair)
# ===========================================================================
if [[ "${PHASE}" == "1" ]]; then
    PAIRS=()
    for src in "${REGIONS[@]}"; do
        for tgt in "${REGIONS[@]}"; do
            [[ "$src" == "$tgt" ]] && continue
            PAIRS+=("${src}|${tgt}")
        done
    done

    echo "=== PHASE 1: Base HP search (vanilla) ==="
    echo "Domain pairs: ${#PAIRS[@]}"
    echo ""

    for i in "${!PAIRS[@]}"; do
        IFS='|' read -r src tgt <<< "${PAIRS[$i]}"
        echo "--- [$((i+1))/${#PAIRS[@]}] ${src} -> ${tgt} | vanilla HP search ---"

        # Check if already completed
        HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
        if [[ -f "${HP_FILE}" ]]; then
            echo "  Base HPs already exist: ${HP_FILE}, skipping."
            continue
        fi

        SUBSET_FLAG=""
        [[ -n "${SUBSET_SIZE}" ]] && SUBSET_FLAG="--subset_size ${SUBSET_SIZE}"

        run_python "${CODE_ROOT}/train_unet.py" \
            --source_path "${DATA_ROOT}/${src}" \
            --target_path "${DATA_ROOT}/${tgt}" \
            --output_dir  "${OUTPUT_DIR}" \
            --data_format "${DATA_FORMAT}" \
            --uda_method  none \
            --epochs      "${EPOCHS}" \
            --patience    "${PATIENCE}" \
            --num_workers "${NUM_WORKERS}" \
            ${SUBSET_FLAG} \
            ${COMPILE} \
            --optuna \
            --optuna_phase 1 \
            --n_trials    "${N_TRIALS_P1}" \
            --optuna_timeout "${OPTUNA_TIMEOUT}" \
            2>&1 | tee "${OUTPUT_DIR}/phase1_${src}__to__${tgt}.log"

        echo ""
    done

    echo "=== PHASE 1 complete ==="

# ===========================================================================
#  PHASE 2: UDA weight search (one per domain pair × method)
# ===========================================================================
elif [[ "${PHASE}" == "2" ]]; then
    RUNS=()
    for src in "${REGIONS[@]}"; do
        for tgt in "${REGIONS[@]}"; do
            [[ "$src" == "$tgt" ]] && continue

            # Verify phase 1 completed for this pair
            HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
            if [[ ! -f "${HP_FILE}" ]]; then
                echo "WARNING: Missing base HPs for ${src} -> ${tgt}, skipping."
                echo "  Expected: ${HP_FILE}"
                echo "  Run PHASE=1 first."
                continue
            fi

            for method in "${METHODS[@]}"; do
                RUNS+=("${src}|${tgt}|${method}")
            done
        done
    done

    echo "=== PHASE 2: UDA weight search ==="
    echo "Total runs: ${#RUNS[@]}"
    echo ""

    for i in "${!RUNS[@]}"; do
        IFS='|' read -r src tgt method <<< "${RUNS[$i]}"
        echo "--- [$((i+1))/${#RUNS[@]}] ${src} -> ${tgt} | ${method} ---"

        # Check if already completed
        BEST_FILE="${OUTPUT_DIR}/best_hp/${src}__to__${tgt}__${method}.json"
        if [[ -f "${BEST_FILE}" ]]; then
            echo "  Best HPs already exist: ${BEST_FILE}, skipping."
            continue
        fi

        SUBSET_FLAG=""
        [[ -n "${SUBSET_SIZE}" ]] && SUBSET_FLAG="--subset_size ${SUBSET_SIZE}"

        run_python "${CODE_ROOT}/train_unet.py" \
            --source_path "${DATA_ROOT}/${src}" \
            --target_path "${DATA_ROOT}/${tgt}" \
            --output_dir  "${OUTPUT_DIR}" \
            --data_format "${DATA_FORMAT}" \
            --uda_method  "${method}" \
            --fda_beta    "${FDA_BETA}" \
            --epochs      "${EPOCHS}" \
            --patience    "${PATIENCE}" \
            --num_workers "${NUM_WORKERS}" \
            ${SUBSET_FLAG} \
            ${COMPILE} \
            --optuna \
            --optuna_phase 2 \
            --n_trials    "${N_TRIALS_P2}" \
            --optuna_timeout "${OPTUNA_TIMEOUT}" \
            2>&1 | tee "${OUTPUT_DIR}/phase2_${src}__to__${tgt}__${method}.log"

        echo ""
    done

    echo "=== PHASE 2 complete ==="

else
    echo "ERROR: PHASE must be 1 or 2 (got: ${PHASE})"
    exit 1
fi