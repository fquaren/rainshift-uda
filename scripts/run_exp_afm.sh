#!/bin/bash -l
# ===========================================================================
#  RainShift UDA — AFM two-phase experiments
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
#SBATCH --cpus-per-task 12
#SBATCH --mem 400G
#SBATCH --time 72:00:00

set -euo pipefail

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0"

CONTAINER="/users/fquareng/singularity/dl_gh200.sif"
CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/rainshift_npy"
OUTPUT_DIR="/scratch/fquareng/rainshift_uda/afm"

PHASE="${PHASE:-1}"

REGIONS=(
    "europe_west"
    "blacksea"
    "horn-of-africa"
    "melanesia"
)

METHODS=("coral" "mmd" "spectral" "fda" "dann" "adabn")

EPOCHS=100
PATIENCE=10
NUM_WORKERS=8
SUBSET_SIZE=5000
BATCH_SIZE=32

N_TRIALS_P1=10
N_TRIALS_P2=10
OPTUNA_TIMEOUT=172800

COMPILE="--compile"

run_python() {
    singularity exec --nv "${CONTAINER}" python "$@"
}

if [[ "${PHASE}" == "1" ]]; then
    PAIRS=()
    for src in "${REGIONS[@]}"; do
        for tgt in "${REGIONS[@]}"; do
            [[ "$src" == "$tgt" ]] && continue
            PAIRS+=("${src}|${tgt}")
        done
    done

    echo "=== AFM PHASE 1: Base HP search ==="
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
            --uda_method  none \
            --epochs      "${EPOCHS}" \
            --batch_size  "${BATCH_SIZE}" \
            --patience    "${PATIENCE}" \
            --num_workers "${NUM_WORKERS}" \
            --subset_size "${SUBSET_SIZE}" \
            ${COMPILE} \
            --optuna --optuna_phase 1 \
            --n_trials    "${N_TRIALS_P1}" \
            --optuna_timeout "${OPTUNA_TIMEOUT}" \
            2>&1 | tee "${OUTPUT_DIR}/afm_phase1_${src}__to__${tgt}.log"
        echo ""
    done
    echo "=== AFM PHASE 1 complete ==="

elif [[ "${PHASE}" == "2" ]]; then
    RUNS=()
    for src in "${REGIONS[@]}"; do
        for tgt in "${REGIONS[@]}"; do
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

    echo "=== AFM PHASE 2: UDA weight search ==="
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
            --uda_method  "${method}" \
            --epochs      "${EPOCHS}" \
            --batch_size  "${BATCH_SIZE}" \
            --patience    "${PATIENCE}" \
            --num_workers "${NUM_WORKERS}" \
            --subset_size "${SUBSET_SIZE}" \
            ${COMPILE} \
            --optuna --optuna_phase 2 \
            --n_trials    "${N_TRIALS_P2}" \
            --optuna_timeout "${OPTUNA_TIMEOUT}" \
            2>&1 | tee "${OUTPUT_DIR}/afm_phase2_${src}__to__${tgt}__${method}.log"
        echo ""
    done
    echo "=== AFM PHASE 2 complete ==="
else
    echo "ERROR: PHASE must be 1 or 2"; exit 1
fi
