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
#SBATCH --cpus-per-task 24
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
METHODS=("joint_ot" "dann" "mmd" "spectral" "fda" "adabn")
# Dropped: "coral" (subsumed by MMD, mean-blind, weakly scaled) and
# "mmd_ms" (same mechanism as MMD, unstable). joint_ot = DeepJDOT, the
# only method aligning the JOINT distribution and so the only one that
# can address conditional shift.

EPOCHS=25
PATIENCE="${PATIENCE:--1}"
NUM_WORKERS=24
BATCH_SIZE=512

FDA_BETA=0.01
LAMBDA_UDA=0.1

run_python() {
    singularity exec --nv "${CONTAINER}" python "$@"
}

# ===========================================================================
#  PHASE 1: source-only baselines — ONE RUN PER SOURCE DOMAIN
#
#  A source-only model never sees the target: with --uda_method none the
#  target loader is built but never drawn from, so europe_west->horn and
#  europe_west->melanesia train byte-identical weights. The old pair loop
#  therefore trained each of the 3 models twice (4 of 6 runs were duplicate
#  compute). We train 3 models and fan the resulting HP file out to the 6
#  pair names that Phase 2 and evaluate.py look up.
#
#  target_path is set to the SOURCE itself rather than to a dummy domain:
#    - weights are unchanged (target is unused),
#    - tgt_mae then reports IN-DOMAIN test error, i.e. the diagonal of the
#      3x3 transfer matrix, which the paper reports anyway,
#    - the checkpoint dir is unambiguously {src}__to__{src}__none,
#    - no I/O is wasted streaming a domain that is never used.
#
#  PATIENCE_P1 defaults to 5: validation bottoms out at epoch 3-6 on all
#  three domains, so 25 epochs spends ~80% of the wall clock past the
#  checkpoint that is actually selected.
#
#  RESIDUAL=1 enables the log-space residual head (predict a correction to
#  the upsampled coarse tp channel). Use it to A/B against the plain head.
# ===========================================================================
if [[ "${PHASE}" == "1" ]]; then
    PATIENCE_P1="${PATIENCE_P1:-5}"
    RESIDUAL_FLAG=""
    [[ -n "${RESIDUAL:-}" ]] && RESIDUAL_FLAG="--residual"

    echo "=== PHASE 1: source-only baselines (${#SOURCE_REGIONS[@]} runs) ==="
    echo "    patience=${PATIENCE_P1}  residual=${RESIDUAL:-0}"

    for i in "${!SOURCE_REGIONS[@]}"; do
        src="${SOURCE_REGIONS[$i]}"
        echo "--- [$((i+1))/${#SOURCE_REGIONS[@]}] source=${src} ---"

        HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${src}.json"
        if [[ -f "${HP_FILE}" ]]; then
            echo "  Base HPs already exist: ${HP_FILE}, skipping training."
        else
            run_python "${CODE_ROOT}/train_afm.py" \
                --source_path "${DATA_ROOT}/${src}" \
                --target_path "${DATA_ROOT}/${src}" \
                --output_dir  "${OUTPUT_DIR}" \
                --data_format "${DATA_FORMAT}" \
                --uda_method  none \
                --epochs      "${EPOCHS}" \
                --batch_size  "${BATCH_SIZE}" \
                --patience    "${PATIENCE_P1}" \
                --num_workers "${NUM_WORKERS}" \
                ${RESIDUAL_FLAG} \
                2>&1 | tee "${OUTPUT_DIR}/afm_phase1_${src}.log"
        fi

        # Fan the source-keyed HP file out to every pair name. Phase 2's
        # skip-guard and train_unet's own base-HP lookup both key on
        # {src}__to__{tgt}.json, so without this Phase 2 finds nothing and
        # silently skips every run.
        if [[ -f "${HP_FILE}" ]]; then
            for tgt in "${TARGET_REGIONS[@]}"; do
                [[ "$src" == "$tgt" ]] && continue
                cp -f "${HP_FILE}" "${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
            done
            echo "  Fanned base HPs out to $((${#TARGET_REGIONS[@]} - 1)) pair name(s)."
        else
            echo "  WARNING: ${HP_FILE} not produced; Phase 2 will skip ${src}."
        fi
    done
    echo "=== PHASE 1 complete ==="

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
