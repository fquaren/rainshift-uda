#!/bin/bash -l
# ===========================================================================
#  RainShift UDA — AFM (flow matching) experiments, per seed.
#
#  Mirrors run_exp_unet.sh so the two arms differ only in the model:
#    PHASE=1       source-only baselines, one run per source domain
#    PHASE=2       UDA methods
#    PHASE=oracle  joint source+target training, separate _oracle tree
#  Every phase runs as a job array, one run per task (PLAN.md 3.6 step 5);
#  without --array it loops over all runs in one job.
#
#  Output tree: afm_seed${SEED}${OUTPUT_SUFFIX}/ (and ..._oracle/), the tree
#  evaluate.sh reads. Run tags are afm_{src}__to__{tgt}__{method}.
#  After PHASE=1, run materialise_baselines.sh with MODEL=afm.
#
#  Usage (task order: DOMAINS for phase 1, src-major pairs for the oracle):
#    SEED=42 PHASE=1 LR=1e-4 ENC_LR_RATIO=1.0 sbatch --array=0-2 scripts/run_exp_afm.sh
#    SEED=42 PHASE=2 METHODS_ENV="fda spectral" sbatch --array=0-11 scripts/run_exp_afm.sh
#    SEED=42 PHASE=oracle sbatch --array=0-5 --time 24:00:00 scripts/run_exp_afm.sh
#  An oracle epoch streams both training splits (~0.7 h), so up to 25 epochs
#  needs the 24 h limit, which is the gpu-long QOS. Phase 1 and 2 fit 12 h.
#  Smoke test (2 epochs, few chunks, side tree): see scripts/smoke_afm.sh.
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name afm_exp
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

# Same defaults as run_exp_unet.sh: one A100 (40 GB), half a node, <= 12 h.
# To fall back to the GH200 node: sbatch --partition gpu-gh --mem 0 ...
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
# dl-torch mamba env on x86 nodes, the GH200 container on gpu-gh.
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
SEED="${SEED:?SEED must be set, e.g. SEED=42}"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
OUTPUT_DIR="${OUTPUT_ROOT}/afm_seed${SEED}${OUTPUT_SUFFIX:-}"
DATA_FORMAT="npy"

PHASE="${PHASE:-1}"
echo "Selected PHASE: ${PHASE}  SEED: ${SEED}"
echo "Output tree:    ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}/base_hp" "${OUTPUT_DIR}/best_hp"

DOMAINS=("europe_west" "horn-of-africa" "melanesia")

# Bash arrays are whitespace-delimited: NO commas.
# Default is the full set; restrict to the Stage 2 survivors with METHODS_ENV.
METHODS=("fda" "spectral" "adabn" "dann" "mmd" "joint_ot")
if [[ -n "${METHODS_ENV:-}" ]]; then
    read -ra METHODS <<< "${METHODS_ENV}"
fi
# Optional restriction to one pair, e.g. PAIR_ENV="europe_west|horn-of-africa".
PAIR_ENV="${PAIR_ENV:-}"

EPOCHS="${EPOCHS:-25}"
# Early stopping as in the UNet arm: 5 for phase 1 and 2, 8 for the oracle.
PATIENCE="${PATIENCE:-5}"
PATIENCE_ORACLE="${PATIENCE_ORACLE:-8}"
NUM_WORKERS=12
# Width 32, so the AFM encoder is exactly the UNet arm's DualEncoderUNet and
# the arms differ in the objective, not the backbone. Passed in every phase
# (common_args): train_afm reads base_features from argv, not from base_hp/.
#
# Phase 1 sets the batch size; phase 2 and the oracle read it back from
# base_hp/. At width 64 the AFM ran out of memory on a 40 GB A100 at batch
# 512 and at batch 128 (smoke tests 65294537, 65308291, 2026-10-02): two
# full-resolution UNets in fp32 plus, for dann / mmd / joint_ot, a second
# encoder pass on the target batch. Width 32 at batch 64 was estimated at
# ~16 GB (source-only) and ~25 GB (two-forward methods).
BASE_FEATURES="${BASE_FEATURES:-32}"
BATCH_SIZE="${BATCH_SIZE:-64}"
# Flow net at LR, encoder at LR * ENC_LR_RATIO. Both come from the LR sweep
# (scripts/tune_afm_lr.sh, PLAN.md 3.1); phase 1 refuses to train on the
# untuned default except in a smoke test. Phase 2 and the oracle read both
# back from base_hp/. --weight_decay is ignored by train_afm.build_optimizer,
# which hard-codes per-group weight decay.
if [[ "${PHASE}" == "1" && -z "${LR:-}" && -z "${SUBSET_CHUNKS:-}" ]]; then
    echo "ERROR: PHASE=1 needs LR (and ENC_LR_RATIO) from the LR sweep."; exit 1
fi
LR="${LR:-1e-5}"
ENC_LR_RATIO="${ENC_LR_RATIO:-0.1}"
# Checkpoint selection and early stopping on the validation CRPS of the
# sampled ensemble, not on the encoder mean's MSE, so the flow network is
# part of what is selected. Same settings as the sweep.
SELECT_ON="${SELECT_ON:-crps}"
# EMA of weights and BatchNorm buffers; validation, selection and best.pt use
# the EMA copy (PLAN.md 3.6 step 2). Every phase gets the same decay.
EMA_DECAY="${EMA_DECAY:-0.999}"

# UDA settings identical to run_exp_unet.sh.
FDA_BETA=0.01
LAMBDA_UDA=0.1
DANN_WEIGHT=5e-4
JDOT_REG=0.001

# Smoke tests: cap the chunks per split.
EXTRA=()
if [[ -n "${SUBSET_CHUNKS:-}" ]]; then
    EXTRA+=(--subset_chunks "${SUBSET_CHUNKS}")
fi

# Optional phase-2 variants, all off by default: UDA_GRAD_RATIO (gradient-
# matched weight for mmd / mmd_ms / dann), MMD_POOL=grid4, MMD_UNBIASED=1.
# The run tag does not record them, so they need their own side tree.
UDA_EXTRA=()
[[ -n "${UDA_GRAD_RATIO:-}" ]] && UDA_EXTRA+=(--uda_grad_ratio "${UDA_GRAD_RATIO}")
[[ -n "${MMD_POOL:-}" ]] && UDA_EXTRA+=(--mmd_pool "${MMD_POOL}")
[[ -n "${MMD_UNBIASED:-}" ]] && UDA_EXTRA+=(--mmd_unbiased)
if (( ${#UDA_EXTRA[@]} )) && [[ -z "${OUTPUT_SUFFIX:-}" ]]; then
    echo "ERROR: UDA_GRAD_RATIO / MMD_POOL / MMD_UNBIASED need an OUTPUT_SUFFIX side tree."; exit 1
fi

common_args() {
    echo --data_format "${DATA_FORMAT}" --epochs "${EPOCHS}" \
         --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
         --base_features "${BASE_FEATURES}" --select_on "${SELECT_ON}" \
         --val_crps_members 4 --val_crps_steps 10 --val_crps_chunks 8 \
         --ema_decay "${EMA_DECAY}"
}

tag() { echo "afm_$1__to__$2__$3"; }

# One run per array task; every run in a plain job. Sets INDICES. Called
# directly, not in $(...), so the exit stops the job.
select_indices() {   # $1 = number of runs
    INDICES=($(seq 0 $(( $1 - 1 ))))
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        (( SLURM_ARRAY_TASK_ID < $1 )) || {
            echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= $1 runs"; exit 1; }
        INDICES=("${SLURM_ARRAY_TASK_ID}")
    fi
}

# A run directory holding best.pt but no completion marker (best_hp/ or
# base_hp/ JSON, or config.json, all written only after training ends) is a
# run that died part-way. Retraining would overwrite that checkpoint, so stop
# instead (CLAUDE.md rules 4 and 7).
refuse_partial() {
    if [[ -e "$1/best.pt" ]]; then
        echo "ERROR: $1 holds a checkpoint from an unfinished run."
        echo "       Move it to a _trash/ folder in the same tree, then rerun."
        exit 1
    fi
}

# ===========================================================================
#  PHASE 1: source-only baselines, one run per source domain
#  (see run_exp_unet.sh for why one run per source is exact).
# ===========================================================================
if [[ "${PHASE}" == "1" ]]; then
    echo "=== AFM PHASE 1: source-only baselines (${#DOMAINS[@]} runs), patience=${PATIENCE} ==="

    select_indices "${#DOMAINS[@]}"
    for i in "${INDICES[@]}"; do
        src="${DOMAINS[$i]}"
        echo "--- [$((i+1))/${#DOMAINS[@]}] source=${src} ---"

        HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${src}.json"
        if [[ -f "${HP_FILE}" ]]; then
            echo "  Base HPs already exist: ${HP_FILE}, skipping training."
        else
            refuse_partial "${OUTPUT_DIR}/$(tag "${src}" "${src}" none)"
            run_python "${CODE_ROOT}/train_afm.py" \
                --source_path "${DATA_ROOT}/${src}" \
                --target_path "${DATA_ROOT}/${src}" \
                --output_dir  "${OUTPUT_DIR}" \
                --uda_method  none \
                --batch_size  "${BATCH_SIZE}" \
                --lr          "${LR}" \
                --encoder_lr_ratio "${ENC_LR_RATIO}" \
                --patience    "${PATIENCE}" \
                $(common_args) "${EXTRA[@]}" \
                2>&1 | tee "${OUTPUT_DIR}/afm_phase1_${src}.log"
        fi

        # Fan the source-keyed HP file out to every pair name, which is what
        # phase 2, the oracle and train_afm's base-HP lookup key on.
        if [[ -f "${HP_FILE}" ]]; then
            for tgt in "${DOMAINS[@]}"; do
                [[ "$src" == "$tgt" ]] && continue
                cp -f "${HP_FILE}" "${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
            done
        else
            echo "  WARNING: ${HP_FILE} not produced; phase 2 will skip ${src}."
        fi
    done
    echo "=== AFM PHASE 1 complete ==="

# ===========================================================================
#  PHASE 2: UDA methods, base HPs from phase 1. One run per array task.
# ===========================================================================
elif [[ "${PHASE}" == "2" ]]; then
    RUNS=()
    for src in "${DOMAINS[@]}"; do
        for tgt in "${DOMAINS[@]}"; do
            [[ "$src" == "$tgt" ]] && continue
            [[ -n "${PAIR_ENV}" && "${src}|${tgt}" != "${PAIR_ENV}" ]] && continue
            HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
            # A side tree (OUTPUT_SUFFIX) reuses the seed's own phase-1 HPs,
            # as in run_exp_unet.sh.
            MAIN_HP="${OUTPUT_ROOT}/afm_seed${SEED}/base_hp/${src}__to__${tgt}.json"
            if [[ ! -f "${HP_FILE}" && -n "${OUTPUT_SUFFIX:-}" && -f "${MAIN_HP}" ]]; then
                cp "${MAIN_HP}" "${HP_FILE}"
            fi
            if [[ ! -f "${HP_FILE}" ]]; then
                echo "ERROR: missing base HPs ${HP_FILE}; run PHASE=1 first."
                exit 1
            fi
            for method in "${METHODS[@]}"; do
                RUNS+=("${src}|${tgt}|${method}")
            done
        done
    done

    echo "=== AFM PHASE 2: ${#RUNS[@]} runs (${METHODS[*]}), patience=${PATIENCE} ==="

    select_indices "${#RUNS[@]}"
    for i in "${INDICES[@]}"; do
        IFS='|' read -r src tgt method <<< "${RUNS[$i]}"
        echo "--- [$((i+1))/${#RUNS[@]}] ${src} -> ${tgt} | ${method} ---"

        BEST_FILE="${OUTPUT_DIR}/best_hp/afm_${src}__to__${tgt}__${method}.json"
        if [[ -f "${BEST_FILE}" ]]; then
            echo "  Already done, skipping."
            continue
        fi
        refuse_partial "${OUTPUT_DIR}/$(tag "${src}" "${tgt}" "${method}")"

        run_python "${CODE_ROOT}/train_afm.py" \
            --source_path "${DATA_ROOT}/${src}" \
            --target_path "${DATA_ROOT}/${tgt}" \
            --output_dir  "${OUTPUT_DIR}" \
            --uda_method  "${method}" \
            --lambda_uda  "${LAMBDA_UDA}" \
            --fda_beta    "${FDA_BETA}" \
            --dann_weight "${DANN_WEIGHT}" \
            --jdot_reg    "${JDOT_REG}" \
            --patience    "${PATIENCE}" \
            $(common_args) "${EXTRA[@]}" "${UDA_EXTRA[@]}" \
            2>&1 | tee "${OUTPUT_DIR}/afm_phase2_${src}__to__${tgt}__${method}.log"
    done

# ===========================================================================
#  PHASE oracle: joint source+target training, kept in ${OUTPUT_DIR}_oracle
#  so it cannot collide with the phase-1 __none checkpoints.
# ===========================================================================
elif [[ "${PHASE}" == "oracle" ]]; then
    ORACLE_DIR="${OUTPUT_DIR}_oracle"
    mkdir -p "${ORACLE_DIR}/base_hp"

    PAIRS=()
    for src in "${DOMAINS[@]}"; do
        for tgt in "${DOMAINS[@]}"; do
            [[ "$src" == "$tgt" ]] && continue
            [[ -n "${PAIR_ENV}" && "${src}|${tgt}" != "${PAIR_ENV}" ]] && continue
            PAIRS+=("${src}|${tgt}")
        done
    done

    echo "=== AFM PHASE oracle: ${#PAIRS[@]} pairs, patience=${PATIENCE_ORACLE}, output ${ORACLE_DIR} ==="

    select_indices "${#PAIRS[@]}"
    for i in "${INDICES[@]}"; do
        IFS='|' read -r src tgt <<< "${PAIRS[$i]}"
        echo "--- [$((i+1))/${#PAIRS[@]}] ${src} + ${tgt} (joint) ---"

        # config.json is written only after training ends; best.pt from the
        # first epoch on, so it cannot tell a finished run from a killed one.
        RUN_DIR="${ORACLE_DIR}/$(tag "${src}" "${tgt}" none)"
        if [[ -f "${RUN_DIR}/config.json" ]]; then
            echo "  Oracle run complete, skipping."
            continue
        fi
        refuse_partial "${RUN_DIR}"

        # Same base HPs as the source-only baseline (fair oracle comparison).
        HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
        [[ -f "${HP_FILE}" ]] || { echo "ERROR: missing ${HP_FILE}; run PHASE=1 first."; exit 1; }
        cp "${HP_FILE}" "${ORACLE_DIR}/base_hp/${src}__to__${tgt}.json"

        run_python "${CODE_ROOT}/train_afm.py" \
            --source_path "${DATA_ROOT}/${src}" \
            --target_path "${DATA_ROOT}/${tgt}" \
            --output_dir  "${ORACLE_DIR}" \
            --uda_method  none \
            --joint_training \
            --patience    "${PATIENCE_ORACLE}" \
            $(common_args) "${EXTRA[@]}" \
            2>&1 | tee "${ORACLE_DIR}/afm_oracle_${src}__to__${tgt}.log"
    done
    echo "=== AFM PHASE oracle complete ==="

else
    echo "ERROR: PHASE must be 1, 2, or oracle (got: ${PHASE})"; exit 1
fi
