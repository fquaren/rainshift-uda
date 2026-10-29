#!/bin/bash -l
# ===========================================================================
#  RainShift UDA — UNet experiments, per seed (run_exp_afm.sh is the AFM twin).
#
#    PHASE=1       source-only baselines, one run per SOURCE domain (3 runs,
#                  looped in one job), fixed HPs (no search). Writes
#                  base_hp/{src}__to__{src}.json and fans it out to the pair
#                  names. Then run materialise_baselines.sh.
#    PHASE=2       UDA methods, base HPs read back from base_hp/. As a job
#                  array, one run per task: 6 pairs x |METHODS|, src-major.
#    PHASE=oracle  joint source+target training (6 pairs, looped), in the
#                  separate unet_seed<N>_oracle/ tree.
#
#  Early stopping as in every existing config.json: PATIENCE (phase 2) and
#  PATIENCE_P1 default to 5, PATIENCE_ORACLE to 8.
#
#  Usage (launch_sbatch.sh chains all three per seed):
#    SEED=42 PHASE=1 sbatch --time 72:00:00 scripts/run_exp_unet.sh
#    SEED=42 PHASE=2 sbatch --array=0-29 scripts/run_exp_unet.sh
#    SEED=42 PHASE=oracle sbatch --time 72:00:00 scripts/run_exp_unet.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name uda_exp
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

# Default: one A100 on the gpu partition, half a node, so two runs share a
# node. <= 12 h keeps the job in the gpu-normal QOS (6 GPUs per user); a
# GH200 epoch takes ~440 s, so even 25 epochs fit on an A100 unless it is
# more than ~4x slower. Phase 1 and oracle loop over several runs and need
# the long limit: pass --time 72:00:00 on the sbatch line.
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

# --- Configuration --------------------------------------------------------
CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
# dl-torch mamba env on x86 nodes, the GH200 container on gpu-gh.
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
# Seed. Every run of every phase is tagged by it, and each seed gets its OWN
# output tree, so repeated seeds never overwrite one another's checkpoints,
# base_hp, best_hp or oracle directories. Launch one job per seed.
SEED="${SEED:-42}"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
# OUTPUT_SUFFIX sends a run to a side tree, e.g. _a100check for the hardware
# calibration, so it cannot collide with the checkpoints in unet_seed<N>/.
OUTPUT_DIR="${OUTPUT_ROOT}/unet_seed${SEED}${OUTPUT_SUFFIX:-}"
DATA_FORMAT="npy"

PHASE="${PHASE:-1}"
echo "Selected PHASE: ${PHASE}  SEED: ${SEED}"
echo "Output tree:    ${OUTPUT_DIR}"

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
METHODS=("fda" "spectral" "adabn" "dann" "mmd") # done: "joint_ot"
# Space-separated override from the environment, e.g. METHODS_ENV="joint_ot".
if [[ -n "${METHODS_ENV:-}" ]]; then
    read -ra METHODS <<< "${METHODS_ENV}"
fi
# Optional restriction to one pair, e.g. PAIR_ENV="europe_west|horn-of-africa".
PAIR_ENV="${PAIR_ENV:-}"
# Dropped: "coral" (subsumed by MMD, mean-blind, weakly scaled) and
# "mmd_ms" (same mechanism as MMD, unstable). joint_ot = DeepJDOT, the
# only method aligning the JOINT distribution and so the only one that
# can address conditional shift.

EPOCHS=25
# Every phase-2 config.json says 5 and every oracle 8; the old default of -1
# (no early stopping) applied only to hand-submitted runs and matched none.
PATIENCE="${PATIENCE:-5}"
PATIENCE_ORACLE="${PATIENCE_ORACLE:-8}"
NUM_WORKERS=12
BATCH_SIZE=128

FDA_BETA=0.01
LAMBDA_UDA=0.1
# DANN uses a separate, small adversarial weight (the GRL alpha already
# ramps the reversal 0->1). ~5e-4 per Wang et al. 2026; a large value
# collapses the discriminator to ln(2). Ignored by non-DANN methods.
DANN_WEIGHT=5e-4

# --------------------------------------------------------------------------


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
# ===========================================================================
# A run directory holding best.pt but no completion marker (best_hp/ or
# base_hp/ JSON, or config.json, all written only after training ends) is a
# run that died part-way: time limit, OOM, node drain. Retraining would
# overwrite that checkpoint, so stop instead (CLAUDE.md rules 4 and 7).
refuse_partial() {
    if [[ -e "$1/best.pt" ]]; then
        echo "ERROR: $1 holds a checkpoint from an unfinished run."
        echo "       Move it to a _trash/ folder in the same tree, then rerun."
        exit 1
    fi
}

if [[ "${PHASE}" == "1" ]]; then
    PATIENCE_P1="${PATIENCE_P1:-5}"

    echo "=== PHASE 1: source-only baselines (${#SOURCE_REGIONS[@]} runs) ==="
    echo "    patience=${PATIENCE_P1}"

    for i in "${!SOURCE_REGIONS[@]}"; do
        src="${SOURCE_REGIONS[$i]}"
        echo "--- [$((i+1))/${#SOURCE_REGIONS[@]}] source=${src} ---"

        HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${src}.json"
        if [[ -f "${HP_FILE}" ]]; then
            echo "  Base HPs already exist: ${HP_FILE}, skipping training."
        else
            refuse_partial "${OUTPUT_DIR}/${src}__to__${src}__none"
            run_python "${CODE_ROOT}/train_unet.py" \
                --source_path "${DATA_ROOT}/${src}" \
                --target_path "${DATA_ROOT}/${src}" \
                --output_dir  "${OUTPUT_DIR}" \
                --data_format "${DATA_FORMAT}" \
                --uda_method  none \
                --epochs      "${EPOCHS}" \
                --batch_size  "${BATCH_SIZE}" \
                --patience    "${PATIENCE_P1}" \
                --num_workers "${NUM_WORKERS}" \
                --seed        "${SEED}" \
                2>&1 | tee "${OUTPUT_DIR}/phase1_${src}.log"
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

# ===========================================================================
#  PHASE 2: UDA application (Fixed HPs, no Optuna)
# ===========================================================================
elif [[ "${PHASE}" == "2" ]]; then
    RUNS=()
    for src in "${SOURCE_REGIONS[@]}"; do
        for tgt in "${TARGET_REGIONS[@]}"; do
            [[ "$src" == "$tgt" ]] && continue
            [[ -n "${PAIR_ENV}" && "${src}|${tgt}" != "${PAIR_ENV}" ]] && continue

            HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
            # A side tree (OUTPUT_SUFFIX) reuses the seed's own phase-1 HPs.
            MAIN_HP="${OUTPUT_ROOT}/unet_seed${SEED}/base_hp/${src}__to__${tgt}.json"
            if [[ ! -f "${HP_FILE}" && -n "${OUTPUT_SUFFIX:-}" && -f "${MAIN_HP}" ]]; then
                cp "${MAIN_HP}" "${HP_FILE}"
            fi
            # Abort rather than skip: a skipped pair shifts every later array
            # index onto the wrong run.
            if [[ ! -f "${HP_FILE}" ]]; then
                echo "ERROR: missing base HPs ${HP_FILE}; run PHASE=1 first."
                exit 1
            fi

            for method in "${METHODS[@]}"; do
                RUNS+=("${src}|${tgt}|${method}")
            done
        done
    done

    echo "=== PHASE 2: Fixed UDA training (${#RUNS[@]} runs: ${METHODS[*]}), patience=${PATIENCE} ==="

    # As a job array, each task trains exactly one run, RUNS[task id]. The
    # order is fixed (source, target, method), so --array=0-$((N-1)) with
    # N = 6 pairs x |METHODS| covers every run once.
    INDICES=("${!RUNS[@]}")
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        (( SLURM_ARRAY_TASK_ID < ${#RUNS[@]} )) || {
            echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#RUNS[@]} runs"; exit 1; }
        INDICES=("${SLURM_ARRAY_TASK_ID}")
    fi

    for i in "${INDICES[@]}"; do
        IFS='|' read -r src tgt method <<< "${RUNS[$i]}"
        echo "--- [$((i+1))/${#RUNS[@]}] ${src} -> ${tgt} | ${method} ---"

        BEST_FILE="${OUTPUT_DIR}/best_hp/${src}__to__${tgt}__${method}.json"
        if [[ -f "${BEST_FILE}" ]]; then
            echo "  Run completed previously, skipping."
            continue
        fi
        refuse_partial "${OUTPUT_DIR}/${src}__to__${tgt}__${method}"

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
            --seed        "${SEED}" \
            --jdot_reg    0.001 \
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
            [[ -n "${PAIR_ENV}" && "${src}|${tgt}" != "${PAIR_ENV}" ]] && continue
            PAIRS+=("${src}|${tgt}")
        done
    done

    echo "=== PHASE oracle: joint source+target training ==="
    echo "Domain pairs: ${#PAIRS[@]}   patience: ${PATIENCE_ORACLE}   output: ${ORACLE_DIR}"

    for i in "${!PAIRS[@]}"; do
        IFS='|' read -r src tgt <<< "${PAIRS[$i]}"
        echo "--- [$((i+1))/${#PAIRS[@]}] ${src} + ${tgt} (joint) ---"

        # Oracle completion marker: config.json, written only after training
        # ends. best.pt alone is written from the first epoch on, so an oracle
        # killed part-way used to be skipped as if complete.
        RUN_DIR="${ORACLE_DIR}/${src}__to__${tgt}__none"
        if [[ -f "${RUN_DIR}/config.json" ]]; then
            echo "  Oracle run complete, skipping."
            continue
        fi
        refuse_partial "${RUN_DIR}"

        # The Phase 1 base HPs for this pair, so lr/bs/wd match the
        # source-only baseline (fair oracle comparison). Without them the
        # oracle would silently train on the argparse defaults.
        HP_FILE="${OUTPUT_DIR}/base_hp/${src}__to__${tgt}.json"
        [[ -f "${HP_FILE}" ]] || { echo "ERROR: missing ${HP_FILE}; run PHASE=1 first."; exit 1; }
        cp "${HP_FILE}" "${ORACLE_DIR}/base_hp/${src}__to__${tgt}.json"

        run_python "${CODE_ROOT}/train_unet.py" \
            --source_path "${DATA_ROOT}/${src}" \
            --target_path "${DATA_ROOT}/${tgt}" \
            --output_dir  "${ORACLE_DIR}" \
            --data_format "${DATA_FORMAT}" \
            --uda_method  none \
            --joint_training \
            --epochs      "${EPOCHS}" \
            --batch_size  "${BATCH_SIZE}" \
            --patience    "${PATIENCE_ORACLE}" \
            --num_workers "${NUM_WORKERS}" \
            --seed        "${SEED}" \
            2>&1 | tee "${ORACLE_DIR}/oracle_${src}__to__${tgt}.log"
    done
    echo "=== PHASE oracle complete ==="

else
    echo "ERROR: PHASE must be 1, 2, or oracle (got: ${PHASE})"
    exit 1
fi
