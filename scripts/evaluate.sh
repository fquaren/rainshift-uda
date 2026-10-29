#!/bin/bash -l
# ===========================================================================
#  Batch evaluation of all trained models × all input transforms.
#
#  For each model (unet, afm), evaluates every FINISHED checkpoint
#  (config.json written, evaluate.py run_finished) under the transforms in
#  TRANSFORMS (none by default; qm_tp, qm_precip, qm_all available).
#
#  Results land in {RESULTS_DIR}/{model}/{transform}/results.csv
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name eval
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

# Default: one A100 on the gpu partition. To fall back to the GH200 node:
# sbatch --partition gpu-gh --mem 0 ...
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 200G
#SBATCH --time 24:00:00

# singularity is on PATH on the GPU nodes; load the module only if it is not.
command -v singularity >/dev/null || module load singularityce/4.1.0

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
# dl-torch mamba env on x86 nodes, the GH200 container on gpu-gh.
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda
OUTPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"

# One seed per job. Each seed reads its own trees (unet_seed<N>/,
# unet_seed<N>_oracle/) and writes its own results/seed<N>/. There is no
# --force: evaluate.py skips any run that already has metrics.json, so a
# re-run resumes instead of overwriting. The old unseeded unet/ tree and
# results/unet/ are never touched.
SEED="${SEED:?SEED must be set, e.g. SEED=42 sbatch scripts/evaluate.sh}"
UNET_DIR="${OUTPUT_DIR}/unet_seed${SEED}"
ORACLE_DIR="${OUTPUT_DIR}/unet_seed${SEED}_oracle"
AFM_DIR="${OUTPUT_DIR}/afm_seed${SEED}"
RESULTS_DIR="${OUTPUT_DIR}/results/seed${SEED}"
# Input-transform cache is seed-independent; share it across seeds.
CACHE_DIR="${OUTPUT_DIR}/results/transforms"

[[ -d "${UNET_DIR}" ]] || { echo "ERROR: ${UNET_DIR} does not exist."; exit 1; }
echo "SEED: ${SEED}   models: ${UNET_DIR}   results: ${RESULTS_DIR}"

# The RAW 1D Wasserstein cube of compute_shift.py (log for tp/cp/precip,
# physical otherwise; no z-score). compute_ngg.py divides it by the SOURCE
# std of normalization_stats.json per channel. The normalized/ cube z-scores
# each domain by its own stats and is no longer used (ngg_v2).
W1_MATRIX="${CODE_ROOT}/covariate_shift_analysis/raw/Wasserstein_1D_test.npy"
W1_META="${CODE_ROOT}/covariate_shift_analysis/raw/Wasserstein_1D_test_variables.json"
NGG_DIR="${RESULTS_DIR}/ngg_v3"

# NOTE: "oracle" is NOT a valid --input_transform. evaluate.py defines
# _TRANSFORM_CHOICES = [none, qm_tp, qm_precip, qm_all], so passing
# "oracle" made argparse abort the whole script on the second iteration.
# The oracle is a different CHECKPOINT DIRECTORY, not an input transform;
# it is evaluated separately below. `ot` was removed on 2026-10-29.
TRANSFORMS=("none") # "qm_precip" "qm_tp" "qm_all")

# AFM probabilistic settings
N_ENSEMBLE=16
SAMPLE_STEPS=20

# -------------------------------------------------------------------
#  UNet: deterministic evaluation across all transforms
# -------------------------------------------------------------------
for tf in "${TRANSFORMS[@]}"; do
    echo "=== UNet | transform: ${tf} ==="
    run_python "${CODE_ROOT}/evaluate.py" batch \
        --exp_root "${UNET_DIR}" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${RESULTS_DIR}/unet/${tf}" \
        --input_transform "${tf}" \
        --transform_cache_dir "${CACHE_DIR}" \
        --save_samples 10 \
        --eval_train
    echo ""
done

# -------------------------------------------------------------------
#  ORACLE (joint source+target training) evaluation.
#  The oracle checkpoints live in a SEPARATE output dir written by
#  `PHASE=oracle`, so they need their own --exp_root. They are ordinary
#  checkpoints: no input transform applies. Their target risk is the
#  addressable-budget denominator.
# -------------------------------------------------------------------
if [[ -d "${ORACLE_DIR}" ]]; then
    echo "=== UNet | ORACLE (joint) ==="
    run_python "${CODE_ROOT}/evaluate.py" batch \
        --exp_root "${ORACLE_DIR}" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${RESULTS_DIR}/unet/oracle" \
        --input_transform none \
        --transform_cache_dir "${CACHE_DIR}" \
        --save_samples 10 \
        --eval_train
    echo ""
else
    echo "WARNING: ${ORACLE_DIR} not found; skipping oracle eval."
fi

# -------------------------------------------------------------------
#  AFM deterministic: across all transforms
# -------------------------------------------------------------------
# run_exp_afm.sh writes afm_seed<N>/; this runs only once
# an afm_seed<N>/ tree exists.
if [[ -d "${AFM_DIR}" ]]; then
for tf in "${TRANSFORMS[@]}"; do
    echo "=== AFM (deterministic) | transform: ${tf} ==="
    run_python "${CODE_ROOT}/evaluate.py" batch \
        --exp_root "${AFM_DIR}" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${RESULTS_DIR}/afm/${tf}" \
        --input_transform "${tf}" \
        --transform_cache_dir "${CACHE_DIR}" \
        --save_samples 10 \
        --eval_train
    echo ""
done

# # -------------------------------------------------------------------
# #  AFM probabilistic: "none" only (avoid the 16-member cost per transform);
# #  add a QM transform here if needed.
# # -------------------------------------------------------------------
for tf in "none"; do
    echo "=== AFM (probabilistic, ${N_ENSEMBLE} members) | transform: ${tf} ==="
    run_python "${CODE_ROOT}/evaluate.py" batch \
        --exp_root "${AFM_DIR}" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${RESULTS_DIR}/afm_prob/${tf}" \
        --input_transform "${tf}" \
        --transform_cache_dir "${CACHE_DIR}" \
        --n_ensemble "${N_ENSEMBLE}" \
        --sample_steps "${SAMPLE_STEPS}" \
        --save_samples 5
    # No --eval_train here: an ensemble over the ~122k-sample training split
    # (x members x steps) per checkpoint does not fit the time limit, and the
    # deterministic pass above already supplies the training-split risk.
    echo ""
done
else
    echo "=== AFM: ${AFM_DIR} not found; skipping AFM eval. ==="
fi

# -------------------------------------------------------------------
#  AFM ORACLE: deterministic and probabilistic, no input transform.
#  Its target risk (MSE and CRPS) is the AFM addressable-budget
#  denominator, as unet/oracle is for the UNet arm.
# -------------------------------------------------------------------
if [[ -d "${AFM_DIR}_oracle" ]]; then
    echo "=== AFM | ORACLE (joint, deterministic) ==="
    run_python "${CODE_ROOT}/evaluate.py" batch \
        --exp_root "${AFM_DIR}_oracle" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${RESULTS_DIR}/afm/oracle" \
        --input_transform none \
        --transform_cache_dir "${CACHE_DIR}" \
        --save_samples 10 \
        --eval_train
    echo ""
    echo "=== AFM | ORACLE (joint, probabilistic, ${N_ENSEMBLE} members) ==="
    run_python "${CODE_ROOT}/evaluate.py" batch \
        --exp_root "${AFM_DIR}_oracle" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${RESULTS_DIR}/afm_prob/oracle" \
        --input_transform none \
        --transform_cache_dir "${CACHE_DIR}" \
        --n_ensemble "${N_ENSEMBLE}" \
        --sample_steps "${SAMPLE_STEPS}" \
        --save_samples 5
    echo ""
fi

# -------------------------------------------------------------------
#  The raw W1 cube is NOT regenerated here: compute_shift.py reads the NPY
#  conversion, which no longer exists. If the cube is missing, NGG is
#  skipped rather than computed from the own-stats normalized/ cube.
# -------------------------------------------------------------------
if [[ ! -f "${W1_MATRIX}" || ! -f "${W1_META}" ]]; then
    echo "=== NGG skipped: raw W1 cube ${W1_MATRIX} missing (cannot be regenerated without NPY) ==="
    echo "=== Evaluation complete (no NGG) ==="
    exit 0
fi

# -------------------------------------------------------------------
#  NGG Computation
#  E_S is the source-only in-domain risk for every method (--e_s_ref
#  baseline); W1 is in source-normalised space (--w1_space source_norm).
#  Output goes to ngg_v3/. ngg_v2/ (own-stats W1) and ngg/ (method's own
#  E_S) are left as they are. An existing ngg_v3/<model>_<tf> is never
#  rewritten: compute_ngg_all.py refuses, and it is skipped here.
# -------------------------------------------------------------------
echo "=== Computing NGG ==="
for model in "unet"; do # "afm"
    for tf in "${TRANSFORMS[@]}"; do
        if [[ -e "${NGG_DIR}/${model}_${tf}" ]]; then
            echo "--- NGG | ${NGG_DIR}/${model}_${tf} exists; not overwriting ---"
            continue
        fi
        echo "--- NGG | model: ${model} | transform: ${tf} ---"
        run_python "${CODE_ROOT}/compute_ngg_all.py" \
            --w1_matrix "${W1_MATRIX}" \
            --w1_variables "${W1_META}" \
            --w1_space source_norm \
            --stats_root "${DATA_ROOT}" \
            --results_csv "${RESULTS_DIR}/${model}/${tf}/results.csv" \
            --model "${model}" \
            --error_metric "mse_std" \
            --w1_agg "mean_inputs" \
            --e_s_ref "baseline" \
            --output_root "${NGG_DIR}/${model}_${tf}"
    done
done

echo "=== Evaluation & NGG complete ==="
