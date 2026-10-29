#!/bin/bash -l
# ===========================================================================
#  Per-checkpoint evaluation as a job array (PLAN.md 3.6 step 5): one
#  checkpoint per task, through `evaluate.py batch --only <tag>`. Each task
#  writes its CSV row to <out>/rows/; `evaluate.py merge` builds results.csv
#  once every task is done. evaluate.py skips a run that already has
#  metrics.json, so a re-submitted task never overwrites.
#
#    MODEL=afm TREE=none     afm_seed<N>/          all runs
#    MODEL=afm TREE=oracle   afm_seed<N>_oracle/   all runs
#      deterministic pass (mu: mse_std, mae_std, floored log MSE, training
#      split) -> results/seed<N>/afm/<TREE>/
#      probabilistic pass (M members, STEPS midpoint steps: crps_std,
#      spread_std, floored log CRPS, rank histograms; no training split)
#      -> results/seed<N>/afm_prob/<TREE>/
#    MODEL=unet TREE=diag    unet_seed<N>/ diagonal runs X__to__X__none
#      deterministic pass on the test splits -> results/seed<N>/unet_accept/
#      (mae_std, the point-forecast CRPS the AFM acceptance compares with;
#      a new directory, so the existing unet/none rows are untouched)
#
#  The task list is the sorted run directories holding best.pt. Print it,
#  and its length, without python (safe on the login node):
#      SEED=42 MODEL=afm TREE=none LIST=1 bash scripts/eval_array.sh
#  then submit with that length, which every task checks against the tree
#  so that a run added after submission cannot shift the indices:
#      SEED=42 MODEL=afm TREE=none N_RUNS=3 sbatch --array=0-2 scripts/eval_array.sh
#  and merge once all tasks are done (CPU, a few seconds):
#      sbatch --dependency=afterok:<array> --partition cpu --time 00:10:00 \
#          --mem 4G --account tbeucler_downscaling --chdir /scratch/fquareng \
#          --output outputs/%j --error job_errors/%j --wrap \
#          'source <CODE_ROOT>/scripts/python_env.sh; run_python <CODE_ROOT>/evaluate.py merge --output_dir <out>'
#  (one merge per output directory: afm/<TREE> and afm_prob/<TREE>, or
#  unet_accept).
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name evalarr
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 100G
#SBATCH --time 12:00:00

set -euo pipefail
# Byte-order sorting of the run directories, the same on every node.
export LC_ALL=C

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
SEED="${SEED:?SEED must be set}"
MODEL="${MODEL:?MODEL=afm or unet}"
TREE="${TREE:?TREE=none, oracle (afm) or diag (unet)}"
# Ensemble settings of PLAN.md 3.5 step 5 and 3.6 (10 steps within 1 % of 40).
N_ENSEMBLE="${N_ENSEMBLE:-8}"
STEPS="${STEPS:-10}"
RESULTS="${OUTPUT_ROOT}/results/seed${SEED}"

case "${MODEL}|${TREE}" in
    afm\|none)   EXP_ROOT="${OUTPUT_ROOT}/afm_seed${SEED}" ;;
    afm\|oracle) EXP_ROOT="${OUTPUT_ROOT}/afm_seed${SEED}_oracle" ;;
    unet\|diag)  EXP_ROOT="${OUTPUT_ROOT}/unet_seed${SEED}" ;;
    *) echo "ERROR: MODEL=${MODEL} TREE=${TREE} is not a known combination."; exit 1 ;;
esac
[[ -d "${EXP_ROOT}" ]] || { echo "ERROR: ${EXP_ROOT} does not exist."; exit 1; }

TAGS=()
for ckpt in "${EXP_ROOT}"/*/best.pt; do
    [[ -e "${ckpt}" ]] || continue
    tag="$(basename "$(dirname "${ckpt}")")"
    if [[ "${MODEL}" == "unet" ]]; then
        # Diagonal only: X__to__X__none.
        [[ "${tag}" =~ ^(.+)__to__(.+)__none$ && "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]] || continue
    else
        [[ "${tag}" == afm_* ]] || continue
    fi
    TAGS+=("${tag}")
done

if [[ -n "${LIST:-}" ]]; then
    for i in "${!TAGS[@]}"; do echo "${i} ${TAGS[$i]}"; done
    echo "N_RUNS=${#TAGS[@]}  (${EXP_ROOT})"
    exit 0
fi

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

N_RUNS="${N_RUNS:?pass N_RUNS from LIST=1}"
(( N_RUNS == ${#TAGS[@]} )) || {
    echo "ERROR: ${EXP_ROOT} has ${#TAGS[@]} runs, the submission expected ${N_RUNS}. Re-list and resubmit."; exit 1; }
TASK="${SLURM_ARRAY_TASK_ID:?submit with --array}"
(( TASK < N_RUNS )) || { echo "ERROR: task ${TASK} >= ${N_RUNS} runs"; exit 1; }
TAG="${TAGS[$TASK]}"
echo "SEED ${SEED}  ${MODEL}/${TREE}  task ${TASK}/${N_RUNS}: ${TAG}"

echo "=== test_evaluate_metrics.py ==="
run_python "${CODE_ROOT}/test_evaluate_metrics.py"

COMMON=(batch --exp_root "${EXP_ROOT}" --data_root "${DATA_ROOT}" --only "${TAG}"
        --input_transform none --transform_cache_dir "${RESULTS}/../transforms")

if [[ "${MODEL}" == "unet" ]]; then
    echo "=== deterministic, test splits -> unet_accept ==="
    run_python "${CODE_ROOT}/evaluate.py" "${COMMON[@]}" \
        --output_dir "${RESULTS}/unet_accept" --save_samples 0
else
    echo "=== deterministic (mu), with the training split -> afm/${TREE} ==="
    run_python "${CODE_ROOT}/evaluate.py" "${COMMON[@]}" \
        --output_dir "${RESULTS}/afm/${TREE}" --save_samples 10 --eval_train
    echo "=== probabilistic, M=${N_ENSEMBLE}, ${STEPS} steps -> afm_prob/${TREE} ==="
    run_python "${CODE_ROOT}/evaluate.py" "${COMMON[@]}" \
        --output_dir "${RESULTS}/afm_prob/${TREE}" --save_samples 5 \
        --n_ensemble "${N_ENSEMBLE}" --sample_steps "${STEPS}"
fi
echo "=== eval task ${TASK} (${TAG}) complete ==="
