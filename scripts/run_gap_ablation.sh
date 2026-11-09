#!/bin/bash -l
# ===========================================================================
#  GAP ablation, the falsification test for claim 3 (PLAN.md, Stage 2 c).
#  europe_west -> melanesia, seeds 42/43/44, four arms, one run per task:
#
#    C  tgtfwd   mmd, lambda 0. The target forward pass in train mode and
#                nothing else: what BatchNorm running-stat mixing alone does.
#    G  gap      unbiased MMD on GAP-pooled bottleneck features.
#    S  grid4    unbiased MMD on 4 x 4-grid mean/std bottleneck features.
#    M  grid4ms  unbiased MMD on grid4 features at enc2, enc3, enc4 and the
#                bottleneck, since the skip connections bypass the bottleneck.
#
#  G, S and M weight the MMD so that its encoder gradient is 0.5 x the task
#  gradient (--uda_grad_ratio), so the alignment is never inert. The seeded
#  reference mmd runs (biased, GAP, lambda 0.1) put ~0.2 % of the loss on it.
#  Every arm logs the unbiased bottleneck MMD under both poolings, with a
#  null, to history.csv. Base HPs are each seed's phase-1 ones; everything
#  else is as in run_exp_unet.sh phase 2. Each task then scores its
#  checkpoint with evaluate.py.
#
#  Task i: arm = ARMS[i / 3], seed = SEEDS[i % 3].
#  Writes gap_ablation/seed<N>/<arm>/ and results/gap_ablation/seed<N>/<arm>/;
#  refuses to run if the run directory exists.
#
#  Usage:  sbatch --array=0-11 scripts/run_gap_ablation.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name gap_ablation
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --exclude=dnagpu008
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --mem 200G
#SBATCH --time 12:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
SRC="europe_west"
TGT="melanesia"
GRAD_RATIO=0.5

# Bash arrays are whitespace-delimited: NO commas.
ARMS=("tgtfwd" "gap" "grid4" "grid4ms")
SEEDS=(42 43 44)

TASK="${SLURM_ARRAY_TASK_ID:?run as an array job: sbatch --array=0-11}"
(( TASK < ${#ARMS[@]} * ${#SEEDS[@]} )) || { echo "ERROR: task ${TASK} out of range"; exit 1; }
ARM="${ARMS[$(( TASK / ${#SEEDS[@]} ))]}"
SEED="${SEEDS[$(( TASK % ${#SEEDS[@]} ))]}"

case "${ARM}" in
    tgtfwd)  METHOD=mmd;    ARM_ARGS=(--lambda_uda 0) ;;
    gap)     METHOD=mmd;    ARM_ARGS=(--mmd_pool gap   --mmd_unbiased --uda_grad_ratio "${GRAD_RATIO}") ;;
    grid4)   METHOD=mmd;    ARM_ARGS=(--mmd_pool grid4 --mmd_unbiased --uda_grad_ratio "${GRAD_RATIO}") ;;
    grid4ms) METHOD=mmd_ms; ARM_ARGS=(--mmd_pool grid4 --mmd_unbiased --uda_grad_ratio "${GRAD_RATIO}"
                                      --mmd_levels enc2 enc3 enc4 bottleneck) ;;
esac

# SMOKE=1: 2 epochs on 8 chunks per split in separate _smoke trees, to run
# the training and evaluation path once, e.g. the heaviest arm:
#   SMOKE=1 sbatch --array=9 --time 01:00:00 scripts/run_gap_ablation.sh
TREE="gap_ablation"
EPOCHS=25
EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
    TREE="gap_ablation_smoke"
    EPOCHS=2
    EXTRA+=(--subset_chunks 8)
fi

OUT="${OUTPUT_ROOT}/${TREE}/seed${SEED}/${ARM}"
RES="${OUTPUT_ROOT}/results/${TREE}/seed${SEED}/${ARM}"
RUN_DIR="${OUT}/${SRC}__to__${TGT}__${METHOD}"
for p in "${RUN_DIR}" "${RES}"; do
    [[ -e "${p}" ]] && { echo "ERROR: ${p} exists; not overwriting."; exit 1; }
done
HP_SRC="${OUTPUT_ROOT}/unet_seed${SEED}/base_hp/${SRC}__to__${TGT}.json"
[[ -f "${HP_SRC}" ]] || { echo "ERROR: missing ${HP_SRC}"; exit 1; }
mkdir -p "${OUT}/base_hp"
cp "${HP_SRC}" "${OUT}/base_hp/"

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda
# The smoke run of grid4ms (65424882_9) ran out of memory at batch 128 on a
# 40 GB A100; the seeded mmd runs already logged allocator OOM warnings.
# Expandable segments reduce fragmentation; numerics are unchanged.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ARM_ARGS come last on the command line: argparse keeps the last value, so
# tgtfwd's --lambda_uda 0 overrides the 0.1 below.
echo "=== GAP ablation | task ${TASK} | arm ${ARM} | seed ${SEED} | ${SRC} -> ${TGT} | ${METHOD} ${ARM_ARGS[*]} ==="

run_python "${CODE_ROOT}/train_unet.py" \
    --source_path "${DATA_ROOT}/${SRC}" \
    --target_path "${DATA_ROOT}/${TGT}" \
    --output_dir  "${OUT}" \
    --data_format npy \
    --uda_method  "${METHOD}" \
    --lambda_uda  0.1 \
    --fda_beta    0.01 \
    --dann_weight 5e-4 \
    --epochs      "${EPOCHS}" \
    --batch_size  128 \
    --patience    5 \
    --num_workers 12 \
    --seed        "${SEED}" \
    --jdot_reg    0.001 \
    --uda_diagnostics \
    "${EXTRA[@]}" \
    "${ARM_ARGS[@]}" \
    2>&1 | tee "${OUT}/train.log"

echo "=== evaluate ==="
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${OUT}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RES}" \
    --input_transform none \
    --save_samples 0

echo "=== done: ${RES}/results.csv ==="
