#!/bin/bash -l
# ===========================================================================
#  Few-shot label curve (PLAN.md P16), one seed per submission, one run per
#  array task: fine-tune the source-only UNet of each off-diagonal pair on
#  N target chunks (finetune_unet.py: 500 steps at base LR / 5, no
#  selection, last.pt), then score last.pt on both test splits.
#
#  Tasks: pair-major, 6 pairs x 3 sizes = 18 (0-17).
#    pairs  europe->horn, europe->mel, horn->europe, horn->mel, mel->europe, mel->horn
#    sizes  N_CHUNKS = 1 10 100   (200, 2,000, 20,000 samples)
#  Output: unet_seed<S>_fewshot/{src}__to__{tgt}__fs<N>/ (refuses to overwrite)
#          results/seed<S>/unet_fewshot/ (rows/; merge with evaluate.py merge)
#
#  Usage:  SEED=42 SMOKE=1 sbatch --array=3 --time 00:30:00 scripts/run_fewshot.sh   # europe->mel fs1: melanesia is the NaN-prone target
#          SEED=42 sbatch --array=0-17 scripts/run_fewshot.sh
#  SMOKE=1: 20 steps into /scratch/fquareng/fewshot_smoke/<jobid>/, no scoring.
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name fewshot
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 100G
#SBATCH --time 03:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
SEED="${SEED:?SEED must be set}"

# Bash arrays are whitespace-delimited: NO commas.
DOMAINS=("europe_west" "horn-of-africa" "melanesia")
SIZES=(1 10 100)
PAIRS=()
for s in "${DOMAINS[@]}"; do for t in "${DOMAINS[@]}"; do
    [[ "$s" == "$t" ]] || PAIRS+=("${s}|${t}")
done; done
TASK="${SLURM_ARRAY_TASK_ID:?submit with --array}"
(( TASK < ${#PAIRS[@]} * ${#SIZES[@]} )) || { echo "ERROR: task ${TASK} out of range"; exit 1; }
IFS='|' read -r SRC TGT <<< "${PAIRS[$(( TASK / ${#SIZES[@]} ))]}"
N="${SIZES[$(( TASK % ${#SIZES[@]} ))]}"
TAG="${SRC}__to__${TGT}__fs${N}"
INIT_DIR="${OUTPUT_ROOT}/unet_seed${SEED}"

if [[ "${SMOKE:-0}" == "1" ]]; then
    OUT_DIR="/scratch/fquareng/fewshot_smoke/${SLURM_JOB_ID}"
    STEPS=20
else
    OUT_DIR="${OUTPUT_ROOT}/unet_seed${SEED}_fewshot"
    STEPS=500
fi
echo "SEED ${SEED}  task ${TASK}: ${TAG}  (${STEPS} steps) -> ${OUT_DIR}"
[[ -e "${OUT_DIR}/${TAG}" ]] && { echo "ERROR: ${OUT_DIR}/${TAG} exists. Not overwriting."; exit 1; }
mkdir -p "${OUT_DIR}"

run_python "${CODE_ROOT}/finetune_unet.py" \
    --source_path "${DATA_ROOT}/${SRC}" \
    --target_path "${DATA_ROOT}/${TGT}" \
    --init_dir    "${INIT_DIR}" \
    --output_dir  "${OUT_DIR}" \
    --n_chunks    "${N}" \
    --steps       "${STEPS}" \
    --seed        "${SEED}"

[[ "${SMOKE:-0}" == "1" ]] && { echo "=== smoke: training only, done ==="; exit 0; }

run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${OUT_DIR}" --data_root "${DATA_ROOT}" \
    --only "${TAG}" --ckpt_name last.pt \
    --output_dir "${OUTPUT_ROOT}/results/seed${SEED}/unet_fewshot" \
    --input_transform none --save_samples 0
echo "=== few-shot task ${TASK} (${TAG}) complete ==="
