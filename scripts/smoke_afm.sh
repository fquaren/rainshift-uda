#!/bin/bash -l
# ===========================================================================
#  AFM smoke test: every phase end to end, small, in a side tree.
#
#  2 epochs on SUBSET_CHUNKS chunks per split (200 samples each), at the FULL
#  batch size, so the peak-memory line in each training log answers whether
#  AFM fits on a 40 GB A100. Exercises:
#    phase 1 (3 sources) -> materialise -> phase 2 on one pair with the two
#    heaviest methods (joint_ot, dann; two forwards each) and adabn (the
#    best_adabn.pt path) -> oracle on that pair -> evaluate.py on 4 of the
#    checkpoints with a 2-member ensemble (sample(), CRPS, AFM tag parsing).
#
#  Writes only to new paths: afm_seed42_smoke/, afm_seed42_smoke_oracle/,
#  results/smoke_afm_seed42/. Refuses to run if any of them exists.
#
#  Usage:  sbatch scripts/smoke_afm.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name afm_smoke
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --mem 200G
#SBATCH --time 06:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

export SEED=42 OUTPUT_SUFFIX=_smoke EPOCHS=2 SUBSET_CHUNKS="${SUBSET_CHUNKS:-8}"
TREE="${OUTPUT_ROOT}/afm_seed${SEED}${OUTPUT_SUFFIX}"
RES="${OUTPUT_ROOT}/results/smoke_afm_seed${SEED}"
PAIR="europe_west|horn-of-africa"

for p in "${TREE}" "${TREE}_oracle" "${RES}"; do
    [[ -e "${p}" ]] && { echo "ERROR: ${p} exists; not overwriting."; exit 1; }
done

echo "=== 1/5 phase 1 ==="
PHASE=1 bash "${CODE_ROOT}/scripts/run_exp_afm.sh"

echo "=== 2/5 materialise ==="
MODEL=afm bash "${CODE_ROOT}/scripts/materialise_baselines.sh"

echo "=== 3/5 phase 2: joint_ot dann adabn on ${PAIR} ==="
PHASE=2 METHODS_ENV="joint_ot dann adabn" PAIR_ENV="${PAIR}" bash "${CODE_ROOT}/scripts/run_exp_afm.sh"

echo "=== 4/5 oracle on ${PAIR} ==="
PHASE=oracle PAIR_ENV="${PAIR}" bash "${CODE_ROOT}/scripts/run_exp_afm.sh"

# The test split is never subset, so scoring all 12 checkpoints with an
# ensemble would take hours. Score three representative runs through a
# directory of symlinks (evaluate.py globs */best.pt and follows them), plus
# the oracle, with the smallest ensemble that still gives a CRPS.
echo "=== 5/5 evaluate (2 members, 2 steps) ==="
EVALSET="${TREE}/_evalset"
mkdir -p "${EVALSET}"
for m in none joint_ot adabn; do
    ln -s "${TREE}/afm_europe_west__to__horn-of-africa__${m}" "${EVALSET}/afm_europe_west__to__horn-of-africa__${m}"
done
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${EVALSET}" --data_root "${DATA_ROOT}" \
    --output_dir "${RES}/afm/none" --input_transform none \
    --n_ensemble 2 --sample_steps 2 --save_samples 2
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${TREE}_oracle" --data_root "${DATA_ROOT}" \
    --output_dir "${RES}/afm/oracle" --input_transform none \
    --n_ensemble 2 --sample_steps 2 --save_samples 2

echo "=== peak GPU memory per run ==="
grep -h "peak GPU memory" "${TREE}"/*.log "${TREE}_oracle"/*.log || true
echo "=== results ==="
cut -d, -f2-5 "${RES}/afm/none/results.csv" "${RES}/afm/oracle/results.csv"
echo "=== AFM smoke test complete ==="
