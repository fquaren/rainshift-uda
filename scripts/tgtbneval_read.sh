#!/bin/bash -l
# ===========================================================================
#  Read of one BN-fixed MMD run (MILESTONES M2, input to the gate).
#  unet_seed<S>_tgtbneval/<SRC>__to__<TGT>__mmd is trained with
#  --target_bn eval: the target pass never touches BatchNorm, so the run is
#  scored AS TRAINED (CLAUDE.md), with no source-BN re-estimation.
#
#  Three steps, one GPU; every checkpoint is read, never written:
#    1. feature_probe.py --pair SRC TGT, plus this run as an extra
#       (eval_trained, train_batch). Recomputes source-only and the seed's
#       method rows of that pair on the same samples, for comparison.
#    2. compare_fields.py SRC -> TGT: source-only, MMD as trained (mixed BN),
#       MMD source BN, MMD target-BN-eval, oracle, in-domain TGT.
#       Wet fraction, max, above-training-max beside mse_log_e01.
#    3. evaluate.py batch --only <run> (source and target test).
#
#  Outputs (each refused if it exists, all checked before any GPU work):
#    results/seed<S>_tgtbneval/feature_probe_<SRC>__to__<TGT>/
#    results/plots/compare_fields_tgtbneval_seed<S>/<SRC>__to__<TGT>/
#    results/seed<S>_tgtbneval/unet/none/<SRC>__to__<TGT>__mmd/  (+ rows/)
#
#  Usage:  SEED=43 SRC=europe_west TGT=melanesia sbatch scripts/tgtbneval_read.sh
#  History: job 20460852 (seed 42, europe_west -> melanesia) ran the first
#  version, which wrote feature_probe/ and evaluated the whole tree into
#  unet/none/results.csv. Those outputs are kept as they are.
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name tgtbn_read
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 48G
#SBATCH --time 03:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"   # read-only

SEED="${SEED:?set SEED}"
SRC="${SRC:?set SRC}"
TGT="${TGT:?set TGT}"
P="${SRC}__to__${TGT}"
U="${OUTPUT_ROOT}/unet_seed${SEED}"
TREE="${U}_tgtbneval"
TAG="${P}__mmd"
RUN="${TREE}/${TAG}"

PROBE="${OUTPUT_ROOT}/results/seed${SEED}_tgtbneval/feature_probe_${P}"
CMPF="${OUTPUT_ROOT}/results/plots/compare_fields_tgtbneval_seed${SEED}/${P}"
RES="${OUTPUT_ROOT}/results/seed${SEED}_tgtbneval/unet/none"

[[ -f "${RUN}/best.pt" && -f "${RUN}/config.json" ]] || { echo "ERROR: ${RUN} incomplete"; exit 1; }
for o in "${PROBE}" "${CMPF}" "${RES}/${TAG}" "${RES}/rows/${TAG}.csv"; do
    [[ -e "${o}" ]] && { echo "ERROR: ${o} exists; not overwriting."; exit 1; }
done

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== 1/3 feature probe, ${P} -> ${PROBE} ==="
run_python "${CODE_ROOT}/feature_probe.py" \
    --data_root "${DATA_ROOT}" --exp_root "${OUTPUT_ROOT}" --seed "${SEED}" --out_dir "${PROBE}" \
    --pair "${SRC}" "${TGT}" \
    --extra "mmd_tgtbneval=${RUN}"

echo "=== 2/3 compare_fields, ${P} -> ${CMPF} ==="
run_python "${CODE_ROOT}/compare_fields.py" \
    --data_root "${DATA_ROOT}" --output_dir "${CMPF}" \
    --source "${SRC}" --target "${TGT}" --stride 10 \
    --run "source-only=${U}/${P}__none" \
    --run "MMD mixed BN=${U}/${P}__mmd" \
    --run "MMD src BN=${U}_srcbn/${P}__mmd" \
    --run "MMD tgt-BN eval=${RUN}" \
    --run "oracle=${U}_oracle/${P}__none" \
    --run "in-domain=${U}/${TGT}__to__${TGT}__none"

echo "=== 3/3 evaluate ${TAG} -> ${RES} ==="
run_python "${CODE_ROOT}/evaluate.py" batch \
    --exp_root "${TREE}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RES}" \
    --only "${TAG}" \
    --input_transform none \
    --transform_cache_dir "${OUTPUT_ROOT}/results/transforms" \
    --save_samples 0
echo "=== done ==="
