#!/bin/bash -l
# ===========================================================================
#  Input ranges against the training range, in the units each model sees
#  (rainshift_input_ranges.py; PLAN.md R5 "Read 2026-11-24"). Hypothesis: the
#  west-africa in-domain UNet blows up because test-period inputs leave the
#  west-africa training range. CPU only, read only.
#
#  One array task per stats region; the first entry is that region's train
#  store, the rest are compared against it:
#    0 west-africa  : wafr train | wafr test (blows up), melanesia test
#                     (wafr->melanesia, blows up worst)
#    1 europe_west  : europe train | europe test (does not blow up), wafr test
#                     and melanesia test (both blow up)
#    2 horn         : horn train | horn test, wafr test (horn->wafr does not
#                     blow up: the contrast)
#    3 melanesia    : melanesia train | melanesia test (in-domain, ~no blow-up)
#
#  Writes results/r5/input_ranges${TAG}/<stats_region>/ and refuses to start
#  if it exists.
#
#  Usage:  sbatch --array=0-3 scripts/rainshift_input_ranges.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name r5_inrange
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16G
#SBATCH --time 08:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_rainshift_r5"   # overlay, read only

# Bash arrays are whitespace-delimited: NO commas.
STATS=(west-africa europe_west horn-of-africa melanesia)
ENTRIES=(
    "west-africa:train west-africa:test melanesia:test"
    "europe_west:train europe_west:test west-africa:test melanesia:test"
    "horn-of-africa:train horn-of-africa:test west-africa:test"
    "melanesia:train melanesia:test"
)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job: --array=0-3}"
(( TASK < ${#STATS[@]} )) || { echo "ERROR: task ${TASK} >= ${#STATS[@]}"; exit 1; }
REGION="${STATS[$TASK]}"
OUT="${OUTPUT_ROOT}/results/r5/input_ranges${TAG:-}/${REGION}"

if [[ -e "${OUT}" ]]; then
    echo "ERROR: ${OUT} exists; not overwriting."
    exit 1
fi
[[ -f "${DATA_ROOT}/${REGION}/normalization_stats.json" ]] || {
    echo "ERROR: ${DATA_ROOT}/${REGION}/normalization_stats.json missing"; exit 1; }

ARGS=()
for e in ${ENTRIES[$TASK]}; do ARGS+=(--entry "${e}"); done

source "${CODE_ROOT}/scripts/python_env.sh"

echo "=== stats ${REGION}: ${ENTRIES[$TASK]} -> ${OUT} ==="
cd "${CODE_ROOT}"
run_python "${CODE_ROOT}/rainshift_input_ranges.py" \
    --data_root "${DATA_ROOT}" --stats_region "${REGION}" --output_dir "${OUT}" "${ARGS[@]}"
echo "=== done: ${OUT} ==="
