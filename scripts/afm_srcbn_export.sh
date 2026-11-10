#!/bin/bash -l
# ===========================================================================
#  Phase-2 AFM runs with source-only BatchNorm statistics -> side tree
#  afm_seed42_srcbn/ (afm_srcbn_export.py; PLAN.md "Read 2026-11-09").
#  One GPU job, no training: per run ~3 min re-estimation + ~3 min check score.
#
#  Usage:  sbatch scripts/afm_srcbn_export.sh
#  Smoke:  SMOKE=1 sbatch --time 00:30:00 scripts/afm_srcbn_export.sh
#          (first run only, 4 BN batches, 2 check batches, _srcbn_smoke tree)
#  Then score with the standard pipeline (12 tasks, ~2.2 h each):
#     EXP_SUFFIX=_srcbn OUT_SUFFIX=_srcbn SEED=42 MODEL=afm TREE=none \
#         N_RUNS=12 sbatch --array=0-11 scripts/eval_array.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name afm_srcbn
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --mem 200G
#SBATCH --time 03:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"
SEED=42
RUNS_ROOT="${OUTPUT_ROOT}/afm_seed${SEED}"
TREE="${OUTPUT_ROOT}/afm_seed${SEED}_srcbn"

# Bash arrays are whitespace-delimited: NO commas.
METHODS=("spectral" "joint_ot")
DOMAINS=("europe_west" "horn-of-africa" "melanesia")

RUNS=()
for m in "${METHODS[@]}"; do
    for s in "${DOMAINS[@]}"; do
        for t in "${DOMAINS[@]}"; do
            [[ "${s}" == "${t}" ]] && continue
            RUNS+=("${RUNS_ROOT}/afm_${s}__to__${t}__${m}")
        done
    done
done

EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
    TREE="${OUTPUT_ROOT}/afm_seed${SEED}_srcbn_smoke"
    RUNS=("${RUNS[0]}")
    EXTRA+=(--n_batches 4 --max_batches 2)
fi

for d in "${RUNS[@]}"; do
    [[ -f "${d}/best.pt" && -f "${d}/config.json" ]] || { echo "ERROR: ${d} incomplete"; exit 1; }
    [[ -e "${TREE}/$(basename "${d}")" ]] && { echo "ERROR: ${TREE}/$(basename "${d}") exists; not overwriting."; exit 1; }
done
mkdir -p "${TREE}"

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda

echo "=== source-BN export: ${#RUNS[@]} runs -> ${TREE} ==="
run_python "${CODE_ROOT}/afm_srcbn_export.py" --runs "${RUNS[@]}" --out_root "${TREE}" "${EXTRA[@]}"
echo "=== done: ${TREE} ==="
