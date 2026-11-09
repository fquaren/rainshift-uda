#!/bin/bash -l
# ===========================================================================
#  Recompute RainShift normalization stats — one domain per array task.
#
#  Streaming Welford accumulation with non-finite scrubbing, so it never
#  OOMs on the full target array and cannot produce a NaN stat from a few
#  bad IMERG pixels (the melanesia failure).
#
#  Writes ${OUT_ROOT}/${domain}.json in your own tree, never into the lab's
#  shared raw_data/rainshift copy (CLAUDE.md rule 3; compute_stats.py refuses
#  that too), and never over an existing file. Training keeps reading the
#  shared {domain}/normalization_stats.json; a recomputed file is only for
#  comparison with it. The old FORCE=1 overwrite of the shared stats is gone.
#
#  Usage (3 domains below -> indices 0-2):
#    sbatch --array=0-2 scripts/compute_stats.sh
#    OUT_ROOT=/path/in/your/tree sbatch --array=0-2 scripts/compute_stats.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name stats
#SBATCH --output outputs/%A_%a
#SBATCH --error  job_errors/%A_%a

# No GPU work: xarray and numpy only, so a CPU node with the x86 env.
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 04:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
# dl-torch mamba env on x86 nodes.
source "${CODE_ROOT}/scripts/python_env.sh"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"   # read-only
OUT_ROOT="${OUT_ROOT:-/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/normalization_stats_recomputed}"

# Domains to compute, in a fixed order so the array index selects one.
# blacksea is a dummy domain and is not computed.
DOMAINS=("europe_west" "horn-of-africa" "melanesia")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= ${#DOMAINS[@]} )); then
    echo "ERROR: array task ${TASK_ID} >= ${#DOMAINS[@]} domains"; exit 1
fi
DOMAIN="${DOMAINS[$TASK_ID]}"
OUT="${OUT_ROOT}/${DOMAIN}.json"

if [[ -e "${OUT}" ]]; then
    echo "ERROR: ${OUT} exists; not overwriting. Move it to a _trash/ folder first."; exit 1
fi

echo "=== Computing stats [${TASK_ID}] ${DOMAIN} -> ${OUT} ==="

run_python "${CODE_ROOT}/data/compute_stats.py" \
    --domain_path "${DATA_ROOT}/${DOMAIN}" \
    --out         "${OUT}"

echo "=== Done: ${DOMAIN} ==="
