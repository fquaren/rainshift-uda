#!/bin/bash -l
# ===========================================================================
#  Precompute RainShift normalization stats — one domain per array task.
#
#  Streaming Welford accumulation with non-finite scrubbing, so it never
#  OOMs on the full target array and cannot produce a NaN stat from a few
#  bad IMERG pixels (the melanesia failure).
#
#  Usage (4 domains listed below -> indices 0-3):
#    sbatch --array=0-3 scripts/compute_stats.sh
#  Force overwrite of existing stats files:
#    FORCE=1 sbatch --array=0-3 scripts/compute_stats.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name stats
#SBATCH --output outputs/%A
#SBATCH --error  job_errors/%A

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 0
#SBATCH --time 02:00:00

set -euo pipefail

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0"

CONTAINER="/users/fquareng/singularity/dl_gh200.sif"
CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"

# Domains to compute, in a fixed order so the array index selects one.
DOMAINS=(
    "europe_west"
    "horn-of-africa"
    "melanesia"
    "blacksea"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" -ge "${#DOMAINS[@]}" ]]; then
    echo "TASK_ID ${TASK_ID} >= number of domains ${#DOMAINS[@]}; nothing to do."
    exit 0
fi
DOMAIN="${DOMAINS[$TASK_ID]}"

FORCE_FLAG=""
[[ -n "${FORCE:-}" ]] && FORCE_FLAG="--force"

echo "=== Computing stats [${TASK_ID}] ${DOMAIN} ${FORCE_FLAG} ==="

singularity exec --nv "${CONTAINER}" python "${CODE_ROOT}/data/compute_stats.py" \
    --domain_path "${DATA_ROOT}/${DOMAIN}" \
    ${FORCE_FLAG}

echo "=== Done: ${DOMAIN} ==="