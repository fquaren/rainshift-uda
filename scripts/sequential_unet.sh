#!/bin/bash
# submit_sequential.sh — submit one SLURM job per (src,tgt) pair, each
# depending on the previous, so they run strictly one at a time.
#
# Usage:  PHASE=1 ./submit_sequential.sh          (or PHASE=2 / PHASE=oracle)
# Requires: run_exp_unet.sh must be the JOB-ARRAY version, where
#           SLURM_ARRAY_TASK_ID selects the work item.
set -euo pipefail

PHASE="${PHASE:-1}"
LAUNCHER="${LAUNCHER:-scripts/run_exp_unet.sh}"

# Must match the domain order inside the launcher's build_pairs().
SOURCE_REGIONS=("europe_west" "horn-of-africa" "melanesia")
TARGET_REGIONS=("europe_west" "horn-of-africa" "melanesia")

# Reconstruct the same ordered pair list the launcher builds, to know indices.
PAIRS=()
for src in "${SOURCE_REGIONS[@]}"; do
    for tgt in "${TARGET_REGIONS[@]}"; do
        [[ "$src" == "$tgt" ]] && continue
        PAIRS+=("${src}|${tgt}")
    done
done

echo "PHASE=${PHASE}: submitting ${#PAIRS[@]} sequential jobs via ${LAUNCHER}"

prev_jobid=""
for idx in "${!PAIRS[@]}"; do
    IFS='|' read -r src tgt <<< "${PAIRS[$idx]}"

    dep_arg=""
    [[ -n "${prev_jobid}" ]] && dep_arg="--dependency=afterany:${prev_jobid}"

    # Submit this single item as a 1-element array (--array=${idx}).
    # --parsable makes sbatch print just the job id.
    jobid=$(PHASE="${PHASE}" sbatch --parsable ${dep_arg} \
        --array="${idx}" "${LAUNCHER}")

    echo "  [${idx}] ${src} -> ${tgt}  job=${jobid}  ${dep_arg:-<no dep>}"
    prev_jobid="${jobid}"
done

echo "Submitted. Chain runs strictly in order; check with: squeue -u \$USER"