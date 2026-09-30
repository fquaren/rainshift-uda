#!/bin/bash -l
# ===========================================================================
#  launch_pipeline.sh -- submit the whole UNet pipeline, for several seeds.
#
#  This is a SUBMITTER, not a batch job: run it on the login node. It calls
#  sbatch and returns immediately; SLURM runs the stages.
#
#  Per seed the stages are chained with --dependency=afterok, so each starts
#  only if the previous succeeded:
#
#      1. phase 1   source-only baselines      (3 runs)
#      2. materialise  copy diagonal -> pair-named dirs, so that
#                      E_T(f_S) exists for the 6 off-diagonal cells
#      3. oracle    joint source+target        (6 runs)
#      4. phase 2   UDA methods                (6 pairs x |METHODS|)
#      5. evaluate  baselines + oracle + UDA, writes results/seed<N>/
#
#  Seeds are independent of one another and run in parallel; each writes to
#  its own tree, unet_seed<N>/ and unet_seed<N>_oracle/, so nothing collides.
#
#  Stage 2 matters and is easy to forget. Phase 1 trains ONE model per source
#  domain into {src}__to__{src}__none/, because the target is unused when
#  uda_method=none. evaluate.py reads the (source, target) pair from the
#  directory name, so without the pair-named copies it only ever scores each
#  model on its own domain and the transfer numerator is never computed.
#
#  Usage
#    ./launch_pipeline.sh                  # seeds 42 43 44, all stages
#    SEEDS="42" ./launch_pipeline.sh       # one seed
#    STAGES="4 5" SEEDS="42" ./launch_pipeline.sh    # resume partway
#    DRY=1 ./launch_pipeline.sh            # print the plan, submit nothing
# ===========================================================================
set -euo pipefail

SEEDS="${SEEDS:-42 43 44}"
STAGES="${STAGES:-1 2 3 4 5}"
SCRIPTS="${SCRIPTS:-scripts}"

# Early stopping. Phase 1 and 2 plateau by epoch 3-6; the oracle has two
# domains to fit, so it gets more slack.
PATIENCE_TRAIN="${PATIENCE_TRAIN:-5}"
PATIENCE_ORACLE="${PATIENCE_ORACLE:-8}"

# Phase 2 runs as a job array, one run per task: 6 pairs x |METHODS|.
METHODS="${METHODS:-fda spectral adabn dann mmd}"
read -ra _methods <<< "${METHODS}"
N_PHASE2=$(( 6 * ${#_methods[@]} ))

DRY="${DRY:-}"

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

# Submit and echo, or just echo under DRY. Returns the job id on stdout.
submit() {
    local desc="$1"; shift
    if [[ -n "${DRY}" ]]; then
        echo "    [dry] ${desc}: $*" >&2
        echo "000000"
        return
    fi
    local jid
    jid=$("$@" | tr -d '\n')
    echo "    ${desc} -> job ${jid}" >&2
    echo "${jid}"
}

echo "seeds:  ${SEEDS}"
echo "stages: ${STAGES}"
echo

for SEED in ${SEEDS}; do
    echo "=== seed ${SEED} ==="
    dep=""          # dependency flag carried from the previous stage

    if has_stage 1; then
        jid=$(submit "phase1  (baselines)" \
              env SEED="${SEED}" PATIENCE="${PATIENCE_TRAIN}" PHASE=1 \
              sbatch --parsable --time 72:00:00 ${dep} "${SCRIPTS}/run_exp_unet.sh")
        dep="--dependency=afterok:${jid}"
    fi

    if has_stage 2; then
        # Not a GPU job: a few file copies. Wrapped in sbatch anyway so it can
        # sit in the dependency chain rather than needing a manual step.
        jid=$(submit "materialise (pair-named baselines)" \
              sbatch --parsable ${dep} \
                     --job-name=matbase --time=00:10:00 --partition=cpu \
                     --wrap="SEED=${SEED} bash ${SCRIPTS}/materialise_baselines.sh")
        dep="--dependency=afterok:${jid}"
    fi

    if has_stage 3; then
        jid=$(submit "oracle  (joint training)" \
              env SEED="${SEED}" PATIENCE="${PATIENCE_ORACLE}" PHASE=oracle \
              sbatch --parsable --time 72:00:00 ${dep} "${SCRIPTS}/run_exp_unet.sh")
        dep="--dependency=afterok:${jid}"
    fi

    if has_stage 4; then
        # afterok on an array job id waits for every task to succeed.
        jid=$(submit "phase2  (UDA methods, array of ${N_PHASE2})" \
              env SEED="${SEED}" PATIENCE="${PATIENCE_TRAIN}" PHASE=2 METHODS_ENV="${METHODS}" \
              sbatch --parsable --array=0-$(( N_PHASE2 - 1 )) ${dep} "${SCRIPTS}/run_exp_unet.sh")
        dep="--dependency=afterok:${jid}"
    fi

    if has_stage 5; then
        jid=$(submit "evaluate" \
              env SEED="${SEED}" \
              sbatch --parsable ${dep} "${SCRIPTS}/evaluate.sh")
        dep="--dependency=afterok:${jid}"
    fi
    echo
done

cat <<'EOF'
Submitted. Monitor with:   squeue --me
Chained with afterok, so a failed stage leaves the rest pending as
DependencyNeverSatisfied rather than running on bad inputs -- cancel those
with scancel once you have fixed the cause.

Results land in  results/seed<N>/  , one tree per seed; aggregate across
seeds only after all of them have finished.
EOF
