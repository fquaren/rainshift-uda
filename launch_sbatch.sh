#!/bin/bash -l
# ===========================================================================
#  launch_sbatch.sh -- submit the whole UNet pipeline, for several seeds.
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
#      4. phase 2   UDA methods                (pairs x |METHODS|, array)
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
#    ./launch_sbatch.sh                    # seeds 42 43 44, all stages
#    SEEDS="42" ./launch_sbatch.sh         # one seed
#    STAGES="4 5" SEEDS="42" ./launch_sbatch.sh      # resume partway
#    PAIR_ENV="europe_west|horn-of-africa" ./launch_sbatch.sh   # one pair
#    DRY=1 ./launch_sbatch.sh              # print the plan, submit nothing
# ===========================================================================
set -euo pipefail

SEEDS="${SEEDS:-42 43 44}"
STAGES="${STAGES:-1 2 3 4 5}"
# Absolute: the --wrap job runs from --chdir, not from here.
SCRIPTS="${SCRIPTS:-/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda/scripts}"

# Early stopping. Phase 1 and 2 plateau by epoch 3-6; the oracle has two
# domains to fit, so it gets more slack. Phase 1 reads PATIENCE_P1, phase 2
# PATIENCE and the oracle PATIENCE_ORACLE (run_exp_unet.sh).
PATIENCE_TRAIN="${PATIENCE_TRAIN:-5}"
PATIENCE_ORACLE="${PATIENCE_ORACLE:-8}"

# Phase 2 runs as a job array, one run per task. RUNS is built exactly as in
# run_exp_unet.sh (src-major pairs, optionally one PAIR_ENV, x METHODS), so
# the array covers every run once. run_exp_unet.sh aborts if a pair's
# base_hp is missing rather than shifting the indices.
DOMAINS=("europe_west" "horn-of-africa" "melanesia")
PAIR_ENV="${PAIR_ENV:-}"
METHODS="${METHODS:-fda spectral adabn dann mmd}"
read -ra _methods <<< "${METHODS}"
N_PAIRS=0
for src in "${DOMAINS[@]}"; do
    for tgt in "${DOMAINS[@]}"; do
        [[ "$src" == "$tgt" ]] && continue
        [[ -n "${PAIR_ENV}" && "${src}|${tgt}" != "${PAIR_ENV}" ]] && continue
        N_PAIRS=$(( N_PAIRS + 1 ))
    done
done
(( N_PAIRS > 0 )) || { echo "ERROR: PAIR_ENV='${PAIR_ENV}' matches no pair."; exit 1; }
N_PHASE2=$(( N_PAIRS * ${#_methods[@]} ))

# Same account and log locations as the #SBATCH headers of the job scripts.
SB_COMMON=(--account tbeucler_downscaling --chdir /scratch/fquareng/
           --output "outputs/%j" --error "job_errors/%j")

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
echo "phase 2: ${N_PAIRS} pair(s) x ${#_methods[@]} method(s) = ${N_PHASE2} runs${PAIR_ENV:+ (PAIR_ENV=${PAIR_ENV})}"
echo

for SEED in ${SEEDS}; do
    echo "=== seed ${SEED} ==="
    dep=""          # dependency flag carried from the previous stage

    if has_stage 1; then
        jid=$(submit "phase1  (baselines)" \
              env SEED="${SEED}" PATIENCE_P1="${PATIENCE_TRAIN}" PHASE=1 \
              sbatch --parsable --time 72:00:00 ${dep} "${SCRIPTS}/run_exp_unet.sh")
        dep="--dependency=afterok:${jid}"
    fi

    if has_stage 2; then
        # Not a GPU job: a few file copies. Wrapped in sbatch anyway so it can
        # sit in the dependency chain rather than needing a manual step.
        jid=$(submit "materialise (pair-named baselines)" \
              sbatch --parsable ${dep} "${SB_COMMON[@]}" \
                     --job-name=matbase --time=00:10:00 --partition=cpu \
                     --wrap="SEED=${SEED} bash ${SCRIPTS}/materialise_baselines.sh")
        dep="--dependency=afterok:${jid}"
    fi

    if has_stage 3; then
        jid=$(submit "oracle  (joint training)" \
              env SEED="${SEED}" PATIENCE_ORACLE="${PATIENCE_ORACLE}" PHASE=oracle PAIR_ENV="${PAIR_ENV}" \
              sbatch --parsable --time 72:00:00 ${dep} "${SCRIPTS}/run_exp_unet.sh")
        dep="--dependency=afterok:${jid}"
    fi

    if has_stage 4; then
        # afterok on an array job id waits for every task to succeed.
        jid=$(submit "phase2  (UDA methods, array of ${N_PHASE2})" \
              env SEED="${SEED}" PATIENCE="${PATIENCE_TRAIN}" PHASE=2 METHODS_ENV="${METHODS}" PAIR_ENV="${PAIR_ENV}" \
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
