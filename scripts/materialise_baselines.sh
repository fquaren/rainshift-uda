#!/bin/bash -l
# ===========================================================================
#  Materialise pair-named source-only checkpoints, then verify.
#
#  Phase 1 trains one model per SOURCE and writes it to
#      {src}__to__{src}__none/
#  because --target_path is set to the source (the target is unused when
#  uda_method=none). evaluate.py, however, parses (source, target) from the
#  DIRECTORY NAME and evaluates on whatever target it finds there -- so with
#  only the diagonal dirs present it produced 3 rows, all src==tgt, and
#  E_T(f_S) (source-only error on the real targets, the numerator of every
#  transfer gap) was never computed.
#
#  The weights are target-independent, so copying the diagonal checkpoint to
#  each pair name is exact, not an approximation.
#
#  Run this AFTER Phase 1 finishes and BEFORE evaluate.sh.
#
#  Works on ONE seed's tree, ${MODEL}_seed${SEED}/ (MODEL=unet default, or afm), which is where
#  run_exp_unet.sh / run_exp_afm.sh write. SEED is required: the old default wrote into the
#  unseeded unet/ tree for every seed. An existing pair-named directory is
#  never overwritten -- identical copies are skipped, anything else aborts.
# ===========================================================================
set -euo pipefail

SEED="${SEED:?SEED must be set, e.g. SEED=42}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda}"
# MODEL=afm handles the AFM tree, whose tags are afm_{src}__to__{tgt}__none.
MODEL="${MODEL:-unet}"
case "${MODEL}" in
    unet) PFX="" ;;
    afm)  PFX="afm_" ;;
    *) echo "ERROR: MODEL must be unet or afm"; exit 1 ;;
esac
# OUTPUT_SUFFIX selects a side tree, e.g. _smoke.
MODEL_DIR="${OUTPUT_DIR}/${MODEL}_seed${SEED}${OUTPUT_SUFFIX:-}"
run_dir() { echo "${MODEL_DIR}/${PFX}$1__to__$2__none"; }

DOMAINS=("europe_west" "horn-of-africa" "melanesia")

[[ -d "${MODEL_DIR}" ]] || { echo "ERROR: ${MODEL_DIR} does not exist."; exit 1; }
echo "=== materialising pair-named source-only checkpoints in ${MODEL_DIR} ==="

missing=0
for src in "${DOMAINS[@]}"; do
    diag="$(run_dir "${src}" "${src}")"
    if [[ ! -f "${diag}/best.pt" ]]; then
        echo "  ERROR: missing ${diag}/best.pt -- run PHASE=1 first."
        missing=1
        continue
    fi
    # The diagonal must be a source-only run, never an oracle.
    if ! grep -q '"joint_training": false' "${diag}/config.json"; then
        echo "  ERROR: ${diag}/config.json is not a source-only run."
        missing=1
        continue
    fi
    for tgt in "${DOMAINS[@]}"; do
        [[ "$src" == "$tgt" ]] && continue
        dst="$(run_dir "${src}" "${tgt}")"
        if [[ -e "${dst}" ]]; then
            if cmp -s "${diag}/best.pt" "${dst}/best.pt"; then
                echo "  ${src} -> ${tgt}  (already present, identical, skipped)"
                continue
            fi
            echo "  ERROR: ${dst} exists and differs from the diagonal. Not overwriting."
            exit 1
        fi
        mkdir -p "${dst}"
        cp "${diag}/best.pt" "${dst}/best.pt"
        cp "${diag}/config.json" "${dst}/config.json"
        echo "  ${src} -> ${tgt}"
    done
done
[[ "${missing}" -eq 1 ]] && { echo "Aborting: incomplete Phase 1."; exit 1; }

echo
echo "=== verification ==="
n_diag=$(ls -d "${MODEL_DIR}"/${PFX}*__to__*__none 2>/dev/null | wc -l)
echo "  checkpoint dirs found: ${n_diag} (expect 9 = 3 diagonal + 6 off-diagonal)"
for d in "${MODEL_DIR}"/${PFX}*__to__*__none; do
    [[ -f "$d/best.pt" ]] || echo "  WARNING: $(basename "$d") has no best.pt"
done

# Byte-identity check: the off-diagonal copies must equal their diagonal source.
echo "  byte-identity of copies:"
fail=0
for src in "${DOMAINS[@]}"; do
    ref="$(run_dir "${src}" "${src}")/best.pt"
    for tgt in "${DOMAINS[@]}"; do
        [[ "$src" == "$tgt" ]] && continue
        if cmp -s "${ref}" "$(run_dir "${src}" "${tgt}")/best.pt"; then
            echo "    OK  ${src}__to__${tgt}"
        else
            echo "    FAIL ${src}__to__${tgt} differs from diagonal"
            fail=1
        fi
    done
done
# Non-zero exit so an afterok-chained evaluate job does not start.
[[ "${fail}" -eq 1 ]] && { echo "Aborting: byte-identity check failed."; exit 1; }

echo
echo "Done. Now run: SEED=${SEED} sbatch scripts/evaluate.sh"
