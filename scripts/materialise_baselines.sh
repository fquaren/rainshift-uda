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
# ===========================================================================
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda}"
MODEL_DIR="${OUTPUT_DIR}/unet"

DOMAINS=("europe_west" "horn-of-africa" "melanesia")

echo "=== materialising pair-named source-only checkpoints in ${MODEL_DIR} ==="

missing=0
for src in "${DOMAINS[@]}"; do
    diag="${MODEL_DIR}/${src}__to__${src}__none"
    if [[ ! -f "${diag}/best.pt" ]]; then
        echo "  ERROR: missing ${diag}/best.pt -- run PHASE=1 first."
        missing=1
        continue
    fi
    for tgt in "${DOMAINS[@]}"; do
        [[ "$src" == "$tgt" ]] && continue
        dst="${MODEL_DIR}/${src}__to__${tgt}__none"
        mkdir -p "${dst}"
        cp -f "${diag}/best.pt" "${dst}/best.pt"
        [[ -f "${diag}/config.json" ]] && cp -f "${diag}/config.json" "${dst}/config.json"
        # Remove any stale metrics.json so evaluate.py does not skip this dir.
        rm -f "${dst}/metrics.json"
        echo "  ${src} -> ${tgt}"
    done
done
[[ "${missing}" -eq 1 ]] && { echo "Aborting: incomplete Phase 1."; exit 1; }

echo
echo "=== verification ==="
n_diag=$(ls -d "${MODEL_DIR}"/*__to__*__none 2>/dev/null | wc -l)
echo "  checkpoint dirs found: ${n_diag} (expect 9 = 3 diagonal + 6 off-diagonal)"
for d in "${MODEL_DIR}"/*__to__*__none; do
    [[ -f "$d/best.pt" ]] || echo "  WARNING: $(basename "$d") has no best.pt"
done

# Byte-identity check: the off-diagonal copies must equal their diagonal source.
echo "  byte-identity of copies:"
for src in "${DOMAINS[@]}"; do
    ref="${MODEL_DIR}/${src}__to__${src}__none/best.pt"
    for tgt in "${DOMAINS[@]}"; do
        [[ "$src" == "$tgt" ]] && continue
        if cmp -s "${ref}" "${MODEL_DIR}/${src}__to__${tgt}__none/best.pt"; then
            echo "    OK  ${src}__to__${tgt}"
        else
            echo "    FAIL ${src}__to__${tgt} differs from diagonal"
        fi
    done
done

echo
echo "Done. Now run: sbatch scripts/evaluate.sh"
