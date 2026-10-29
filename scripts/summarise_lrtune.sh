#!/bin/bash
# ===========================================================================
#  One line per AFM LR-sweep run, from the history.csv files written by
#  train_afm.py. Text processing only (awk), so it is safe on the login node.
#
#  best_crps   lowest validation CRPS (standardised space) and its epoch
#  mse@best    encoder-mean validation MSE at that epoch
#  best_mse    lowest encoder-mean validation MSE and its epoch
#  last_crps   CRPS at the final epoch; well above best_crps means the run
#              was already getting worse inside the 12 epochs
#  s/epoch     mean wall-clock per epoch, training plus validation
#
#  Usage:  bash scripts/summarise_lrtune.sh [tree]
# ===========================================================================

TREE="${1:-/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/afm_lrtune_seed42}"

printf "%-16s %-18s %6s %-14s %9s %-14s %9s %8s\n" \
    domain config epochs best_crps mse@best best_mse last_crps s/epoch
for h in "${TREE}"/*/*/afm_*__none/history.csv; do
    [[ -f "$h" ]] || continue
    run=$(dirname "$(dirname "$h")")
    cfg=$(basename "$run")
    dom=$(basename "$(dirname "$run")")
    awk -F, -v dom="$dom" -v cfg="$cfg" '
        NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
        {
            n++; t += $c["time_s"]
            crps = $c["val_crps"]; mse = $c["val_mse"]; last = crps
            # awk reads "nan" as 0, so non-finite values are skipped by name;
            # a run that diverged shows nan rather than a perfect score.
            if (crps != "nan" && crps != "inf" && (bc == "" || crps + 0 < bc + 0)) { bc = crps; be = $c["epoch"]; mb = mse }
            if (mse != "nan" && mse != "inf" && (bm == "" || mse + 0 < bm + 0)) { bm = mse; me = $c["epoch"] }
        }
        END {
            b1 = (bc == "") ? "nan" : sprintf("%.4f@%s", bc, be)
            b2 = (bm == "") ? "nan" : sprintf("%.4f@%s", bm, me)
            m1 = (bc == "" || mb == "nan") ? "nan" : sprintf("%.4f", mb)
            printf "%-16s %-18s %6d %-14s %9s %-14s %9s %8.0f\n", dom, cfg, n, b1, m1, b2, last, t / n
        }' "$h"
done
