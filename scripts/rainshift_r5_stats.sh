#!/bin/bash -l
# ===========================================================================
#  R5 (PLAN.md 2026-11-17): normalisation stats for northamerica-east and
#  west-africa, written into the overlay root data_rainshift_r5/ (symlinks to
#  the shared copy; nothing is written into raw_data/rainshift), plus target
#  checks (non-finite pixels, max, mean, wet fraction, time range) on all five
#  R5 regions. First recomputes europe_west and stops unless it matches the
#  cached file to 1e-6. CPU only.
#
#  Outputs (refused if they exist):
#    data_rainshift_r5/{northamerica-east,west-africa}/normalization_stats.json
#    results_rainshift_uda/results/r5/target_check.json
#
#  Usage:  sbatch scripts/rainshift_r5_stats.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name r5_stats
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 10:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
OV="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_rainshift_r5"
CHECK="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda/results/r5/target_check.json"
NEW=(northamerica-east west-africa)
ALL=(europe_west horn-of-africa melanesia northamerica-east west-africa)

[[ -e "${CHECK}" ]] && { echo "ERROR: ${CHECK} exists; not overwriting."; exit 1; }
for r in "${NEW[@]}"; do
    [[ -e "${OV}/${r}/normalization_stats.json" || -L "${OV}/${r}/normalization_stats.json" ]] && {
        echo "ERROR: ${OV}/${r}/normalization_stats.json exists"; exit 1; }
done
for r in "${ALL[@]}"; do
    [[ -d "${OV}/${r}/train_data_out.zarr" ]] || { echo "ERROR: ${OV}/${r} incomplete"; exit 1; }
done

source "${CODE_ROOT}/scripts/python_env.sh"
cd "${CODE_ROOT}"
run_python "${CODE_ROOT}/rainshift_r5_stats.py" \
    --overlay_root "${OV}" \
    --new_regions "${NEW[@]}" \
    --check_regions "${ALL[@]}" \
    --verify_region europe_west \
    --check_out "${CHECK}"
