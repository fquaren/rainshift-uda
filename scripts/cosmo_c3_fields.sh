#!/bin/bash -l
# ===========================================================================
#  COSMO C3 follow-up: wet fraction, maxima and above-training-max pixels on
#  the four large-gap cells (PLAN_COSMO.md C3 read 2026-11-17), which
#  results.csv does not carry. No training.
#
#  compare_fields.py once per (seed, target), targets british_isles and
#  scandinavia_south, seeds 42/43/44 = 6 calls in one job. Runs per call, on
#  the same target test samples (stride 10):
#    poland->T, spain_central->T      the large-gap cells
#    other wet region->T              no-gap control into the same target
#                                     (scandinavia_south<->british_isles)
#    T->T                             in-domain
#  summary.csv: mean, wet fraction, p99, max, mse_log_e01, fraction above the
#  run's source training maximum and above 100 mm/h, clamped rows.
#  Plot labels say "ERA5 tp" / "IMERG obs"; on COSMO they are the coarse tp
#  input and the 2.2 km COSMO target.
#
#  Outputs (all checked before any GPU work, each refused if it exists):
#    results_rainshift_uda/results/plots/compare_fields_cosmo_c3_seed<S>/<T>/
#
#  Usage:  sbatch scripts/cosmo_c3_fields.sh
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type FAIL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cosmo_c3f
#SBATCH --output outputs/%j
#SBATCH --error  job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 06:00:00

set -euo pipefail

CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/rainshift-uda"
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data_cosmo_pm10"
OUTPUT_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"

# Bash arrays are whitespace-delimited: NO commas.
SEEDS=(42 43 44)
TARGETS=(british_isles scandinavia_south)
control_for() { [[ "$1" == british_isles ]] && echo scandinavia_south || echo british_isles; }

# Check every input and output before starting anything.
for S in "${SEEDS[@]}"; do
    U="${OUTPUT_ROOT}/unet_cosmo_seed${S}"
    for T in "${TARGETS[@]}"; do
        for SRC in poland spain_central "$(control_for "$T")" "$T"; do
            R="${U}/${SRC}__to__${T}__none"
            [[ -f "${R}/best.pt" && -f "${R}/config.json" ]] || { echo "ERROR: ${R} incomplete"; exit 1; }
        done
        O="${OUTPUT_ROOT}/results/plots/compare_fields_cosmo_c3_seed${S}/${T}"
        [[ -e "${O}" ]] && { echo "ERROR: ${O} exists; not overwriting."; exit 1; }
    done
done

source "${CODE_ROOT}/scripts/python_env.sh"
require_cuda
cd "${CODE_ROOT}"

for S in "${SEEDS[@]}"; do
    U="${OUTPUT_ROOT}/unet_cosmo_seed${S}"
    for T in "${TARGETS[@]}"; do
        C="$(control_for "$T")"
        O="${OUTPUT_ROOT}/results/plots/compare_fields_cosmo_c3_seed${S}/${T}"
        echo "=== seed ${S} | target ${T} -> ${O} ==="
        run_python "${CODE_ROOT}/compare_fields.py" \
            --data_root "${DATA_ROOT}" --output_dir "${O}" \
            --source poland --target "${T}" --stride 10 \
            --run "poland=${U}/poland__to__${T}__none" \
            --run "spain_central=${U}/spain_central__to__${T}__none" \
            --run "${C}=${U}/${C}__to__${T}__none" \
            --run "in-domain=${U}/${T}__to__${T}__none"
        cat "${O}/summary.csv"
    done
done
echo "=== done ==="
