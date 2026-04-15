#!/bin/bash -l

# ===========================================================================
#  RainShift UDA experiment grid
#
#  Runs all (source, target, method) combinations.
#  Adjust REGIONS, DATA_ROOT, and GPU settings below.
# ===========================================================================

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type NONE
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name convert
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH --mem 350G
#SBATCH --time 72:00:00

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0" 
container_path="/users/fquareng/singularity/dl_gh200.sif"

set -euo pipefail

# --- Configuration --------------------------------------------------------
DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift"
CODE_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR"
OUTPUT_DIR="/scratch/fquareng/rainshift_uda/unet"

# RainShift regions
# REGIONS=(
#   "amazon-basin" 
#   "arabian-peninsula" 
#   "australasia-east"
#   "blacksea" 
#   "cape-horn" 
#   "caribbean" 
#   "east-asia-north-east"
#   "east-asia-south" 
#   "europe_west" 
#   "horn-of-africa" 
#   "melanesia"
#   "northamerica-east" 
#   "northamerica-west" 
#   "southamerica-east"
#   "southeastasia-west" 
#   "tibetan-plateau" 
#   "west-africa"
# )

REGIONS=(
  # "europe_west"       # Baseline
  "blacksea"          # Easy (Norm. W_1≈0.00)
  "horn-of-africa"    # Medium (Norm. W_1≈0.05)
  "melanesia"         # Hard (Norm. W_1≈0.16)
)


singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/data/convert_zarr_to_npy.py \
    --zarr_root /work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift \
    --out_root  /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/rainshift_npy \
    --regions ${REGIONS[@]} \