# ===========================================================================
#  Sourced by the job scripts: defines run_python for the node's architecture.
#
#    x86_64  (gpu, gpu-h100, gpu-l40, cpu)  the dl-torch mamba env, called by
#            absolute path; no activation needed.
#    aarch64 (gpu-gh)                       the GH200 Singularity container;
#            the mamba env is x86 only.
#
#  require_cuda aborts the job if torch cannot see a GPU. evaluate.py falls
#  back to CPU silently, so without it a mis-set job would run for days on CPU.
# ===========================================================================

DL_TORCH_PYTHON="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/mamba_root/envs/dl-torch/bin/python"
GH200_CONTAINER="/users/fquareng/singularity/dl_gh200.sif"

# Keep ~/.local site-packages out of the interpreter.
export PYTHONNOUSERSITE=1

case "$(uname -m)" in
    x86_64)
        [[ -x "${DL_TORCH_PYTHON}" ]] || { echo "ERROR: ${DL_TORCH_PYTHON} not found"; exit 1; }
        PY=("${DL_TORCH_PYTHON}")
        ;;
    aarch64)
        command -v singularity >/dev/null || { echo "ERROR: singularity not on PATH"; exit 1; }
        [[ -f "${GH200_CONTAINER}" ]] || { echo "ERROR: ${GH200_CONTAINER} not found"; exit 1; }
        export SINGULARITY_BINDPATH="/work,/scratch,/users"
        export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0"
        PY=(singularity exec --nv "${GH200_CONTAINER}" python)
        ;;
    *)
        echo "ERROR: unsupported architecture $(uname -m)"; exit 1 ;;
esac
echo "node: $(hostname)  arch: $(uname -m)  python: ${PY[*]}"

run_python() {
    "${PY[@]}" "$@"
}

require_cuda() {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    run_python -c "
import sys, torch
ok = torch.cuda.is_available()
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', ok,
      '|', torch.cuda.get_device_name(0) if ok else '-')
sys.exit(0 if ok else 1)
" || { echo "ERROR: torch cannot see a GPU; aborting rather than running on CPU."; exit 1; }
}
