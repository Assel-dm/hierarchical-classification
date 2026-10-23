#!/usr/bin/env bash
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "Use runGHSession.sh or a GPU batch job before activating this environment." >&2
    return 1
fi
: "${WORK:?WORK must be defined by Kairos}"
module purge || return 1
module load miniforge/26.1.0 || return 1
source "$(conda info --base)/etc/profile.d/conda.sh" || return 1
conda activate "$HOME/envs/mbm-kairos-arm" || return 1
export HF_HOME="$WORK/cache/huggingface"
export TORCH_HOME="$WORK/cache/torch"
export MPLCONFIGDIR="$WORK/cache/matplotlib"
export PIP_CACHE_DIR="$WORK/cache/pip"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$PIP_CACHE_DIR"
