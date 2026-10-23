#!/usr/bin/env bash
set -eo pipefail
[[ "$(uname -m)" == "aarch64" ]] || { echo "An ARM/GPU session is required." >&2; exit 1; }
: "${WORK:?WORK must be defined by Kairos}"
cd "$(dirname "$0")/.."
test -f requirement.txt
module purge
module load miniforge/26.1.0
module load proxychains
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS="$WORK/cache/conda-arm"
export PIP_CACHE_DIR="$WORK/cache/pip"
export PYTHONNOUSERSITE=1
mkdir -p "$HOME/envs" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
if [[ ! -f "$HOME/envs/mbm-kairos-arm/conda-meta/history" ]]; then
    proxychains conda create -y --prefix "$HOME/envs/mbm-kairos-arm" --clone pytorch-2.10.0
fi
conda activate "$HOME/envs/mbm-kairos-arm"
proxychains conda install -y -c conda-forge \
    'pytorch=2.10.0=*cuda*' 'torchvision=0.25.0' \
    'numpy>=1.26,<3' 'pandas>=2.2,<3' 'scikit-learn>=1.4,<2' \
    'pillow>=10,<13' 'matplotlib>=3.8,<4' pip

python - <<'PY'
from pathlib import Path
import torch, torchvision
assert torch.cuda.is_available(), 'The cloned environment must support CUDA'
Path('calmip/torch-constraints.txt').write_text(
    f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n'
)
print('CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))
PY

proxychains python -m pip install -c calmip/torch-constraints.txt -r requirement.txt
python -m pip check
python - <<'PY'
import platform, torch, torchvision, timm
assert platform.machine() == 'aarch64'
assert torch.cuda.is_available()
assert timm.__version__ == '1.0.29'
print('Environment ready:', torch.__version__, torchvision.__version__, timm.__version__)
PY
