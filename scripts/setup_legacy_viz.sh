#!/usr/bin/env bash
# Install the RTX 2080 Ti-compatible legacy cuRobo/Viser environment.
# Software installation only: does not connect to or command the robot.
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${VIZ_CONDA_ENV:-molmobot-viz}"
CUROBO_DIR="${CUROBO_DIR:-$HOME/.local/src/curobo-v0.7.8}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"

if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "Missing conda initialization: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "Missing CUDA compiler: $CUDA_HOME/bin/nvcc" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.10 pip
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1
python -m pip install "numpy<2" viser yourdfpy requests

mkdir -p "$(dirname "$CUROBO_DIR")"
if [ ! -d "$CUROBO_DIR/.git" ]; then
    git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
else
    git -C "$CUROBO_DIR" fetch --depth 1 origin tag v0.7.8
    git -C "$CUROBO_DIR" checkout --detach v0.7.8
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.5"
python -m pip install --no-build-isolation -e "$CUROBO_DIR"

python - <<'PY'
import torch, viser, yourdfpy, curobo
from curobo.wrap.reacher.motion_gen import MotionGen
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("curobo:", curobo.__file__)
print("viser:", viser.__version__)
print("legacy MotionGen import: OK")
PY

echo "[setup] Complete. Activate with: conda activate $ENV_NAME"
