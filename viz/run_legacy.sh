#!/usr/bin/env bash
# Run the legacy cuRobo Viser UI on the workstation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${VIZ_CONDA_ENV:-molmobot-viz}"
ENV_FILE="${ROBOT_ENV_FILE:-$HERE/configs/robot.env}"

# shellcheck disable=SC1090
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
export FRANKY_SERVICE_URL="${FRANKY_SERVICE_URL:-${FRANKY_URL:-http://192.168.123.249:54321}}"
export FRANKA_HAND_SERVICE_URL="${FRANKA_HAND_SERVICE_URL:-${HAND_URL:-http://192.168.123.249:54324}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
export PATH="$CUDA_HOME/bin:$PATH"

cd "$HERE"
echo "[legacy-ui] arm service: $FRANKY_SERVICE_URL"
echo "[legacy-ui] open http://$(hostname -I | awk '{print $1}'):8080"
exec python viz/server_legacy.py
