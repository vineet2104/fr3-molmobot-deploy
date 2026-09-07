#!/usr/bin/env bash
# Launch the mixed-camera MolmoBot deployment web console on the workstation.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$HERE/configs/robot.env" ] && source "$HERE/configs/robot.env"

export FRANKY_URL="${FRANKY_URL:-http://192.168.123.249:54321}"
export HAND_URL="${HAND_URL:-http://192.168.123.249:54324}"
export ZED_EXO_FRONT_SN="${ZED_EXO_FRONT_SN:-28576947}"
export REALSENSE_WRIST_SN="${REALSENSE_WRIST_SN:-216322074479}"

cd "$HERE"
exec python3 deployment_console.py --host "${DEPLOYMENT_UI_HOST:-0.0.0.0}" --port "${DEPLOYMENT_UI_PORT:-7071}"
