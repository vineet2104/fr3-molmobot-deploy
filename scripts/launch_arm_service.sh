#!/usr/bin/env bash
# Launch the FR3 arm (position) service on the RT/robot host.
# Requires: RT kernel, `franky` + libfranka installed, FCI activated in Desk,
# joints unlocked, robot on the FCI network. Runs EXACTLY ONE worker (one robot owner).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$HERE/configs/robot.env" ] && source "$HERE/configs/robot.env"

ROBOT_IP="${FRANKY_ROBOT_IP:-172.16.0.3}"
PORT="${FRANKY_SERVICE_PORT:-54321}"
echo "[arm] robot_ip=$ROBOT_IP port=$PORT rel_dyn=${FRANKY_REL_DYN:-0.05}"
echo "[arm] SAFETY: keep the e-stop within reach. Service starts STOPPED/latched until first target."
exec env FRANKY_ROBOT_IP="$ROBOT_IP" \
     uvicorn --app-dir "$HERE/services" franky_service:app \
     --host "${FRANKY_SERVICE_HOST:-0.0.0.0}" --port "$PORT" --workers 1
