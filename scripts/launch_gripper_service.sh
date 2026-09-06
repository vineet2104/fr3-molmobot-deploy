#!/usr/bin/env bash
# Launch the gripper service on the RT/robot host (same host as the arm service).
# Default: Franka Hand (:54324). Set GRIPPER_BACKEND=robotiq for the 2F-85 (:54323).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$HERE/configs/robot.env" ] && source "$HERE/configs/robot.env"
BACKEND="${GRIPPER_BACKEND:-franka_hand}"

if [ "$BACKEND" = "franka_hand" ]; then
  echo "[gripper] Franka Hand on :${HAND_SERVICE_PORT:-54324} (robot ${FRANKY_ROBOT_IP:-172.16.0.3})"
  exec python "$HERE/services/franka_hand_service.py" \
       --robot_ip "${FRANKY_ROBOT_IP:-172.16.0.3}" --host 0.0.0.0 --port "${HAND_SERVICE_PORT:-54324}"
elif [ "$BACKEND" = "robotiq" ]; then
  echo "[gripper] Robotiq 2F-85 on :54323"
  exec uvicorn --app-dir "$HERE/services" robotiq_gripper_service:app --host 0.0.0.0 --port 54323 --workers 1
else
  echo "[gripper] backend '$BACKEND' unknown (use franka_hand|robotiq); or run with no gripper (--no_gripper on the client)."; exit 1
fi
