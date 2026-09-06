#!/usr/bin/env bash
# Run a MolmoBot-FT rollout from the WORKSTATION.
# Prereqs: arm+gripper services up on the RT host; model server up on the GPU host;
# ZED cameras free (stop any camera_service). Edit configs/robot.env first.
#
# Usage: scripts/run_molmobot_ft.sh "Pick up the pineapple slices can" [extra client flags]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$HERE/configs/robot.env" ] && source "$HERE/configs/robot.env"
TASK="${1:?usage: run_molmobot_ft.sh \"<task>\" [extra flags]}"; shift || true
TS="$(date +%Y%m%d-%H%M%S)"
LOG="$HERE/logs/${TS}_molmobot_ft"
mkdir -p "$HERE/logs"
echo "[run] task=$TASK  model=${MOLMOBOTFT_A100_HOST:-?}:${MOLMOBOTFT_A100_PORT:-8000}  log=$LOG"
exec python "$HERE/client/run_molmobot_ft.py" \
  --task "$TASK" \
  --gripper panda \
  --front_view_cam_id exo_front \
  --chunk_executor auto \
  --log_dir "$LOG" "$@"
