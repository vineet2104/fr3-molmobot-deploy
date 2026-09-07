#!/usr/bin/env bash
# Launch exactly one isolated Franky arm service on the RT host.
# The service opens an FCI connection but starts stop-latched and sends no target.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROBOT_ENV_FILE:-$HERE/configs/robot.env}"
IMAGE="${FRANKY_SERVICE_IMAGE:-fr3-molmobot-arm:libfranka-0.13.3}"
CONTAINER="${FRANKY_SERVICE_CONTAINER:-fr3-molmobot-arm}"

if [ ! -f "$ENV_FILE" ]; then
    echo "[arm] Missing $ENV_FILE" >&2
    echo "[arm] Copy configs/env.example to configs/robot.env and set FRANKY_ROBOT_IP." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"
: "${FRANKY_ROBOT_IP:?Set FRANKY_ROBOT_IP in $ENV_FILE}"

if [ "$FRANKY_ROBOT_IP" = "172.16.0.3" ]; then
    echo "[arm] Refusing example robot IP 172.16.0.3; configure the lab FR3 IP." >&2
    exit 1
fi

if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER"; then
    echo "[arm] Container $CONTAINER is already running." >&2
    exit 1
fi

# Refuse to launch if a known previous bridge is active anywhere on the host.
if pgrep -af 'fr3_joint_interface|fr3_joint_velocity|fr3_joint_torque|fr3_task_interface' \
    | grep -v -E 'pgrep|launch_arm_service_docker' >/dev/null; then
    echo "[arm] REFUSING: an FR3 bridge process appears to be active:" >&2
    pgrep -af 'fr3_joint_interface|fr3_joint_velocity|fr3_joint_torque|fr3_task_interface' >&2 || true
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[arm] Missing image $IMAGE; run ./scripts/build_arm_service_image.sh first." >&2
    exit 1
fi

echo "[arm] robot_ip=$FRANKY_ROBOT_IP port=${FRANKY_SERVICE_PORT:-54321}"
echo "[arm] control_hz=${CONTROL_HZ:-50} timeout=${COMMAND_TIMEOUT_S:-0.5} rel_dyn=${FRANKY_REL_DYN:-0.05}"
echo "[arm] Starts STOP-LATCHED with no motion target. Keep the physical e-stop within reach."

docker run --rm \
    --name "$CONTAINER" \
    --network host \
    --cap-add SYS_NICE \
    --ulimit rtprio=99:99 \
    --ulimit memlock=-1:-1 \
    --env FRANKY_ROBOT_IP="$FRANKY_ROBOT_IP" \
    --env FRANKY_REL_DYN="${FRANKY_REL_DYN:-0.05}" \
    --env CONTROL_HZ="${CONTROL_HZ:-50}" \
    --env COMMAND_TIMEOUT_S="${COMMAND_TIMEOUT_S:-0.5}" \
    --env FRANKY_SERVICE_PORT="${FRANKY_SERVICE_PORT:-54321}" \
    --volume "$HERE:/app:ro" \
    "$IMAGE" \
    python3 -m uvicorn --app-dir /app/services franky_service:app \
        --host "${FRANKY_SERVICE_HOST:-0.0.0.0}" \
        --port "${FRANKY_SERVICE_PORT:-54321}" --workers 1
