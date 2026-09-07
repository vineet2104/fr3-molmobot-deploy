#!/usr/bin/env bash
# Launch the Franka Hand HTTP service in the pinned robot-service image.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROBOT_ENV_FILE:-$HERE/configs/robot.env}"
IMAGE="${FRANKY_SERVICE_IMAGE:-fr3-molmobot-arm:libfranka-0.13.3}"
CONTAINER="${FRANKY_HAND_CONTAINER:-fr3-molmobot-hand}"

if [ ! -f "$ENV_FILE" ]; then
    echo "[hand] Missing $ENV_FILE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"
: "${FRANKY_ROBOT_IP:?Set FRANKY_ROBOT_IP in $ENV_FILE}"

if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER"; then
    echo "[hand] Container $CONTAINER is already running." >&2
    exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[hand] Missing image $IMAGE; run ./scripts/build_arm_service_image.sh first." >&2
    exit 1
fi

echo "[hand] robot_ip=$FRANKY_ROBOT_IP port=${HAND_SERVICE_PORT:-54324}"
echo "[hand] Commands are asynchronous. Keep fingers and objects clear during testing."
exec docker run --rm \
    --name "$CONTAINER" \
    --network host \
    --env FRANKY_ROBOT_IP="$FRANKY_ROBOT_IP" \
    --env HAND_SERVICE_PORT="${HAND_SERVICE_PORT:-54324}" \
    --env HAND_MAX_WIDTH_M="${HAND_MAX_WIDTH_M:-0.08}" \
    --volume "$HERE:/app:ro" \
    "$IMAGE" \
    python3 /app/services/franka_hand_service.py \
        --robot_ip "$FRANKY_ROBOT_IP" --host 0.0.0.0 \
        --port "${HAND_SERVICE_PORT:-54324}"
