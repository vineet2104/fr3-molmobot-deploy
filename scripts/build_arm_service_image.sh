#!/usr/bin/env bash
# Build the isolated, pinned robot-service image. This does not connect to the robot.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FRANKY_SERVICE_IMAGE:-fr3-molmobot-arm:libfranka-0.13.3}"

echo "[build] image=$IMAGE"
echo "[build] This only builds software; it does not connect to or command the robot."
docker build --pull \
    --file "$HERE/deploy/robot-service.Dockerfile" \
    --tag "$IMAGE" \
    "$HERE"

echo "[build] Verifying pinned package versions..."
docker run --rm --entrypoint python3 "$IMAGE" -c \
    'import franky, fastapi, uvicorn; print("franky:", franky.__file__); print("fastapi:", fastapi.__version__); print("uvicorn:", uvicorn.__version__)'

echo "[build] OK: $IMAGE"
