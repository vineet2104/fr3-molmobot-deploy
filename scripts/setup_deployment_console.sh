#!/usr/bin/env bash
# Install lightweight web-console dependencies into the workstation's Python 3.8 user site.
# Keeps the existing system/user pyzed and pyrealsense2 installations visible.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m pip install --user \
    'fastapi==0.116.1' \
    'uvicorn==0.33.0' \
    'pydantic==2.10.6' \
    'msgpack-numpy==0.4.8'

"$PYTHON_BIN" - <<'PY'
import cv2, fastapi, msgpack_numpy, numpy, pyrealsense2, pyzed.sl, uvicorn, websockets
print("deployment console imports: OK")
print("fastapi:", fastapi.__version__, "uvicorn:", uvicorn.__version__)
PY
