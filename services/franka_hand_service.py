"""Franka Hand HTTP service (the FR3 *default* gripper) — port 54324.

Mirrors the `HAND_URL` contract the MolmoBot rollout clients use with the
`--gripper panda` preset:

    GET  /health          -> {ok, connected, width_m, is_grasped, busy, error}
    GET  /gripper_state   -> {width_m, max_width_m, is_grasped, busy}
    POST /gripper_move    {width_m, speed}                    (async open/close to width)
    POST /gripper_grasp   {width_m, speed, force, epsilon_inner, epsilon_outer}  (async)
    POST /gripper_open    {speed?}                            (open to max width)
    POST /gripper_stop

The service OWNS the gripper connection, runs blocking libfranka gripper ops in
a background worker (so the 15 Hz rollout loop never blocks), and enforces one
active operation at a time.

Backend: the `franky` library's gripper (`franky.Gripper`). This runs on the
RT/robot host where `franky` + libfranka are installed (same host as the arm
`franky_service`).

Run:
    FRANKY_ROBOT_IP=172.16.0.3 \
    python services/franka_hand_service.py --host 0.0.0.0 --port 54324
(robot IP via --robot_ip or $FRANKY_ROBOT_IP; must match the arm service's robot.)
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import franky
except Exception as e:  # pragma: no cover - only importable on the robot host
    franky = None
    _IMPORT_ERR = e

DEFAULT_ROBOT_IP = os.environ.get("FRANKY_ROBOT_IP", "172.16.0.3")
DEFAULT_PORT = int(os.environ.get("HAND_SERVICE_PORT", "54324"))
DEFAULT_MAX_WIDTH_M = float(os.environ.get("HAND_MAX_WIDTH_M", "0.08"))  # Franka Hand ~0.08 m


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.gripper = None
        self.max_width = DEFAULT_MAX_WIDTH_M
        self.busy = False
        self.last_error: Optional[str] = None
        self.last_state = {"width": DEFAULT_MAX_WIDTH_M, "is_grasped": False}
        self.worker: Optional[threading.Thread] = None


STATE = _State()
app = FastAPI(title="Franka Hand Service")


def _connect(robot_ip: str) -> None:
    if franky is None:
        raise RuntimeError(f"`franky` not importable on this host: {_IMPORT_ERR!r}")
    # franky exposes the Franka Hand as franky.Gripper(<robot_ip>)
    STATE.gripper = franky.Gripper(robot_ip)
    try:
        STATE.max_width = float(STATE.gripper.max_width)
    except Exception:
        STATE.max_width = DEFAULT_MAX_WIDTH_M


def _read_state() -> dict:
    g = STATE.gripper
    if g is None:
        return {"width": None, "is_grasped": None}
    try:
        # franky.Gripper exposes .width and .is_grasped (SDK-dependent); fall back gracefully
        width = float(getattr(g, "width"))
        is_grasped = bool(getattr(g, "is_grasped", False))
        STATE.last_state = {"width": width, "is_grasped": is_grasped}
    except Exception as e:
        STATE.last_error = f"read: {type(e).__name__}: {e}"
    return STATE.last_state


def _run_async(action: str, fn) -> dict:
    """Start a blocking gripper op in a worker; reject if one is already running."""
    with STATE.lock:
        if STATE.busy:
            raise HTTPException(409, "gripper busy")
        STATE.busy = True

    def _target():
        try:
            fn()
            STATE.last_error = None
        except Exception as e:
            STATE.last_error = f"{action}: {type(e).__name__}: {e}"
        finally:
            with STATE.lock:
                STATE.busy = False

    STATE.worker = threading.Thread(target=_target, daemon=True)
    STATE.worker.start()
    return {"ok": True, "accepted": True, "action": action}


class MoveIn(BaseModel):
    width_m: float = Field(..., ge=0.0)
    speed: float = Field(default=0.1, gt=0.0)


class GraspIn(BaseModel):
    width_m: float = Field(..., ge=0.0)
    speed: float = Field(default=0.05, gt=0.0)
    force: float = Field(default=20.0, ge=0.0)
    epsilon_inner: float = Field(default=0.02, ge=0.0)
    epsilon_outer: float = Field(default=0.05, ge=0.0)


class OpenIn(BaseModel):
    speed: float = Field(default=0.1, gt=0.0)


@app.get("/health")
def health():
    s = _read_state()
    return {"ok": STATE.gripper is not None, "connected": STATE.gripper is not None,
            "width_m": s.get("width"), "is_grasped": s.get("is_grasped"),
            "busy": STATE.busy, "error": STATE.last_error}


@app.get("/gripper_state")
def gripper_state():
    s = _read_state()
    return {"width_m": s.get("width"), "max_width_m": STATE.max_width,
            "is_grasped": s.get("is_grasped"), "busy": STATE.busy}


@app.post("/gripper_move")
def gripper_move(req: MoveIn):
    g = STATE.gripper
    if g is None:
        raise HTTPException(503, "gripper not connected")
    w = min(max(req.width_m, 0.0), STATE.max_width)
    return _run_async("move", lambda: g.move(w, req.speed))


@app.post("/gripper_grasp")
def gripper_grasp(req: GraspIn):
    g = STATE.gripper
    if g is None:
        raise HTTPException(503, "gripper not connected")
    w = min(max(req.width_m, 0.0), STATE.max_width)
    return _run_async("grasp", lambda: g.grasp(w, req.speed, req.force,
                                                epsilon_inner=req.epsilon_inner,
                                                epsilon_outer=req.epsilon_outer))


@app.post("/gripper_open")
def gripper_open(req: OpenIn = OpenIn()):
    g = STATE.gripper
    if g is None:
        raise HTTPException(503, "gripper not connected")
    return _run_async("open", lambda: g.move(STATE.max_width, req.speed))


@app.post("/gripper_stop")
def gripper_stop():
    g = STATE.gripper
    if g is None:
        raise HTTPException(503, "gripper not connected")
    try:
        g.stop()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"stop failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_ip", default=DEFAULT_ROBOT_IP)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    print(f"[franka-hand] connecting to robot {args.robot_ip} …")
    _connect(args.robot_ip)
    print(f"[franka-hand] serving on http://{args.host}:{args.port}  (max_width={STATE.max_width:.3f} m)")
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
