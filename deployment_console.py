"""MolmoBot deployment console for the lab FR3 workstation.

Owns the ZED exterior and RealSense wrist cameras, serves both live views, and
runs sequential closed-loop MolmoBot rollouts:

    observe -> infer one 16-action chunk -> execute first 8 at 15 Hz -> repeat

The model runs on a remote GPU host. The arm and hand HTTP services run on the
RT host. Run exactly one console process because cameras are exclusive devices.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
import uvicorn

from client.client_example import MolmoBotClient

ROOT = Path(__file__).resolve().parent
LOG_ROOT = ROOT / "logs" / "deployments"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

FRANKY_URL = os.environ.get("FRANKY_URL", "http://192.168.123.249:54321")
HAND_URL = os.environ.get("HAND_URL", "http://192.168.123.249:54324")
ZED_EXO_SN = int(os.environ.get("ZED_EXO_FRONT_SN", "28576947"))
REALSENSE_WRIST_SN = os.environ.get("REALSENSE_WRIST_SN", "216322074479")
POLICY_SIZE = 480
CAMERA_STALE_S = 1.0
CONTROL_HZ = 15.0
EXECUTE_HORIZON = 8
EXPECTED_ACTION_HORIZON = 16

# Existing deployment limits; verify against the target FR3 before widening.
JOINT_LIMITS = np.asarray([
    [-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
    [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525],
    [-2.8973, 2.8973],
], dtype=np.float32)


def _square_rgb(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    side = min(h, w)
    crop = bgr[(h - side) // 2:(h + side) // 2,
               (w - side) // 2:(w + side) // 2]
    resized = cv2.resize(crop, (POLICY_SIZE, POLICY_SIZE), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


class CameraStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frames = {"exo": None, "wrist": None}
        self.timestamps = {"exo": 0.0, "wrist": 0.0}
        self.errors = {"exo": None, "wrist": None}
        self.threads = []

    def start(self) -> None:
        self.stop.clear()
        self.threads = [
            threading.Thread(target=self._zed_loop, daemon=True, name="zed-exo"),
            threading.Thread(target=self._realsense_loop, daemon=True, name="realsense-wrist"),
        ]
        for thread in self.threads:
            thread.start()

    def close(self) -> None:
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=4.0)

    def _put(self, name: str, bgr: np.ndarray) -> None:
        with self.lock:
            self.frames[name] = bgr.copy()
            self.timestamps[name] = time.monotonic()
            self.errors[name] = None

    def _error(self, name: str, exc: Exception) -> None:
        with self.lock:
            self.errors[name] = f"{type(exc).__name__}: {exc}"

    def get_bgr(self, name: str) -> tuple[Optional[np.ndarray], float]:
        with self.lock:
            frame = self.frames.get(name)
            age = time.monotonic() - self.timestamps.get(name, 0.0)
            return (None if frame is None else frame.copy()), age

    def get_policy_pair(self) -> tuple[np.ndarray, np.ndarray]:
        exo, exo_age = self.get_bgr("exo")
        wrist, wrist_age = self.get_bgr("wrist")
        if exo is None or wrist is None:
            raise RuntimeError("both camera frames are not available")
        if exo_age > CAMERA_STALE_S or wrist_age > CAMERA_STALE_S:
            raise RuntimeError(f"stale camera frame: exo={exo_age:.2f}s wrist={wrist_age:.2f}s")
        return _square_rgb(exo), _square_rgb(wrist)

    def status(self) -> dict:
        now = time.monotonic()
        with self.lock:
            return {
                name: {
                    "ok": self.frames[name] is not None and now - self.timestamps[name] <= CAMERA_STALE_S,
                    "age_s": round(now - self.timestamps[name], 3) if self.timestamps[name] is not None else None,
                    "error": self.errors[name],
                }
                for name in ("exo", "wrist")
            }

    def _zed_loop(self) -> None:
        import pyzed.sl as sl
        camera = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.set_from_serial_number(ZED_EXO_SN)
        try:
            status = camera.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"ZED SN {ZED_EXO_SN} open failed: {status}")
            mat = sl.Mat()
            runtime = sl.RuntimeParameters()
            while not self.stop.is_set():
                if camera.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                    camera.retrieve_image(mat, sl.VIEW.LEFT)
                    rgba = np.asarray(mat.get_data())
                    self._put("exo", cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR))
                else:
                    time.sleep(0.01)
        except Exception as exc:
            self._error("exo", exc)
        finally:
            camera.close()

    def _realsense_loop(self) -> None:
        import pyrealsense2 as rs
        context = rs.context()
        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(REALSENSE_WRIST_SN)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        started = False
        try:
            pipeline.start(config)
            started = True
            while not self.stop.is_set():
                frames = pipeline.wait_for_frames(2000)
                color = frames.get_color_frame()
                if color:
                    self._put("wrist", np.asanyarray(color.get_data()))
        except Exception as exc:
            self._error("wrist", exc)
        finally:
            if started:
                pipeline.stop()


CAMERAS = CameraStore()


def _safe_experiment_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_.")
    if not value:
        raise ValueError("experiment name is required")
    return value[:80]


def _json_get(url: str, timeout: float = 2.0) -> dict:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _stop_hardware() -> None:
    try:
        requests.post(f"{FRANKY_URL}/stop", timeout=2.0)
    except Exception:
        pass
    try:
        requests.post(f"{HAND_URL}/gripper_stop", timeout=1.0)
    except Exception:
        pass


def _gripper_state(width_m: float, mode: str) -> float:
    width = float(np.clip(width_m, 0.0, 0.08))
    closed = 1.0 - width / 0.08
    if mode == "robotiq_angle":
        return closed * 0.87
    if mode == "per_finger_m":
        return width * 0.5
    if mode == "normalized":
        return closed
    if mode == "meters":
        return width
    raise ValueError(f"unknown gripper state mode: {mode}")


class RolloutManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.phase = "idle"
        self.message = ""
        self.error: Optional[str] = None
        self.inference_count = 0
        self.action_count = 0
        self.log_dir: Optional[Path] = None
        self.log_lines = []
        self.started_at: Optional[float] = None

    def log(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(f"[deploy] {line}", flush=True)
        with self.lock:
            self.log_lines.append(line)
            self.log_lines = self.log_lines[-500:]
            self.message = message
            log_dir = self.log_dir
        if log_dir is not None:
            with (log_dir / "console.log").open("a") as fp:
                fp.write(line + "\n")

    def start(self, config: dict) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("a rollout is already running")
            self.stop_event.clear()
            self.running = True
            self.phase = "starting"
            self.message = "starting"
            self.error = None
            self.inference_count = 0
            self.action_count = 0
            self.log_lines = []
            self.started_at = time.time()
        self.thread = threading.Thread(target=self._worker, args=(config,), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        _stop_hardware()
        self.log("STOP requested; robot stop sent immediately")

    def status(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "phase": self.phase,
                "message": self.message,
                "error": self.error,
                "inference_count": self.inference_count,
                "action_count": self.action_count,
                "log_dir": str(self.log_dir) if self.log_dir else None,
                "elapsed_s": round(time.time() - self.started_at, 1) if self.started_at else 0,
                "log": "\n".join(self.log_lines[-100:]),
            }

    def _set_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase

    def _worker(self, cfg: dict) -> None:
        client = None
        steps_fp = None
        try:
            experiment = _safe_experiment_name(cfg["experiment"])
            stamp = time.strftime("%Y%m%d-%H%M%S")
            log_dir = LOG_ROOT / f"{stamp}_{experiment}"
            (log_dir / "frames_exo").mkdir(parents=True)
            (log_dir / "frames_wrist").mkdir(parents=True)
            with self.lock:
                self.log_dir = log_dir
            (log_dir / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
            steps_fp = (log_dir / "steps.jsonl").open("a")

            model_host = cfg["model_host"].strip().replace("ws://", "").replace("http://", "").rstrip("/")
            model_port = int(cfg["model_port"])
            ws_url = f"ws://{model_host}:{model_port}"
            self.log(f"experiment={experiment} model={ws_url}")

            # Fail before connecting the policy if hardware/cameras are unavailable.
            health = _json_get(f"{FRANKY_URL}/health")
            if not health.get("robot_connected"):
                raise RuntimeError(f"arm service not connected: {health}")
            CAMERAS.get_policy_pair()

            self._set_phase("connecting_model")
            client = MolmoBotClient(ws_url, jpeg_obs=bool(cfg.get("jpeg_obs", False)), jpeg_quality=90)
            metadata = client.connect()
            self.log(f"model metadata={metadata}")
            if not isinstance(metadata, dict):
                raise RuntimeError("model metadata is not a dictionary")
            if metadata.get("action_type", "joint_pos") != "joint_pos":
                raise RuntimeError(f"requires absolute joint_pos actions; got {metadata.get('action_type')}")
            if cfg.get("jpeg_obs", False) and not metadata.get("accepts_jpeg", False):
                raise RuntimeError("JPEG observations requested but model metadata does not advertise accepts_jpeg=True")
            camera_names = metadata.get("camera_names")
            if camera_names and list(camera_names) != ["exo_front", "wrist"]:
                raise RuntimeError(f"model camera_names must be ['exo_front','wrist']; got {camera_names}")

            last_gripper_vote = None
            gripper_votes = 0
            gripper_commanded = None
            chunk_index = 0
            while not self.stop_event.is_set():
                self._set_phase("observing")
                exo_rgb, wrist_rgb = CAMERAS.get_policy_pair()
                joint = _json_get(f"{FRANKY_URL}/joint_state")
                arm_q = np.asarray(joint["positions"][:7], dtype=np.float32)
                if arm_q.shape != (7,) or not np.isfinite(arm_q).all():
                    raise RuntimeError(f"invalid arm state: {arm_q}")
                try:
                    hand = _json_get(f"{HAND_URL}/gripper_state", timeout=1.0)
                    width = float(hand["width_m"])
                except Exception:
                    if cfg.get("execute_gripper", False):
                        raise RuntimeError("gripper execution requested but hand state is unavailable")
                    width = float(cfg.get("dummy_gripper_width_m", 0.08))
                grip_state = _gripper_state(width, cfg.get("gripper_state_mode", "robotiq_angle"))
                obs = {
                    "exo_front": exo_rgb,
                    "wrist": wrist_rgb,
                    "qpos": {"arm": arm_q, "gripper": np.asarray([grip_state], dtype=np.float32)},
                    "task": cfg["task"].strip(),
                }

                cv2.imwrite(str(log_dir / "frames_exo" / f"{chunk_index:05d}.jpg"), cv2.cvtColor(exo_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(log_dir / "frames_wrist" / f"{chunk_index:05d}.jpg"), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))

                self._set_phase("inference")
                infer_start = time.monotonic()
                response = client.get_action(obs)
                infer_s = time.monotonic() - infer_start
                if self.stop_event.is_set():
                    break
                arm_chunk = np.asarray(response.get("arm"), dtype=np.float32)
                grip_chunk = np.asarray(response.get("gripper"), dtype=np.float32)
                if arm_chunk.ndim != 2 or arm_chunk.shape[1] != 7:
                    raise RuntimeError(f"expected chunked arm response (N,7), got {arm_chunk.shape}")
                if arm_chunk.shape[0] < EXECUTE_HORIZON:
                    raise RuntimeError(f"need at least {EXECUTE_HORIZON} actions, got {arm_chunk.shape[0]}")
                if not np.isfinite(arm_chunk).all() or not np.isfinite(grip_chunk).all():
                    raise RuntimeError("model returned NaN or infinity")
                with self.lock:
                    self.inference_count += 1
                self.log(f"chunk {chunk_index}: inference={infer_s:.2f}s returned={arm_chunk.shape[0]} actions; executing first {EXECUTE_HORIZON}")
                if arm_chunk.shape[0] != EXPECTED_ACTION_HORIZON:
                    self.log(f"WARNING expected {EXPECTED_ACTION_HORIZON}-action prediction, received {arm_chunk.shape[0]}")

                self._set_phase("dry_run" if cfg.get("dry_run", True) else "executing")
                previous = arm_q.copy()
                deadline = time.monotonic()
                for action_i in range(EXECUTE_HORIZON):
                    if self.stop_event.is_set():
                        break
                    raw_q = arm_chunk[action_i]
                    target_q = np.clip(raw_q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
                    max_delta = float(cfg.get("max_joint_delta", 0.10))
                    delta_clipped = False
                    if max_delta > 0:
                        delta = target_q - previous
                        if np.max(np.abs(delta)) > max_delta:
                            target_q = previous + np.clip(delta, -max_delta, max_delta)
                            delta_clipped = True
                    target_q = np.clip(target_q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

                    if not cfg.get("dry_run", True):
                        r = requests.post(
                            f"{FRANKY_URL}/target_joint_state",
                            json={"positions": target_q.tolist(), "velocities": [0.0] * 7,
                                  "seq": self.action_count}, timeout=2.0,
                        )
                        r.raise_for_status()

                    grip_value = float(grip_chunk[action_i].reshape(-1)[0]) if grip_chunk.size else 0.0
                    vote = grip_value > 0.5
                    if vote == last_gripper_vote:
                        gripper_votes += 1
                    else:
                        last_gripper_vote = vote
                        gripper_votes = 1
                    if (cfg.get("execute_gripper", False) and not cfg.get("dry_run", True)
                            and gripper_votes >= 3 and vote != gripper_commanded):
                        endpoint = "gripper_grasp" if vote else "gripper_open"
                        payload = ({"width_m": float(cfg.get("grasp_width_m", 0.02)),
                                    "speed": 0.05, "force": float(cfg.get("grasp_force_n", 10.0)),
                                    "epsilon_inner": 0.02, "epsilon_outer": 0.02}
                                   if vote else {"speed": 0.05})
                        gr = requests.post(f"{HAND_URL}/{endpoint}", json=payload, timeout=1.0)
                        if gr.status_code not in (200, 409):
                            gr.raise_for_status()
                        gripper_commanded = vote

                    record = {
                        "time": time.time(), "chunk": chunk_index, "action_in_chunk": action_i,
                        "inference_s": infer_s if action_i == 0 else None,
                        "arm_observed": arm_q.tolist(), "arm_raw": raw_q.tolist(),
                        "arm_sent": target_q.tolist(), "delta_clipped": delta_clipped,
                        "gripper_raw": grip_value, "dry_run": bool(cfg.get("dry_run", True)),
                    }
                    steps_fp.write(json.dumps(record) + "\n")
                    steps_fp.flush()
                    previous = target_q
                    with self.lock:
                        self.action_count += 1
                    deadline += 1.0 / CONTROL_HZ
                    time.sleep(max(0.0, deadline - time.monotonic()))
                chunk_index += 1

            self.log("rollout stopped")
        except Exception as exc:
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
            self.log(f"ERROR: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            _stop_hardware()
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            if steps_fp is not None:
                steps_fp.close()
            with self.lock:
                self.running = False
                self.phase = "idle"


ROLLOUT = RolloutManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    CAMERAS.start()
    try:
        yield
    finally:
        ROLLOUT.stop()
        CAMERAS.close()


app = FastAPI(title="FR3 MolmoBot Deployment Console", lifespan=lifespan)


@app.get("/api/status")
def api_status():
    out = ROLLOUT.status()
    out["cameras"] = CAMERAS.status()
    return out


@app.get("/api/preflight")
def api_preflight(model_host: str, model_port: int):
    checks = {"cameras": CAMERAS.status()}
    try:
        checks["arm"] = {"ok": True, "detail": _json_get(f"{FRANKY_URL}/health")}
    except Exception as exc:
        checks["arm"] = {"ok": False, "detail": str(exc)}
    try:
        checks["hand"] = {"ok": True, "detail": _json_get(f"{HAND_URL}/health")}
    except Exception as exc:
        checks["hand"] = {"ok": False, "detail": str(exc)}
    try:
        r = requests.get(f"http://{model_host}:{model_port}/healthz", timeout=3.0)
        checks["model"] = {"ok": r.ok, "detail": r.text.strip()}
    except Exception as exc:
        checks["model"] = {"ok": False, "detail": str(exc)}
    return checks


@app.post("/api/run")
async def api_run(request: Request):
    cfg = await request.json()
    for key in ("experiment", "model_host", "model_port", "task"):
        if not str(cfg.get(key, "")).strip():
            raise HTTPException(400, f"{key} is required")
    try:
        ROLLOUT.start(cfg)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


@app.post("/api/stop")
def api_stop():
    ROLLOUT.stop()
    return {"ok": True}


def _mjpeg(name: str):
    while True:
        frame, age = CAMERAS.get_bgr(name)
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Waiting for {name} camera", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, f"{name}  age={age:.2f}s", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        time.sleep(1.0 / 15.0)


@app.get("/api/camera/{name}/stream")
def camera_stream(name: str):
    if name not in ("exo", "wrist"):
        raise HTTPException(404, "unknown camera")
    return StreamingResponse(_mjpeg(name), media_type="multipart/x-mixed-replace; boundary=frame")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>FR3 MolmoBot Deployment</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{color-scheme:dark}body{font-family:system-ui;margin:0;background:#0d1117;color:#e6edf3}.wrap{max-width:1400px;margin:auto;padding:18px}
.card{background:#161b22;border:1px solid #30363d;border-radius:9px;padding:14px;margin-bottom:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.row{display:flex;gap:10px;align-items:end}.grow{flex:1}
label{display:block;color:#9da7b3;font-size:12px;margin:7px 0 3px}input,textarea,select{width:100%;box-sizing:border-box;padding:9px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px}
img{width:100%;background:#000;border-radius:7px}.run,.stop,.pre{border:0;border-radius:6px;padding:11px 20px;color:white;font-weight:700;cursor:pointer}.run{background:#238636}.stop{background:#da3633}.pre{background:#1f6feb}button:disabled{opacity:.5}.status{white-space:pre-wrap;font-family:monospace;font-size:12px}.ok{color:#3fb950}.bad{color:#f85149}.warn{color:#d29922}details{margin-top:8px}@media(max-width:800px){.grid{grid-template-columns:1fr}.row{display:block}}
</style></head><body><div class="wrap"><h2>FR3 MolmoBot Deployment</h2>
<div class="grid"><div class="card"><h3>Exterior — ZED 2</h3><img src="/api/camera/exo/stream"></div><div class="card"><h3>Wrist — RealSense D435i</h3><img src="/api/camera/wrist/stream"></div></div>
<div class="card"><div class="row"><div class="grow"><label>Experiment name</label><input id="experiment" placeholder="pineapple_pick_01"></div><div class="grow"><label>Model server IP / hostname</label><input id="host" placeholder="10.x.x.x"></div><div><label>Port</label><input id="port" value="8000"></div></div>
<label>Task instruction</label><textarea id="task" rows="2" placeholder="Pick up the pineapple slices can"></textarea>
<details><summary>Safety and model I/O settings</summary><div class="row"><div><label>Max per-action joint delta (rad; 0 disables)</label><input id="delta" type="number" value="0.10" step="0.01"></div><div><label>Gripper state encoding</label><select id="gmode"><option value="robotiq_angle">Robotiq-equivalent angle (V3)</option><option value="per_finger_m">Per-finger meters</option><option value="normalized">Normalized closed fraction</option><option value="meters">Full width meters</option></select></div><div><label>Dummy gripper width when disabled (m)</label><input id="dummy" type="number" value="0.08" step="0.01"></div></div>
<label><input id="dry" type="checkbox" checked style="width:auto"> Dry run — infer and log but do not move arm or hand</label>
<label><input id="gripper" type="checkbox" style="width:auto"> Execute model gripper actions (10 N grasp, 20 mm target)</label>
<label><input id="jpeg" type="checkbox" style="width:auto"> JPEG-compress observations (only if model metadata advertises support)</label></details>
<div class="row"><button class="pre" onclick="preflight()">Preflight</button><button id="run" class="run" onclick="runRollout()">Run continuously</button><button class="stop" onclick="stopRollout()">STOP</button></div></div>
<div class="card"><h3>Status</h3><div id="status" class="status">Loading...</div></div><div class="card"><h3>Log</h3><div id="log" class="status"></div></div>
</div><script>
const $=id=>document.getElementById(id);async function api(url,opt){const r=await fetch(url,opt);let d;try{d=await r.json()}catch(e){d={detail:await r.text()}}if(!r.ok)throw new Error(d.detail||JSON.stringify(d));return d}
async function preflight(){try{const d=await api(`/api/preflight?model_host=${encodeURIComponent($('host').value)}&model_port=${$('port').value}`);$('log').textContent=JSON.stringify(d,null,2)}catch(e){$('log').textContent=e}}
async function runRollout(){const body={experiment:$('experiment').value,model_host:$('host').value,model_port:Number($('port').value),task:$('task').value,dry_run:$('dry').checked,execute_gripper:$('gripper').checked,jpeg_obs:$('jpeg').checked,max_joint_delta:Number($('delta').value),gripper_state_mode:$('gmode').value,dummy_gripper_width_m:Number($('dummy').value),grasp_width_m:0.02,grasp_force_n:10};try{await api('/api/run',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)})}catch(e){alert(e)}}
async function stopRollout(){await api('/api/stop',{method:'POST'})}async function poll(){try{const s=await api('/api/status');$('status').textContent=`${s.running?'RUNNING':'IDLE'} | phase=${s.phase} | inference chunks=${s.inference_count} | actions=${s.action_count} | elapsed=${s.elapsed_s}s\n${s.message}${s.error?'\nERROR: '+s.error:''}\nlog_dir=${s.log_dir||''}\ncameras=${JSON.stringify(s.cameras)}`;$('log').textContent=s.log;$('run').disabled=s.running}catch(e){$('status').textContent=e}}setInterval(poll,500);poll();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7071)
    args = parser.parse_args()
    print(f"[deploy] UI: http://{args.host}:{args.port}")
    print(f"[deploy] arm={FRANKY_URL} hand={HAND_URL}")
    print(f"[deploy] exo ZED SN={ZED_EXO_SN}; wrist RealSense SN={REALSENSE_WRIST_SN}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
