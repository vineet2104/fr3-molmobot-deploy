"""REFERENCE ONLY — the original single-box console from sim2real-deploy.
It hardcodes local conda paths, checkpoint locations, and auto-deploys model
servers on the SAME machine. It imports modules NOT shipped here
(sim_serve/, ../serve, sim2real-ft-factory, openpi) and will not run in this
repo as-is. Kept for reference. For the 3-host lab use `service_console.py`
(endpoint-driven, subprocesses the client/ scripts). See docs/SETUP.md.
"""

"""Sim2Real deployment web service — one-stop console.

A FastAPI app that (a) on startup runs a PREFLIGHT that gets the whole stack
ready, and (b) gives a browser UI to launch/stop policy rollouts and watch the
live camera views.

Preflight (so the engineer only runs once everything is green):
  1. Stop camera_service        (frees the ZEDs; the rollout opens them directly)
  2. Check franky arm service   (54321 @ robot NUC 10.20.12.110 — check-only,
                                 it runs on another host and can't be started here)
  3. Ensure robotiq gripper svc (54323 @ localhost — auto-start if down)
  4. Deploy the selected model  (local: auto-launch serve_bridge in the molmobot
                                 env if /healthz is down; remote: check-only)
  5. Dummy inference            (send a synthetic obs, confirm a valid action chunk)

Run it in the franky env (has the client deps):
    /home/franka/miniforge3/envs/franky/bin/python run_s2r_service.py --port 7070
Open http://<this-host-ip>:7070 .

Only ONE rollout runs at a time (one robot). Local model servers are deployed
one-at-a-time (they don't co-fit on the GPU); preparing a new local policy stops
the previously-managed one.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import glob
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from openpi_client import msgpack_numpy as opmn

ROOT = Path(__file__).resolve().parent
REAL_SERVE = ROOT / "real_serve"
LOGS = ROOT / "logs"
FTF = ROOT / "sim2real-ft-factory"
MOLMOBOT_CWD = FTF / "MolmoBot"
FRANKY_PY = "/home/franka/miniforge3/envs/franky/bin/python"
MOLMOBOT_PY = "/home/franka/miniforge3/envs/molmobot/bin/python"
FRANKY_SVC_DIR = "/home/franka/franky_service"
UV = "/home/franka/.local/bin/uv"
OPENPI_CWD = ROOT / "openpi"

FRANKY_URL = "http://10.20.12.110:54321"     # arm, robot NUC (check-only)
ROBOTIQ_URL = "http://localhost:54323"        # gripper, this box (auto-start)

sys.path.insert(0, str(ROOT / "sim_serve"))
from client_example import MolmoBotClient  # noqa: E402  (standard-msgpack molmobot client)

# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------
POLICIES: dict[str, dict] = {
    "molmobot_official": {
        "label": "MolmoBot Official — Img-DROID (local :8000)",
        "script": "run_molmobot_faster_benchmarks_vineet.py",
        "args": ["--dual_camera", "--gripper", "robotiq", "--chunk_executor", "auto", "--grasp_width_m", "0.0"],
        "views": [("exo", "frames"), ("wrist", "frames_wrist")],
        "location": "local", "supports_predicted_video": False,
        "default_task": "Pick up the pineapple slices can",
        "host": "127.0.0.1", "port": 8000, "endpoint_style": "ws_url",
        "dummy": "molmobot",
        "deploy": {"port": 8000, "cmd": [
            "../serve/serve_bridge.py",
            "--checkpoint_path", "../checkpoints/MolmoBot-Img-DROID",
            "--host", "127.0.0.1", "--port", "8000",
            "--camera_names", "exo_camera_1", "wrist_camera",
            "--action_type", "joint_pos", "--action_horizon", "16",
            "--execute_horizon", "8", "--gripper_representation_count", "1",
            "--clamp_gripper", "--chunk_response"]},
    },
    "molmobot_ft_v4full": {
        "label": "MolmoBot Finetuned — v4full 10ep (local :8005)",
        "script": "run_molmobotft_faster_benchmark_vineet.py",
        "args": ["--gripper", "robotiq", "--front_view_cam_id", "exo_front",
                 "--chunk_executor", "auto", "--grasp_width_m", "0.0"],
        "views": [("exo_front", "frames"), ("wrist", "frames_wrist")],
        "location": "local", "supports_predicted_video": False,
        "default_task": "Pick up the pineapple slices can",
        "host": "127.0.0.1", "port": 8005, "endpoint_style": "a100",
        "dummy": "molmobot",
        "deploy": {"port": 8005, "cmd": [
            "../serve/serve_bridge.py",
            "--checkpoint_path", "../checkpoints/v4full_twoview_step14500_epoch10",
            "--host", "127.0.0.1", "--port", "8005",
            "--camera_names", "exo_front", "wrist",
            "--action_type", "joint_pos", "--action_horizon", "16",
            "--execute_horizon", "8", "--gripper_representation_count", "1",
            "--chunk_response"]},
    },
    "molmobot_custom": {
        "label": "MolmoBot Custom — any finetuned ckpt (set endpoint + id)",
        "script": "run_molmobotft_benchmark_vineet.py",
        "args": ["--gripper", "robotiq", "--front_view_cam_id", "exo_front",
                 "--chunk_executor", "auto", "--grasp_width_m", "0.0"],
        "views": [("exo_front", "frames"), ("wrist", "frames_wrist")],
        "location": "remote", "supports_predicted_video": False,
        "default_task": "Pick up the box of farm fresh butter",
        # Placeholder endpoint — set the real host:port + optional id in the UI.
        "host": "0.0.0.0", "port": 8005, "endpoint_style": "a100",
        "dummy": "molmobot", "custom": True,
    },
    "pi05_droid": {
        "label": "Pi0.5-DROID (openpi, local :8000)",
        "script": "run_pi_droid_benchmark.py",
        "args": ["--exterior", "exo_left", "--open_loop_horizon", "8", "--vel_scale", "1.0", "--gripper_debounce", "1"],
        "views": [("exo_left", "frames_exo_left"), ("exo_right", "frames_exo_right"), ("wrist", "frames_wrist")],
        "location": "local", "supports_predicted_video": False,
        "default_task": "pick up the pineapple slices can",
        "host": "127.0.0.1", "port": 8001, "endpoint_style": "hostport",
        "dummy": "pi_droid",
        "deploy": {"port": 8001, "runner": "uv", "cmd": [
            "run", "scripts/serve_policy.py", "--port", "8001", "policy:checkpoint",
            "--policy.config=pi05_droid",
            "--policy.dir=gs://openpi-assets/checkpoints/pi05_droid"]},
    },
    "pi0_fast_droid": {
        "label": "Pi0-FAST-DROID (openpi, local :8000)",
        "script": "run_pi_droid_benchmark.py",
        "args": ["--exterior", "exo_left", "--open_loop_horizon", "8", "--vel_scale", "1.0", "--gripper_debounce", "1"],
        "views": [("exo_left", "frames_exo_left"), ("exo_right", "frames_exo_right"), ("wrist", "frames_wrist")],
        "location": "local", "supports_predicted_video": False,
        "default_task": "pick up the pineapple slices can",
        "host": "127.0.0.1", "port": 8002, "endpoint_style": "hostport",
        "dummy": "pi_droid",
        "deploy": {"port": 8002, "runner": "uv", "cmd": [
            "run", "scripts/serve_policy.py", "--port", "8002", "policy:checkpoint",
            "--policy.config=pi0_fast_droid",
            "--policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid"]},
    },
    "pi0_droid": {
        "label": "Pi0-DROID (openpi, local :8000)",
        "script": "run_pi_droid_benchmark.py",
        "args": ["--exterior", "exo_left", "--open_loop_horizon", "8", "--vel_scale", "1.0", "--gripper_debounce", "1"],
        "views": [("exo_left", "frames_exo_left"), ("exo_right", "frames_exo_right"), ("wrist", "frames_wrist")],
        "location": "local", "supports_predicted_video": False,
        "default_task": "pick up the pineapple slices can",
        "host": "127.0.0.1", "port": 8003, "endpoint_style": "hostport",
        "dummy": "pi_droid",
        "deploy": {"port": 8003, "runner": "uv", "cmd": [
            "run", "scripts/serve_policy.py", "--port", "8003", "policy:checkpoint",
            "--policy.config=pi0_droid",
            "--policy.dir=gs://openpi-assets/checkpoints/pi0_droid"]},
    },
    "cosmos3": {
        "label": "Cosmos3-Nano — WAM (REMOTE — ask HAMILTON)",
        "script": "run_cosmos3_benchmark_vineet.py",
        "args": [],
        "views": [("wrist", "frames_wrist"), ("exo_left", "frames_exo_left"),
                  ("exo_right", "frames_exo_right")],
        "location": "remote", "supports_predicted_video": True,
        "default_task": "pick up the pineapple slices can",
        "host": "10.49.164.253", "port": 8010, "endpoint_style": "ws_url",
        "dummy": "cosmos",
    },
}
DEFAULT_POLICY = "molmobot_official"

# Live, user-overridable endpoint per policy (host, port). Init from registry.
_ENDPOINT: dict[str, list] = {k: [v["host"], int(v["port"])] for k, v in POLICIES.items()}


def _ep(policy: str):
    h, p = _ENDPOINT[policy]
    return str(h), int(p)


def _ws_url(policy: str) -> str:
    h, p = _ep(policy)
    return f"ws://{h}:{p}"


def _healthz(policy: str) -> str:
    h, p = _ep(policy)
    return f"http://{h}:{p}/healthz"


def _endpoint_args(policy: str) -> list:
    """CLI endpoint args for the rollout script, from the live endpoint."""
    h, p = _ep(policy)
    style = POLICIES[policy]["endpoint_style"]
    if style == "a100":
        return ["--a100_host", h, "--a100_port", str(p)]
    if style == "hostport":
        return ["--host", h, "--port", str(p)]   # openpi pi-droid client
    return ["--ws_url", f"ws://{h}:{p}"]     # ws_url style

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_PREP_LOCK = threading.Lock()
_RUN: dict = {"proc": None, "policy": None, "task": None, "log_dir": None,
              "start_ts": None, "log_fp": None}
_PREFLIGHT: dict = {"running": False, "done": False,
                    "camera": {"status": "pending", "detail": ""},
                    "franky": {"status": "pending", "detail": ""},
                    "robotiq": {"status": "pending", "detail": ""}}
_MODEL: dict[str, dict] = {k: {"status": "unknown", "detail": ""} for k in POLICIES}
_MANAGED_SERVERS: dict[int, subprocess.Popen] = {}   # port -> serve_bridge proc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _http_ok(url: str, timeout: float = 3.0):
    try:
        r = requests.get(url, timeout=timeout)
        js = None
        try:
            js = r.json()
        except Exception:
            pass
        return (r.status_code == 200), js
    except requests.RequestException as e:
        return False, str(e)


def _is_running() -> bool:
    p = _RUN["proc"]
    return p is not None and p.poll() is None


def _killpg(proc: subprocess.Popen, grace: float = 5.0):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except Exception:
        pass
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < grace:
        time.sleep(0.1)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def _stop_camera_service() -> None:
    try:
        subprocess.run(["pkill", "-f", "camera_service:app"], timeout=5)
    except Exception:
        pass


# ---------- preflight steps ----------
def _check_camera() -> None:
    _stop_camera_service()
    time.sleep(1.0)
    try:
        r = subprocess.run(["pgrep", "-f", "camera_service:app"],
                           capture_output=True, timeout=5)
        still = bool(r.stdout.strip())
    except Exception:
        still = False
    if still:
        _PREFLIGHT["camera"] = {"status": "warn", "detail": "camera_service still running"}
    else:
        _PREFLIGHT["camera"] = {"status": "ok", "detail": "stopped — ZEDs free"}


def _check_franky() -> None:
    ok, js = _http_ok(f"{FRANKY_URL}/health")
    if not ok:
        _PREFLIGHT["franky"] = {"status": "fail",
                                "detail": f"unreachable at {FRANKY_URL} — start franky_service on the robot NUC"}
        return
    connected = bool(js.get("robot_connected")) if isinstance(js, dict) else False
    latched = bool(js.get("stop_latched")) if isinstance(js, dict) else False
    ctrl_err = (js.get("last_control_error") if isinstance(js, dict) else None) or ""
    if not connected:
        _PREFLIGHT["franky"] = {"status": "fail", "detail": "robot_connected=false"}
        return
    # /health can report robot_connected=true even when the libfranka RT link
    # has dropped, so actually probe /joint_state (the read the rollout needs).
    js_ok, js_body = _http_ok(f"{FRANKY_URL}/joint_state", timeout=4.0)
    if not js_ok:
        hint = ("arm state read FAILING (/joint_state not 200) — the "
                "franky<->robot FCI link is likely down. Fix on the NUC: in "
                "Franka Desk unlock brakes + activate FCI + clear E-stop, then "
                "restart franky_service.")
        if "Exception" in ctrl_err or "error" in ctrl_err.lower():
            hint += f"  last_control_error: {ctrl_err[:160]}"
        _PREFLIGHT["franky"] = {"status": "fail", "detail": hint}
        return
    if latched:
        _PREFLIGHT["franky"] = {"status": "warn",
                                "detail": "joint_state OK — but STOP LATCHED (clear with /go_home before moving)"}
    else:
        _PREFLIGHT["franky"] = {"status": "ok", "detail": "connected, joint_state OK, not latched"}


def _check_robotiq() -> None:
    ok, js = _http_ok(f"{ROBOTIQ_URL}/health")
    if ok and isinstance(js, dict) and js.get("ok"):
        _PREFLIGHT["robotiq"] = {"status": "ok", "detail": "activated & calibrated"}
        return
    # auto-start
    _PREFLIGHT["robotiq"] = {"status": "starting", "detail": "launching gripper_service…"}
    try:
        LOGS.mkdir(exist_ok=True)
        fp = (LOGS / "gripper_service.log").open("w")
        subprocess.Popen(
            [FRANKY_PY, "-m", "uvicorn", "droid_plus.services.gripper_service:app",
             "--host", "0.0.0.0", "--port", "54323", "--workers", "1"],
            cwd=FRANKY_SVC_DIR, stdout=fp, stderr=subprocess.STDOUT,
            start_new_session=True)
    except Exception as e:
        _PREFLIGHT["robotiq"] = {"status": "fail", "detail": f"launch failed: {e}"}
        return
    for _ in range(30):
        time.sleep(1.0)
        ok, js = _http_ok(f"{ROBOTIQ_URL}/health")
        if ok and isinstance(js, dict) and js.get("ok"):
            _PREFLIGHT["robotiq"] = {"status": "ok", "detail": "auto-started"}
            return
    _PREFLIGHT["robotiq"] = {"status": "fail", "detail": "gripper_service did not become healthy"}


def _run_preflight() -> None:
    _PREFLIGHT["running"] = True
    _PREFLIGHT["done"] = False
    for k in ("camera", "franky", "robotiq"):
        _PREFLIGHT[k] = {"status": "running", "detail": ""}
    _check_camera()
    _check_franky()
    _check_robotiq()
    _PREFLIGHT["running"] = False
    _PREFLIGHT["done"] = True


def _preflight_ok() -> bool:
    return (_PREFLIGHT["done"]
            and _PREFLIGHT["camera"]["status"] in ("ok", "warn")
            and _PREFLIGHT["franky"]["status"] in ("ok", "warn")
            and _PREFLIGHT["robotiq"]["status"] == "ok")


# ---------- model server deploy + dummy inference ----------
def _local_ports() -> set:
    return {POLICIES[k]["deploy"]["port"] for k in POLICIES if "deploy" in POLICIES[k]}


def _stop_other_local_servers(keep_port) -> None:
    """Free the GPU by stopping every local serve_bridge whose --port is a known
    local model port other than keep_port (keep_port=None -> stop ALL local
    servers). Matches by port so it works even across service restarts (when
    _MANAGED_SERVERS lost track). One 24 GB GPU can't hold two ~11 GB models +
    the ZED CUDA context, so we keep at most the one we need."""
    for port in _local_ports():
        if keep_port is not None and port == keep_port:
            continue
        # kill the tracked process group first (covers `uv run` -> child python + JAX/torch GPU)
        pr = _MANAGED_SERVERS.pop(port, None)
        if pr is not None:
            _killpg(pr)
        # belt-and-suspenders: kill ANY known model server on this port (survives
        # service restarts, and covers BOTH molmobot serve_bridge AND openpi serve_policy)
        for pat in (f"serve_bridge.py.*--port {port}", f"serve_policy.py.*--port {port}"):
            subprocess.run(["pkill", "-9", "-f", pat], stderr=subprocess.DEVNULL)


def _deploy_model(policy: str):
    spec = POLICIES[policy]
    dep = spec.get("deploy")
    if dep is None:
        return False, "no local deploy spec (remote model)"
    port = dep["port"]
    _stop_other_local_servers(keep_port=port)   # free GPU for this model
    if _http_ok(_healthz(policy))[0]:
        return True, "already up"
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    LOGS.mkdir(exist_ok=True)
    if dep.get("runner") == "uv":       # openpi pi-droid: `uv run scripts/serve_policy.py ...`
        argv = [UV] + dep["cmd"]; cwd = str(OPENPI_CWD)
    else:                                # molmobot: serve_bridge in the molmobot env
        argv = [MOLMOBOT_PY] + dep["cmd"]; cwd = str(MOLMOBOT_CWD)
    fp = (LOGS / f"server_{policy}.log").open("w")
    fp.write(f"# cwd={cwd}\n# {' '.join(argv)}\n"); fp.flush()
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=fp,
                            stderr=subprocess.STDOUT, start_new_session=True)
    _MANAGED_SERVERS[port] = proc
    for _ in range(300):  # openpi first-load (checkpoint + JIT) can take a few min
        if proc.poll() is not None:
            return False, f"server exited early (see logs/server_{policy}.log)"
        if _http_ok(_healthz(policy))[0]:
            return True, "deployed"
        time.sleep(1.0)
    return False, "timeout waiting for /healthz (~180s)"


def _dummy_infer_molmobot(ws_url: str):
    c = MolmoBotClient(ws_url)
    try:
        meta = c.connect()
        cams = (meta.get("camera_names") if isinstance(meta, dict) else None) or ["exo_front"]
        obs = {cam: np.random.randint(0, 255, (480, 480, 3), np.uint8) for cam in cams}
        obs["qpos"] = {"arm": np.zeros(7, np.float32), "gripper": np.array([0.0], np.float32)}
        obs["task"] = "preflight"
        r = c.get_action(obs)
        arm = np.asarray(r["arm"])
        ok = arm.shape[-1] == 7 and arm.ndim in (1, 2)
        return ok, f"arm={arm.shape}"
    finally:
        try:
            c.close()
        except Exception:
            pass


def _dummy_infer_cosmos(ws_url: str):
    async def go():
        ws = await websockets.connect(ws_url, compression=None, max_size=None,
                                      ping_interval=None, ping_timeout=None, close_timeout=5)
        await ws.recv()
        obs = {"prompt": "preflight",
               "observation/wrist_image_left": np.random.randint(0, 255, (720, 1280, 3), np.uint8),
               "observation/exterior_image_1_left": np.random.randint(0, 255, (720, 1280, 3), np.uint8),
               "observation/exterior_image_2_left": np.random.randint(0, 255, (720, 1280, 3), np.uint8),
               "observation/joint_position": np.zeros(7, np.float32),
               "observation/gripper_position": np.float32(0.0)}
        await ws.send(opmn.packb(obs))
        msg = await asyncio.wait_for(ws.recv(), timeout=90)
        await ws.close()
        if isinstance(msg, str):
            raise RuntimeError(msg[:200])
        return np.asarray(opmn.unpackb(msg)["action"]).shape
    shape = asyncio.run(go())
    return (shape[-1] == 8), f"action={shape}"


def _dummy_infer_pi_droid(host: str, port: int):
    from openpi_client import websocket_client_policy as wcp
    c = wcp.WebsocketClientPolicy(host=host, port=port)
    obs = {
        "observation/exterior_image_1_left": np.random.randint(0, 255, (224, 224, 3), np.uint8),
        "observation/wrist_image_left": np.random.randint(0, 255, (224, 224, 3), np.uint8),
        "observation/joint_position": np.zeros(7, np.float32),
        "observation/gripper_position": np.array([0.0], np.float32),
        "prompt": "preflight",
    }
    a = np.asarray(c.infer(obs)["actions"])
    return (a.ndim == 2 and a.shape[-1] == 8), f"actions={a.shape}"


def _prepare(policy: str) -> None:
    with _PREP_LOCK:
        spec = POLICIES[policy]
        _MODEL[policy] = {"status": "checking", "detail": "checking server /healthz"}
        if not _http_ok(_healthz(policy))[0]:
            if spec["location"] == "local" and "deploy" in spec:
                _MODEL[policy] = {"status": "deploying", "detail": "loading model (~1-2 min)…"}
                ok, detail = _deploy_model(policy)
                if not ok:
                    _MODEL[policy] = {"status": "fail", "detail": detail}
                    return
            else:
                _MODEL[policy] = {"status": "fail",
                                  "detail": f"remote server down at {_healthz(policy)} — set the endpoint / ask HAMILTON to deploy"}
                return
        _MODEL[policy] = {"status": "checking", "detail": "dummy inference…"}
        try:
            if spec["dummy"] == "molmobot":
                ok, detail = _dummy_infer_molmobot(_ws_url(policy))
            elif spec["dummy"] == "pi_droid":
                ok, detail = _dummy_infer_pi_droid(*_ep(policy))
            else:
                ok, detail = _dummy_infer_cosmos(_ws_url(policy))
            _MODEL[policy] = {"status": "ready" if ok else "fail",
                              "detail": ("dummy ok: " if ok else "dummy bad: ") + detail}
        except Exception as e:
            _MODEL[policy] = {"status": "fail", "detail": f"dummy infer error: {e}"[:200]}


def _prepare_async(policy: str) -> None:
    if _MODEL.get(policy, {}).get("status") in ("deploying", "checking"):
        return  # already in progress
    threading.Thread(target=_prepare, args=(policy,), daemon=True).start()


# ---------- rollout ----------
def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in (s or "").strip().lower()).strip("-")[:40]


def _launch_rollout(policy: str, task: str, save_pred: bool, identifier: str = "",
                    gripper_threshold: str = "") -> Path:
    spec = POLICIES[policy]
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    idslug = _slug(identifier)
    name = f"{ts}_{policy}" + (f"_{idslug}" if idslug else "")
    log_dir = LOGS / name
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [FRANKY_PY, str(REAL_SERVE / spec["script"]),
           "--task", task, "--no_preview", "--log_dir", str(log_dir)] + list(spec["args"]) + _endpoint_args(policy)
    # Optional custom gripper close threshold (molmobot FT: --binary_threshold).
    if gripper_threshold not in (None, ""):
        try:
            cmd += ["--binary_threshold", str(float(gripper_threshold))]
        except (TypeError, ValueError):
            pass
    if save_pred and spec.get("supports_predicted_video"):
        cmd += ["--save_predicted_video"]
    _stop_camera_service()
    # Free GPU for the ZED CUDA context: keep only THIS policy's local model
    # server (a local policy needs its server; a remote policy needs none locally).
    keep = spec["deploy"]["port"] if "deploy" in spec else None
    _stop_other_local_servers(keep_port=keep)
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
    fp = (log_dir / "service_run.log").open("w")
    fp.write("# cmd: " + " ".join(cmd) + "\n"); fp.flush()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fp,
                            stderr=subprocess.STDOUT, start_new_session=True)
    _RUN.update({"proc": proc, "policy": policy, "task": task, "log_dir": log_dir,
                 "start_ts": time.time(), "log_fp": fp})
    return log_dir


def _latest_frame(sub: str):
    ld = _RUN["log_dir"]
    if ld is None:
        return None
    files = glob.glob(str(ld / sub / "*.jpg"))
    return Path(max(files)) if files else None


def _tail(path: Path, n: int = 25):
    try:
        with path.open("r", errors="replace") as f:
            return [l.rstrip("\n") for l in f.readlines()[-n:]]
    except Exception:
        return []


# ---------------------------------------------------------------------------
app = FastAPI(title="Sim2Real Deploy Service")


@app.get("/api/policies")
def api_policies():
    out = {}
    for k, v in POLICIES.items():
        h, p = _ep(k)
        out[k] = {"label": v["label"], "location": v["location"],
                  "views": [lbl for lbl, _ in v["views"]],
                  "supports_predicted_video": v["supports_predicted_video"],
                  "default_task": v["default_task"],
                  "host": h, "port": p, "endpoint_style": v["endpoint_style"],
                  "custom": bool(v.get("custom", False))}
    return out


@app.post("/api/endpoint")
def api_endpoint(payload: dict):
    policy = payload.get("policy")
    host = (payload.get("host") or "").strip()
    port = payload.get("port")
    if policy not in POLICIES:
        raise HTTPException(400, f"unknown policy {policy!r}")
    if not host:
        raise HTTPException(400, "host is required")
    try:
        port = int(port)
    except (TypeError, ValueError):
        raise HTTPException(400, "port must be an integer")
    if _is_running():
        raise HTTPException(409, "stop the running rollout before changing the endpoint")
    _ENDPOINT[policy] = [host, port]
    _MODEL[policy] = {"status": "unknown", "detail": f"endpoint set to {host}:{port}"}
    _prepare_async(policy)   # re-check the new endpoint (health + dummy)
    return {"ok": True, "policy": policy, "host": host, "port": port,
            "ws_url": _ws_url(policy)}


@app.get("/api/preflight")
def api_preflight():
    return {**_PREFLIGHT, "ok": _preflight_ok()}


@app.post("/api/preflight")
def api_preflight_run():
    if _PREFLIGHT["running"]:
        return {"ok": True, "already_running": True}
    threading.Thread(target=_run_preflight, daemon=True).start()
    return {"ok": True}


@app.get("/api/model/{policy}")
def api_model(policy: str):
    if policy not in POLICIES:
        raise HTTPException(404, f"unknown policy {policy!r}")
    return {"policy": policy, **_MODEL[policy]}


@app.post("/api/prepare")
def api_prepare(payload: dict):
    policy = payload.get("policy")
    if policy not in POLICIES:
        raise HTTPException(400, f"unknown policy {policy!r}")
    _prepare_async(policy)
    return {"ok": True, "policy": policy}


@app.post("/api/run")
def api_run(payload: dict):
    policy = payload.get("policy")
    task = (payload.get("task") or "").strip()
    save_pred = bool(payload.get("save_predicted_video", False))
    identifier = payload.get("identifier") or ""
    gripper_threshold = payload.get("gripper_threshold") or ""
    if policy not in POLICIES:
        raise HTTPException(400, f"unknown policy {policy!r}")
    if not task:
        raise HTTPException(400, "task/instruction is required")
    if not _preflight_ok():
        raise HTTPException(409, "preflight not passed — check camera/franky/robotiq")
    if _MODEL[policy]["status"] != "ready":
        raise HTTPException(409, f"model for '{policy}' not ready ({_MODEL[policy]['status']}: {_MODEL[policy]['detail']})")
    with _LOCK:
        if _is_running():
            raise HTTPException(409, "a rollout is already running — stop it first")
        log_dir = _launch_rollout(policy, task, save_pred, identifier, gripper_threshold)
    return {"ok": True, "log_dir": str(log_dir)}


@app.post("/api/stop")
def api_stop():
    with _LOCK:
        _killpg(_RUN["proc"])
    return {"ok": True}


@app.post("/api/reset")
def api_reset():
    """Stop any running rollout and clear the experiment state so the UI is
    fresh for a new run. Does NOT move the robot or redeploy models."""
    with _LOCK:
        _killpg(_RUN["proc"])
        try:
            if _RUN["log_fp"] is not None:
                _RUN["log_fp"].close()
        except Exception:
            pass
        _RUN.update({"proc": None, "policy": None, "task": None, "log_dir": None,
                     "start_ts": None, "log_fp": None})
    _stop_camera_service()                 # free the ZEDs for the next rollout
    _stop_other_local_servers(keep_port=None)  # free ALL GPU: unload every local model server
    for k in _MODEL:                        # mark models as needing re-prepare
        if _MODEL[k].get("status") == "ready":
            _MODEL[k] = {"status": "unknown", "detail": "reset — model unloaded, re-select to reload"}
    return {"ok": True}


@app.get("/api/status")
def api_status():
    running = _is_running()
    p = _RUN["proc"]
    step_count = 0
    if _RUN["policy"]:
        for _, sub in POLICIES[_RUN["policy"]]["views"]:
            f = _latest_frame(sub)
            if f is not None:
                try:
                    step_count = max(step_count, int(f.stem))
                except ValueError:
                    pass
    return {
        "running": running,
        "policy": _RUN["policy"], "task": _RUN["task"],
        "log_dir": str(_RUN["log_dir"]) if _RUN["log_dir"] else None,
        "elapsed_s": round(time.time() - _RUN["start_ts"], 1) if _RUN["start_ts"] else 0,
        "views": [lbl for lbl, _ in POLICIES[_RUN["policy"]]["views"]] if _RUN["policy"] else [],
        "step_count": step_count,
        "exit_code": (p.poll() if p is not None else None),
        "log_tail": _tail(_RUN["log_dir"] / "service_run.log") if _RUN["log_dir"] else [],
        "preflight": {**_PREFLIGHT, "ok": _preflight_ok()},
        "models": _MODEL,
    }


@app.get("/api/frame/{view}")
def api_frame(view: str):
    if not _RUN["policy"]:
        raise HTTPException(404, "no active rollout")
    sub = dict(POLICIES[_RUN["policy"]]["views"]).get(view)
    if sub is None:
        raise HTTPException(404, f"view {view!r} not in current policy")
    f = _latest_frame(sub)
    if f is None:
        raise HTTPException(404, "no frame yet")
    return Response(content=f.read_bytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Sim2Real Deploy</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#111;color:#eee}
 header{background:#1b1b1b;padding:12px 18px;border-bottom:1px solid #333}
 h1{font-size:18px;margin:0}
 .wrap{padding:18px;max-width:1200px;margin:0 auto}
 .card{background:#181818;border:1px solid #333;border-radius:8px;padding:12px 14px;margin-bottom:14px}
 .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 select,input,button{font-size:15px;padding:8px 10px;border-radius:6px;border:1px solid #444;background:#222;color:#eee}
 input#task{flex:1;min-width:280px}
 button{cursor:pointer} #run{background:#1f7a34;border-color:#2a9d47}
 #stop{background:#7a1f1f;border-color:#9d2a2a} button:disabled{opacity:.45;cursor:not-allowed}
 .badge{padding:2px 8px;border-radius:10px;font-size:12px}
 .local{background:#22432a;color:#8f8}.remote{background:#443322;color:#fc8}
 .chk{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:14px}
 .ic{width:16px;height:16px;border-radius:50%;flex:none}
 .ok{background:#3f3}.fail{background:#f55}.warn{background:#fc4}.run{background:#4af}.pending{background:#666}
 #views{display:flex;gap:14px;flex-wrap:wrap;margin-top:6px}
 .view{background:#000;border:1px solid #333;border-radius:8px;padding:6px}
 .view h3{margin:4px 6px;font-size:13px;color:#9cf}
 .view img{display:block;width:420px;max-width:44vw;background:#000;border-radius:4px}
 pre#log{background:#000;border:1px solid #333;border-radius:8px;padding:10px;height:150px;overflow:auto;font-size:12px;color:#9d9}
 .muted{color:#999;font-size:13px}
</style></head><body>
<header><h1>Sim2Real Deployment Console</h1></header>
<div class="wrap">
  <div class="card">
    <b>System preflight</b> <button id="recheck" style="float:right">↻ re-check</button>
    <div class="chk"><span class="ic pending" id="ic_camera"></span><span id="tx_camera">camera_service…</span></div>
    <div class="chk"><span class="ic pending" id="ic_franky"></span><span id="tx_franky">franky arm service…</span></div>
    <div class="chk"><span class="ic pending" id="ic_robotiq"></span><span id="tx_robotiq">robotiq gripper service…</span></div>
    <div class="chk"><span class="ic pending" id="ic_model"></span><span id="tx_model">model server…</span></div>
  </div>
  <div class="card">
    <div class="row">
      <label>Policy</label>
      <select id="policy"></select>
      <span id="loc" class="badge"></span>
    </div>
    <div class="row" style="margin-top:10px">
      <label>Endpoint</label>
      <input id="host" style="width:150px" placeholder="host/IP">
      <span>:</span>
      <input id="port" style="width:80px" placeholder="port">
      <input id="ident" style="width:150px" placeholder="id (optional, for logs)">
      <input id="gthr" style="width:130px" placeholder="grip close thr (0.5)">
      <span id="epstyle" class="muted"></span>
      <button id="applyep">Apply &amp; re-check</button>
    </div>
    <div class="row" style="margin-top:10px">
      <label>Task</label>
      <input id="task" placeholder="instruction">
      <label style="font-size:13px"><input type="checkbox" id="predvid"> save predicted video</label>
      <button id="run" disabled>▶ Run</button>
      <button id="stop" disabled>■ Stop</button>
      <button id="reset">⟳ Reset</button>
    </div>
    <div id="status" class="muted" style="margin-top:8px">idle</div>
  </div>
  <div class="card"><b>Live views</b><div id="views"></div></div>
  <pre id="log"></pre>
</div>
<script>
let POLICIES={}, CUR=null;
const $=id=>document.getElementById(id);
function setIc(name,st,txt){ $('ic_'+name).className='ic '+(st||'pending'); if(txt!==undefined)$('tx_'+name).textContent=txt; }
async function loadPolicies(){
  POLICIES=await (await fetch('/api/policies')).json();
  const sel=$('policy'); sel.innerHTML='';
  for(const [k,v] of Object.entries(POLICIES)){ const o=document.createElement('option'); o.value=k; o.textContent=v.label; sel.appendChild(o); }
  onPolicyChange();
}
function buildViews(views){ const c=$('views'); c.innerHTML=''; for(const v of views){ const d=document.createElement('div'); d.className='view'; d.innerHTML=`<h3>${v}</h3><img id="img_${v}">`; c.appendChild(d); } }
async function onPolicyChange(){
  CUR=$('policy').value; const p=POLICIES[CUR]; if(!p) return;
  $('loc').textContent=p.location; $('loc').className='badge '+p.location;
  $('task').value=p.default_task||'';
  $('host').value=p.host; $('port').value=p.port;
  $('ident').style.display = p.custom ? '' : 'none';   // id box only for custom policy
  $('gthr').style.display  = p.custom ? '' : 'none';   // gripper-threshold box only for custom
  if(!p.custom){ $('ident').value=''; $('gthr').value=''; }
  $('epstyle').textContent='('+(p.endpoint_style==='a100'?'--a100_host/--a100_port':'--ws_url')+')';
  $('predvid').disabled=!p.supports_predicted_video; if(!p.supports_predicted_video)$('predvid').checked=false;
  buildViews(p.views);
  await fetch('/api/prepare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({policy:CUR})});
}
async function applyEndpoint(){
  const r=await fetch('/api/endpoint',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({policy:CUR,host:$('host').value,port:$('port').value})});
  if(!r.ok){ const e=await r.json(); alert('Endpoint change failed: '+(e.detail||r.status)); return; }
  const j=await r.json(); POLICIES[CUR].host=j.host; POLICIES[CUR].port=j.port;
}
async function run(){
  const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({policy:CUR,task:$('task').value,save_predicted_video:$('predvid').checked,identifier:$('ident').value,gripper_threshold:$('gthr').value})});
  if(!r.ok){ const e=await r.json(); alert('Run failed: '+(e.detail||r.status)); }
}
async function stop(){ await fetch('/api/stop',{method:'POST'}); }
async function reset(){
  if(!confirm('Reset: stop any running rollout and clear the experiment for a new run?')) return;
  await fetch('/api/reset',{method:'POST'});
  document.querySelectorAll('#views img').forEach(i=>i.removeAttribute('src'));
  $('status').textContent='reset — ready for a new experiment';
}
function poll(){ fetch('/api/status').then(r=>r.json()).then(s=>{
  const pf=s.preflight||{};
  for(const k of ['camera','franky','robotiq']){ const c=pf[k]||{}; setIc(k,c.status,`${k}: ${c.status}${c.detail?' — '+c.detail:''}`); }
  const m=(s.models||{})[CUR]||{status:'unknown',detail:''};
  const mst=m.status==='ready'?'ok':(m.status==='fail'?'fail':(m.status==='deploying'||m.status==='checking'?'run':'pending'));
  setIc('model',mst,`model (${CUR}): ${m.status}${m.detail?' — '+m.detail:''}`);
  const ready=pf.ok && m.status==='ready';
  $('run').disabled=s.running||!ready; $('stop').disabled=!s.running; $('policy').disabled=s.running;
  if(s.running){ $('status').textContent=`RUNNING ${s.policy} — "${s.task}" | step ${s.step_count} | ${s.elapsed_s}s | ${s.log_dir}`;
    for(const v of s.views){ const img=$('img_'+v); if(img) img.src='/api/frame/'+v+'?ts='+Date.now(); } }
  else{ $('status').textContent = ready?'ready to run':(s.log_dir?`stopped (exit=${s.exit_code}) — ${s.log_dir}`:'waiting for checks…'); }
  $('log').textContent=(s.log_tail||[]).join('\\n'); $('log').scrollTop=$('log').scrollHeight;
}).catch(()=>{ $('status').textContent='service unreachable'; }); }
$('policy').addEventListener('change',onPolicyChange);
$('applyep').addEventListener('click',applyEndpoint);
$('run').addEventListener('click',run); $('stop').addEventListener('click',stop);
$('reset').addEventListener('click',reset);
$('recheck').addEventListener('click',async()=>{ await fetch('/api/preflight',{method:'POST'}); if(CUR) await fetch('/api/prepare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({policy:CUR})}); });
loadPolicies(); setInterval(poll,800);
</script></body></html>"""


def _startup_preflight():
    _run_preflight()
    if _preflight_ok():
        _prepare(DEFAULT_POLICY)   # deploy + dummy-check the default local model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7070)
    args = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    print(f"[s2r-service] http://{args.host}:{args.port}  — running startup preflight…")
    threading.Thread(target=_startup_preflight, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
