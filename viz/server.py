"""Viser-based UI for cuRobo motion planning + Franka execution.

- Loads the Franka Panda URDF in a viser scene.
- Polls franky_service for live joint state and updates the visualization.
- Goal gizmo snaps to current EE on connect (and via "Snap to EE" button).
- Plan: runs cuRobo, draws a translucent polyline of the planned EE path.
- Execute: streams the planned trajectory to /target_joint_state.

Open http://localhost:8080 in a browser.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import requests
import torch
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState

# ---------- Config ----------

FRANKY_URL = os.environ.get("FRANKY_SERVICE_URL", "http://127.0.0.1:54321")
HAND_URL = os.environ.get("FRANKA_HAND_SERVICE_URL", "http://127.0.0.1:54324")
# --- Gripper (Robotiq 2F-85 by default; falls back to Franka Hand) ---------
# The rig currently runs the Robotiq HTTP service locally on :54323; the
# Franka Hand service on the NUC (:54324) is the legacy path.
GRIPPER_KIND = os.environ.get("GRIPPER_KIND", "franka_hand").strip().lower()
ROBOTIQ_URL = os.environ.get(
    "ROBOTIQ_SERVICE_URL", "http://localhost:54323"
)
# 2F-85 register defaults for UI clicks (0..255 each). 255 = max.
ROBOTIQ_SPEED_255 = int(os.environ.get("ROBOTIQ_SPEED_255", "255"))
ROBOTIQ_FORCE_255 = int(os.environ.get("ROBOTIQ_FORCE_255", "255"))
URDF_PATH = Path(os.environ.get(
    "FRANKA_URDF_PATH",
    str(Path(__file__).resolve().parent / "assets" / "franka_panda.urdf"),
))
COMMAND_TIMEOUT_S = 1.0
JOINT_POLL_HZ = float(os.environ.get("JOINT_POLL_HZ", "10.0"))
# Franky HTTP timeouts (seconds). The service serialises libfranka reads on a
# single mutex, so under contention with e.g. run_policy.py the response can
# spike to ~1 s. Old defaults (0.5 s) tripped constantly; these are looser.
FRANKY_READ_TIMEOUT_S = float(os.environ.get("FRANKY_READ_TIMEOUT_S", "2.0"))
FRANKY_WRITE_TIMEOUT_S = float(os.environ.get("FRANKY_WRITE_TIMEOUT_S", "3.0"))
# Joint-space "home" pose — must match HOME_POSITION in franky_service.py.
HOME_Q: list[float] = [0.0, -0.40, 0.0, -1.9, 0.0, 1.5, 0.0]


# ---------- Planner ----------

print("[ui] loading cuRobo Franka planner...")
mp_cfg = MotionPlannerCfg.create(robot="franka.yml")
planner = MotionPlanner(mp_cfg)
planner.warmup(enable_graph=True, num_warmup_iterations=5)
print(f"[ui] tool frames: {planner.tool_frames}")
print(f"[ui] joint names: {planner.joint_names}")


# ---------- Franky helpers ----------

# Rate-limit noisy "joint_state error" prints under contention with e.g.
# run_policy.py. Log once per N seconds instead of on every failed poll.
_JS_LAST_ERR_TS: float = 0.0
_JS_ERR_STREAK: int = 0
_JS_ERR_PRINT_EVERY_S: float = 5.0


def get_joint_state() -> tuple[list[float] | None, bool]:
    global _JS_LAST_ERR_TS, _JS_ERR_STREAK
    try:
        r = requests.get(f"{FRANKY_URL}/joint_state",
                         timeout=FRANKY_READ_TIMEOUT_S)
        r.raise_for_status()
        pos = list(r.json()["positions"])[:7]
        h = requests.get(f"{FRANKY_URL}/health",
                         timeout=FRANKY_READ_TIMEOUT_S).json()
        if _JS_ERR_STREAK > 0:
            print(f"[franky] joint_state recovered "
                  f"(after {_JS_ERR_STREAK} failed poll(s))")
            _JS_ERR_STREAK = 0
        return pos, bool(h.get("stop_latched", False))
    except Exception as e:
        _JS_ERR_STREAK += 1
        now = time.time()
        if now - _JS_LAST_ERR_TS > _JS_ERR_PRINT_EVERY_S:
            print(f"[franky] joint_state error x{_JS_ERR_STREAK} "
                  f"(silencing similar errors for {_JS_ERR_PRINT_EVERY_S:.0f}s): {e}")
            _JS_LAST_ERR_TS = now
        return None, True


def get_health() -> dict | None:
    try:
        r = requests.get(f"{FRANKY_URL}/health",
                         timeout=FRANKY_READ_TIMEOUT_S)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[franky] health error: {e}")
        return None


def send_target(positions: list[float]) -> bool:
    try:
        r = requests.post(
            f"{FRANKY_URL}/target_joint_state",
            json={"positions": positions, "velocities": [0.0] * 7},
            timeout=FRANKY_WRITE_TIMEOUT_S,
        )
        return r.ok
    except Exception as e:
        print(f"[franky] send_target error: {e}")
        return False


def franky_stop() -> None:
    try:
        requests.post(f"{FRANKY_URL}/stop", timeout=FRANKY_WRITE_TIMEOUT_S)
    except Exception as e:
        print(f"[franky] stop error: {e}")


def franky_go_home() -> None:
    try:
        # /go_home returns after the motion starts, not after it completes;
        # but under contention the accept-and-return itself can take 1-2 s.
        requests.post(f"{FRANKY_URL}/go_home",
                      timeout=max(3.0, FRANKY_WRITE_TIMEOUT_S))
    except Exception as e:
        print(f"[franky] go_home error: {e}")


def _robotiq_open() -> None:
    """Robotiq 2F-85: POST /open?wait=false to the local gripper_service."""
    try:
        requests.post(
            f"{ROBOTIQ_URL}/open?wait=false",
            json={"position": 0,
                  "speed": ROBOTIQ_SPEED_255,
                  "force": ROBOTIQ_FORCE_255},
            timeout=1.0,
        )
    except Exception as e:
        print(f"[robotiq] open error: {e}")


def _robotiq_close() -> None:
    """Robotiq 2F-85: POST /close?wait=false to the local gripper_service."""
    try:
        requests.post(
            f"{ROBOTIQ_URL}/close?wait=false",
            json={"position": 255,
                  "speed": ROBOTIQ_SPEED_255,
                  "force": ROBOTIQ_FORCE_255},
            timeout=1.0,
        )
    except Exception as e:
        print(f"[robotiq] close error: {e}")


def _panda_hand_open(speed: float = 0.1) -> None:
    try:
        requests.post(
            f"{HAND_URL}/gripper_open",
            json={"speed": speed},
            timeout=1.0,
        )
    except Exception as e:
        print(f"[hand] open error: {e}")


def _panda_hand_close(width_m: float = 0.0, speed: float = 0.1,
                      force_n: float = 20.0) -> None:
    try:
        requests.post(
            f"{HAND_URL}/gripper_grasp",
            json={"width_m": width_m, "speed": speed,
                  "force": force_n,
                  "epsilon_inner": 0.02, "epsilon_outer": 0.05},
            timeout=1.0,
        )
    except Exception as e:
        print(f"[hand] close error: {e}")


def hand_open(speed: float = 0.1) -> None:
    """Kind-aware gripper-open dispatch used by UI buttons."""
    if GRIPPER_KIND == "robotiq":
        _robotiq_open()
    else:
        _panda_hand_open(speed=speed)


def hand_close(speed: float = 0.1) -> None:
    """Kind-aware gripper-close dispatch used by UI buttons."""
    if GRIPPER_KIND == "robotiq":
        _robotiq_close()
    else:
        _panda_hand_close(speed=speed)


def set_command_timeout(t: float) -> None:
    try:
        requests.post(
            f"{FRANKY_URL}/command_timeout",
            json={"command_timeout_s": t},
            timeout=FRANKY_WRITE_TIMEOUT_S,
        )
    except Exception as e:
        print(f"[franky] command_timeout error: {e}")


HOME_Q = [0.0, -0.4, 0.0, -1.9, 0.0, 1.5, 0.0]
HOME_TOL = 0.02   # rad; per-joint inf-norm tolerance to declare "at home"


def monitor_motion(label: str, max_seconds: float = 15.0,
                   target_q: list[float] | None = None,
                   target_tol: float = HOME_TOL) -> None:
    """Poll /health while a motion is in flight and log notable transitions.

    Stops when:
      - target_q is given AND ||q - target_q||_inf < target_tol
        (send /stop early to cut the deceleration tail), or
      - the joint state stops changing significantly for ~1 s, or
      - stop_latched flips True, or
      - max_seconds elapses.
    """
    print(f"[mon:{label}] starting health monitor")
    t0 = time.time()
    last_q: list[float] | None = None
    still_since: float | None = None
    last_stop_latched: bool | None = None
    last_err: object = object()
    while time.time() - t0 < max_seconds:
        try:
            h = requests.get(f"{FRANKY_URL}/health",
                             timeout=FRANKY_READ_TIMEOUT_S).json()
            js = requests.get(f"{FRANKY_URL}/joint_state",
                              timeout=FRANKY_READ_TIMEOUT_S).json()
        except Exception as e:
            print(f"[mon:{label}] health/joint poll failed: {e}")
            time.sleep(0.2)
            continue

        sl = bool(h.get("stop_latched", False))
        err = h.get("last_control_error")
        q = list(js["positions"])[:7]
        vel = list(js.get("velocities", [0.0] * 7))[:7]
        speed = float(max(abs(v) for v in vel)) if vel else 0.0

        if last_stop_latched is None or sl != last_stop_latched:
            print(f"[mon:{label}] t={time.time()-t0:5.2f}s stop_latched={sl}")
            last_stop_latched = sl
        if err != last_err:
            print(f"[mon:{label}] t={time.time()-t0:5.2f}s last_control_error={err!r}")
            last_err = err
        if sl:
            print(f"[mon:{label}] stop latched after {time.time()-t0:.2f}s — bailing")
            return

        if target_q is not None:
            err_to_target = max(abs(a - b) for a, b in zip(q, target_q))
            if err_to_target < target_tol:
                print(f"[mon:{label}] within {target_tol} rad of target at "
                      f"t={time.time()-t0:.2f}s (err={err_to_target:.4f}) — "
                      f"sending /stop")
                try:
                    requests.post(f"{FRANKY_URL}/stop",
                                  timeout=FRANKY_WRITE_TIMEOUT_S)
                except Exception as e:
                    print(f"[mon:{label}] stop failed: {e}")
                return

        if last_q is not None:
            delta = max(abs(a - b) for a, b in zip(q, last_q))
        else:
            delta = float("inf")
        last_q = q

        if delta < 1e-3 and speed < 1e-3:
            if still_since is None:
                still_since = time.time()
            elif time.time() - still_since > 1.0:
                print(
                    f"[mon:{label}] settled at t={time.time()-t0:.2f}s "
                    f"q={[round(x,3) for x in q]}"
                )
                return
        else:
            still_since = None
        time.sleep(0.1)
    print(f"[mon:{label}] timed out after {max_seconds}s")


# ---------- Kinematics helpers ----------

def fk(q_arm: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (position[3], quaternion_wxyz[4]) for the EE."""
    js = JointState.from_position(
        torch.tensor([q_arm], device="cuda", dtype=torch.float32),
        joint_names=planner.joint_names,
    )
    tp = planner.compute_kinematics(js).tool_poses
    # tp.position: [batch=1, horizon=1, num_links=1, 3]; same for quaternion
    pos = tp.position[0, 0, 0].detach().cpu().numpy()
    quat = tp.quaternion[0, 0, 0].detach().cpu().numpy()
    return pos, quat


def fk_traj(q_traj: np.ndarray) -> np.ndarray:
    """FK over a [T, dof] joint trajectory -> [T, 3] EE positions."""
    # cuRobo plans on 7 active arm joints; trim any extras (e.g. locked fingers).
    q7 = q_traj[:, : len(planner.joint_names)]
    js = JointState.from_position(
        torch.tensor(q7, device="cuda", dtype=torch.float32),
        joint_names=planner.joint_names,
    )
    tp = planner.compute_kinematics(js).tool_poses
    # tp.position: [T, 1, 1, 3]
    return tp.position[:, 0, 0, :].detach().cpu().numpy()


# ---------- Policy server health check ----------
# Lets the UI point at a locally- or remotely-served policy (MolmoBot bridge or
# Cosmos3 RoboLab server), enter its host/port, and confirm it works by sending
# the SAME dummy observation the smoke tests use and validating the response.
import asyncio

import msgpack_numpy as _msgpack_numpy
import websockets as _websockets

_PS_CONNECT_TIMEOUT_S = 5.0
_PS_INFER_TIMEOUT_S = 40.0   # first dummy inference (cold/compile) can be slow
_PS_OK = "\u2705"
_PS_FAIL = "\u274c"


def _ps_rand_img(h: int = 480, w: int = 480) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _ps_dummy_molmobot(exo_key: str, wrist_key: str) -> dict:
    rng = np.random.default_rng(0)
    return {
        exo_key: _ps_rand_img(),
        wrist_key: _ps_rand_img(),
        "qpos": {
            "arm": rng.standard_normal(7).astype(np.float32),
            "gripper": np.array([0.02], dtype=np.float32),
        },
        "task": "health check",
    }


def _ps_dummy_cosmos() -> dict:
    rng = np.random.default_rng(0)
    return {
        "prompt": "health check",
        "observation/wrist_image_left": _ps_rand_img(),
        "observation/exterior_image_1_left": _ps_rand_img(),
        "observation/exterior_image_2_left": _ps_rand_img(),
        "observation/joint_position": rng.standard_normal(7).astype(np.float32),
        "observation/gripper_position": np.array([0.0], dtype=np.float32),
    }


# Registry of servable policies. host/port are just DEFAULTS that prefill the
# editable Host/Port fields; a "server" (remote) model like Cosmos3 is reached by
# typing its box's IP into the Host field.
POLICY_MODELS: dict[str, dict] = {
    "MolmoBot Official (DROID)": dict(
        host="127.0.0.1", port=8000, protocol="molmobot",
        make_obs=lambda: _ps_dummy_molmobot("exo_camera_1", "wrist_camera"),
    ),
    "MolmoBot v4full (finetuned)": dict(
        host="127.0.0.1", port=8005, protocol="molmobot",
        make_obs=lambda: _ps_dummy_molmobot("exo_front", "wrist"),
    ),
    "Cosmos3-Nano-Policy-DROID (server)": dict(
        host="127.0.0.1", port=8010, protocol="cosmos",
        make_obs=_ps_dummy_cosmos,
    ),
}


def _ps_healthz(host: str, port: int, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        r = requests.get(f"http://{host}:{port}/healthz", timeout=timeout)
        return r.ok, (r.text.strip() or f"HTTP {r.status_code}")
    except requests.exceptions.ConnectionError:
        return False, "connection refused (server not up / wrong host:port)"
    except requests.exceptions.Timeout:
        return False, f"timeout after {timeout:.0f}s"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


async def _ps_ws_roundtrip(url: str, obs: dict) -> tuple[object, object]:
    async with _websockets.connect(
        url, compression=None, max_size=None, open_timeout=_PS_CONNECT_TIMEOUT_S
    ) as ws:
        meta = _msgpack_numpy.unpackb(await asyncio.wait_for(ws.recv(), _PS_INFER_TIMEOUT_S))
        await ws.send(_msgpack_numpy.Packer().pack(obs))
        resp = _msgpack_numpy.unpackb(await asyncio.wait_for(ws.recv(), _PS_INFER_TIMEOUT_S))
        return meta, resp


def _ps_check(name: str, host: str, port: int) -> str:
    """Full health check: /healthz + one dummy WS inference. Returns markdown."""
    spec = POLICY_MODELS[name]
    lines = [f"**{name}** @ `ws://{host}:{port}`"]

    ok, msg = _ps_healthz(host, port)
    lines.append("- healthz: " + (_PS_OK + " OK" if ok else _PS_FAIL + " " + msg))

    try:
        obs = spec["make_obs"]()
        t0 = time.time()
        loop = asyncio.new_event_loop()
        try:
            meta, resp = loop.run_until_complete(
                _ps_ws_roundtrip(f"ws://{host}:{port}", obs)
            )
        finally:
            loop.close()
        dt_ms = (time.time() - t0) * 1000.0

        if not isinstance(resp, dict):
            raise ValueError(f"non-dict response ({type(resp).__name__}) "
                             f"- server likely errored: {str(resp)[:120]}")

        if spec["protocol"] == "cosmos":
            act = np.asarray(resp["action"])
            lines.append(f"- dummy inference: \u2705 action {act.shape} in {dt_ms:.0f} ms")
        else:
            arm = np.asarray(resp["arm"])
            grip = np.asarray(resp["gripper"])
            chunk = bool(meta.get("chunk_response")) if isinstance(meta, dict) else False
            cams = meta.get("camera_names") if isinstance(meta, dict) else None
            lines.append(f"- dummy inference: \u2705 arm {arm.shape} gripper {grip.shape} "
                         f"in {dt_ms:.0f} ms (chunk_response={chunk}, cams={cams})")
        lines.append("**Result: SERVER OK \u2705**")
    except Exception as e:
        lines.append(f"- dummy inference: \u274c {type(e).__name__}: {e}")
        lines.append("**Result: FAILED \u274c**")
    return "  \n".join(lines)


# ---------- Viser scene ----------

server = viser.ViserServer(host="0.0.0.0", port=8080, label="Franka cuRobo UI")
server.scene.world_axes.visible = True

urdf_model = URDF.load(str(URDF_PATH))
viser_urdf = ViserUrdf(server, urdf_or_path=urdf_model, root_node_name="/robot")

current_q_arm: list[float] | None = None
gizmo_initialized = False

goal_gizmo = server.scene.add_transform_controls(
    "/goal", position=(0.4, 0.0, 0.5), wxyz=(1.0, 0.0, 0.0, 0.0), scale=0.25
)
server.scene.add_frame(
    "/goal/marker", axes_length=0.08, axes_radius=0.004, origin_radius=0.012
)

# Planned-path preview (a translucent polyline, redrawn each plan)
preview_handle = None
latest_traj: np.ndarray | None = None  # [T, 7] arm joint targets
latest_dt: float | None = None

# GUI
status = server.gui.add_markdown("**Status:** waiting for joint state…")
btn_snap = server.gui.add_button("Snap goal to current EE")
btn_plan = server.gui.add_button("Plan")
btn_exec = server.gui.add_button("Execute")
btn_exec.disabled = True
btn_home = server.gui.add_button("Go Home (franky)")
btn_open_grip = server.gui.add_button("Open gripper")
btn_close_grip = server.gui.add_button("Close gripper")
btn_stop = server.gui.add_button("STOP")

# Waypoints (session-only). Each entry is a 7-dof joint configuration captured
# from the live robot at the moment "Add" was pressed.
wp_folder = server.gui.add_folder("Waypoints")
with wp_folder:
    btn_wp_add = server.gui.add_button("Add current pose")
    btn_wp_cycle = server.gui.add_button("Cycle: start")
    btn_wp_clear = server.gui.add_button("Clear all")
    wp_status = server.gui.add_markdown("_(empty)_")

# Policy server: pick a model, enter host/port (defaults prefilled per model),
# and health-check it with a dummy observation matching that server's schema.
ps_folder = server.gui.add_folder("Policy Server")
with ps_folder:
    _ps_first = next(iter(POLICY_MODELS))
    ps_model = server.gui.add_dropdown(
        "Model", options=tuple(POLICY_MODELS.keys()), initial_value=_ps_first
    )
    ps_host = server.gui.add_text("Host / IP", initial_value=POLICY_MODELS[_ps_first]["host"])
    ps_port = server.gui.add_number(
        "Port", initial_value=POLICY_MODELS[_ps_first]["port"], min=1, max=65535, step=1
    )
    btn_ps_check = server.gui.add_button("Check health")
    ps_status = server.gui.add_markdown("_(idle)_")

waypoints: list[list[float]] = []
cycle_stop = threading.Event()
cycle_stop.set()  # not running
cycle_thread: threading.Thread | None = None

exec_lock = threading.Lock()


# ---------- Polling ----------

def poll_joint_state() -> None:
    global current_q_arm, gizmo_initialized
    actuated = viser_urdf.get_actuated_joint_names()
    period = 1.0 / JOINT_POLL_HZ
    while True:
        pos, stopped = get_joint_state()
        if pos is not None:
            current_q_arm = pos
            cfg_map = {name: v for name, v in zip(planner.joint_names, pos)}
            cfg_map.setdefault("panda_finger_joint1", 0.04)
            cfg_map.setdefault("panda_finger_joint2", 0.04)
            try:
                viser_urdf.update_cfg(
                    np.array([cfg_map.get(n, 0.0) for n in actuated])
                )
            except Exception as e:
                print(f"[ui] update_cfg error: {e}")
            # On first joint state, snap gizmo to current EE pose.
            if not gizmo_initialized:
                try:
                    p, q = fk(pos)
                    goal_gizmo.position = (float(p[0]), float(p[1]), float(p[2]))
                    goal_gizmo.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    gizmo_initialized = True
                    print(f"[ui] snapped gizmo to current EE: {p}")
                except Exception as e:
                    print(f"[ui] initial FK failed: {e}")
            tag = " [STOPPED]" if stopped else ""
            status.content = (
                f"**Status:** connected{tag}  \n"
                f"q = {[round(x, 3) for x in pos]}"
            )
        time.sleep(period)


threading.Thread(target=poll_joint_state, daemon=True).start()


# ---------- Plan / Execute ----------

def do_snap(_) -> None:
    if current_q_arm is None:
        status.content = "**Status:** no joint state yet."
        return
    try:
        p, q = fk(current_q_arm)
        goal_gizmo.position = (float(p[0]), float(p[1]), float(p[2]))
        goal_gizmo.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        status.content = "**Status:** goal snapped to current EE."
    except Exception as e:
        status.content = f"**Status:** snap failed: {e}"


def do_plan(_) -> None:
    global latest_traj, latest_dt, preview_handle
    if current_q_arm is None:
        status.content = "**Status:** no joint state — cannot plan."
        return

    q_start = JointState.from_position(
        torch.tensor([current_q_arm], device="cuda", dtype=torch.float32),
        joint_names=planner.joint_names,
    )
    p = goal_gizmo.position
    w = goal_gizmo.wxyz
    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor(
            [[[[[float(p[0]), float(p[1]), float(p[2])]]]]],
            device="cuda", dtype=torch.float32,
        ),
        quaternion=torch.tensor(
            [[[[[float(w[0]), float(w[1]), float(w[2]), float(w[3])]]]]],
            device="cuda", dtype=torch.float32,
        ),
    )

    status.content = "**Status:** planning…"
    btn_exec.disabled = True
    t0 = time.time()
    result = planner.plan_pose(goal, q_start)
    plan_t = time.time() - t0
    if result is None or not result.success.any():
        status.content = f"**Status:** plan FAILED ({plan_t:.2f}s)"
        return

    interp = result.get_interpolated_plan()
    # interp.position is [..., T, dof] (4D in current cuRobo); squeeze to [T, dof].
    pos_t = interp.position.detach().cpu().numpy()
    while pos_t.ndim > 2:
        pos_t = pos_t[0]
    q_traj = pos_t  # [T, dof] — dof may include locked fingers
    dt = planner.trajopt_solver.config.interpolation_dt

    # Compute EE polyline for preview.
    ee_pts = fk_traj(q_traj).astype(np.float32)  # [T, 3]
    # add_spline_catmull_rom would be ideal; viser has add_point_cloud + lines via add_line_segments.
    # Use add_spline_catmull_rom if available, else fall back to line segments.
    global preview_handle
    if preview_handle is not None:
        try:
            preview_handle.remove()
        except Exception:
            pass
        preview_handle = None
    try:
        preview_handle = server.scene.add_spline_catmull_rom(
            "/plan_preview", positions=ee_pts, color=(0, 200, 255), line_width=3.0
        )
    except AttributeError:
        # Fallback: connect consecutive points as line segments.
        segs = np.stack([ee_pts[:-1], ee_pts[1:]], axis=1).astype(np.float32)
        colors = np.tile(np.array([0, 200, 255], dtype=np.uint8), (segs.shape[0], 1))
        preview_handle = server.scene.add_line_segments(
            "/plan_preview", points=segs, colors=colors, line_width=3.0
        )

    latest_traj = q_traj
    latest_dt = dt
    btn_exec.disabled = False
    status.content = (
        f"**Status:** plan OK ({plan_t:.2f}s) — {q_traj.shape[0]} waypoints, "
        f"{q_traj.shape[0] * dt:.2f}s. Click **Execute** to run."
    )


def do_execute(_) -> None:
    if latest_traj is None or latest_dt is None:
        status.content = "**Status:** no plan to execute. Click **Plan** first."
        return
    threading.Thread(target=_execute_worker, daemon=True).start()


def _stream_trajectory(q_traj: np.ndarray, dt: float, label: str) -> None:
    """Stream a [T, dof>=7] joint trajectory to franky_service at `dt` cadence."""
    traj = q_traj[:, : len(planner.joint_names)]  # 7 arm joints only
    status.content = f"**Status:** {label} ({traj.shape[0]} waypoints)…"
    set_command_timeout(COMMAND_TIMEOUT_S)
    with exec_lock:
        for i in range(traj.shape[0]):
            send_target(traj[i].tolist())
            time.sleep(dt)
        for _ in range(int(0.3 / dt)):
            send_target(traj[-1].tolist())
            time.sleep(dt)
    status.content = f"**Status:** {label} done."


def _execute_worker() -> None:
    assert latest_traj is not None and latest_dt is not None
    threading.Thread(target=monitor_motion, args=("execute",), kwargs={"max_seconds": 20.0}, daemon=True).start()
    _stream_trajectory(latest_traj, latest_dt, "executing")


def do_home(_) -> None:
    """Delegate to franky_service /go_home.

    Same behavior as the franky dashboard's Go Home: set a single static
    target and let the franky control loop drive there. Bump the command
    timeout so the motion can actually finish before the loop emits a
    JointStopMotion. Restored to the streaming-safe value when the user
    next runs Plan/Execute.
    """
    status.content = "**Status:** Go-home sent."
    set_command_timeout(1e6)
    franky_go_home()
    threading.Thread(
        target=monitor_motion, args=("home",),
        kwargs={"max_seconds": 20.0, "target_q": HOME_Q, "target_tol": HOME_TOL},
        daemon=True,
    ).start()


def do_stop(_) -> None:
    status.content = "**Status:** STOP sent."
    franky_stop()


def do_open_grip(_) -> None:
    status.content = (
        f"**Status:** opening gripper… ({GRIPPER_KIND})"
    )
    hand_open()


def do_close_grip(_) -> None:
    status.content = (
        f"**Status:** closing gripper… ({GRIPPER_KIND})"
    )
    hand_close()


# ---------- Waypoints ----------

def _render_waypoint_list() -> None:
    if not waypoints:
        wp_status.content = "_(empty)_"
        return
    lines = ["**Waypoints:**"]
    for i, q in enumerate(waypoints):
        lines.append(f"{i}. q = {[round(x, 3) for x in q]}")
    wp_status.content = "  \n".join(lines)


def do_wp_add(_) -> None:
    if current_q_arm is None:
        wp_status.content = "**no joint state — cannot capture.**"
        return
    waypoints.append(list(current_q_arm))
    _render_waypoint_list()


def do_wp_clear(_) -> None:
    cycle_stop.set()
    btn_wp_cycle.label = "Cycle: start"
    waypoints.clear()
    _render_waypoint_list()


def _plan_leg(q_from: list[float], q_to: list[float]) -> np.ndarray | None:
    """Plan cspace q_from -> q_to. Returns a [T, dof] trajectory or None."""
    q_start = JointState.from_position(
        torch.tensor([q_from], device="cuda", dtype=torch.float32),
        joint_names=planner.joint_names,
    )
    q_goal = JointState.from_position(
        torch.tensor([q_to], device="cuda", dtype=torch.float32),
        joint_names=planner.joint_names,
    )
    result = planner.plan_cspace(q_goal, q_start)
    if result is None or not result.success.any():
        return None
    interp = result.get_interpolated_plan()
    pos_t = interp.position.detach().cpu().numpy()
    while pos_t.ndim > 2:
        pos_t = pos_t[0]
    return pos_t


def _stream_no_hold(q_traj: np.ndarray, dt: float, label: str) -> None:
    """Like _stream_trajectory but without the 0.3 s end-hold.

    Also bails out if franky_service reports a control error mid-stream.
    """
    traj = q_traj[:, : len(planner.joint_names)]
    status.content = f"**Status:** {label} ({traj.shape[0]} waypoints)…"
    set_command_timeout(COMMAND_TIMEOUT_S)
    last_health_check = time.time() + 0.3  # let the first send clear stop latch
    with exec_lock:
        for i in range(traj.shape[0]):
            if cycle_stop.is_set():
                return
            now = time.time()
            if now - last_health_check > 0.25:
                last_health_check = now
                h = get_health()
                if h and h.get("last_control_error"):
                    err = h["last_control_error"]
                    status.content = f"**Status:** {label} ABORTED — reflex: `{err[:120]}…`"
                    cycle_stop.set()
                    return
            send_target(traj[i].tolist())
            time.sleep(dt)


def _cycle_worker() -> None:
    print(f"[cycle] starting with {len(waypoints)} waypoints")
    dt = planner.trajopt_solver.config.interpolation_dt
    while not cycle_stop.is_set():
        if not waypoints:
            break
        # Build one full loop: current -> wp[0] -> wp[1] -> ... -> wp[N-1] -> wp[0]
        # so the next iteration's start equals the previous end.
        if current_q_arm is None:
            print("[cycle] no joint state; aborting")
            break
        legs_from = [list(current_q_arm)] + [list(w) for w in waypoints]
        legs_to = [list(w) for w in waypoints] + [list(waypoints[0])]
        segments: list[np.ndarray] = []
        for idx, (qf, qt) in enumerate(zip(legs_from, legs_to)):
            seg = _plan_leg(qf, qt)
            if seg is None:
                print(f"[cycle] plan failed for leg {idx} ({qf} -> {qt}) — skipping")
                continue
            if segments:
                # Smooth the seam: linearly interpolate ~5 samples between the
                # end of the previous leg and the start of this one. Each leg
                # ends and starts at ~zero velocity in its own plan, but the
                # next leg's second sample already has non-trivial velocity, so
                # a hard concat trips libfranka's velocity/accel discontinuity
                # reflex. Drop seg[0] (duplicate of prev end) and prepend a
                # short linear bridge.
                seam_steps = 5
                q_prev_end = segments[-1][-1, : len(planner.joint_names)]
                q_next_start = seg[0, : len(planner.joint_names)]
                bridge_arm = np.linspace(
                    q_prev_end, q_next_start, seam_steps + 2
                )[1:-1]  # drop the two endpoints (already present)
                # Match the leg's dof by padding with the leg's first row for
                # any locked finger columns.
                if seg.shape[1] > len(planner.joint_names):
                    extras = np.tile(
                        seg[0, len(planner.joint_names):], (seam_steps, 1)
                    )
                    bridge = np.concatenate([bridge_arm, extras], axis=1)
                else:
                    bridge = bridge_arm
                segments.append(bridge.astype(seg.dtype))
                seg = seg[1:]
            segments.append(seg)
        if not segments:
            print("[cycle] all legs failed; stopping")
            break
        full = np.concatenate(segments, axis=0)
        status.content = (
            f"**Status:** cycling — {full.shape[0]} pts, {full.shape[0]*dt:.2f}s, "
            f"{len(segments)} legs"
        )
        _stream_no_hold(full, dt, "cycling")
    print("[cycle] stopped")
    btn_wp_cycle.label = "Cycle: start"


def do_wp_cycle(_) -> None:
    global cycle_thread
    if not cycle_stop.is_set():
        # currently running -> stop
        cycle_stop.set()
        btn_wp_cycle.label = "Cycle: start"
        return
    if not waypoints:
        wp_status.content = "**no waypoints to cycle.**"
        return
    cycle_stop.clear()
    btn_wp_cycle.label = "Cycle: stop"
    cycle_thread = threading.Thread(target=_cycle_worker, daemon=True)
    cycle_thread.start()


btn_snap.on_click(do_snap)
btn_plan.on_click(do_plan)
btn_exec.on_click(do_execute)
btn_home.on_click(do_home)
btn_open_grip.on_click(do_open_grip)
btn_close_grip.on_click(do_close_grip)
btn_stop.on_click(do_stop)
btn_wp_add.on_click(do_wp_add)
btn_wp_cycle.on_click(do_wp_cycle)
btn_wp_clear.on_click(do_wp_clear)


def _on_ps_model(_) -> None:
    """Prefill Host/Port with the selected model's defaults."""
    spec = POLICY_MODELS[ps_model.value]
    ps_host.value = spec["host"]
    ps_port.value = spec["port"]
    ps_status.content = "_(idle)_"


def do_ps_check(_) -> None:
    name = ps_model.value
    host = ps_host.value.strip()
    port = int(ps_port.value)
    ps_status.content = f"**Checking** {name} @ `ws://{host}:{port}` \u2026"

    def worker() -> None:
        try:
            ps_status.content = _ps_check(name, host, port)
        except Exception as e:  # never let the worker die silently
            ps_status.content = f"**Result: FAILED \u274c**  \n- {type(e).__name__}: {e}"

    threading.Thread(target=worker, daemon=True).start()


ps_model.on_update(_on_ps_model)
btn_ps_check.on_click(do_ps_check)


if GRIPPER_KIND == "robotiq":
    print(f"[ui] gripper: Robotiq 2F-85 via {ROBOTIQ_URL}  "
          f"(speed_255={ROBOTIQ_SPEED_255} force_255={ROBOTIQ_FORCE_255})")
else:
    print(f"[ui] gripper: Franka Hand via {HAND_URL}")
print("[ui] ready — open http://localhost:8080")
server.sleep_forever()
