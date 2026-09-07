"""Minimal Viser UI for legacy cuRobo v0.7.8 on a Turing workstation.

Starts read-only. Planning never moves the robot. Execution requires explicitly
checking the enable box and is limited to goals near the state used to plan.
No lab obstacles are modeled: visually verify the path and keep the e-stop ready.
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

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

FRANKY_URL = os.environ.get("FRANKY_SERVICE_URL", "http://127.0.0.1:54321")
URDF_PATH = Path(os.environ.get("FRANKA_URDF_PATH", str(Path(__file__).parent / "assets/franka_panda.urdf")))
POLL_HZ = 10.0
READ_TIMEOUT = 2.0
WRITE_TIMEOUT = 3.0
MAX_START_ERROR_RAD = 0.05
MAX_TOTAL_JOINT_DELTA_RAD = 0.35
MAX_GOAL_TRANSLATION_M = 0.10

print("[legacy-ui] loading cuRobo v0.7.8 MotionGen...")
tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
mg_cfg = MotionGenConfig.load_from_robot_config(
    "franka.yml", None, tensor_args,
    interpolation_dt=0.05,
    num_ik_seeds=32,
    num_trajopt_seeds=4,
    velocity_scale=0.20,
    acceleration_scale=0.20,
    jerk_scale=0.20,
)
motion_gen = MotionGen(mg_cfg)
motion_gen.warmup(enable_graph=True)
JOINT_NAMES = list(motion_gen.joint_names)
print(f"[legacy-ui] joints: {JOINT_NAMES}")


def get_q() -> list[float] | None:
    try:
        r = requests.get(f"{FRANKY_URL}/joint_state", timeout=READ_TIMEOUT)
        r.raise_for_status()
        q = [float(x) for x in r.json()["positions"][:7]]
        return q if len(q) == 7 and np.isfinite(q).all() else None
    except Exception as exc:
        print(f"[legacy-ui] joint read failed: {exc}")
        return None


def fk(q: list[float]) -> tuple[np.ndarray, np.ndarray]:
    js = JointState.from_position(tensor_args.to_device([q]), joint_names=JOINT_NAMES)
    state = motion_gen.compute_kinematics(js)
    return (
        state.ee_pos_seq.detach().cpu().numpy().reshape(-1, 3)[0],
        state.ee_quat_seq.detach().cpu().numpy().reshape(-1, 4)[0],
    )


def send_q(q: np.ndarray) -> None:
    r = requests.post(
        f"{FRANKY_URL}/target_joint_state",
        json={"positions": q[:7].tolist(), "velocities": [0.0] * 7},
        timeout=WRITE_TIMEOUT,
    )
    r.raise_for_status()


def stop() -> None:
    try:
        requests.post(f"{FRANKY_URL}/stop", timeout=WRITE_TIMEOUT).raise_for_status()
    except Exception as exc:
        print(f"[legacy-ui] stop failed: {exc}")


server = viser.ViserServer(host="0.0.0.0", port=8080, label="FR3 legacy cuRobo")
robot_urdf = ViserUrdf(server, urdf_or_path=URDF.load(str(URDF_PATH)), root_node_name="/robot")
actuated = robot_urdf.get_actuated_joint_names()
goal = server.scene.add_transform_controls("/goal", position=(0.4, 0.0, 0.5), wxyz=(1, 0, 0, 0), scale=0.2)
status = server.gui.add_markdown("**Status:** waiting for arm service")
execute_enabled = server.gui.add_checkbox("ENABLE REAL EXECUTION", initial_value=False)
btn_snap = server.gui.add_button("Snap goal to current EE")
btn_plan = server.gui.add_button("Plan (no motion)")
btn_execute = server.gui.add_button("Execute planned path")
btn_execute.disabled = True
btn_stop = server.gui.add_button("STOP")
warning = server.gui.add_markdown(
    "**Warning:** no table or lab obstacles are modeled. Execution is limited but still real motion."
)

current_q: list[float] | None = None
planned_q: np.ndarray | None = None
planned_dt = 0.05
plan_start_q: np.ndarray | None = None
plan_start_ee: np.ndarray | None = None
preview = None
exec_lock = threading.Lock()


def update_robot(q: list[float]) -> None:
    cfg = {name: value for name, value in zip(JOINT_NAMES, q)}
    cfg["panda_finger_joint1"] = 0.04
    cfg["panda_finger_joint2"] = 0.04
    robot_urdf.update_cfg(np.array([cfg.get(name, 0.0) for name in actuated]))


def poll() -> None:
    global current_q
    initialized = False
    while True:
        q = get_q()
        if q is not None:
            current_q = q
            update_robot(q)
            if not initialized:
                p, quat = fk(q)
                goal.position = tuple(float(x) for x in p)
                goal.wxyz = tuple(float(x) for x in quat)
                initialized = True
            status.content = f"**Status:** connected, q={[round(x, 3) for x in q]}"
        time.sleep(1.0 / POLL_HZ)


def snap(_) -> None:
    if current_q is None:
        status.content = "**Status:** no current state"
        return
    p, quat = fk(current_q)
    goal.position = tuple(float(x) for x in p)
    goal.wxyz = tuple(float(x) for x in quat)
    status.content = "**Status:** goal snapped; planning/execution has not occurred"


def plan(_) -> None:
    global planned_q, planned_dt, plan_start_q, plan_start_ee, preview
    btn_execute.disabled = True
    planned_q = None
    if current_q is None:
        status.content = "**Status:** no current state"
        return
    q0 = np.asarray(current_q, dtype=np.float64)
    p0, _ = fk(current_q)
    gp = np.asarray(goal.position, dtype=np.float64)
    if np.linalg.norm(gp - p0) > MAX_GOAL_TRANSLATION_M:
        status.content = f"**Status:** rejected: goal is over {MAX_GOAL_TRANSLATION_M:.2f} m from current EE"
        return
    start = JointState.from_position(tensor_args.to_device([q0.tolist()]), joint_names=JOINT_NAMES)
    target = Pose(
        position=tensor_args.to_device([list(goal.position)]),
        quaternion=tensor_args.to_device([list(goal.wxyz)]),
    )
    status.content = "**Status:** planning (no motion)..."
    result = motion_gen.plan_single(start, target, MotionGenPlanConfig(enable_graph=True, max_attempts=4))
    if not bool(result.success.item()):
        status.content = f"**Status:** plan failed: {result.status}"
        return
    traj = result.get_interpolated_plan()
    qtraj = traj.position.detach().cpu().numpy()
    while qtraj.ndim > 2:
        qtraj = qtraj[0]
    qtraj = qtraj[:, :7]
    total_delta = float(np.max(np.abs(qtraj - q0[None, :])))
    if total_delta > MAX_TOTAL_JOINT_DELTA_RAD:
        status.content = f"**Status:** rejected: planned joint displacement {total_delta:.3f} rad exceeds {MAX_TOTAL_JOINT_DELTA_RAD:.3f}"
        return
    ee = motion_gen.compute_kinematics(
        JointState.from_position(tensor_args.to_device(qtraj), joint_names=JOINT_NAMES)
    ).ee_pos_seq.detach().cpu().numpy().reshape(-1, 3)
    if preview is not None:
        preview.remove()
    segs = np.stack([ee[:-1], ee[1:]], axis=1).astype(np.float32)
    preview = server.scene.add_line_segments(
        "/plan_preview", points=segs,
        colors=np.tile(np.array([0, 200, 255], dtype=np.uint8), (len(segs), 1)), line_width=3.0,
    )
    planned_q = qtraj
    planned_dt = float(result.interpolation_dt)
    plan_start_q = q0
    plan_start_ee = p0
    btn_execute.disabled = False
    status.content = f"**Status:** plan OK: {len(qtraj)} points, {len(qtraj)*planned_dt:.2f}s, max displacement {total_delta:.3f} rad"


def execute_worker() -> None:
    if planned_q is None or plan_start_q is None:
        return
    if not execute_enabled.value:
        status.content = "**Status:** execution blocked; check ENABLE REAL EXECUTION first"
        return
    qnow = get_q()
    if qnow is None or np.max(np.abs(np.asarray(qnow) - plan_start_q)) > MAX_START_ERROR_RAD:
        status.content = "**Status:** execution blocked; robot moved since planning—replan"
        return
    status.content = "**Status:** EXECUTING REAL MOTION"
    try:
        with exec_lock:
            deadline = time.monotonic()
            for q in planned_q:
                if not execute_enabled.value:
                    raise RuntimeError("execution enable was cleared")
                send_q(q)
                deadline += planned_dt
                time.sleep(max(0.0, deadline - time.monotonic()))
            stop()
        status.content = "**Status:** execution complete; software stop latched"
        execute_enabled.value = False
    except Exception as exc:
        stop()
        execute_enabled.value = False
        status.content = f"**Status:** execution aborted and stop sent: {exc}"


def execute(_) -> None:
    threading.Thread(target=execute_worker, daemon=True).start()


def do_stop(_) -> None:
    execute_enabled.value = False
    stop()
    status.content = "**Status:** software STOP sent"


btn_snap.on_click(snap)
btn_plan.on_click(plan)
btn_execute.on_click(execute)
btn_stop.on_click(do_stop)
threading.Thread(target=poll, daemon=True).start()
print("[legacy-ui] ready at http://0.0.0.0:8080 (execution disabled by default)")
server.sleep_forever()
