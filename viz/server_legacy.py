"""Minimal Viser UI for legacy cuRobo v0.7.8 on a Turing workstation.

Starts read-only. Planning never moves the robot. Execution requires explicitly
checking the enable box and verifies the robot is still at the plan's start state.
No lab obstacles are modeled: visually verify the path and keep the e-stop ready.
Optional displacement guards can be enabled through environment variables.
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
HAND_URL = os.environ.get(
    "FRANKA_HAND_SERVICE_URL",
    os.environ.get("HAND_URL", "http://127.0.0.1:54324"),
)
URDF_PATH = Path(os.environ.get("FRANKA_URDF_PATH", str(Path(__file__).parent / "assets/franka_panda.urdf")))
POLL_HZ = 10.0
READ_TIMEOUT = 2.0
WRITE_TIMEOUT = 3.0
MAX_START_ERROR_RAD = 0.05
# Initial joint configuration hard-coded by the previous FR3Py C++ bridge
# (fr3_bridge/src/fr3_joint_interface/fr3_joint_interface.cpp).
FR3PY_HOME_Q = np.array(
    [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0],
    dtype=np.float64,
)
# Lab-selected Cartesian home in the robot base frame. Its orientation is
# computed by FK from FR3PY_HOME_Q, so only the position is overridden.
HOME_POSITION_XYZ = np.array([0.378, 0.432, 0.317], dtype=np.float64)
# Optional site/operator guards. A value <= 0 disables the guard and lets
# cuRobo determine reachability, joint limits, and self-collision feasibility.
MAX_TOTAL_JOINT_DELTA_RAD = float(os.environ.get("VIZ_MAX_TOTAL_JOINT_DELTA_RAD", "0"))
MAX_GOAL_TRANSLATION_M = float(os.environ.get("VIZ_MAX_GOAL_TRANSLATION_M", "0"))

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
connection_status = server.gui.add_markdown("**Robot:** waiting for arm service")
planner_status = server.gui.add_markdown("**Planner:** idle — no plan generated")

with server.gui.add_folder("Goal pose in robot base frame"):
    goal_x = server.gui.add_number("X (m)", initial_value=0.4, step=0.001)
    goal_y = server.gui.add_number("Y (m)", initial_value=0.0, step=0.001)
    goal_z = server.gui.add_number("Z (m)", initial_value=0.5, step=0.001)
    btn_apply_xyz = server.gui.add_button("Apply XYZ (keeps current goal orientation)")
    btn_up_5mm = server.gui.add_button("Nudge +Z by 5 mm")
    goal_orientation = server.gui.add_markdown("**Goal orientation (wxyz):** waiting")

execute_enabled = server.gui.add_checkbox("ENABLE REAL EXECUTION", initial_value=False)
btn_snap = server.gui.add_button("Snap goal to current EE")
btn_plan = server.gui.add_button("Plan (no motion)")
btn_execute = server.gui.add_button("Execute planned path")
btn_execute.disabled = True
btn_home = server.gui.add_button("Go Home (configured Cartesian pose)")
home_note = server.gui.add_markdown(
    "Home XYZ: `[0.378, 0.432, 0.317] m` in robot base frame; orientation is "
    "taken from the default FR3Py home q. Plans first and requires ENABLE REAL EXECUTION."
)
btn_stop = server.gui.add_button("STOP")

with server.gui.add_folder("Franka Hand"):
    gripper_status = server.gui.add_markdown("**Hand:** waiting for service")
    gripper_enabled = server.gui.add_checkbox("ENABLE GRIPPER COMMANDS", initial_value=False)
    grasp_width = server.gui.add_number("Grasp width (m)", initial_value=0.02, min=0.0, max=0.08, step=0.005)
    grasp_force = server.gui.add_number("Grasp force (N)", initial_value=10.0, min=1.0, max=40.0, step=1.0)
    btn_gripper_open = server.gui.add_button("Open hand")
    btn_gripper_close = server.gui.add_button("Close / grasp")
    btn_gripper_stop = server.gui.add_button("STOP hand")

warning = server.gui.add_markdown(
    "**Warning:** cuRobo checks its robot model, joint limits, reachability, and self-collision, "
    "but no table, floor, cameras, cables, or lab obstacles are modeled. A successful plan is "
    "not automatically safe in the physical workspace."
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


def sync_goal_fields() -> None:
    goal_x.value = float(goal.position[0])
    goal_y.value = float(goal.position[1])
    goal_z.value = float(goal.position[2])
    goal_orientation.content = (
        "**Goal orientation (wxyz):** "
        + str([round(float(x), 5) for x in goal.wxyz])
    )


def apply_xyz(_) -> None:
    goal.position = (float(goal_x.value), float(goal_y.value), float(goal_z.value))
    planner_status.content = (
        f"**Planner:** goal updated to base-frame XYZ "
        f"[{goal_x.value:.4f}, {goal_y.value:.4f}, {goal_z.value:.4f}] m; click Plan"
    )


def nudge_up_5mm(_) -> None:
    goal_z.value = float(goal_z.value) + 0.005
    apply_xyz(None)


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
                sync_goal_fields()
                initialized = True
            connection_status.content = f"**Robot:** connected, q={[round(x, 3) for x in q]}"
        time.sleep(1.0 / POLL_HZ)


def snap(_) -> None:
    if current_q is None:
        planner_status.content = "**Planner:** cannot snap — no current robot state"
        return
    p, quat = fk(current_q)
    goal.position = tuple(float(x) for x in p)
    goal.wxyz = tuple(float(x) for x in quat)
    sync_goal_fields()
    planner_status.content = (
        f"**Planner:** snapped to current EE XYZ "
        f"[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m; no plan generated yet"
    )


def plan(_) -> None:
    global planned_q, planned_dt, plan_start_q, plan_start_ee, preview
    btn_execute.disabled = True
    planned_q = None
    if current_q is None:
        planner_status.content = "**Planner:** cannot plan — no current robot state"
        return
    q0 = np.asarray(current_q, dtype=np.float64)
    p0, _ = fk(current_q)
    gp = np.asarray(goal.position, dtype=np.float64)
    goal_distance = float(np.linalg.norm(gp - p0))
    if not np.isfinite(gp).all():
        planner_status.content = "**Planner:** REJECTED — goal XYZ contains NaN or infinity"
        return
    if MAX_GOAL_TRANSLATION_M > 0 and goal_distance > MAX_GOAL_TRANSLATION_M:
        planner_status.content = (
            f"**Planner:** REJECTED by optional operator guard — goal distance "
            f"{goal_distance:.3f} m exceeds {MAX_GOAL_TRANSLATION_M:.3f} m"
        )
        return
    start = JointState.from_position(tensor_args.to_device([q0.tolist()]), joint_names=JOINT_NAMES)
    target = Pose(
        position=tensor_args.to_device([list(goal.position)]),
        quaternion=tensor_args.to_device([list(goal.wxyz)]),
    )
    planner_status.content = "**Planner:** PLANNING on GPU — no robot command is being sent..."
    print(f"[legacy-ui] planning to XYZ={list(goal.position)}, wxyz={list(goal.wxyz)}")
    result = motion_gen.plan_single(start, target, MotionGenPlanConfig(enable_graph=True, max_attempts=4))
    if not bool(result.success.item()):
        planner_status.content = f"**Planner:** PLAN FAILED — {result.status}"
        print(f"[legacy-ui] PLAN FAILED: {result.status}")
        return
    traj = result.get_interpolated_plan()
    qtraj = traj.position.detach().cpu().numpy()
    while qtraj.ndim > 2:
        qtraj = qtraj[0]
    qtraj = qtraj[:, :7]
    total_delta = float(np.max(np.abs(qtraj - q0[None, :])))
    if MAX_TOTAL_JOINT_DELTA_RAD > 0 and total_delta > MAX_TOTAL_JOINT_DELTA_RAD:
        planner_status.content = (
            f"**Planner:** REJECTED by optional operator guard — planned joint displacement "
            f"{total_delta:.3f} rad exceeds {MAX_TOTAL_JOINT_DELTA_RAD:.3f} rad"
        )
        return
    ee = motion_gen.compute_kinematics(
        JointState.from_position(tensor_args.to_device(qtraj), joint_names=JOINT_NAMES)
    ).ee_pos_seq.detach().cpu().numpy().reshape(-1, 3)
    if preview is not None:
        preview.remove()
    segs = np.stack([ee[:-1], ee[1:]], axis=1).astype(np.float32)
    preview = server.scene.add_line_segments(
        "/plan_preview",
        points=segs,
        colors=np.array([0, 200, 255], dtype=np.uint8),
        line_width=3.0,
    )
    planned_q = qtraj
    planned_dt = float(result.interpolation_dt)
    plan_start_q = q0
    plan_start_ee = p0
    btn_execute.disabled = False
    planner_status.content = (
        f"**Planner: PLAN SUCCESS (no motion sent)**  \n"
        f"Goal XYZ: `[{gp[0]:.4f}, {gp[1]:.4f}, {gp[2]:.4f}] m`  \n"
        f"Cartesian goal distance: `{goal_distance:.3f} m`  \n"
        f"Trajectory: `{len(qtraj)} points`, `{len(qtraj)*planned_dt:.2f} s`; "
        f"max joint displacement: `{total_delta:.3f} rad`  \n"
        "Cyan line = planned EE path. Check ENABLE REAL EXECUTION only after reviewing it."
    )
    print(f"[legacy-ui] PLAN SUCCESS: points={len(qtraj)}, duration={len(qtraj)*planned_dt:.2f}s, max_joint_delta={total_delta:.3f}")


def go_home(_) -> None:
    """Plan to configured XYZ with the prior FR3Py home's orientation, then execute."""
    global planned_q, planned_dt, plan_start_q, plan_start_ee, preview
    if not execute_enabled.value:
        planner_status.content = (
            "**Go Home:** blocked — check ENABLE REAL EXECUTION, then click Go Home again"
        )
        return
    if current_q is None:
        planner_status.content = "**Go Home:** blocked — no current robot state"
        execute_enabled.value = False
        return

    q0 = np.asarray(current_q, dtype=np.float64)
    start = JointState.from_position(
        tensor_args.to_device([q0.tolist()]), joint_names=JOINT_NAMES
    )
    # Compute the orientation of the old FR3Py home configuration, but replace
    # its translation with the operator-selected Cartesian home position.
    _, home_quat = fk(FR3PY_HOME_Q.tolist())
    home_pose = Pose(
        position=tensor_args.to_device([HOME_POSITION_XYZ.tolist()]),
        quaternion=tensor_args.to_device([home_quat.tolist()]),
    )
    planner_status.content = (
        "**Go Home:** planning to configured Cartesian home with FR3Py home orientation "
        "(no motion yet)..."
    )
    print(
        f"[legacy-ui] planning Cartesian home: xyz={HOME_POSITION_XYZ.tolist()}, "
        f"wxyz={home_quat.round(6).tolist()}"
    )
    result = motion_gen.plan_single(
        start,
        home_pose,
        MotionGenPlanConfig(enable_graph=True, max_attempts=4),
    )
    if not bool(result.success.item()):
        planner_status.content = f"**Go Home:** PLAN FAILED — {result.status}"
        execute_enabled.value = False
        print(f"[legacy-ui] HOME PLAN FAILED: {result.status}")
        return

    traj = result.get_interpolated_plan()
    qtraj = traj.position.detach().cpu().numpy()
    while qtraj.ndim > 2:
        qtraj = qtraj[0]
    qtraj = qtraj[:, :7]
    total_delta = float(np.max(np.abs(qtraj - q0[None, :])))
    if MAX_TOTAL_JOINT_DELTA_RAD > 0 and total_delta > MAX_TOTAL_JOINT_DELTA_RAD:
        planner_status.content = (
            f"**Go Home:** rejected by optional operator guard — {total_delta:.3f} rad exceeds "
            f"{MAX_TOTAL_JOINT_DELTA_RAD:.3f} rad"
        )
        execute_enabled.value = False
        return

    ee = motion_gen.compute_kinematics(
        JointState.from_position(tensor_args.to_device(qtraj), joint_names=JOINT_NAMES)
    ).ee_pos_seq.detach().cpu().numpy().reshape(-1, 3)
    if preview is not None:
        preview.remove()
    segs = np.stack([ee[:-1], ee[1:]], axis=1).astype(np.float32)
    preview = server.scene.add_line_segments(
        "/plan_preview",
        points=segs,
        colors=np.array([255, 190, 0], dtype=np.uint8),
        line_width=3.0,
    )
    planned_q = qtraj
    planned_dt = float(result.interpolation_dt)
    plan_start_q = q0
    plan_start_ee = ee[0]
    planner_status.content = (
        f"**Go Home:** plan success — executing `{len(qtraj)}` points over "
        f"`{len(qtraj) * planned_dt:.2f} s`; max joint displacement `{total_delta:.3f} rad`"
    )
    print(
        f"[legacy-ui] HOME PLAN SUCCESS: points={len(qtraj)}, "
        f"duration={len(qtraj) * planned_dt:.2f}s, max_joint_delta={total_delta:.3f}"
    )
    threading.Thread(target=execute_worker, daemon=True).start()


def execute_worker() -> None:
    if planned_q is None or plan_start_q is None:
        return
    if not execute_enabled.value:
        planner_status.content = "**Execution:** BLOCKED — check ENABLE REAL EXECUTION first"
        return
    qnow = get_q()
    if qnow is None or np.max(np.abs(np.asarray(qnow) - plan_start_q)) > MAX_START_ERROR_RAD:
        planner_status.content = "**Execution:** BLOCKED — robot moved since planning; snap and replan"
        return
    planner_status.content = "**Execution:** EXECUTING REAL MOTION"
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
        planner_status.content = "**Execution:** complete; software stop latched"
        execute_enabled.value = False
    except Exception as exc:
        stop()
        execute_enabled.value = False
        planner_status.content = f"**Execution:** aborted and stop sent — {exc}"


def execute(_) -> None:
    threading.Thread(target=execute_worker, daemon=True).start()


def gripper_request(path: str, payload: dict | None = None) -> None:
    if not gripper_enabled.value:
        gripper_status.content = "**Hand:** command blocked — check ENABLE GRIPPER COMMANDS"
        return
    try:
        r = requests.post(f"{HAND_URL}/{path}", json=payload, timeout=WRITE_TIMEOUT)
        r.raise_for_status()
        gripper_status.content = f"**Hand:** `{path}` accepted; waiting for completion"
        gripper_enabled.value = False
    except Exception as exc:
        gripper_status.content = f"**Hand:** command failed — {exc}"
        gripper_enabled.value = False


def open_gripper(_) -> None:
    gripper_request("gripper_open", {"speed": 0.05})


def close_gripper(_) -> None:
    gripper_request(
        "gripper_grasp",
        {
            "width_m": float(grasp_width.value),
            "speed": 0.05,
            "force": float(grasp_force.value),
            "epsilon_inner": 0.02,
            "epsilon_outer": 0.02,
        },
    )


def stop_gripper(_) -> None:
    try:
        requests.post(f"{HAND_URL}/gripper_stop", timeout=WRITE_TIMEOUT).raise_for_status()
        gripper_status.content = "**Hand:** stop sent"
    except Exception as exc:
        gripper_status.content = f"**Hand:** stop failed — {exc}"
    finally:
        gripper_enabled.value = False


def poll_gripper() -> None:
    while True:
        try:
            data = requests.get(f"{HAND_URL}/health", timeout=READ_TIMEOUT).json()
            error = data.get("error")
            gripper_status.content = (
                f"**Hand:** {'busy' if data.get('busy') else 'ready'}, "
                f"width={data.get('width_m')}, grasped={data.get('is_grasped')}"
                + (f", error={error}" if error else "")
            )
        except Exception:
            gripper_status.content = f"**Hand:** service unavailable at `{HAND_URL}`"
        time.sleep(0.5)


def do_stop(_) -> None:
    execute_enabled.value = False
    stop()
    planner_status.content = "**Execution:** software STOP sent"


btn_apply_xyz.on_click(apply_xyz)
btn_up_5mm.on_click(nudge_up_5mm)
btn_snap.on_click(snap)
btn_plan.on_click(plan)
btn_execute.on_click(execute)
btn_home.on_click(go_home)
btn_stop.on_click(do_stop)
btn_gripper_open.on_click(open_gripper)
btn_gripper_close.on_click(close_gripper)
btn_gripper_stop.on_click(stop_gripper)
threading.Thread(target=poll, daemon=True).start()
threading.Thread(target=poll_gripper, daemon=True).start()
print(f"[legacy-ui] Franka Hand service: {HAND_URL}")
print("[legacy-ui] ready at http://0.0.0.0:8080 (execution disabled by default)")
server.sleep_forever()
