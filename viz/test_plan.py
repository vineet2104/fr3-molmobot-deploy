"""Offline simulate-only test:
- Read current joint state from franky_service.
- Compute current EE via FK.
- Set goal = current pose, z -= 0.20 m (20 cm down).
- Plan with cuRobo.
- Print summary, EE polyline, dump trajectory to /tmp.
NO targets are sent to the robot.
"""

from __future__ import annotations

import os

import numpy as np
import requests
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState

FRANKY_URL = os.environ.get("FRANKY_SERVICE_URL", "http://10.20.12.110:54321")

print("[test] loading planner...")
mp = MotionPlanner(MotionPlannerCfg.create(robot="franka.yml"))
mp.warmup(enable_graph=True, num_warmup_iterations=3)

print("[test] reading current joint state...")
js_data = requests.get(f"{FRANKY_URL}/joint_state", timeout=2.0).json()
q = list(js_data["positions"])[:7]
print(f"  q = {[round(x, 3) for x in q]}")

start = JointState.from_position(
    torch.tensor([q], device="cuda", dtype=torch.float32),
    joint_names=mp.joint_names,
)

# Current EE via FK
tp = mp.compute_kinematics(start).tool_poses
cur_pos = tp.position[0, 0, 0].detach().cpu().numpy()
cur_quat = tp.quaternion[0, 0, 0].detach().cpu().numpy()
print(f"  current EE position : {cur_pos.tolist()}")
print(f"  current EE quaternion (wxyz): {cur_quat.tolist()}")

# Goal = same orientation, z - 20 cm
goal_pos = cur_pos.copy()
goal_pos[2] -= 0.20
print(f"  goal     EE position : {goal_pos.tolist()}")

goal = GoalToolPose(
    tool_frames=mp.tool_frames,
    position=torch.tensor(
        [[[[goal_pos.tolist()]]]], device="cuda", dtype=torch.float32
    ),
    quaternion=torch.tensor(
        [[[[cur_quat.tolist()]]]], device="cuda", dtype=torch.float32
    ),
)

print("[test] planning...")
result = mp.plan_pose(goal, start)
if result is None or not result.success.any():
    print("[test] PLAN FAILED")
    raise SystemExit(1)

interp = result.get_interpolated_plan()
# interp.position is [..., T, dof]; squeeze leading singleton dims.
pos_t = interp.position.detach().cpu().numpy()
while pos_t.ndim > 2:
    pos_t = pos_t[0]
q_traj = pos_t  # [T, dof]
dt = mp.trajopt_solver.config.interpolation_dt
T, dof = q_traj.shape
print(f"[test] PLAN OK: {T} waypoints, dt={dt:.4f}, duration={T*dt:.2f}s, dof={dof}")
print(f"  start q: {q_traj[0].round(3).tolist()}")
print(f"  end   q: {q_traj[-1].round(3).tolist()}")

# Trim to the 7 active arm joints for FK and execution.
q7 = q_traj[:, : len(mp.joint_names)]
js_traj = JointState.from_position(
    torch.tensor(q7, device="cuda", dtype=torch.float32),
    joint_names=mp.joint_names,
)
ee_traj = mp.compute_kinematics(js_traj).tool_poses.position[:, 0, 0, :].detach().cpu().numpy()
print(f"[test] EE polyline ({ee_traj.shape[0]} pts):")
for i in range(0, ee_traj.shape[0], max(1, ee_traj.shape[0] // 10)):
    print(f"  t={i*dt:5.2f}s  ee={ee_traj[i].round(4).tolist()}")
print(f"  final EE: {ee_traj[-1].round(4).tolist()}  (target was {goal_pos.round(4).tolist()})")
err = np.linalg.norm(ee_traj[-1] - goal_pos)
print(f"  position error: {err*1000:.2f} mm")

# Dump for later
np.save("/tmp/sim_q_traj.npy", q_traj)
np.save("/tmp/sim_ee_traj.npy", ee_traj)
print("[test] saved /tmp/sim_q_traj.npy and /tmp/sim_ee_traj.npy")
