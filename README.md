# fr3-molmobot-deploy

A small, self-contained deployment repo to test **MolmoBot** and **MolmoBot-FT**
VLA policies on an FR3 arm + Franka Hand, reproducing the proven
`franky_service` (joint-position) + sequential 15 Hz chunk executor setup.

> This is a **migration/bring-up** repo. It reproduces a stack known to work on
> one FR3; it is **not** a claim of safety on a different robot. Follow the
> incremental hardware bring-up in `docs/MIGRATION_PLAN.md` §14 before trusting
> policy motion. Keep the e-stop within reach.

## Start here (docs)
- **`docs/AGENT_HANDOFF.md`** — orientation for the agent that clones this repo.
- **`docs/SETUP.md`** — which machine runs what + **"do I clone twice?"** (yes:
  once per machine role). Read this before installing anything.
- **`docs/MIGRATION_PLAN.md`** — authoritative, safety-ordered bring-up.
- **`docs/V3_GRIPPER_AND_IO_CONVENTIONS.md`** — gripper/IO contract (re-validate
  for the Franka Hand).
- **`viz/README.md`** — optional viser 3D UI + cuRobo planning.
- **`server/README.md`** — MolmoBot model server on the GPU host.
- **`LICENSES.md`** — provenance to review before pushing to a public remote.

## Do I clone it twice?
You clone the **same repo on each machine that has a role** — normally **two**
(RT/robot host for the services, workstation for the client/webpage/viz), plus a
**third** on the GPU host if you launch the model server from `server/` here.
Each machine only runs its part and keeps its own git-ignored
`configs/robot.env`. Full walkthrough: `docs/SETUP.md`.

## Three machine roles

```
RT/robot host          Workstation (this repo)         GPU host
franky_service :54321  run_molmobot_*.py + console      serve_bridge.py (MolmoBot)
franka_hand    :54324  cameras (ZED) + logs             ws://GPU:8000
   │ libfranka/FCI            │ HTTP to RT host                 │ websocket
   └────── FR3 ───────────────┴───────── LAN ───────────────────┘
```

- **RT/robot host** (real-time kernel, `franky`+libfranka, FCI): runs the arm +
  gripper services. This is where your lab currently runs the FR3Py velocity
  controller — here it instead runs `franky_service` (position control).
- **Workstation** (where you code / open the webpage): runs the rollout client,
  cameras, and the web console.
- **GPU host** (2×L40): runs the MolmoBot model server (`server/serve_bridge.py`).

## Layout

```
services/   franky_service.py (arm :54321), franka_hand_service.py (:54324),
            robotiq_gripper_service.py (optional :54323)
client/     run_molmobot_official.py, run_molmobot_ft.py, client_example.py
server/     serve_bridge.py, bridge_policy.py (+ README) — runs on the GPU host
deployment_console.py  mixed ZED+RealSense live UI and 16→8 closed-loop rollout (workstation)
service_console.py   older subprocess-based FastAPI console (kept for reference)
service_console_reference.py  original single-box console (reference only, not runnable here)
viz/        server.py (viser 3D UI + cuRobo), bundled Panda URDF+meshes (workstation)
configs/    robot.example.yaml, env.example  (copy -> robot.yaml / robot.env)
scripts/    launch_arm_service.sh, launch_gripper_service.sh, run_molmobot_ft.sh,
            preflight.py, dummy_inference.py, list_cameras.py
tools/      stitch_task_videos.py (rollout mp4s)
docs/       MIGRATION_PLAN.md (authoritative bring-up), gripper/IO conventions,
            GPU model-server setup
```

## Quick start (after the incremental bring-up in docs/MIGRATION_PLAN.md)

1. **Configure** (no source edits needed):
   ```bash
   cp configs/env.example configs/robot.env
   $EDITOR configs/robot.env      # FR3 IP, RT_HOST_IP, GPU_HOST_IP, ZED serials
   source configs/robot.env
   ```
2. **RT host** — start services (FCI activated, joints unlocked, e-stop ready):
   ```bash
   ./scripts/launch_arm_service.sh          # arm :54321
   ./scripts/launch_gripper_service.sh      # Franka Hand :54324
   ```
3. **GPU host** — start the MolmoBot server (see `server/README.md`), e.g.
   `python server/serve_bridge.py --checkpoint_path <ckpt> --host 0.0.0.0 --port 8000 --chunk_response ...`
4. **Workstation** — verify everything (no motion):
   ```bash
   python scripts/list_cameras.py           # get ZED serials -> put in robot.env
   python scripts/dummy_inference.py        # model contract ok?
   python scripts/preflight.py              # arm/gripper/model/cameras reachable?
   ```
5. **Rollout** (start safe):
   ```bash
   # gated, tiny motion, no gripper first:
   ./scripts/run_molmobot_ft.sh "Pick up the pineapple slices can" --step_confirm --no_gripper --max_joint_delta 0.05
   # then a real run:
   ./scripts/run_molmobot_ft.sh "Pick up the pineapple slices can"
   ```
6. **Deployment console** on the workstation (mixed ZED exterior + RealSense wrist):
   ```bash
   ./scripts/setup_deployment_console.sh    # once
   ./scripts/launch_deployment_console.sh   # open http://<workstation-ip>:7071
   ```
   Enter the remote model IP/port, experiment name and task in the UI. Start
   with **Dry run** enabled. See `docs/DEPLOYMENT_CONSOLE.md`.

## Gripper note (IMPORTANT)
The lab uses the **Franka Hand** (`--gripper panda`), not the Robotiq 2F-85 the
reference used. The gripper **state encoding** and **action threshold** must be
re-validated for the Franka Hand — do NOT assume the 2F-85 kinematics map. See
`docs/V3_GRIPPER_AND_IO_CONVENTIONS.md` and validate with gripper-only tests
(`--no_gripper` off only after direction/force are confirmed).

## What is NOT included
- Model checkpoints (obtain on the GPU host; see `docs/LOCAL_GPU_INFERENCE_SETUP.md`).
- The MolmoBot model repo (`olmo`) — install on the GPU host per that doc.
- Any secrets/hostnames — put those only in `configs/robot.env` (git-ignored).

See `docs/MIGRATION_PLAN.md` for the full, safety-first bring-up and validation.
