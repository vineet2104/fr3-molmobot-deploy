# SETUP — which machine runs what (and how many times to clone)

> **Short answer to "do I clone twice?"** You clone the *same repo* on **each
> machine that has a role** — normally **two** (RT/robot host + workstation),
> and a **third** on the GPU host if you run the model server from this repo's
> `server/` glue. Each machine only runs its part; each keeps its own
> (git-ignored) `configs/robot.env`. Nothing here auto-syncs between machines —
> they communicate over HTTP/websocket on the LAN.

```
                         ┌──────────────────────────────────────────────┐
   clone #1  RT/robot    │ services/franky_service.py      :54321 (arm)  │
   host (RT kernel,      │ services/franka_hand_service.py :54324 (grip) │
   franky+libfranka,     │ scripts/launch_arm_service.sh                 │
   FCI)                  │ scripts/launch_gripper_service.sh             │
                         └───────────────▲──────────────────────────────┘
                                         │ HTTP (joint_state / target_joint_state / gripper)
                         ┌───────────────┴──────────────────────────────┐
   clone #2  workstation │ client/run_molmobot_ft.py  (rollout)          │
   (your dev box, ZED    │ service_console.py         (webpage :7070)    │
   cameras, GPU for viz) │ viz/server.py              (viser :8080)      │
                         │ scripts/preflight.py, run_molmobot_ft.sh      │
                         └───────────────▲──────────────────────────────┘
                                         │ websocket (obs -> action chunk)
                         ┌───────────────┴──────────────────────────────┐
   clone #3  GPU host    │ server/serve_bridge.py  ws://:8000            │
   (2×L40, MolmoBot env) │ (+ external MolmoBot `olmo` repo + checkpts)  │
                         └───────────────────────────────────────────────┘
```

## Per-machine setup

### Machine 1 — RT / robot host (runs the robot services)
This replaces "start the FR3Py velocity controller" — here you start the
`franky` **position** service instead.
```bash
git clone <repo> fr3-molmobot-deploy && cd fr3-molmobot-deploy
python -m pip install -r requirements-robot.txt      # needs franky-control + libfranka
cp configs/env.example configs/robot.env
$EDITOR configs/robot.env      # set FRANKY_ROBOT_IP (the FR3 FCI IP)
source configs/robot.env
# FCI active in Desk, joints unlocked, e-stop ready:
./scripts/launch_arm_service.sh          # terminal 1 -> arm :54321
./scripts/launch_gripper_service.sh      # terminal 2 -> Franka Hand :54324
```
Leave these running. Verify from a browser: `http://<rt-host>:54321` (arm
service status page).

### Machine 2 — Workstation (runs the policy client + webpage + viz)
```bash
git clone <repo> fr3-molmobot-deploy && cd fr3-molmobot-deploy
python -m pip install -r requirements-client.txt     # + ZED SDK / pyzed
cp configs/env.example configs/robot.env
$EDITOR configs/robot.env      # set RT_HOST_IP (-> FRANKY_URL/HAND_URL) and GPU_HOST_IP (-> model)
source configs/robot.env
python scripts/list_cameras.py           # copy the ZED serials into robot.env, re-source
python scripts/dummy_inference.py        # model reachable + chunk shape ok?
python scripts/preflight.py              # arm/gripper/model/cameras all green?
# first, SAFE motion test (gated, no gripper, tiny deltas):
./scripts/run_molmobot_ft.sh "Pick up the pineapple slices can" --step_confirm --no_gripper --max_joint_delta 0.05
```
Webpage / viz (optional): `python service_console.py --port 7070` and
`python viz/server.py` (see `viz/README.md`; needs cuRobo).

### Machine 3 — GPU host (runs the MolmoBot model server)
```bash
git clone <repo> fr3-molmobot-deploy && cd fr3-molmobot-deploy
# install the external MolmoBot `olmo` repo + its env + checkpoints (docs/LOCAL_GPU_INFERENCE_SETUP.md)
# then launch (from the olmo repo so it's importable): see server/README.md
```

## The one config file per machine
`configs/robot.env` is **git-ignored** and **different on each machine**:
- RT host: `FRANKY_ROBOT_IP` matters (FCI IP); URLs can stay localhost.
- Workstation: `FRANKY_URL`/`HAND_URL` point at **RT_HOST_IP**; model vars point
  at **GPU_HOST_IP**; ZED serials are this box's cameras.
- GPU host: mostly uses the model repo's own launch flags.

## Ports reference
| service | port | host |
|---|---|---|
| arm (`franky_service`) | 54321 | RT host |
| Franka Hand | 54324 | RT host |
| Robotiq (optional) | 54323 | RT host |
| MolmoBot model | 8000 | GPU host |
| web console | 7070 | workstation |
| viser UI | 8080 | workstation |
