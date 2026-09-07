# viz/ — Viser 3D UI + cuRobo motion planning

A browser-based 3D scene (viser) that:
- Loads the Franka Panda URDF and shows the **live** robot pose (polls the arm
  service `/joint_state`).
- Lets you drag a goal gizmo, **plan** a collision-free trajectory with cuRobo,
  preview the EE path, and **execute** it by streaming to
  `/target_joint_state`.
- Buttons for **go home / stop** and **gripper open/close** (Franka Hand or
  Robotiq).

Runs on the **workstation** (same machine as the rollout client). It talks to
the same arm + gripper services on the RT host — it does not need the GPU host.

```
browser  ──http──▶  viz/server.py (viser :8080)  ──http──▶  franky_service :54321 (RT host)
                          │                                   franka_hand   :54324 (RT host)
                          └── cuRobo planner (local GPU)
```

## Dependencies (heavier than the rollout client)
- `viser`, `yourdfpy`, `numpy`, `requests`
- `torch` (CUDA) + **cuRobo** installed with its content assets — the planner
  loads `MotionPlannerCfg.create(robot="franka.yml")`, which ships with cuRobo.
  Install cuRobo per its docs (needs CUDA toolkit; `CUDA_HOME` set). This is the
  main setup cost of the viz tool.
- The Franka Panda URDF + meshes are **bundled** here in `viz/assets/` (no
  external asset path needed). Override with `FRANKA_URDF_PATH` if you want your
  own URDF.

## Config (env; defaults shown)
| var | default | meaning |
|---|---|---|
| `FRANKY_SERVICE_URL` | `http://127.0.0.1:54321` | arm service (set to `http://RT_HOST_IP:54321`) |
| `FRANKA_HAND_SERVICE_URL` | `http://127.0.0.1:54324` | Franka Hand service |
| `GRIPPER_KIND` | `franka_hand` | `franka_hand` or `robotiq` |
| `ROBOTIQ_SERVICE_URL` | `http://127.0.0.1:54323` | only if `GRIPPER_KIND=robotiq` |
| `FRANKA_URDF_PATH` | bundled `assets/franka_panda.urdf` | override URDF |
| `JOINT_POLL_HZ` | `10.0` | live-state poll rate |
| `HOME_Q` (in file) | matches `franky_service.HOME_POSITION` | keep in sync |

## Run
```bash
source ../configs/robot.env          # sets FRANKY_URL etc; export the SERVICE_URL vars too
export FRANKY_SERVICE_URL=$FRANKY_URL FRANKA_HAND_SERVICE_URL=$HAND_URL
python server.py                     # open http://<workstation-ip>:8080
# or: ./run.sh  (tmux session, activates the `franky` conda env + CUDA_HOME)
```

### Legacy RTX 2080 Ti path

The lab workstation's Turing GPU/driver cannot run current cuRobo V2. A
minimal, execution-disabled-by-default v0.7.8 UI is available temporarily:

```bash
./scripts/setup_legacy_viz.sh  # once, from the repository root
./viz/run_legacy.sh
# open http://<workstation-ip>:8080
```

See `docs/BRINGUP_PROGRESS_2026-09-06.md`. Use `server_legacy.py` only with the
pinned environment created by the setup script.

## Safety
Planning is offline; **Execute streams real motion** to the arm. Same rules as a
rollout: e-stop within reach, joints unlocked, FCI active, start with the
conservative `relative_dynamics` in the arm service. Verify `HOME_Q` matches the
arm service's home before using "go home".
