# FR3 bring-up progress — 2026-09-06

## Verified hardware/topology

- RT host: `franka_computer@crrl-HP-Z440-Workstation`
  - Ubuntu 20.04.6, kernel `5.9.1-rt20` (`PREEMPT_RT`)
  - robot-facing `eno1`: `192.168.123.249/24`
- FR3/Desk/FCI: `192.168.123.250`
  - ping approximately 0.2 ms; Desk HTTPS reachable
- Workstation: `crrl-rstation`, `192.168.123.203/24`
  - RTX 2080 Ti, driver 550.127.08, CUDA toolkit 12.2
- RT service is reachable from the workstation at
  `http://192.168.123.249:54321`.

## Prior FR3 control stack

`../FR3DataCollection` used an FR3Py Python client over LCM to a C++
`fr3_joint_interface`, then libfranka/FCI. The known container uses Ubuntu
22.04, Python 3.10 and `libfranka.so.0.13`. Existing FR3 containers contained
only idle shells during inspection; no prior FCI owner was running.

The previous bridge's initial pose was approximately
`[0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854]`. It differs from this repo's
hard-coded home pose. **Do not use `/go_home` until home is validated.**

## Arm service completed

A pinned Docker image was added and built on the RT host:

- image: `fr3-molmobot-arm:libfranka-0.13.3`
- Ubuntu 22.04 / Python 3.10
- `franky-control==1.1.4`, wheel built for libfranka 0.13.3
- FastAPI 0.116.1 / uvicorn 0.35.0

Launch: `./scripts/launch_arm_service_docker.sh`.

Service fixes:

- `FRANKY_REL_DYN` is now applied instead of ignored;
- initial dynamics are 0.05 for velocity/acceleration/jerk;
- command watchdog defaults to 0.5 seconds.

Read-only state verification passed:

- `robot_connected=true`
- `stop_latched=true`
- `has_target=false`
- `last_control_error=null`
- seven finite joint positions and velocities returned
- command timeout reported as 0.5 seconds

## Legacy Viser/cuRobo workstation setup

Current cuRobo V2 is unsuitable for the RTX 2080 Ti/driver 550, so a temporary
legacy path was added:

- Conda environment: `molmobot-viz` (Python 3.10)
- PyTorch 2.4.1+cu121
- cuRobo v0.7.8 from `~/.local/src/curobo-v0.7.8`
- CUDA extensions compiled for the RTX 2080 Ti (`sm_75`)
- Viser 1.1.0

Setup: `./scripts/setup_legacy_viz.sh`

Run: `./viz/run_legacy.sh`

`viz/server_legacy.py` successfully:

- loaded and warmed legacy cuRobo MotionGen;
- connected to the RT arm service;
- loaded the seven Panda joints;
- started Viser on `http://192.168.123.203:8080`;
- remained execution-disabled by default.

The legacy UI limits a goal to 10 cm from the planning start, rejects plans
whose total joint displacement exceeds 0.35 rad, verifies that the robot has
not moved more than 0.05 rad before execution, and latches `/stop` after a
trajectory. These are supplemental checks, not a safety certification.

## Stop point / tomorrow's first actions

No Viser-planned robot motion has been executed yet.

1. Start the arm service and confirm it is stop-latched.
2. Run `./viz/run_legacy.sh` and open `http://192.168.123.203:8080`.
3. Verify the rendered configuration agrees with the physical arm.
4. Snap the goal to the current EE and plan the zero-motion goal.
5. Test a 5 mm upward goal in read-only planning mode and inspect the preview.
6. Only after review, enable real execution and perform one 5 mm motion with
   the physical e-stop held.
7. Add the real table/workspace collision geometry before larger moves.

Important: the planner currently models the robot/self-collision but not the
lab table, cameras, cables, or other workspace obstacles.

## 2026-09-07 UI follow-up

The initial legacy UI used one status widget for both the 10 Hz robot-state
poll and planner results. The poll immediately overwrote `planning`, `failed`,
or `success`, making planning appear to produce no feedback. This was fixed by
separating persistent **Robot** and **Planner/Execution** status widgets.

Base-frame numeric X/Y/Z fields, an **Apply XYZ** button, and a precise
**Nudge +Z by 5 mm** button were added. Snap now fills these coordinates and
planning prints success/failure both in the browser and terminal. Restart
`./viz/run_legacy.sh` after pulling the update.
