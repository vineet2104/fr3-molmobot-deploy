# AGENT_HANDOFF — read this first

You (a coding agent) are being handed this repo to bring MolmoBot / MolmoBot-FT
up on a **new** FR3 lab. This file orients you; the authoritative,
safety-ordered procedure is `docs/MIGRATION_PLAN.md`.

## What this repo is
A copy of a **known-working** FR3 + gripper VLA stack (`franky_service` joint
**position** control + a sequential 15 Hz action-chunk executor + a MolmoBot
websocket contract), parameterized for a new site. It is a *reproduction* task,
not a redesign. Do **not** invent a new arm controller here (that long-term work
is a separate plan: `FR3Py/MOLMOBOT_VLA_INTEGRATION_PLAN.md`, not in this repo).

## Mental model (3 machines)
See `docs/SETUP.md` for the diagram. RT host runs the robot services; the
workstation runs the client + webpage + viz; the GPU host runs the model. They
talk over the LAN. Each machine has its own git-ignored `configs/robot.env`.

## The data/control contract (don't break these)
- **Arm service** `franky_service.py`: `GET /joint_state` (7 rad), `POST
  /target_joint_state` (absolute 7-rad target), `/go_home`, `/stop`, `/health`,
  short `command_timeout_s` watchdog, one worker = one robot owner. 50 Hz loop.
- **Policy → arm**: the model returns an **action chunk** `arm:(K,7)` absolute
  joint radians + `gripper:(K,1)`. The client executes it **sequentially**
  (MolmoBot `RealRobotVLAPolicy`: predict 16 → execute first 8 from a fresh obs
  → re-infer). Do **not** re-enable the double-buffer path (it oscillates on
  stale-obs chunk boundaries).
- **Gripper action**: normalized `[0,1]`, `> close_threshold` (≈0.5) ⇒ close,
  with a debounce of ~3 votes. **Gripper STATE encoding is hardware-specific**
  (see below).
- **Websocket**: one connection = one episode; server sends metadata on connect
  (`camera_names`, `action_type=joint_pos`, `action_horizon`, `execute_horizon`,
  `chunk_response`).

## Biggest gotchas for the new site (spend your attention here)
1. **`franky` / libfranka compatibility** on the RT host is the go/no-go. If
   `franky.Robot(<ip>)` won't connect (firmware/libfranka mismatch), fix that
   before anything else. This whole approach assumes `franky` works here.
2. **Gripper is a Franka Hand here, not a Robotiq 2F-85.** The reference's
   `robotiq_state_kine` four-bar mapping and thresholds **do not transfer**. Use
   `--gripper panda` (Franka Hand service). Re-validate: (a) does `> 0.5` close
   and `< 0.5` open? (b) what state value does the model expect (radians? meters?
   normalized?) — check the checkpoint's training convention and the
   `configure_real_robot`/gripper preset. Run gripper-only tests before combined
   rollouts. Ref: `docs/V3_GRIPPER_AND_IO_CONVENTIONS.md`.
3. **Camera identity + framing.** The model was trained on a specific view
   geometry (front exo + wrist, square crop). Reproduce the crop/resize in the
   client and map ZED serials via env (`ZED_*_SN`). Wrong view = silent failure.
4. **Home pose + workspace limits** differ. Set `HOME_POSITION` in
   `services/franky_service.py` (and `HOME_Q` in `viz/server.py`) to a validated
   pose. Start with tiny `relative_dynamics` (0.05) and `--max_joint_delta`.
5. **One robot owner.** Only one process may hold the FCI. Only one process may
   open a given ZED (stop any camera service before a rollout).

## Bring-up order (do NOT skip; full detail in MIGRATION_PLAN.md §14)
1. RT host: `franky` connects; arm service `/joint_state` reads live; `/stop`
   works. No motion yet.
2. Tiny gated motion: `--step_confirm --no_gripper --max_joint_delta 0.05`.
3. Gripper-only validation (direction, force, state readout).
4. Cameras enumerate + framing matches training.
5. `dummy_inference.py` → chunk shape ok. Then a recorded (non-executing) chunk.
6. First confirmed rollout at conservative dynamics; then relax.

## Where things live
- Arm/gripper services: `services/`
- Rollout clients + websocket client lib: `client/`
- Model server glue (GPU host): `server/`
- Webpage: `service_console.py`; 3D viz: `viz/`
- Launch/preflight scripts: `scripts/`
- Conventions & setup docs: `docs/`
- **Licensing/provenance to review before publishing: `LICENSES.md`**

## Definition of done
A confirmed pick with MolmoBot-FT: correct camera views, gripper closes on the
object and lifts, arm tracks the chunk at ~15 Hz with no runaway, logs saved.
Then repeat for the official checkpoint. Document any site-specific values you
changed (IPs, serials, home pose, gripper mapping) back into `configs/` + docs.
