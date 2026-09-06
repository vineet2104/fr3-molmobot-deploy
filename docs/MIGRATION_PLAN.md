# Franky-Service ⇄ MolmoBot New-Robot Migration Plan

**Audience:** handoff document for the agent responsible for creating a small,
self-contained deployment repository.

**Goal:** reproduce the known working FR3 `franky_service`-based MolmoBot and
MolmoBot-FT setup on a new robot system with the fewest new control components
possible. Preserve the proven joint-position service, sequential policy
executor, websocket contract, gripper behavior, and logs while parameterizing
all site-specific details.

**Non-goal:** redesign FR3Py or introduce a new arm controller. The separate
long-term FR3Py plan is in
[`MOLMOBOT_VLA_INTEGRATION_PLAN.md`](MOLMOBOT_VLA_INTEGRATION_PLAN.md).

**Important:** this is a migration and repository-construction plan, not a claim
that the copied stack is safe on a different robot without incremental hardware
validation.

---

## 1. Why migrate the Franky stack first

The reference stack already implements the core functionality MolmoBot needs:

- absolute joint-position targets through
  `franky.JointMotion(franky.JointState(positions, velocities))`;
- a 50 Hz arm service loop;
- live joint-state reads;
- stop, home, health, and configurable command timeout;
- asynchronous/non-blocking HTTP gripper services;
- tested MolmoBot/MolmoBot-FT websocket clients;
- sequential chunk playback at 15 Hz;
- camera preprocessing, gripper debounce, safety clipping, and episode logs.

This makes the migration primarily an environment, configuration, and hardware
validation task. In contrast, a new FR3Py path would also require a new
position-control implementation.

---

## 2. Known reference topology

The working arrangement separated real-time robot control, policy I/O, and
model inference:

```text
New RT control host/NUC
┌────────────────────────────────────────────────────────────────┐
│ franky_service :54321                                         │
│   franky-control → libfranka/FCI → FR3                         │
│ optional Franka-Hand service :54324                            │
└────────────────────────▲───────────────────────────────────────┘
                         │ HTTP
Policy client/workstation│
┌────────────────────────┴───────────────────────────────────────┐
│ MolmoBot rollout client                                       │
│ cameras + optional Robotiq service :54323                      │
│ sequential 15 Hz chunk executor + logs                        │
└────────────────────────▲───────────────────────────────────────┘
                         │ websocket/msgpack-numpy
GPU inference host       │
┌────────────────────────┴───────────────────────────────────────┐
│ published MolmoBot checkout/checkpoint + serve_bridge.py      │
└────────────────────────────────────────────────────────────────┘
```

The three roles may run on fewer machines, but the robot-control process must
run on the RT-capable host connected to the FR3 FCI network.

---

## 3. First decision gate: establish compatibility

Before building the new repository, inventory the **actual working robot-control
host**. A workstation environment is not an authoritative substitute if the
service ran on a separate NUC.

Capture:

```bash
uname -a
cat /etc/os-release
uname -r
python --version
python -m pip freeze
python -m pip show franky-control
python -m pip show numpy fastapi uvicorn pydantic
```

Also capture:

- exact repository commit containing the working arm and gripper services;
- exact `franky-control` package version and installation source;
- libfranka version if it is independently installed;
- FR3 system/firmware version shown by Desk;
- kernel boot entry and real-time scheduling setup;
- robot and control-host network configuration;
- launch scripts, systemd units, tmux commands, and environment files;
- whether the working system uses Franka Hand or Robotiq 2F-85;
- camera SDK and camera serial numbers;
- model-server checkout commit, server shim commit, and checkpoint identifiers.

### Go/no-go probe on the new system

The decisive early test is:

1. Install a pinned minimal Franky environment on the new RT host.
2. Import `franky`.
3. Connect to the target FR3.
4. Recover/read robot state without commanding motion.

If Franky cannot connect because the available package/libfranka combination is
incompatible with the target firmware or operating system, stop the migration
and resolve that compatibility issue before copying the policy stack. Do not
mask it by changing policy code.

---

## 4. Repository the Franky agent should create

Suggested repository layout:

```text
fr3-molmobot-deploy/
├── README.md
├── LICENSES.md
├── pyproject.toml                  # client/common package, optional extras
├── configs/
│   ├── robot.example.yaml
│   ├── cameras.example.yaml
│   ├── molmobot_official.yaml
│   └── molmobot_ft.yaml
├── requirements/
│   ├── robot-lock.txt              # exact tested robot-host packages
│   ├── client-lock.txt
│   └── model-server-notes.md
├── services/
│   ├── franky_service.py
│   ├── franka_hand_service.py      # include only if authorized/needed
│   └── clients.py
├── client/
│   ├── molmobot_client.py
│   ├── run_molmobot.py
│   ├── cameras.py
│   ├── grippers.py
│   ├── safety.py
│   ├── chunk_executor.py
│   └── logging.py
├── server/
│   ├── serve_bridge.py
│   ├── bridge_policy.py
│   └── README.md
├── scripts/
│   ├── install_robot_host.sh
│   ├── install_client_host.sh
│   ├── launch_arm_service.sh
│   ├── launch_gripper_service.sh
│   ├── preflight.py
│   ├── dummy_inference.py
│   └── capture_environment.sh
├── deploy/
│   ├── systemd/franky-service.service
│   └── env/robot.env.example
└── tests/
    ├── test_safety.py
    ├── test_gripper_encoding.py
    ├── test_chunk_executor.py
    ├── test_fake_arm_service.py
    └── test_fake_model_server.py
```

A smaller layout is acceptable, but the robot service, rollout client, and GPU
server setup must remain separable because they have different dependencies and
hardware access.

### Licensing requirement

Before copying implementation files from any private/internal repository, the
agent must determine whether they may be redistributed to the user's personal
repository. Record origins and licenses in `LICENSES.md`.

If a component cannot be redistributed:

- reimplement only the necessary interface from its documented behavior;
- require the user to obtain/install the dependency separately; or
- include a patch/extraction script that operates on an authorized local
  checkout without committing restricted source.

Do not commit model checkpoints, credentials, internal host names, tokens, or
private package indexes.

---

## 5. Configuration contract

No reference-machine address or serial number should be hard-coded in runtime
code. Use a YAML configuration plus environment overrides.

A robot configuration should include at least:

```yaml
robot:
  ip: "172.16.0.3"
  service_host: "0.0.0.0"
  service_port: 54321
  control_hz: 50
  command_timeout_s: 0.5
  home_position: [0.0, -0.4, 0.0, -1.9, 0.0, 1.5, 0.0]
  relative_dynamics:
    velocity: 0.3
    acceleration: 0.2
    jerk: 0.1
  joint_limits: null              # populate from target FR3 verification
  joint_limit_margin_rad: 0.05

gripper:
  type: "franka_hand"             # or robotiq_2f85 / none
  service_url: "http://ROBOT_HOST:54324"
  max_width_m: 0.08

model:
  ws_url: "ws://GPU_HOST:8000"
  profile: "molmobot_ft"

cameras:
  backend: "zed"                  # zed or realsense
  exterior_serial: "..."
  wrist_serial: "..."
```

Secrets and host-specific values belong in an ignored `robot.yaml` or `.env`.
Commit only examples.

### Correct configuration loading

The current reference service reads environment values such as
`FRANKY_REL_DYN`, but the inspected initialization assigns a hard-coded
`RelativeDynamicsFactor(0.3, 0.2, 0.1)`. The migration repository should have
one clear source of truth and verify that displayed configuration is the
configuration actually applied to the robot.

Do not rely on the existing default command timeout of 10 seconds. Set a short,
explicit launch value; start around `0.5 s` for direct 15 Hz streaming or use a
carefully justified value up to `1.0 s` if inference/client timing requires it.

---

## 6. Arm-service behavior to preserve and harden

### Required HTTP API

At minimum preserve:

| Method | Endpoint | Behavior |
|---|---|---|
| GET | `/health` | robot connection, stop latch, target presence, last error |
| GET | `/joint_state` | `positions[7]`, `velocities[7]` |
| GET | `/target_joint_state` | current target, age, sequence, latch state |
| GET | `/command_timeout` | active timeout |
| POST | `/target_joint_state` | accept position and velocity arrays |
| POST | `/stop` | stop and latch |
| POST | `/go_home` | controlled move to configured home |
| POST | `/command_timeout` | set watchdog timeout |

Run exactly one service worker and one robot owner. Multiple uvicorn workers
must not create multiple robot connections.

### Streaming behavior

The reference service loop runs around 50 Hz and repeatedly applies:

```python
target = franky.JointState(positions, velocities)
motion = franky.JointMotion(target)
robot.move(motion, asynchronous=True)
```

When target age exceeds the configured timeout, it sends a
`JointStopMotion`, clears the target, and latches stopped.

Keep that behavior initially; avoid changing a known working controller during
migration.

### Minimal hardening

The new small repository should additionally:

- validate length seven for positions and velocities;
- reject NaN and infinity;
- serialize all access to the robot object correctly;
- reject or clamp targets according to **verified target-FR3 limits**;
- return explicit rejection reasons;
- expose the applied home pose, dynamics factors, control rate, and timeout in a
  diagnostics/config endpoint;
- prevent a second motion client or implement a lease/owner token;
- preserve and expose the last control exception;
- perform best-effort error recovery without hiding the primary exception;
- latch stopped on controller errors;
- default to stopped until the first valid target;
- avoid automatic startup motion;
- support an IP allowlist or bind to a trusted control network;
- document that HTTP is unauthenticated unless deployment adds authentication.

### Target limits

Do not blindly copy Panda limits or assume the bundled URDF is authoritative.
Before enabling motion, obtain the applicable limits from the target FR3's
supported model/URDF/libfranka setup and store the verified source and date.
Use a conservative interior margin for initial policy tests.

Grossly invalid policy targets should abort a chunk rather than merely be
clamped into an unrelated pose. Log every clamp or rejection.

---

## 7. Gripper migration

Select exactly one backend in configuration.

### Franka Hand

Preserve or recreate the HTTP interface:

| Method | Endpoint | Behavior |
|---|---|---|
| GET | `/health` | connected, width, grasp state, busy, error |
| GET | `/gripper_state` | width, maximum width, grasp state, busy |
| POST | `/gripper_move` | asynchronous width move |
| POST | `/gripper_grasp` | asynchronous force grasp |
| POST | `/gripper_open` | open to measured maximum |
| POST | `/gripper_stop` | abort motion |

The service should own the gripper connection, execute blocking SDK operations
in its worker, enforce one active operation, and return immediately to the
rollout client.

### Robotiq 2F-85

If the new robot uses Robotiq, preserve the existing service contract for
state, activation, `go_to`, open, close, and health. Parameterize serial device,
RS-485 settings, speed, force, and width/bit conversion.

### Checkpoint-space state and actions

Profiles must specify independently:

- hardware backend;
- measured-state encoding sent to the model;
- model action decoding or binary threshold;
- open/closed direction;
- threshold (`0.5`, `128`, or profile-specific);
- debounce count;
- grasp width, speed, force, and epsilon.

Never infer checkpoint encoding solely from hardware type. Validate direction
and units with gripper-only tests before policy arm motion.

---

## 8. Camera migration

Support the hardware actually installed on the new system rather than assuming
the reference serial numbers.

### MolmoBot requirements

- RGB, not BGR;
- contiguous `uint8` arrays;
- one or two views according to server metadata/profile;
- commonly square center crop resized to `480×480`;
- explicit mapping to names such as `exo_front`/`wrist` or
  `exo_camera_1`/`wrist_camera`.

### ZED

- Pin/document the ZED SDK version.
- Enumerate and record serial numbers.
- Ensure cameras run over the required USB bandwidth.
- Only one process may own a ZED. Stop any camera service before a rollout
  client opens cameras directly.

### RealSense

If the new rig uses RealSense, provide a color-only adapter unless depth is
required. Record serial numbers and timestamps, use thread-safe snapshots, and
detect stale frames.

### Validation

Provide a camera-preview/preflight command that saves labelled frames without
connecting to the model or robot motion. The operator must verify:

- physical camera identity;
- orientation;
- RGB color order;
- crop and field of view;
- wrist/exterior key mapping;
- no stale or repeated frames.

---

## 9. Model-server migration

Keep model inference on a suitable GPU host. The small repository should carry
only lightweight serving glue and instructions; use an external, pinned
MolmoBot checkout and externally obtained checkpoints.

Record:

- published MolmoBot repository URL and exact commit;
- Python/CUDA/PyTorch environment;
- model checkpoint ID or path convention;
- expected camera names;
- `action_type`;
- `action_horizon` and `execute_horizon`;
- whether `chunk_response` and JPEG observations are supported;
- launch command and health check.

### Required websocket contract

On connect, the server sends metadata. The client sends a msgpack-numpy
observation and receives either:

```python
{"arm": (7,), "gripper": (G,)}
```

or a stacked response:

```python
{
    "arm": (K, 7),
    "gripper": (K, G),
    "chunk_size": K,
    "action_type": "joint_pos",
}
```

For MolmoBot/MolmoBot-FT, require `joint_pos` and treat it as absolute radians.
One websocket connection corresponds to one episode/policy reset.

Provide:

- HTTP health probe;
- metadata-only probe;
- dummy inference using synthetic observations;
- live-observation inference with motion disabled.

No server response should be able to arm the robot automatically.

---

## 10. Rollout behavior to preserve

The default executor must remain faithful to the working reference:

1. Read current arm and gripper state.
2. Capture fresh camera image(s).
3. Construct the exact checkpoint observation.
4. Perform one blocking inference.
5. Validate the complete response.
6. Execute the first `execute_horizon` actions sequentially on a monotonic
   15 Hz clock.
7. Re-observe and re-infer only after that execution horizon.

Do not enable double-buffered inference by default. It changes observation age
and previously reintroduced chunk-boundary oscillation.

### Per-action checks

Before sending each action:

- verify `(7,)` shape and finite values;
- confirm the configured action type is absolute position;
- enforce verified FR3 hard limits and interior margins;
- enforce a conservative delta from the previous target;
- optionally use joint-wise velocity limit × `1/rate` bounds;
- apply profile-specific gripper voting/debounce;
- log raw, clipped, and sent values.

### Required operator modes

- `--dry-run`: cameras, state, inference, validation, and logs; no commands;
- `--step-confirm`: confirm each chunk before sending;
- `--recorded-actions`: test arm service without a model server;
- `--no-gripper`;
- `--steps N`;
- conservative initial `--max-joint-delta`;
- optional image preview;
- clean SIGINT that stops the arm and gripper.

---

## 11. Logging and reproducibility

Preserve the reference-compatible output layout:

```text
run_meta.json
steps.jsonl
frames/
frames_wrist/
service_run.log
```

`run_meta.json` should include:

- repository commits for deployment and model server;
- robot model, firmware/system version, and service package versions;
- sanitized configuration;
- server metadata and checkpoint ID;
- selected policy profile;
- camera serials and mappings;
- gripper backend and encoding;
- initial robot/gripper state;
- safety limits, margins, timeout, control rate, and policy rate.

Each JSONL action should include:

- observed arm and gripper state;
- raw model action;
- sent target;
- every clamp/rejection;
- gripper vote, state, and operation;
- inference latency;
- dispatch jitter;
- chunk/inference indices;
- service stop/error state.

Do not log credentials or private access tokens.

---

## 12. Preflight requirements

Create one command that checks all non-motion prerequisites and exits nonzero on
failure:

1. Robot-host OS and RT kernel recorded.
2. Target FR3 reachable on its dedicated interface.
3. Desk reports joints unlocked and FCI can be activated.
4. `franky_service /health` is reachable.
5. Joint state has correct shape and changes timestamps.
6. Service starts stopped/latched and reports expected configuration.
7. Exactly one robot-control service is running.
8. Gripper service is healthy or explicitly disabled.
9. Each configured camera opens and produces fresh frames.
10. Camera shapes, RGB dtype, and key mapping match the profile.
11. GPU health endpoint is reachable.
12. Websocket metadata is compatible with the selected profile.
13. Dummy inference returns finite expected-shape actions.
14. Initial predicted action is checked against robot state and verified limits.
15. Logging directory is writable.

Preflight must not post arm or gripper motion commands.

---

## 13. Offline validation before moving to the robot

The new repository should include fake services and tests sufficient to run the
whole control flow without hardware.

### Unit tests

- finite and vector-length validation;
- limit and delta handling;
- gross-target rejection;
- normalized/raw/metres/gripper-angle encoding endpoints;
- gripper vote, direction, and debounce;
- camera RGB/crop/resize behavior;
- response shape and metadata validation;
- monotonic chunk scheduler ordering.

### Integration tests

Run:

```text
fake arm HTTP service
+ fake gripper HTTP service
+ synthetic cameras
+ fake MolmoBot websocket server
+ real rollout client
+ real logger
```

Verify:

- one fresh observation per chunk;
- one inference per chunk;
- actions remain absolute and are not integrated;
- exactly the configured execution horizon is dispatched;
- default executor is sequential, not double-buffered;
- service timeout/stop and websocket failure abort the episode;
- malformed and non-finite actions result in zero motion posts;
- logs are complete and parseable.

### Clean-clone test

On clean machines representative of each role:

- robot host installs from the documented lock and imports Franky;
- client host installs without robot-control dependencies;
- GPU host follows pinned external MolmoBot setup;
- all launch and preflight commands work without editing source files.

---

## 14. Incremental hardware migration sequence

### Phase A — Network and read-only robot connection

1. Boot the RT kernel.
2. Connect the dedicated robot Ethernet interface.
3. Confirm host-to-robot connectivity.
4. Open Desk, unlock joints, and activate FCI when instructed.
5. Start only `franky_service`.
6. Check `/health` and repeatedly read `/joint_state`.
7. Verify the correct seven-joint order and plausible units.

**Gate:** no motion test until state reads and service configuration are
correct.

### Phase B — Small arm motions without model/cameras

1. Keep the user stop within reach and clear the workspace.
2. Post the measured position back as a target; verify no jump.
3. Move one joint by `0.01 rad` within a conservative interior range.
4. Return to the original pose.
5. Test `/stop` during a small move.
6. Stop the client/command stream and verify the configured timeout physically.
7. Only after these tests, validate the configured home pose and `/go_home`.

**Gate:** stop, timeout, and small-position tracking must be reliable.

### Phase C — Gripper only

1. Read width/state.
2. Open at low speed.
3. Close/move without an object.
4. Grasp a safe test object at low force.
5. Verify stop, busy state, and open/closed direction.
6. Validate each checkpoint-space state encoding numerically.

**Gate:** model actions remain disabled until direction and safe force are
known.

### Phase D — Cameras and inference without motion

1. Capture labelled frames from every camera.
2. Verify physical mapping, RGB, crop, and orientation.
3. Connect to the model and record server metadata.
4. Run synthetic dummy inference.
5. Build a real observation and run live inference in `--dry-run`.
6. Inspect predicted target positions, deltas, limits, and gripper values.

**Gate:** reject any action-space, camera-key, or gripper-scale ambiguity.

### Phase E — Recorded chunk

1. Create a hand-reviewed eight-action target chunk containing very small
   changes from current position.
2. Execute with `--recorded-actions --step-confirm`.
3. Measure/inspect tracking, timeout behavior, and logs.
4. Repeat at 15 Hz only after individual steps are satisfactory.

### Phase F — MolmoBot confirmed rollout

1. Start `--step-confirm` with a small maximum delta and gripper disabled.
2. Execute one chunk.
3. Review motion and logs.
4. Enable gripper after arm behavior is acceptable.
5. Validate predicted gripper direction before allowing an edge.
6. Increase motion bounds gradually.
7. Only then run free sequential rollouts.

---

## 15. Acceptance criteria

### Reproducibility

- A new clone installs each machine role from documented commands.
- Versions, external checkouts, and checkpoint identifiers are pinned.
- No source edits are needed for host addresses, serials, or robot configuration.
- Restricted/private source and credentials are absent.

### Arm service

- Equal-to-current target causes no jump.
- Small absolute targets track smoothly.
- Invalid or stale commands stop/latch safely.
- `/stop` and command timeout are physically verified.
- Service diagnostics show the configuration actually applied.
- Only one process owns the robot.

### MolmoBot

- Metadata, camera keys, image shapes, and checkpoint profile match.
- Arm outputs are interpreted as absolute radians.
- Sequential execution reproduces the intended 15 Hz horizon behavior.
- No unexpected target is silently transformed across incompatible FR3 limits.
- Gripper state/action conventions are physically verified.
- Reference-compatible logs and videos are generated.

---

## 16. Information the repository agent must request from the user

Before finalizing defaults, obtain:

- target FR3 robot IP and control-host network interface;
- target host OS and RT-kernel status;
- Desk/robot firmware or system version;
- whether Franky is already installed and can connect;
- Franka Hand, Robotiq 2F-85, or no gripper;
- gripper IP or serial device;
- ZED or RealSense cameras, serial numbers, and physical roles;
- location of the rollout client relative to the robot network;
- GPU type and model-server host;
- checkpoint(s): official MolmoBot, MolmoBot-FT, or both;
- authorized source repositories and redistribution constraints;
- desired home pose and whether it has been physically validated;
- task/workspace differences from the reference setup.

Unknown values must remain explicit placeholders, not guessed defaults.

---

## 17. Suggested execution order for the Franky-service agent

1. Audit source licenses and capture the working NUC environment.
2. Create the minimal repository and machine-role documentation.
3. Parameterize and test the arm service; fix configuration truthfulness.
4. Add fake arm/model/gripper services and offline tests.
5. Add the appropriate gripper backend.
6. Add configured ZED/RealSense capture and preprocessing.
7. Port the canonical MolmoBot websocket client and sequential executor.
8. Add logging, preflight, dry-run, step-confirm, and recorded actions.
9. Prove installation from a clean clone.
10. Perform the incremental robot bring-up in Section 14.

This route intentionally minimizes new control research: reproduce the known
working Franky position-control behavior first, isolate hardware-specific
configuration, and validate each layer before model-generated motion is
permitted.
