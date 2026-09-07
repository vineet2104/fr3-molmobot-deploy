# Mixed-camera MolmoBot deployment console

`deployment_console.py` is the workstation web interface for the lab's ZED 2
exterior camera and RealSense D435i wrist camera. It talks to the arm/hand HTTP
services on the RT host and a MolmoBot websocket server on a remote GPU host.

## Install and launch

```bash
./scripts/setup_deployment_console.sh   # once
./scripts/launch_deployment_console.sh
```

Open `http://192.168.123.203:7071` (or the workstation's current IP).
Only one process may own each camera, so stop other ZED/RealSense capture
programs first.

Defaults for the current lab rig are supplied by the launch script. They can be
overridden in the workstation's ignored `configs/robot.env`:

```bash
export FRANKY_URL=http://192.168.123.249:54321
export HAND_URL=http://192.168.123.249:54324
export ZED_EXO_FRONT_SN=28576947
export REALSENSE_WRIST_SN=216322074479
export DEPLOYMENT_UI_PORT=7071
```

## Rollout contract

For every policy query the console sends raw contiguous `uint8` RGB images:

```python
{
    "exo_front": (480, 480, 3),
    "wrist": (480, 480, 3),
    "qpos": {"arm": (7,), "gripper": (1,)},
    "task": "operator text",
}
```

It expects absolute joint-position output with `arm.shape == (N, 7)`, at least
8 actions, and normally `N == 16`. It executes actions 0 through 7 sequentially
at 15 Hz, then captures fresh images/state and performs the next blocking
inference. Inference delay is allowed; there is no overlapping/double-buffered
inference.

The GPU server should advertise `camera_names=["exo_front", "wrist"]` and
`action_type="joint_pos"`. Launch the bridge with action horizon 16, execute
horizon 8, and chunk response enabled.

## Interface

- Both camera streams remain live while idle, inferring, and executing.
- Experiment name creates `logs/deployments/<timestamp>_<name>/`.
- Model host and port are supplied per run.
- Task text is sent verbatim each inference.
- **Dry run is enabled by default**: cameras, state, websocket inference and
  logs run, but no arm or hand commands are sent.
- Stop immediately posts arm and hand stop requests and prevents a response
  from being executed if inference was in progress.

Logs include run configuration, inference timing, raw actions, sent/clipped
actions, both camera frames at each query, and whether the run was dry.

## Safety and current limitations

- Start with dry-run and inspect all predicted chunks before real execution.
- Arm actions are clamped to the configured position limits and, by default,
  each action is limited to 0.10 rad from the preceding target. Set this value
  to zero only after validating the checkpoint.
- Gripper execution is separately disabled by default. The current close path
  uses a 20 mm, 10 N force grasp after three consecutive close votes.
- Select the gripper-state encoding that matches the exact checkpoint. V3 uses
  `robotiq_angle`; some finetuned checkpoints use `per_finger_m`.
- HTTP and the web interface have no authentication. Bind/use only on the
  trusted lab network.
- A software Stop is not an emergency stop. Keep the physical e-stop ready.
