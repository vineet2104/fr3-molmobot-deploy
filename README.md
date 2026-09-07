# FR3 MolmoBot deployment

Deployment stack for testing MolmoBot on the lab FR3 with a Franka Hand, ZED 2
exterior camera, and RealSense D435i wrist camera. The full pipeline runs; task
completion is currently model/checkpoint-dependent and still under evaluation.

> Keep the physical e-stop within reach. This is a lab bring-up stack, not a
> safety-certified controller. Only one application should command the arm at a
> time.

## Lab topology

| Machine | Address | Runs |
|---|---:|---|
| RT robot host | `192.168.123.249` | Franky arm and Franka Hand Docker services |
| FR3 / Desk / FCI | `192.168.123.250` | Robot |
| Robot workstation | `192.168.123.203` | Viser and deployment console |
| GPU server | entered in the deployment UI | MolmoBot model server |

## One-time setup

### RT robot host

```bash
cd ~/fr3-molmobot-deploy
cp configs/env.example configs/robot.env
# Set FRANKY_ROBOT_IP=192.168.123.250 in configs/robot.env
./scripts/build_arm_service_image.sh
```

### Robot workstation

```bash
cd ~/Desktop/vineet/fr3-molmobot-deploy
./scripts/setup_legacy_viz.sh
./scripts/setup_deployment_console.sh
```

## Normal startup

### 1. Prepare the robot

In Franka Desk:

1. Activate FCI.
2. Unlock the joints.
3. Clear the workspace and keep the e-stop ready.
4. Ensure the old FR3Py `fr3_joint_interface` is not running.

### 2. Start both services on the RT host

RT terminal 1 — arm:

```bash
cd ~/fr3-molmobot-deploy
./scripts/launch_arm_service_docker.sh
```

RT terminal 2 — Franka Hand:

```bash
cd ~/fr3-molmobot-deploy
./scripts/launch_gripper_service_docker.sh
```

### 3. Start the model on the GPU server

From the MolmoBot/model repository on the GPU server:

```bash
bash scripts/serve_molmobot_img_droid.sh
```

Keep it running and note the GPU server IP and port. The server must return a
16-action chunk with absolute joint-position actions.

### 4. Start Viser on the robot workstation

```bash
cd ~/Desktop/vineet/fr3-molmobot-deploy
./viz/run_legacy.sh
```

Open:

```text
http://192.168.123.203:8080
```

Use Viser to inspect live joints, plan poses, test the hand, or move to the
configured home pose. The current home is:

```text
XYZ = [0.378, 0.432, 0.317] m
orientation = FK orientation of the original FR3Py home pose
```

Viser does not model the table or other lab obstacles. Do not execute Viser
motions while a deployment rollout is running.

### 5. Start the deployment console on the robot workstation

Workstation terminal 2:

```bash
cd ~/Desktop/vineet/fr3-molmobot-deploy
./scripts/launch_deployment_console.sh
```

Open:

```text
http://192.168.123.203:7071
```

In the page:

1. Confirm both live camera views.
2. Enter an experiment name.
3. Enter the GPU model-server IP and port.
4. Enter the task instruction.
5. Run **Preflight**.
6. First run with **Dry run** enabled.
7. For real motion, stop any Viser motion, uncheck **Dry run**, and press
   **Run continuously**.
8. Press **STOP** to end the rollout.

The rollout loop is:

```text
capture two RGB images + robot state
    → blocking model inference (16 actions)
    → execute the first 8 actions at 15 Hz
    → capture fresh observations and repeat
```

Inference delay between chunks is expected. Logs are saved under:

```text
logs/deployments/<timestamp>_<experiment>/
```

## Shutdown

1. Press **STOP** in the deployment console if a rollout is active.
2. Stop Viser and the deployment console with `Ctrl+C`.
3. Stop the two RT service terminals with `Ctrl+C`.
4. Stop the model server on the GPU host.
5. Lock/deactivate the robot according to normal lab procedure.

## Current camera mapping

```text
exo_camera_1 / exo_front  → ZED 2 SN 28576947
wrist_camera / wrist      → RealSense D435i SN 216322074479
```

The deployment console reads the two camera key names from model metadata, so
both Img-DROID and finetuned naming conventions are supported.

## More detail

- `docs/DEPLOYMENT_CONSOLE.md` — deployment UI and 16→8 execution contract
- `docs/BRINGUP_PROGRESS_2026-09-06.md` — validated setup and bring-up record
- `viz/README.md` — Viser and legacy cuRobo notes
- `server/README.md` — model websocket contract
- `docs/V3_GRIPPER_AND_IO_CONVENTIONS.md` — checkpoint gripper conventions
- `docs/MIGRATION_PLAN.md` — full safety and migration background
