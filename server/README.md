# GPU host — MolmoBot model server

Runs the MolmoBot policy over a websocket the workstation client connects to.
Only lightweight *glue* lives here; the model code (`olmo`) and checkpoints are
external and installed on the GPU host.

## One-time
- Install the MolmoBot repo (`olmo` package providing
  `SynthManipMolmoInferenceWrapper` + `WebsocketPolicyServer`) and its env
  (the reference used a conda env `molmobot`, PyTorch/CUDA matching your GPUs).
- Obtain the checkpoint(s): official (`MolmoBot-Img-DROID`) and/or finetuned.
- Full details: `../docs/LOCAL_GPU_INFERENCE_SETUP.md`.

## Launch (from the MolmoBot repo so `olmo` is importable)
```bash
# MolmoBot-FT (two-view), port 8000, chunk-response on:
PYTHONPATH=. python /path/to/fr3-molmobot-deploy/server/serve_bridge.py \
    --checkpoint_path /local/ckpts/<ft_checkpoint> \
    --host 0.0.0.0 --port 8000 \
    --camera_names exo_front wrist \
    --action_type joint_pos --action_horizon 16 --execute_horizon 8 \
    --gripper_representation_count 1 --chunk_response

# MolmoBot official (Img-DROID): --camera_names exo_camera_1 wrist_camera --clamp_gripper
```
Health check: `curl http://<gpu-host>:8000/healthz` → OK.

Point the workstation at it via `configs/robot.env`:
`MOLMOBOTFT_A100_HOST=<gpu-host> MOLMOBOTFT_A100_PORT=8000`.

## Contract (must match the client)
- On connect the server sends metadata (`camera_names`, `action_type`,
  `action_horizon`, `execute_horizon`, `chunk_response`).
- Per request: msgpack-numpy obs → stacked action chunk
  `{arm:(K,7), gripper:(K,1), chunk_size:K, action_type:"joint_pos"}`.
- `joint_pos` = **absolute joint radians**. One connection = one episode.
