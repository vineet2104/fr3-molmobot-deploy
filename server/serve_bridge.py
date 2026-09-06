"""Serve a MolmoBot SynthManip checkpoint over WebSocket for sim OR real robot.

Run on a GPU node (see serve_bridge.slurm). The same service works for both the
simulator and the real robot -- clients send the same msgpack-numpy observation
dict and receive an action dict (see CLIENT_GUIDE.md).

Example:
    cd /lustre/fs12/.../MolmoBot/MolmoBot
    PYTHONPATH=. python ../serve/serve_bridge.py \
        --checkpoint_path /lustre/fsw/.../checkpoints/molmobot_pilot_pineapple_pick_v1/step5833 \
        --port 8000
"""
import argparse
import logging
import os
import socket
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serve_bridge")

# Make the standalone bridge_policy importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="Serve MolmoBot bridge policy (sim + real)")
    p.add_argument("--checkpoint_path", required=True, help="Path to checkpoint dir (e.g. .../step5833)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--camera_names", nargs="+", default=["exo_front"])
    p.add_argument("--action_type", default="joint_pos",
                   choices=["joint_pos", "joint_pos_rel"],
                   help="joint_pos = absolute (this pilot). joint_pos_rel = delta.")
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--execute_horizon", type=int, default=8,
                   help="Actions executed open-loop before re-querying the model.")
    p.add_argument("--gripper_representation_count", type=int, default=1)
    p.add_argument("--clamp_gripper", action="store_true",
                   help="Binarize gripper output (>128 -> 255 else 0).")
    p.add_argument("--default_task", default="Pick up the pineapple slices can")
    p.add_argument("--num_flow_steps", type=int, default=None,
                   help="Flow-matching steps; default uses checkpoint value (10).")
    args = p.parse_args()

    from bridge_policy import BridgePolicy, BridgePolicyConfig
    from olmo.eval.websocket_server import WebsocketPolicyServer

    cfg = BridgePolicyConfig(
        checkpoint_path=args.checkpoint_path,
        camera_names=args.camera_names,
        action_type=args.action_type,
        action_horizon=args.action_horizon,
        execute_horizon=args.execute_horizon,
        gripper_representation_count=args.gripper_representation_count,
        clamp_gripper=args.clamp_gripper,
        default_task=args.default_task,
        num_flow_steps=args.num_flow_steps,
    )
    policy = BridgePolicy(cfg)

    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "<node-ip>"

    logger.info("=" * 70)
    logger.info(f"Loading model (this can take ~1-2 min for the 4B checkpoint)...")
    # WebsocketPolicyServer calls policy.prepare_model() in serve_forever(),
    # but we load up-front so 'READY' only prints once the model is on the GPU.
    policy.prepare_model()
    logger.info("Model ready.")
    logger.info("=" * 70)
    logger.info(f"Serving on  ws://{hostname}:{args.port}   (node IP: {ip})")
    logger.info(f"Health check: http://{hostname}:{args.port}/healthz")
    logger.info(f"Camera(s): {args.camera_names} | action_type: {args.action_type} "
                f"| horizon: {args.action_horizon} | execute: {args.execute_horizon}")
    logger.info("=" * 70)

    server = WebsocketPolicyServer(
        [policy],
        model_name="molmobot-pilot-pineapple-pick",
        host=args.host,
        port=args.port,
        metadata={
            "checkpoint_path": args.checkpoint_path,
            "camera_names": args.camera_names,
            "action_type": args.action_type,
            "action_horizon": args.action_horizon,
            "execute_horizon": args.execute_horizon,
        },
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
