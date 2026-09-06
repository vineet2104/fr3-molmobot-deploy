"""Dummy-inference the MolmoBot server (no robot/cameras). Confirms the websocket
contract + action-chunk shape. Reads ws from --ws_url or $MOLMOBOTFT_A100_HOST:PORT."""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "client"))
from client_example import MolmoBotClient
ap = argparse.ArgumentParser()
ap.add_argument("--ws_url", default=None)
ap.add_argument("--task", default="Pick up the pineapple slices can")
a = ap.parse_args()
ws = a.ws_url or f"ws://{os.environ.get('MOLMOBOTFT_A100_HOST','127.0.0.1')}:{os.environ.get('MOLMOBOTFT_A100_PORT','8000')}"
print("connecting", ws)
c = MolmoBotClient(ws); meta = c.connect(); print("metadata:", meta)
cams = (meta.get("camera_names") if isinstance(meta, dict) else None) or ["exo_front"]
obs = {cam: np.random.randint(0,255,(480,480,3),np.uint8) for cam in cams}
obs["qpos"] = {"arm": np.zeros(7,np.float32), "gripper": np.array([0.0],np.float32)}
obs["task"] = a.task
r = c.get_action(obs); c.close()
print("action arm shape:", np.asarray(r["arm"]).shape, " gripper:", np.asarray(r["gripper"]).shape)
