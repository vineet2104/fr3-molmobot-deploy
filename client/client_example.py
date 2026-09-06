"""Reference WebSocket client for the MolmoBot bridge service.

IDENTICAL usage for simulator and real robot -- both build the same observation
dict and consume the same action dict. Copy `MolmoBotClient` into your sim / robot
control loop.

Protocol (msgpack-numpy over WebSocket):
  1. connect -> server sends a metadata dict (once).
  2. each control step: send obs dict -> receive action (or action-chunk) dict.

CANONICAL COPY. The repo-root ./client_example.py is kept byte-identical to
this file; edit here and re-sync to avoid drift.

Dependencies on the CLIENT side: `websockets`, `msgpack_numpy`, `numpy`, `opencv`.
    pip install websockets msgpack_numpy numpy opencv-python
"""
import asyncio

import cv2
import msgpack_numpy
import numpy as np
import websockets


def encode_jpeg_obs(obs: dict, quality: int = 90) -> dict:
    """Return a shallow copy of `obs` with every top-level HxWx3 uint8 image
    replaced by a compact JPEG blob:  {"__jpeg__": True, "data": <bytes>,
    "shape": [H,W,3]}.  Non-image entries (qpos dict, task str, etc.) pass
    through unchanged.

    The array is encoded AS-IS (channel order preserved through
    imencode/imdecode), so an RGB array round-trips back to RGB on the server
    with no color conversion required on either side. This cuts a two-view
    480x480 observation from ~1.38 MB to ~60-90 KB (~15-24x), which is the
    dominant cost on a bandwidth-limited link to the GPU node.
    """
    out = {}
    for k, v in obs.items():
        if (isinstance(v, np.ndarray) and v.ndim == 3
                and v.shape[2] == 3 and v.dtype == np.uint8):
            ok, enc = cv2.imencode(".jpg", v,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                raise RuntimeError(f"cv2.imencode failed for obs['{k}']")
            out[k] = {"__jpeg__": True, "data": enc.tobytes(),
                      "shape": list(v.shape)}
        else:
            out[k] = v
    return out


class MolmoBotClient:
    """Synchronous-feeling wrapper around the async websocket policy client."""

    def __init__(self, ws_url: str, jpeg_obs: bool = False, jpeg_quality: int = 90):
        self.ws_url = ws_url
        self._ws = None
        self._loop = asyncio.new_event_loop()
        self._packer = msgpack_numpy.Packer()
        self.metadata = None
        # JPEG observation compression (needs a server that decodes it; the
        # server advertises support via metadata["accepts_jpeg"]).
        self.jpeg_obs = bool(jpeg_obs)
        self.jpeg_quality = int(jpeg_quality)

    def connect(self):
        self._loop.run_until_complete(self._connect())
        return self.metadata

    async def _connect(self):
        self._ws = await websockets.connect(self.ws_url, compression=None, max_size=None)
        # Server sends metadata immediately on connect.
        self.metadata = msgpack_numpy.unpackb(await self._ws.recv())

    def _prep(self, obs: dict) -> dict:
        return encode_jpeg_obs(obs, self.jpeg_quality) if self.jpeg_obs else obs

    def get_action(self, obs: dict) -> dict:
        """Send one observation, return one action dict.

        A NEW connection == a policy reset (action buffer cleared). Keep this
        client connected for the whole episode; reconnect to start a fresh episode.
        """
        return self._loop.run_until_complete(self._get_action(obs))

    async def _get_action(self, obs: dict) -> dict:
        await self._ws.send(self._packer.pack(self._prep(obs)))
        return msgpack_numpy.unpackb(await self._ws.recv())

    def get_action_chunk(self, obs: dict) -> list:
        """15Hz FIX: send ONE observation, receive the FULL execute_horizon chunk.

        When the server was started with chunk_response=True (default), the
        response contains STACKED actions:
            {"arm": (n,7), "gripper": (n,G), "chunk_size": n, "action_type": ...}
        This returns a list of n per-step action dicts [{"arm":(7,),"gripper":(G,)}, ...]
        which the client plays out LOCALLY (one every 66.7 ms) with NO network in
        the per-action path -> true 15 Hz. Re-query (call this again with a fresh
        obs) only every n steps.

        Falls back gracefully if the server is in legacy single-action mode
        (returns a 1-element list).
        """
        resp = self._loop.run_until_complete(self._get_action(obs))
        arm = np.asarray(resp["arm"])
        grip = np.asarray(resp["gripper"])
        timing = resp.get("server_timing")
        if arm.ndim == 1:  # legacy single-action response
            return [{"arm": arm, "gripper": grip, "server_timing": timing}]
        n = int(resp.get("chunk_size", arm.shape[0]))
        return [
            {"arm": arm[t], "gripper": grip[t],
             "server_timing": timing if t == 0 else None}
            for t in range(n)
        ]

    def close(self):
        if self._ws is not None:
            self._loop.run_until_complete(self._ws.close())
        self._loop.close()


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws_url", required=True, help="e.g. ws://<node-hostname>:8000")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--task", default="Pick up the pineapple slices can")
    ap.add_argument("--jpeg_obs", action="store_true",
                    help="JPEG-compress image observations before sending.")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args()

    client = MolmoBotClient(args.ws_url, jpeg_obs=args.jpeg_obs,
                            jpeg_quality=args.jpeg_quality)
    meta = client.connect()
    print("Connected. Server metadata:", meta)
    chunked = bool(meta.get("chunk_response", False)) if isinstance(meta, dict) else False
    accepts_jpeg = bool(meta.get("accepts_jpeg", False)) if isinstance(meta, dict) else False
    print(f"Server chunk_response mode: {chunked}   accepts_jpeg: {accepts_jpeg}")
    if args.jpeg_obs and not accepts_jpeg:
        print("WARN: --jpeg_obs set but server does not advertise accepts_jpeg; "
              "the server must decode the JPEG blobs or it will error.")

    def make_obs():
        rng = np.random.RandomState(0)
        # wrist view included for two-view models; harmless extra key otherwise.
        return {
            "exo_front": rng.randint(0, 255, (480, 480, 3), dtype=np.uint8),  # RGB uint8
            "wrist":     rng.randint(0, 255, (480, 480, 3), dtype=np.uint8),  # RGB uint8
            "qpos": {
                "arm": rng.randn(7).astype(np.float32),       # 7 joint positions (rad)
                "gripper": rng.randn(1).astype(np.float32),   # gripper state (>=1 dim)
            },
            "task": args.task,
        }

    CONTROL_HZ = 15.0
    DT = 1.0 / CONTROL_HZ
    try:
        if chunked:
            # -------- 15Hz path: one net round-trip per CHUNK, play out locally --------
            step = 0
            while step < args.steps:
                obs = make_obs()  # observation only needed at re-query boundaries
                chunk = client.get_action_chunk(obs)  # ONE network round-trip
                for a in chunk:
                    t0 = time.monotonic()
                    arm = np.asarray(a["arm"]); grip = np.asarray(a["gripper"])
                    # ---- SEND arm/grip to robot here (absolute joint targets) ----
                    print(f"step {step:3d}  arm={np.array2string(arm, precision=3)}  "
                          f"gripper={grip}  timing={a.get('server_timing')}")
                    step += 1
                    if step >= args.steps:
                        break
                    # pace to 15Hz locally (no network in this loop)
                    sleep = DT - (time.monotonic() - t0)
                    if sleep > 0:
                        time.sleep(sleep)
        else:
            # -------- legacy path: one net round-trip per ACTION (RTT-capped) --------
            for step in range(args.steps):
                obs = make_obs()
                action = client.get_action(obs)
                arm = np.asarray(action["arm"]); grip = np.asarray(action["gripper"])
                print(f"step {step:3d}  arm={np.array2string(arm, precision=3)}  "
                      f"gripper={grip}  timing={action.get('server_timing')}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
