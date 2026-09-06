# MolmoBot Inference Pipeline — Reference for Local-GPU Setup

**Audience:** an agent porting the MolmoBot inference server from the remote
SLURM/A100 deployment onto a **local GPU** on this workstation, so the policy
can run at a true **15 Hz** without the network bottleneck.

**Why we are doing this:** the model itself infers in ~520 ms/chunk on an A100,
but every control cycle currently ships a **1.38 MB raw-image observation** over
a WebSocket to a remote SLURM node. On a healthy link that was ~0.7 s/chunk; on a
congested link it ballooned to 5–12 s/chunk. The per-request network round trip
(≥80 ms even for a buffered pop) makes a strict 15 Hz (66.7 ms/step) impossible
remotely. Running the server **on localhost** removes the WAN entirely: the obs
never leaves the box, so upload is ~free and the only cost is GPU compute.

This document describes the **existing pipeline** (server contract + the two
client scripts) precisely enough to reproduce it locally.

---

## 0. TL;DR of what must be reproduced locally

1. A **WebSocket policy server** that loads a MolmoBot checkpoint and, per
   request, returns an action **chunk** (`arm (8,7)`, `gripper (8,1)`).
2. The **exact observation contract** the two client scripts already send
   (camera keys, `qpos`, `task`, dtypes, image size).
3. The **chunk-response** behaviour (`chunk_response: True` in metadata) so the
   client-side strict-15 Hz executor works unchanged.
4. Ideally bind the server to `ws://127.0.0.1:8005` (or any local port); then
   the existing client scripts work with `--a100_host 127.0.0.1`.

If all four hold, **no client changes are needed** — just repoint the host.

---

## 1. Repository layout (client + server both live here)

```
sim2real-deploy/
├── sim_serve/                         # SERVER side (GPU) + shared client lib
│   ├── serve_bridge.py                # entrypoint: load ckpt, start WS server
│   ├── serve_bridge.slurm             # how it's launched on the A100 (ref only)
│   ├── bridge_policy.py               # BridgePolicy: chunk buffering + inference
│   ├── client_example.py              # CANONICAL client lib (MolmoBotClient)
│   └── CLIENT_GUIDE.md                # authoritative obs/action contract
├── client_example.py                  # byte-identical copy of the above (root)
├── real_serve/                        # CLIENT side (robot)
│   ├── run_molmobot_benchmarks_vineet.py   # OFFICIAL models (DROID etc.)
│   ├── run_molmobotft_benchmark_vineet.py  # FINETUNED models (v3/v4 pineapple)
│   └── test_dummy_molmobotft.py            # contract smoke test (no robot)
├── V3_GRIPPER_AND_IO_CONVENTIONS.md   # gripper/state I/O spec for FT ckpts
└── ROBOTIQ_KINEMATICS_NOTES.md        # 2F-85 four-bar angle<->width fit
```

The server needs the **MolmoBot model code** (the `olmo` package with
`olmo.models.molmobot.inference_wrapper.SynthManipMolmoInferenceWrapper` and
`olmo.eval.websocket_server.WebsocketPolicyServer`). That lives in Vineet's
`MolmoBot` repo, **not** in this repo — it must be installed/available on the
local GPU box (see §6).

---

## 2. The server: `serve_bridge.py` + `bridge_policy.py`

### 2.1 Entry point — `serve_bridge.py`
Loads a checkpoint and serves it over WebSocket. Key CLI args:

| arg | default | meaning |
|---|---|---|
| `--checkpoint_path` | (required) | path to a checkpoint dir, e.g. `.../v4full_twoview_step14500_epoch10` |
| `--port` | 8000 | WS port (we use **8005** for FT) |
| `--host` | 0.0.0.0 | bind address (**use 127.0.0.1 for local-only**) |
| `--camera_names` | `["exo_front"]` | ordered obs image keys; **two-view FT = `exo_front wrist`** |
| `--action_type` | `joint_pos` | **absolute** joint targets (NOT `joint_pos_rel`) |
| `--action_horizon` | 16 | actions predicted per inference |
| `--execute_horizon` | 8 | actions executed before re-query (= chunk size) |
| `--gripper_representation_count` | 1 | gripper state dims fed to the model |
| `--default_task` | "Pick up the pineapple slices can" | fallback instruction |

It builds a `BridgePolicyConfig`, constructs `BridgePolicy`, calls
`policy.prepare_model()` (loads weights onto the GPU — ~1–2 min for the 4B
checkpoint), then hands the policy to `WebsocketPolicyServer(...).serve_forever()`.

**Metadata** sent to the client on connect (the client keys off this):
```python
{
  "checkpoint_path": ...,
  "camera_names": ["exo_front", "wrist"],
  "action_type": "joint_pos",
  "action_horizon": 16,
  "execute_horizon": 8,
  "chunk_response": True,          # <-- server returns whole chunk per request
  "model_name": "molmobot-...",
  # "accepts_jpeg": True,          # <-- add this if JPEG-obs decode is wired
}
```

### 2.2 The policy — `bridge_policy.py` (`BridgePolicy`)
Duck-typed inference policy with `prepare_model / reset / get_action`.

- `prepare_model()` — instantiates
  `SynthManipMolmoInferenceWrapper(checkpoint_path, device="cuda",
  num_flow_steps=None, use_bfloat16=True)`.
- `reset()` — called on **each new WS connection** (= new episode); clears the
  action buffer. **One connection per episode.**
- `_populate_action_buffer(obs)` — the actual inference:
  1. `_extract_images(obs)` — pulls `obs[cam]` for each name in `camera_names`,
     in order. **This is where JPEG-obs decode must be added** (see §5).
  2. `_extract_state(obs)` — concatenates `qpos["arm"]` (7) + `qpos["gripper"]`
     (first `gripper_representation_count` dims) → state vector.
  3. `pred = self.agent.get_action_chunk(images, task, state)` →
     `(action_horizon, action_dim)` **un-normalized** actions.
  4. Splits each row into `{"arm": (7,), "gripper": (1,)}` and fills
     `self.action_buffer`.
- `get_action(obs)` — pops from the buffer; re-runs inference when
  `buffer_index >= execute_horizon`.

> **Chunk-response mode (already deployed):** Vineet modified the server so that
> instead of returning one action per request, it returns the whole
> `execute_horizon` chunk stacked: `{"arm": (8,7), "gripper": (8,1),
> "chunk_size": 8, "action_type": "joint_pos", "server_timing": {...}}`, and
> sets `chunk_response: True` in metadata. **Preserve this** — the client's
> strict-15 Hz executor depends on it. For an **absolute** (`joint_pos`) model
> this is safe because each action is a self-contained joint target; do NOT do
> this for a relative model.

### 2.3 Timing characteristics (measured, A100)
- First inference after connect: ~2–3 s (warmup/compile).
- Warm inference per chunk: **~520–525 ms server compute**.
- Chunk covers `8 × 66.7 ms = 533 ms` of execution.
- So locally: dispatch 8 actions at 15 Hz (533 ms), then ~525 ms inference gap →
  the inference is the only visible pause. If local inference < 533 ms you can
  even hide it with double-buffering later (not required now).

---

## 3. The observation / action contract (authoritative)

Encoded with **msgpack-numpy** over WebSocket. See `sim_serve/CLIENT_GUIDE.md`
for the full spec; summary:

### Observation (client → server), per inference request
| key | type | notes |
|---|---|---|
| `exo_front` | `np.ndarray (480,480,3)` uint8 **RGB** | first camera |
| `wrist` | `np.ndarray (480,480,3)` uint8 **RGB** | second camera (two-view FT only) |
| `qpos["arm"]` | `np.ndarray (7,)` float32 | joint positions, **radians** |
| `qpos["gripper"]` | `np.ndarray (>=1,)` float32 | gripper state (see §4) |
| `task` | `str` | language instruction |

- Camera keys **must match** the server's `--camera_names` order.
- Images are **square 480×480 RGB uint8**; the model's Molmo preprocessor does
  its own resize to the ViT size internally. Do NOT letterbox/pad to 224.

### Action (server → client)
Chunk-response mode:
```python
{"arm": (8,7) float32,        # ABSOLUTE joint targets (radians)
 "gripper": (8,1) float32,    # gripper target, per convention below
 "chunk_size": 8,
 "action_type": "joint_pos",
 "server_timing": {"total_ms": int, "prev_total_ms": float}}
```

---

## 4. Gripper & state I/O conventions (critical, easy to get wrong)

Full detail in `V3_GRIPPER_AND_IO_CONVENTIONS.md` and
`ROBOTIQ_KINEMATICS_NOTES.md`. The model side is unchanged by the local move,
but the local-setup agent should know these so state encoding stays consistent:

- **State gripper** (`qpos["gripper"][0]`) fed to the model is a **Robotiq 2F-85
  finger-joint angle in radians**, ~0 open → ~0.8 closed. The client computes it
  from the measured jaw width via the **true four-bar kinematics**
  (`robotiq_state_kine`), NOT a linear map. This was the fix that made grasped
  objects actually lift (linear under-reported by ~0.11 rad in the hold range).
- **Action gripper** returned by the model is a **normalized 0..1 scalar**
  (0 = open, 1 = closed), decoded client-side with threshold 0.5.
- **Arm**: absolute joint positions in radians, both directions.

These are applied entirely **client-side**; the server just passes the model's
raw outputs. Moving the server local does not change any of this.

---

## 5. Optional: JPEG-obs (largely moot once local, but implemented)

To fight the remote-link bandwidth we added `--jpeg_obs` on the client:
`sim_serve/client_example.py:encode_jpeg_obs()` replaces each HxWx3 uint8 image
with `{"__jpeg__": True, "data": <jpeg bytes>, "shape": [H,W,3]}` (~23× smaller,
1.38 MB → ~59 KB). **Once the server is on localhost, upload is ~free, so JPEG
is unnecessary** — but if you keep it, the server must decode it. Minimal server
addition to `_extract_images`:

```python
import cv2, numpy as np
def _decode_camera(v):
    if isinstance(v, dict) and v.get("__jpeg__"):
        buf = np.frombuffer(v["data"], dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)   # channel order preserved: RGB->RGB
    return np.asarray(v)
# images.append(_decode_camera(obs[cam]))
```
and advertise `accepts_jpeg: True` in metadata. For a local server, **skip JPEG**
and keep raw obs (simplest, lossless).

---

## 6. The two client scripts (what connects to the server)

Both live in `real_serve/`, share the same helpers, and build the **same obs
contract**. The only difference is which checkpoint/endpoint and obs keys.

### 6.1 OFFICIAL models — `run_molmobot_benchmarks_vineet.py`
- For the **permanent/official** MolmoBot checkpoints (e.g. MolmoBot-DROID).
- Endpoint presets in `MODEL_ENDPOINTS`, e.g.
  `molmobot_droid_official -> ("10.176.215.144", 8000)`; some are dual-camera
  (`DUAL_CAMERA_MODELS`).
- Dual-camera obs keys: **`exo_camera_1`** + **`wrist_camera`** (note: DIFFERENT
  key names than the FT script — the official DROID server expects these).
- Selected via `--model <preset>`.

### 6.2 FINETUNED models — `run_molmobotft_benchmark_vineet.py`  ← primary for local work
- For our finetuned pineapple-pick checkpoints (v3/v4 two-view).
- Endpoint: `--a100_host` / `$MOLMOBOTFT_A100_HOST` + `--a100_port` /
  `$MOLMOBOTFT_A100_PORT` (default port 8005). **This is the one to repoint to
  `127.0.0.1` for local.**
- Dual-camera obs keys: **`exo_front`** + **`wrist`** (matches the FT server's
  `--camera_names exo_front wrist`).
- Has the full deploy feature set:
  - `--gripper {robotiq,panda}` with `robotiq_state_kine` default encoding.
  - `--front_view_cam_id {exo_front,exo_left,exo_right}` camera-view selection.
  - `--chunk_executor {auto,on,off}` — **strict 66.7 ms dispatch** using
    `client.get_action_chunk()`; `auto` engages whenever the server advertises
    `chunk_response: True`.
  - `--step_confirm`, `--manual_gripper_override`, `--crop_bottom_5`,
    `--flip_cams`, `--jpeg_obs`, `--dry_run`.

### 6.3 Control loop shape (FT script, chunk-executor path)
```
connect  →  read metadata (chunk_response?, camera_names, horizons)
loop:
   capture obs (arm+gripper state, 2 ZED frames -> 480² RGB)
   chunk = client.get_action_chunk(obs)     # ONE request -> 8 actions  [BLOCKING inference]
   for k in 0..7:                            # dispatch on a STRICT monotonic 66.7 ms clock
       safety_clip + send absolute arm target to franky_service
       gripper vote/debounce -> Robotiq grasp/open on edges
       log step (incl. dispatch jitter, chunk id)
       sleep to next 66.7 ms deadline
   # (inference for the next chunk happens after this, sequentially)
```

### 6.4 Contract smoke test — `test_dummy_molmobotft.py`
No robot/cameras needed. Connects, sends random obs, validates the response
shape (accepts both legacy `(7,)` and chunk `(K,7)`), prints server metadata and
timing. **Run this first against the new local server** to confirm the contract.

---

## 7. Concrete steps to bring the server up locally

1. **Install the MolmoBot model code + deps** on the local GPU box (the `olmo`
   package providing `SynthManipMolmoInferenceWrapper` and
   `WebsocketPolicyServer`, plus torch/CUDA matching the local GPU). Use Vineet's
   `MolmoBot` repo + its conda env (`molmobot`).
2. **Copy a checkpoint locally** (e.g. `v4full_twoview_step14500_epoch10/`) — the
   dir the A100 served via `--checkpoint_path`. Check it fits in local VRAM
   (4B model in bf16 ≈ 8–10 GB weights + activations; confirm the local GPU has
   enough — the A100 deployment used an 80 GB card but the model itself is far
   smaller).
3. **Launch the server bound to localhost:**
   ```bash
   cd /path/to/MolmoBot/MolmoBot            # so `olmo` is importable
   PYTHONPATH=. python /path/to/sim2real-deploy/sim_serve/serve_bridge.py \
       --checkpoint_path /local/ckpts/v4full_twoview_step14500_epoch10 \
       --host 127.0.0.1 --port 8005 \
       --camera_names exo_front wrist \
       --action_type joint_pos \
       --action_horizon 16 --execute_horizon 8 \
       --gripper_representation_count 1
   ```
   Confirm it prints `chunk_response`-capable metadata and `curl
   http://127.0.0.1:8005/healthz` → OK.
4. **Smoke test the contract:**
   ```bash
   cd /path/to/sim2real-deploy
   MOLMOBOTFT_A100_HOST=127.0.0.1 MOLMOBOTFT_A100_PORT=8005 \
     python real_serve/test_dummy_molmobotft.py --steps 12
   ```
   Expect `CHUNK protocol: arm.shape=(8,7)` and warm latency now dominated by
   **local GPU compute only** (no multi-second network gaps).
5. **Run a rollout** (kill `camera_service` first so ZEDs are free):
   ```bash
   python real_serve/run_molmobotft_benchmark_vineet.py \
       --a100_host 127.0.0.1 --a100_port 8005 \
       --gripper robotiq --front_view_cam_id exo_front \
       --chunk_executor auto \
       --log_dir logs/exp_localgpu_test1
   ```
6. **Verify 15 Hz** from the log:
   ```bash
   python - <<'PY'
   import json, numpy as np
   r=[json.loads(l) for l in open("logs/exp_localgpu_test1/steps.jsonl")]
   j=np.array([x["chunk"]["dispatch_jitter_ms"] for x in r if x["chunk"].get("mode")=="client_executor"])
   print(f"dispatch jitter median={np.median(j):.1f}ms p99={np.percentile(j,99):.1f}ms n={len(j)}")
   PY
   ```
   Median dispatch interval should be ≈66.7 ms with small jitter; the only gaps
   are the ~inference-length pauses between chunks (which shrink to local GPU
   compute time).

---

## 8. Acceptance criteria (what "done" looks like)

- `test_dummy_molmobotft.py` connects to `127.0.0.1:8005`, gets `(8,7)` chunks,
  warm per-chunk wall ≈ local GPU compute (no WAN).
- Real rollout: **median dispatch interval ≈ 66.7 ms**, low p99 jitter, actions
  land on the 15 Hz grid within each chunk.
- Between-chunk pause = local inference time only (target: ≤ chunk duration so it
  can later be hidden by double-buffering).
- Gripper still lifts objects (i.e. `robotiq_state_kine` encoding preserved on
  the client; nothing about the local move changes it).

---

## 9. Gotchas / notes for the setup agent

- **Obs key names differ between scripts**: FT uses `exo_front`/`wrist`; official
  uses `exo_camera_1`/`wrist_camera`. The server's `--camera_names` must match
  whichever client you run. For local FT work, use `exo_front wrist`.
- **One WS connection per episode** — the server `reset()`s (clears buffers) on
  connect; the client already keeps one connection for a whole rollout.
- **Absolute actions**: `action_type=joint_pos`. The chunk-response local-play is
  only valid for absolute targets; keep it that way.
- **First inference is slow** (warmup/compile) — the client already tolerates a
  long first call; don't mistake it for a hang.
- **VRAM**: confirm the local GPU fits the 4B checkpoint in bf16. If not, that's
  the blocker to raise before anything else.
- **No robot needed to validate the server** — use `test_dummy_molmobotft.py`;
  bring in the ZEDs + franky_service only for the full rollout.
- Keep `sim_serve/client_example.py` and root `client_example.py` **in sync**
  (they're byte-identical today; edit one, copy to the other).
```
