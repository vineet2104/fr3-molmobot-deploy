# V3 training I/O conventions — deployment alignment guide

**Purpose:** exact input/output conventions (shapes, ranges, normalization) used
by the **V3 two-view** MolmoBot run (`checkpoints/molmobot_v3_twoview`,
`action_preset=franka_joint`), so the robot client can align its obs encoding
and action decoding to what the model actually saw in training.

Covers **all four modalities**: RGB images (§7), language/task (§8), robot state
(§1–4), and actions (§1–5). All facts below are read directly from:
- V3 H5 data: `exports/v3_molmobot_partial/train/house_*/trajectories_batch_1_of_1.h5`
- The training dataset code: `MolmoBot/olmo/data/synthmanip_dataset.py`, `synthmanip_presets.py`, `synthmanip_config.py`
- **The checkpoint config (authoritative norm stats): `checkpoints/molmobot_v3_twoview/step1205/config.yaml`** (`robot_preprocessor`/`robot_postprocessor`). `norm_stats.yaml` is a training-time input only — **not read at serve time** (see §4.3).
- The image preprocessing: `MolmoBot/olmo/preprocessing/image_preprocessor.py`
- The normalization math: `MolmoBot/olmo/data/robot_processing.py`
- The serving path: `MolmoBot/olmo/models/molmobot/inference_wrapper.py`

### What the server does vs. what the client must supply
The serving wrapper (`inference_wrapper.py`) runs the **full training preprocessor**
on your raw inputs: it resizes+normalizes the images, templates the task string,
normalizes the state, runs the model, and **un-normalizes the action**. So the
client must supply **raw, un-normalized** inputs in the **correct units/format**
(uint8 RGB, plain task text, rad state) and will receive **physical-unit actions
back**. Do not pre-normalize on the client. The rest of this doc specifies those
raw formats exactly.

---

## TL;DR — what the client must send / decode

| modality | client sends (RAW) | server does | §  |
|---|---|---|---|
| **images** | 2× **uint8 RGB** HWC, **`exo_front` then `wrist`**, ~square | resize→378², SigLIP-norm `(x/255)*2−1` | §7 |
| **task** | plain text, e.g. `"Pick up the pineapple slices can"` | templates via `style="demo"` | §8 |
| **state** | `[arm(7) rad, gripper(1) rad]` = **dim 8**, raw units | min_max → [-1,1] | §3.1, §4.1 |
| **action** (returned) | — | model [-1,1] → **un-norm quantiles** → physical units | §3.2, §4.2 |

Action returned = `[arm(7) rad absolute, gripper(1) 0..1]`. Two things the client
still owns: the **gripper decode** (§3.2, `normalized` not `raw`) and the **arm
joint-limit clamp**.

**Most important gotcha:** the gripper **state** and gripper **action** are in
**different spaces** (rad angle vs 0..1 scalar). See §3.

> **The `step1205` deployment bug:** the client decoded the action gripper with
> `--gripper_output raw` (0..255). Training action gripper is a **normalized 0..1
> scalar**. So a model output of `1.0` ("close") decoded to `(1−1/255)*0.08 ≈ 80 mm`
> ≈ fully open, and the hand never closed. Correct decode is the **normalized**
> convention. See §3.2.

---

## 1. What the model consumes and produces

`franka_joint` preset (`synthmanip_presets.py`):
```
ACTION_SPECS["franka_joint"]      = {"arm": 7, "gripper": 1}   # action  dim = 8
ACTION_DATASET_KEYS["franka_joint"] = "joint_pos"              # actions read from actions/joint_pos
gripper_representation_count       = 1                          # state gripper truncated to 1 value
# state spec falls back to action spec -> {"arm":7,"gripper":1} = state dim 8
```
- **State vector** (fed into obs): concat in move-group order `arm, gripper` →
  **`[q0..q6, g_state]`**, dim **8**.
- **Action vector** (predicted, per timestep, horizon=16): concat `arm, gripper` →
  **`[a0..a6, g_cmd]`**, dim **8**. Absolute joint targets (not deltas — that
  would be `franka_jointdelta`/`joint_pos_rel`).

---

## 2. Raw fields in the H5 (before the dataset touches them)

All the "action/qpos" H5 datasets are **JSON dicts serialized to uint8 bytes**
(shape `(T,1024)` etc.); decode with
`json.loads(bytes(row.astype(uint8)).decode().rstrip("\x00"))`.

### 2.1 `obs.agent.qpos` (the STATE source)
```
keys: {arm: 7, base: 0, gripper: 6}
arm     : 7 joint angles (rad), absolute
gripper : 6 values = Robotiq 2F-85 finger-joint angles (rad), SIGNED
          (2 fingers × 3 joints). At open pose all ≈ 0; closing swings to ±0.8.
          per-column ranges (one traj): c0[-0.02,0.74] c1[-0.95,0.02] c2[-0.00,0.88]
                                         c3[-0.01,0.74] c4[-0.94,0.08] c5[-0.01,0.81]
```
The dataset keeps **only `gripper[:1]`** (index 0) because
`gripper_representation_count=1`. So the **state gripper = `qpos.gripper[0]`**, a
single Robotiq finger-joint angle in radians (~0.0 open → ~0.7–0.9 closed).

### 2.2 `actions.joint_pos` (the ACTION source for `franka_joint`)
```
keys: {arm: 7, gripper: 1}
arm     : 7 target joint angles (rad), absolute
gripper : 1 scalar in [0.0, 1.0].  0.0 = fully OPEN, 1.0 = fully CLOSED.
          Distribution across V3: ~35% steps exactly 0.0, ~54% exactly 1.0,
          ~11% on the short 0→1 ramp during the close (0.133,0.267,…,0.933).
          Effectively a binary open→close command with a brief interpolation.
```

### 2.3 Other action keys present (NOT used by `franka_joint`, for reference)
- `commanded_action`: `{arm:7, base:0, gripper:1}`, gripper 0..1 (same as joint_pos).
- `joint_pos_rel`: `{arm:7}` — **delta** joints, no gripper (used by `franka_jointdelta`).
- `ee_pose`: `{arm:7, gripper:1}` — arm = **3 pos + 4 quat** (last-4 norm = 1.0).
- `ee_twist`: `{arm:6, gripper:6}` — velocities.

> ⚠️ The gripper is `0..1` in `joint_pos`/`commanded_action`/`ee_pose`, but the
> gripper *state* in `qpos` is a **radian angle**. Do not conflate them.

---

## 3. The gripper convention in detail (the deployment-critical part)

### 3.1 STATE side — what to feed as `qpos.gripper`
- Training feeds **one scalar**: `qpos.gripper[0]` = a **Robotiq finger-joint
  angle in radians**, ~`0.0` when open and ~`0.7–0.9` when closed.
- Checkpoint state stat (index 7, min_max): **min = -0.13128, max = 0.92728**.
- **Deploy:** encode your real gripper openness as a radian angle on this scale
  (open ≈ 0, closed ≈ +0.87). If your hardware reports width in meters or a
  0..255 bit, **convert to this radian scale first**, otherwise the state token
  is out-of-distribution. A reasonable mapping if you only have a 0..1 openness
  `c` (0=open,1=closed): `g_state ≈ c * 0.87`.

### 3.2 ACTION side — how to decode the predicted gripper
- Model predicts a **normalized 0..1 scalar**: **0 = open, 1 = closed**.
- Checkpoint action stat (index 7, quantiles): **q01 = 0.0, q99 = 1.0** (identity
  range → un-normalization returns the same 0..1 value; see §4).
- **Correct client decode:** treat output as the **`normalized`** convention:
  ```
  width_m   = (1.0 - clip(g, 0, 1)) * 0.08     # 0→80mm open, 1→0mm closed
  close_cmd = g > 0.5                            # binary threshold at 0.5
  ```
- **WRONG (the step1205 bug):** `--gripper_output raw` → `width=(1−g/255)*0.08`,
  which maps g=1.0 to ≈80 mm (open). Never fires the close vote.
- **Fix for the client:** `--gripper_output normalized --binary_threshold 0.5`.

---

## 4. Normalization — exact formulas the model expects

Modes (from `checkpoint/config.yaml → robot_preprocessor/robot_postprocessor`,
math in `robot_processing.py`):
```
action_norm_mode = "quantiles"   (uses q01 / q99)
state_norm_mode  = "min_max"     (uses min / max)
```
**Source of truth = the checkpoint's `config.yaml`, not `norm_stats.yaml` — see §4.3.**

### 4.1 State (client → model input): **min_max to [-1, 1]**
For each of the 8 dims:
```
x_norm = 2.0 * (x - min) / max(max - min, 1e-6) - 1.0
```
`min`/`max` = `config.yaml → robot_preprocessor.stats_by_repo.synthmanip.observation.state.{min,max}`:
```
min = [-2.86769, -2.24775, -2.86100, -3.35337, -3.14299, -0.29311, -2.89730, -0.13128]
max = [ 3.03077,  1.77282,  2.89208,  0.14301,  3.09627,  4.13344,  2.89643,  0.92728]
       └───────────────────── 7 arm joints (rad) ─────────────────────┘  └ gripper ┘
```
The serving wrapper calls `normalize_state()` for you **iff** the checkpoint
carries a `robot_preprocessor` — but the client must still provide the **raw**
8-dim state in the correct units (rad arm, rad gripper). Do **not** pre-normalize
on the client.

### 4.2 Action (model output → robot): **un-normalize with quantiles**
The model emits values in ~[-1, 1]; the serving wrapper
(`inference_wrapper._unnormalize_action`) maps them back with:
```
x = (x_norm + 1.0) * (q99 - q01) / 2.0 + q01
```
`q01`/`q99` from `checkpoint/config.yaml → robot_postprocessor.stats_by_repo.synthmanip.action`
(**these are the served values — see §4.3**):
```
q01 = [-1.58988, -1.63471, -1.59693, -3.04338, -1.22467,  1.18898, -1.95098, 0.0]
q99 = [ 1.29644,  0.90400,  1.84377, -1.27786,  1.77361,  3.18059,  1.97851, 1.0]
       └──────────────────── 7 arm joints (rad) ────────────────────┘  └gripper┘
```
So the **returned action is already in physical units**: arm = absolute joint
angles (rad), gripper = **0..1 scalar** (q01=0, q99=1 → identity). The client
receives arm-rad + gripper-0..1 and only needs the §3.2 gripper decode + the
arm joint-limit clamp.

### 4.3 ⚠️ Where the served stats actually come from (do NOT trust `norm_stats.yaml`)
Serving does **not** read `norm_stats.yaml` at all. The flow is:
1. `norm_stats.yaml` is a **training-time input** — the trainer reads it once and
   **embeds the numbers into the checkpoint's `config.yaml`** (`robot_preprocessor`
   for state, `robot_postprocessor` for action) at save time.
2. At serve time, `SynthManipMolmoInferenceWrapper` loads **`config.yaml`** from
   the checkpoint dir (`inference_wrapper.py:101`) and calls
   `build_preprocessor()` / `build_postprocessor()` off `model_config`
   (line 168). **No YAML stats file is read.** There is no `norm_stats.yaml`
   inside the frozen checkpoint dir — only in the training folder.
3. → The stats are **intrinsically tied to the checkpoint**; you cannot serve
   with a stale/default stats file because the serving path never opens one.

**Consequence:** the authoritative numbers are in
`checkpoints/molmobot_v3_twoview/step1205/config.yaml` under `robot_postprocessor`
/ `robot_preprocessor`, **not** `norm_stats.yaml`. In this checkpoint the state
min/max match between the two files, but the **action q01/q99 differ at ~1e-2**
(the training YAML was evidently recomputed/updated after the value was baked in).
The §4.1/§4.2 numbers above are taken from the **config.yaml** (the served ones).
To verify a running server, check its startup log — it prints the loaded stats:
```
action_norm_mode: quantiles
repos with stats: ['synthmanip']
  synthmanip/observation.state/min: len=8
  synthmanip/observation.state/max: len=8
```

---

## 5. Worked examples

### 5.1 Building the state you send (open hand at a mid-reach pose)
```python
arm_rad = [0.0178, -0.3957, 0.0082, -1.9167, -0.0185, 1.4912, 0.0067]  # 7 joints
g_state = 0.0        # radian gripper angle, ~0 = open  (closed ≈ 0.87)
state = arm_rad + [g_state]        # dim 8, raw units — server normalizes it
# server does: x_norm = 2*(state-min)/(max-min)-1  (min_max, §4.1)
```

### 5.2 Decoding an action the model returns
```python
# server already un-normalized -> arm in rad, gripper in 0..1
a = server_action            # dim 8
arm_targets_rad = a[:7]      # absolute joint targets -> clamp to joint_limits
g = a[7]                     # 0..1, 0=open 1=closed
width_m   = (1.0 - min(max(g,0.0),1.0)) * 0.08
close_cmd = g > 0.5
# g=1.0 -> width 0mm (closed);  g=0.0 -> width 80mm (open)
```

### 5.3 A real V3 grasp trajectory (action gripper over time, traj_0)
```
step:    0 ....  21   22    23    24    25    26    27    28    29  30..62
gripper: 0 ....  0.0  0.133 0.267 0.400 0.533 0.667 0.800 0.933 1.0 1.0 (held closed)
                 └ open during approach ┘└──── 0→1 close ramp ────┘└ closed ┘
```
The model keeps `g≈0` while approaching, ramps `0→1` over ~7 steps to grasp,
then holds `1.0`. With the §3.2 decode, `g>0.5` first trips around step 26–27 —
that is when the physical gripper should close.

---

## 7. RGB image inputs

### 7.1 Views, order, count
- V3 is **two-view**: `max_images=2`, `single_frame=true`, `max_frames=1`
  (one timestep, no frame history). So the client sends **exactly 2 images per
  step**: **`exo_front` first, then `wrist`** (this order matters — it matches the
  `--camera_names exo_front wrist` training order).
- Source frames in training are **480×480 RGB**; deploy frames are also 480×480
  (the runs used 480×480). Any resolution is accepted — the preprocessor resizes
  (§7.3) — but keep aspect ratio close to square to avoid distortion.

### 7.2 Pixel format the client sends
- **`uint8`, RGB, HWC, range 0–255.** The serving wrapper (`_prepare_images`)
  accepts `np.ndarray` HxWx3; if you pass float in [0,1] it will `*255`, and if
  float >1 it just casts — so **send uint8 RGB to be safe.** No normalization,
  no BGR. (If you read frames with OpenCV, convert BGR→RGB first.)

### 7.3 What the server does to each image (SigLIP2 pipeline)
From `config.yaml` (`vit.image_model_type=siglip`, `resize_mode=siglip`,
`normalize=siglip`, `image_default_input_size=[378,378]`, `patch=14`,
`image.crop_mode=resize`, `max_crops=1`, `single_frame=true`) and
`image_preprocessor.py`:
1. **Resize** the whole image to **378×378** with **bilinear** interpolation.
   Because `crop_mode=resize` (not the multi-crop tiling mode), the image is a
   **plain stretch-resize to 378²** — a single crop, no aspect-padding, no tiling.
2. **Normalize (SigLIP convention):** `x_norm = (uint8 / 255.0) * 2 − 1` →
   range **[-1, 1]**. (This is the SigLIP mean/std=0.5 convention, i.e.
   `(x−0.5)/0.5`, done as `*2−1`.) **Not** the OpenAI-CLIP mean/std.
3. Split into 14×14 patches → 27×27 = **729 patch tokens per image**, 1152-dim.
4. With 2 images + pooling, both views' tokens are concatenated into the LLM
   context (this is why `seq_len=528` covers 2 images at 1 frame).

**The client does NOT do any of this** — send raw uint8 RGB and the server handles
resize+normalize. The only client responsibility is: correct **view order**,
**RGB (not BGR)**, and **uint8**.

### 7.4 Appearance-alignment note (sim vs real)
The frozen SigLIP2 encoder was trained on the sim look (see the sim exo/wrist
frames). Large real-world appearance shifts — heavy specular glare, very dark
tabletop, different lighting — are a genuine domain gap for the encoder. This is
not a format issue (nothing to "fix" in the client), but worth minimizing
physically (diffuse lighting, less glare) to stay closer to the training
distribution.

---

## 8. Language / task instruction input

### 8.1 What the client sends
- A **single plain-text task string**, passed as `task_description` (becomes the
  `question` field with `style="demo"`). No special tokens, no chat wrapper, no
  system prompt needed on the client — the preprocessor templates it.

### 8.2 What the model was trained on (the actual text distribution)
The V3 run used **`--randomize_prompts`**, so the model did **not** see a single
fixed string. Per step, the dataset (`_sample_randomized_prompt`) composed:
```
prompt = random_template.format(pickup_obj_name = sampled_referral_expression)
```
- **task_type** = `pick`; **canonical `task_description`** =
  `"Pick up the pineapple slices can"`.
- **Templates** (`DEFAULT_PROMPT_TEMPLATES["pick"]`): `"Pick up the {x}."`,
  `"Lift the {x}."`, `"Pick the {x}."`, `"Grab the {x}."`, `"Get the {x}."`
- **Referral expressions** for `{x}` (sampled, length-biased,
  temperature=4.0, prob_threshold=0.1): `"pineapple slices can"` variants —
  in this dataset the referral list stores full phrases
  (`"Pick up the pineapple slices can"`, `"Grasp …"`, `"Lift …"`,
  `"… from the table"`).
- Plus **casing jitter** (50% lowercased) and **punctuation jitter**
  (50% trailing period removed).

So the model saw many surface forms: `"Pick up the pineapple slices can."`,
`"grab the pineapple slices can"`, `"lift the pineapple slices can from the table"`,
etc. (the referral phrasing can nest inside the template, producing some
redundant compositions). **Practical implication: the exact wording is NOT
critical** — the policy is robust to reasonable paraphrases and casing.

### 8.3 Recommended deploy string
Any of these are in-distribution and safe:
- `"Pick up the pineapple slices can"`  ← closest to the canonical `task_description`
- `"pick up the pineapple slices can"`
- `"Grab the pineapple slices can"`

The `step1205` runs used `"pick up the can of pineapple slices"` (re-ordered
phrasing). That's a mild paraphrase — likely fine given prompt randomization, but
for tightest alignment prefer **"Pick up the pineapple slices can"** (the exact
noun-phrase order the model saw). This was **not** a cause of the failures (the
gripper decode in §3.2 was), but it's a free alignment win.

---

## 9. Deployment checklist

1. **State units:** send `[7 arm rad, 1 gripper rad]` (gripper 0≈open, ~0.87≈closed).
   Convert hardware width/bit → radian before sending. Do not pre-normalize.
2. **Action gripper decode:** `normalized` convention, threshold 0.5
   (`--gripper_output normalized --binary_threshold 0.5`). **Not `raw`.**
3. **Action arm:** absolute joint targets in rad; clamp to `joint_limits`.
4. **Camera obs:** **`exo_front` first, then `wrist`**; **uint8 RGB** (not BGR),
   HWC, 0–255; ~square (480×480 fine). Server resizes to 378² + SigLIP-normalizes.
   2 images/step, 1 frame (no history).
5. **Task string:** plain text, prefer `"Pick up the pineapple slices can"`.
6. **Do not pre-normalize** images/state/action on the client — the server runs
   the training preprocessor and returns physical-unit actions.
7. Stats source = the **checkpoint's `config.yaml`** (`robot_preprocessor` /
   `robot_postprocessor`), **not** `norm_stats.yaml` (§4.3). To verify a running
   server, read its startup log (it prints the loaded state min/max len + norm
   modes). The §4 numbers are from `molmobot_v3_twoview/step1205/config.yaml`.
```
