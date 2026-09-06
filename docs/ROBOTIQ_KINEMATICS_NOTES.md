# Robotiq 2F-85 true kinematics — deploy-side encoding

## Why this exists

Old deploy code (`--gripper_input robotiq_state`) encoded the gripper state as
a **linear** proxy:

```
θ_linear = (1 - width_m / 0.08) * 0.87       # rad
```

This is not what the sim training data contained. The sim exports the raw
2F-85 knuckle-joint qpos from MuJoCo, which follows the actual 2F-85
parallel-jaw four-bar kinematics. On the same physical gap, linear and true
differ by up to ~0.06 rad — enough to shift where in the trained
open/close-ramp/hold distribution the deploy observations land.

`--gripper_input robotiq_state_kine` (now the default for `--gripper robotiq`)
uses the closed-form parallel-jaw kinematics derived from the Robotiq URDF.

## Derivation

From `robotiq_description/urdf/robotiq_2f_85_macro.urdf.xacro`:

| link chain (LEFT side) | translation in parent frame (m) |
|---|---|
| base → knuckle joint       | `(0.03060114, 0, 0.05490452)`, revolute about `-y`, limit `[0, 0.8]` |
| knuckle → finger link      | `(0.03152616, 0, -0.00376347)`, fixed |
| finger → finger_tip link   | `(0.00563134, 0, 0.04718515)`, `finger_tip_joint` mimics knuckle with multiplier `-1` |

Because the fingertip joint mimics the knuckle joint with a negative
multiplier, the pad face stays parallel to the base throughout the motion —
the tip translates but does not rotate relative to the base. This is the
"parallel jaw" property of the 2F-85 four-bar linkage.

Define the rigid vector from the knuckle joint origin to the pad face on the
fingertip:

```
KN2TIP = knuckle→finger + finger→finger_tip
       = (0.0371575, 0, 0.04342168)

pad_offset_x = (0.085 / 2) - (KNUCKLE_X + KN2TIP.x)      # back-solved so w(0)=0.085
             = -0.025259
```

Rotating `KN2TIP` about `-y` by knuckle angle θ:

```
x_after = cosθ * KN2TIP.x - sinθ * KN2TIP.z
```

Full gap width is `2 * pad_face_x_in_base_frame`:

```
w(θ) = 2 * (KNUCKLE_X + x_after + pad_offset_x)
     = A + B*cosθ - C*sinθ
     = A + R*cos(θ + φ)                        # trig identity
```

with

```
A = 2*(KNUCKLE_X + pad_offset_x) = 0.010685
B = 2*KN2TIP.x                   = 0.074315
C = 2*KN2TIP.z                   = 0.086843
R = √(B² + C²)                   = 0.114300
φ = atan2(C, B)                  = 0.8630 rad
```

Inversion is closed-form:

```
θ(w) = arccos((w - A) / R) - φ
```

Endpoints check exactly to numerical precision:
`w(0)=85.0 mm`, `w(0.8)=0.16 mm` (URDF's own 0.16 mm residual at the
closed-position joint limit).

## Numerical table (linear vs true)

| gap w  | linear θ (old default) | true θ (new default) | Δ = new − old |
|--------|------------------------|-----------------------|---------------|
| 80 mm  | 0.000                  | 0.056                 | +0.056        |
| 70 mm  | 0.109                  | 0.162                 | +0.053        |
| 60 mm  | 0.217                  | 0.262                 | +0.044        |
| 57 mm  | 0.250                  | 0.291                 | +0.040        |
| 50 mm  | 0.326                  | 0.357                 | +0.030        |
| 40 mm  | 0.435                  | 0.448                 | +0.013        |
| 30 mm  | 0.544                  | 0.538                 | −0.006        |
| 20 mm  | 0.652                  | 0.626                 | −0.026        |
| 10 mm  | 0.761                  | 0.714                 | −0.047        |
|  0 mm  | 0.870                  | 0.801                 | −0.069        |

## Interaction with the "sim over-closure" bug

Sim training median HOLD angle is **~0.55 rad** because MuJoCo soft contact
lets the fingers over-close into the object. On the real robot the fingers
stop at the geometric surface — at ~57 mm gap for the pineapple can — so
they can only ever report **θ ≈ 0.29 rad** (was 0.25 under linear). The
kinematics fix removes the convention-mismatch component but does **not**
close the gap to the sim HOLD band (0.47–0.69 rad). Expect this fix alone to
help slightly with grasp-timing sensitivity but not to unstick the LIFT
phase; that requires either retraining with realistic contact or a deploy
saturation hack that reports a HOLD-band value once a stable grasp is
detected.

## How to run

New default for `--gripper robotiq` (no flag needed):

```bash
python real_serve/run_molmobotft_benchmark_vineet.py --gripper robotiq
# equivalent to  --gripper_input robotiq_state_kine
```

Revert to the old linear encoding (A/B baseline):

```bash
python real_serve/run_molmobotft_benchmark_vineet.py --gripper robotiq \
    --gripper_input robotiq_state
```

The encoded scalar is logged per-step in `steps.jsonl` under
`obs.gripper_state_scalar`, so you can diff between runs without re-deriving.
