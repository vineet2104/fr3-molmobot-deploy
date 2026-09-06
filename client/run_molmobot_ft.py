"""Closed-loop MolmoBot-FT benchmark rollouts (Vineet).

Fork of run_molmobot_benchmarks_vineet.py that targets a *single* finetuned
MolmoBot checkpoint (our own dataset). The observation contract is the
same shape as the other MolmoBot policies but with different key names:

  Observation (per step):
      exo_front   : (H, W, 3) uint8 RGB     ← exo camera, FIRST
      wrist       : (H, W, 3) uint8 RGB     ← wrist camera, SECOND
      qpos        : {"arm": (7,) float32 rad,
                     "gripper": (>=1,) float32}
      task        : str (optional)

  Action (returned per step):
      arm     : (7,) float32   ABSOLUTE joint targets, radians
      gripper : (1,) float32

One WebSocket connection per episode (server calls .reset() on connect).

Default endpoint (override with $MOLMOBOTFT_A100_HOST / $MOLMOBOTFT_A100_PORT
or --a100_host / --a100_port):

    ws://10.49.163.26:8005      preset --model molmobot_ft

All frames (exo + wrist) and predicted actions are logged as before:
  <log_dir>/frames/0000.jpg          exo view
  <log_dir>/frames_wrist/0000.jpg    wrist view
  <log_dir>/steps.jsonl              obs, raw action, sent action, timings

Everything else (gripper wiring, ZED handling, --flip_cams, --exo/--wrist,
--dry_run, --step_confirm) is identical to the other benchmark script.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim_serve"))
from client_example import MolmoBotClient  # noqa: E402

# ---------- Config ----------
FRANKY_URL = os.environ.get("FRANKY_URL", "http://127.0.0.1:54321")
HAND_URL = os.environ.get("HAND_URL", "http://127.0.0.1:54324")          # Franka Hand (panda preset)
ROBOTIQ_URL = os.environ.get("ROBOTIQ_URL", "http://127.0.0.1:54323")          # Robotiq 2F-85 service on this box
ROBOTIQ_STROKE_M = 0.085                        # 2F-85 max opening (mm/1000)
# HTTP timeouts (seconds). franky_service serialises libfranka reads under a
# single mutex, so under contention (e.g. another client polling, or model
# chunk-refresh spikes) response time can exceed 500 ms. Values below are the
# safe defaults; override via env if you know what you're doing.
FRANKY_READ_TIMEOUT_S  = float(os.environ.get("FRANKY_READ_TIMEOUT_S",  "2.0"))
FRANKY_WRITE_TIMEOUT_S = float(os.environ.get("FRANKY_WRITE_TIMEOUT_S", "2.0"))
GRIPPER_READ_TIMEOUT_S = float(os.environ.get("GRIPPER_READ_TIMEOUT_S", "1.0"))
GRIPPER_WRITE_TIMEOUT_S = float(os.environ.get("GRIPPER_WRITE_TIMEOUT_S", "1.0"))
ZED_CAMERA_FPS = 30
ZED_RESOLUTION = sl.RESOLUTION.HD720         # 1280 x 720
POLICY_IMG_SIZE = 480                         # 480x480 square per CLIENT_GUIDE
# Known ZED cameras on this rig, keyed by short human name for --exo/--wrist
# CLI overrides. Values are pyzed serial numbers (uint32).
ZED_SN_BY_NAME: dict[str, int] = {
    # --- Exo view candidates (any of these can be used as the exo camera) ---
    "exo_front":     int(os.environ.get("ZED_EXO_FRONT_SN", "28110525")),  # ZED 2   — tripod, direct front-of-arm view
    "exo_left":      int(os.environ.get("ZED_EXO_LEFT_SN", "36202768")),  # ZED 2i  — mounted on the LEFT of the workspace
    "exo_right":     int(os.environ.get("ZED_EXO_RIGHT_SN", "33325293")),  # ZED 2i  — mounted on the RIGHT of the workspace
    # --- Wrist (fixed) ---
    "zedm_wrist":    int(os.environ.get("ZED_WRIST_SN", "15855030")),  # ZED-M   — mounted on the gripper (default wrist)
    # --- Legacy aliases kept for back-compat with older scripts / logs ---
    "zed2_tripod":     int(os.environ.get("ZED_EXO_FRONT_SN", "28110525")),  # alias for exo_front (was the sole default exo)
    "zed2i_a":     int(os.environ.get("ZED_EXO_RIGHT_SN", "33325293")),  # alias for exo_right
}
# Named presets that can be passed to --front_view_cam_id (exo view selection).
FRONT_VIEW_PRESETS = ("exo_front", "exo_left", "exo_right")
# Back-compat: these are the defaults used when no CLI override is passed.
EXO_ZED_SN = ZED_SN_BY_NAME["exo_front"]      # ZED 2, tripod, third-person view
WRIST_ZED_SN = ZED_SN_BY_NAME["zedm_wrist"]   # ZED-M, on the gripper
GRIPPER_MAX_WIDTH_M = 0.08                    # Franka Hand fully open
GRIPPER_SPEED = 0.1                           # m/s for /gripper_move (default)
# Robotiq preset — trained state gripper q99 ≈ 0.87 rad (Robotiq 2F-85 finger-joint angle)
ROBOTIQ_CLOSED_RAD = 0.87

# --- True Robotiq 2F-85 four-bar / parallel-jaw kinematics ----------------
# Derived from the official ros2_robotiq_gripper URDF
# (robotiq_2f_85_macro.urdf.xacro):
#   knuckle joint origin in base:      (0.03060114, 0, 0.05490452)
#   finger link origin in knuckle:     (0.03152616, 0, -0.00376347)
#   finger_tip origin in finger link:  (0.00563134, 0, 0.04718515)
#   finger_tip_joint mimic multiplier = -1  →  fingertip stays parallel to base
#   knuckle joint limit = [0.0, 0.8] rad  (0=open, 0.8=closed)
# Pad-face x-offset in the tip-link frame is back-solved so w(θ=0) = 0.085 m.
# Full closed-form is derived and unit-tested in
# tools/robotiq_2f85_kine_notes.md (in this repo).
# True 2F-85 four-bar has the analytical form
#     gap(θ) = 2·(A·sin(θ0 − θ) + B)
# so its inverse is
#     θ    = θ0 − arcsin((gap/2 − B) / A)
# The straight-URDF FK is off from the raw MuJoCo qpos the sim actually
# exports (the URDF's finger_tip revolute vs sim's parallel-jaw finger_joint
# have subtly different parameterisations). We therefore fit (A, B, θ0) to
# five anchor points supplied by the sim-side FK analysis, which is the
# authoritative source of what the *policy* was trained on:
#     (θ_rad,  gap_m):
#         (0.00, 0.085)   fully open  (Robotiq catalogue)
#         (0.28, 0.066)   grasp seated on 66 mm real can
#         (0.36, 0.057)   grasp tight on 57 mm real can
#         (0.55, 0.036)   sim "held" band centre
#         (0.80, 0.000)   fully closed
# scipy.optimize.least_squares fit (see tools/robotiq_2f85_kine_notes.md):
#   A     =  0.088188 m
#   B     = -0.042572 m
#   θ0    =  1.304308 rad
#   max |residual| = 0.7 mm across the five anchors
# Curve is analytically monotone on [0, 0.8] and hits gap(0)=85.00 mm,
# gap(0.8)=0.08 mm — well within Robotiq's stated tolerance.
_R2F85_A_M       = 0.088188
_R2F85_B_M       = -0.042572
_R2F85_TH0_RAD   =  1.304308
ROBOTIQ_KINE_MAX_RAD = 0.8   # closed at 0.8 rad, not 0.87 (0.87 is the LINEAR
                              # convention max used by the older
                              # robotiq_state encoding; keep both for A/B).


def robotiq_2f85_width_to_angle(width_m: float) -> float:
    """True 2F-85 knuckle-joint angle (rad) given the full finger-to-finger
    gap in metres.

    Closed-form inverse of  gap(θ) = 2·(A·sin(θ0 − θ) + B).
    Matches the raw finger_joint qpos that the MuJoCo sim exports as
    obs.qpos.gripper[0] (verified against sim-side FK anchors to ≤1 mm).

    Endpoints:
        w = 0.085 m -> θ = 0.0 rad     (fully open)
        w = 0.000 m -> θ ≈ 0.80 rad    (fully closed, URDF joint limit)
    """
    # Clip to [0, gap(θ=0)]. The 2F-85 max is 0.085 m but the caller may
    # send widths up to Franka Hand's 0.08 m; both map into arcsin's domain.
    w = float(np.clip(width_m, 0.0, 0.085))
    arg = (0.5 * w - _R2F85_B_M) / _R2F85_A_M
    # Numerical safety against fp rounding pushing arg outside [-1, 1].
    arg = float(np.clip(arg, -1.0, 1.0))
    return _R2F85_TH0_RAD - float(np.arcsin(arg))

# Hardware joint position limits (Panda; FR3 nearly identical). Always enforced.
JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
], dtype=np.float32)


# ---------- Helpers ----------

abort_flag = threading.Event()


def sigint_handler(signum, frame):
    print("\n[run] caught SIGINT — aborting episode")
    abort_flag.set()


_JS_LAST_ERR_TS: float = 0.0
_JS_ERR_STREAK: int = 0


def get_joint_state(_retries: int = 2) -> tuple[np.ndarray, bool]:
    """Read /joint_state + /health. Retries transient timeouts once so an
    isolated slow response (franky serialises libfranka reads under a mutex,
    so we occasionally see 500-1500 ms spikes when the model chunk-refresh
    lines up with a control-loop grab) does not kill the episode.
    """
    global _JS_LAST_ERR_TS, _JS_ERR_STREAK
    last_exc: Exception | None = None
    for attempt in range(_retries):
        try:
            r = requests.get(f"{FRANKY_URL}/joint_state",
                             timeout=FRANKY_READ_TIMEOUT_S)
            r.raise_for_status()
            js = r.json()
            arm = np.asarray(js["positions"][:7], dtype=np.float32)
            health = requests.get(f"{FRANKY_URL}/health",
                                  timeout=FRANKY_READ_TIMEOUT_S).json()
            if _JS_ERR_STREAK > 0:
                print(f"[run] joint_state recovered "
                      f"(after {_JS_ERR_STREAK} failed poll(s))")
                _JS_ERR_STREAK = 0
            return arm, bool(health.get("stop_latched", False))
        except (requests.RequestException, ValueError, KeyError) as e:
            last_exc = e
            _JS_ERR_STREAK += 1
            now = time.time()
            if now - _JS_LAST_ERR_TS > 5.0:
                print(f"[run] joint_state slow/failed "
                      f"(attempt {attempt + 1}/{_retries}, streak={_JS_ERR_STREAK}): {e}")
                _JS_LAST_ERR_TS = now
    # All retries exhausted — surface the failure to the caller.
    raise last_exc if last_exc is not None else RuntimeError("joint_state failed")


def get_gripper_width() -> float:
    r = requests.get(f"{HAND_URL}/gripper_state", timeout=GRIPPER_READ_TIMEOUT_S)
    r.raise_for_status()
    return float(r.json()["width_m"])


def send_arm_target(positions: list[float]) -> None:
    requests.post(
        f"{FRANKY_URL}/target_joint_state",
        json={"positions": positions, "velocities": [0.0] * 7},
        timeout=FRANKY_WRITE_TIMEOUT_S,
    )


def send_gripper_move(width_m: float, speed: float = GRIPPER_SPEED) -> None:
    width_m = float(np.clip(width_m, 0.0, GRIPPER_MAX_WIDTH_M))
    try:
        requests.post(
            f"{HAND_URL}/gripper_move",
            json={"width_m": width_m, "speed": speed},
            timeout=GRIPPER_WRITE_TIMEOUT_S,
        )
    except requests.RequestException as e:
        print(f"[run] gripper_move failed: {e}")


def send_gripper_grasp(width_m: float, force_n: float, speed: float = 0.05) -> None:
    """Blocking-force grasp (returns immediately; motion runs in background)."""
    width_m = float(np.clip(width_m, 0.0, GRIPPER_MAX_WIDTH_M))
    try:
        requests.post(
            f"{HAND_URL}/gripper_grasp",
            json={
                "width_m": width_m,
                "speed": speed,
                "force": float(force_n),
                "epsilon_inner": 0.02,
                "epsilon_outer": 0.05,
            },
            timeout=GRIPPER_WRITE_TIMEOUT_S,
        )
    except requests.RequestException as e:
        print(f"[run] gripper_grasp failed: {e}")


def send_gripper_open(speed: float = 0.1) -> None:
    try:
        requests.post(
            f"{HAND_URL}/gripper_open",
            json={"speed": speed},
            timeout=GRIPPER_WRITE_TIMEOUT_S,
        )
    except requests.RequestException as e:
        print(f"[run] gripper_open failed: {e}")


# ---------- Robotiq 2F-85 (local gripper_service on :54323) ----------
# Robotiq convention: position_bits 0 = fully open (~85 mm), 255 = fully closed.
# We expose the same "width_m" [0..0.085] semantics as the Franka Hand so the
# gripper_encode_input(width_m, mode) function keeps working unchanged.

def robotiq_bits_to_width_m(bits: int) -> float:
    b = int(np.clip(bits, 0, 255))
    return (1.0 - b / 255.0) * ROBOTIQ_STROKE_M


def robotiq_width_m_to_bits(width_m: float) -> int:
    w = float(np.clip(width_m, 0.0, ROBOTIQ_STROKE_M))
    return int(round((1.0 - w / ROBOTIQ_STROKE_M) * 255.0))


def get_robotiq_width(url: str = ROBOTIQ_URL) -> float:
    """Query /gripper_state and return the current opening in metres."""
    r = requests.get(f"{url}/gripper_state", timeout=GRIPPER_READ_TIMEOUT_S)
    r.raise_for_status()
    js = r.json()
    bits = js.get("position_bits")
    if bits is None:  # motion in progress
        return float(np.nan)
    return robotiq_bits_to_width_m(int(bits))


def send_robotiq_grasp(width_m: float, force_255: int = 255,
                       speed_255: int = 255,
                       url: str = ROBOTIQ_URL) -> None:
    """Command a target opening in metres. `force_255` and `speed_255` are
    Robotiq's 0..255 register units, not SI — we keep the wire format simple.
    Fire-and-forget: `wait=false` so we don't block the 15 Hz loop.
    """
    bits = robotiq_width_m_to_bits(width_m)
    try:
        requests.post(
            f"{url}/go_to?wait=false",
            json={"position": bits,
                  "speed": int(np.clip(speed_255, 1, 255)),
                  "force": int(np.clip(force_255, 1, 255))},
            timeout=GRIPPER_WRITE_TIMEOUT_S,
        )
    except requests.RequestException as e:
        print(f"[run] robotiq go_to failed: {e}")


def send_robotiq_open(speed_255: int = 255, url: str = ROBOTIQ_URL) -> None:
    try:
        requests.post(
            f"{url}/open?wait=false",
            json={"position": 0, "speed": int(speed_255), "force": 255},
            timeout=GRIPPER_WRITE_TIMEOUT_S,
        )
    except requests.RequestException as e:
        print(f"[run] robotiq open failed: {e}")


def set_command_timeout(t: float) -> None:
    requests.post(
        f"{FRANKY_URL}/command_timeout",
        json={"command_timeout_s": t}, timeout=FRANKY_WRITE_TIMEOUT_S,
    )


def stop_robot() -> None:
    try:
        requests.post(f"{FRANKY_URL}/stop", timeout=FRANKY_WRITE_TIMEOUT_S)
    except requests.RequestException:
        pass


# ---------- ZED ----------

def open_zed(serial_number: int | None = None,
             tag: str = "zed") -> sl.Camera:
    """Open a ZED camera. If serial_number is given, pin to that specific
    device (needed when multiple ZEDs are connected — e.g. exo + wrist).
    """
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = ZED_RESOLUTION
    init.camera_fps = ZED_CAMERA_FPS
    init.depth_mode = sl.DEPTH_MODE.NONE   # RGB only; saves CPU/GPU
    init.coordinate_units = sl.UNIT.METER
    if serial_number is not None:
        init.set_from_serial_number(int(serial_number))
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"ZED open failed for SN {serial_number}: {status}. "
            "If the camera service is running (make camera_service), stop it "
            "so pyzed can claim the device."
        )
    info = zed.get_camera_information()
    res = info.camera_configuration.resolution
    print(f"[{tag}] {info.camera_model} SN {info.serial_number} "
          f"{res.width}x{res.height}")
    return zed


def grab_rgb_square(zed: sl.Camera, buf: sl.Mat) -> np.ndarray:
    """Grab one frame, center-crop to square, resize to POLICY_IMG_SIZE, BGR→RGB."""
    if zed.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED grab failed")
    zed.retrieve_image(buf, sl.VIEW.LEFT)
    bgra = buf.get_data()                       # (H, W, 4) uint8 BGRA
    bgr = bgra[:, :, :3]
    h, w, _ = bgr.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = bgr[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(crop, (POLICY_IMG_SIZE, POLICY_IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


# ---------- Gripper encoding ----------
#
# `--gripper_input` controls what scalar we put into obs["qpos"]["gripper"][0]:
#   normalized : 0.0 (fully open) .. 1.0 (fully closed)        [sim's output convention]
#   meters     : physical width in meters (0..0.08)             [Franka Hand native]
#   raw        : Robotiq-style bit-pos (0=open, 255=closed)     [old datasets used this]
#
# `--gripper_output` controls how we read action["gripper"][0]:
#   normalized : value in ~[0,1], clip then width = (1-g)*0.08  [sim convention]
#   meters     : value already in meters
#   raw        : 0..255, width = (1 - g/255) * 0.08

def gripper_encode_input(width_m: float, mode: str) -> float:
    w = float(np.clip(width_m, 0.0, GRIPPER_MAX_WIDTH_M))
    if mode == "normalized":
        return 1.0 - w / GRIPPER_MAX_WIDTH_M
    if mode == "meters":
        return w
    if mode == "raw":
        return (1.0 - w / GRIPPER_MAX_WIDTH_M) * 255.0
    if mode == "robotiq_state":
        # LINEAR proxy: width -> finger-joint angle.
        # width 0.08 (open) -> 0.0 rad ;  width 0.0 (closed) -> ~0.87 rad
        # KEPT FOR BACK-COMPAT with v3-era runs. Prefer robotiq_state_kine
        # for new checkpoints — see convention doc §4.1.
        return (1.0 - w / GRIPPER_MAX_WIDTH_M) * ROBOTIQ_CLOSED_RAD
    if mode == "robotiq_state_kine":
        # TRUE Robotiq 2F-85 kinematics — closed-form inverse of the URDF
        # parallel-jaw four-bar. Matches what a sim exporter that dumps the
        # raw knuckle-joint qpos would produce for the same physical gap.
        # See ROBOTIQ_KINEMATICS_NOTES.md for derivation.
        return robotiq_2f85_width_to_angle(w)
    if mode == "per_finger_m":
        # Training data stored per-finger width (m). Franka `width_m` is the full
        # gap (0..0.08); per-finger displacement is half of that: 0..0.04.
        # Matches the training range ~0.02-0.04 mentioned by the model author.
        return 0.5 * w
    raise ValueError(f"bad gripper_input mode: {mode}")


def gripper_decode_output(g: float, mode: str) -> float:
    g = float(g)
    if mode == "normalized":
        g = float(np.clip(g, 0.0, 1.0))
        return (1.0 - g) * GRIPPER_MAX_WIDTH_M
    if mode == "meters":
        return float(np.clip(g, 0.0, GRIPPER_MAX_WIDTH_M))
    if mode == "raw":
        g = float(np.clip(g, 0.0, 255.0))
        return (1.0 - g / 255.0) * GRIPPER_MAX_WIDTH_M
    raise ValueError(f"bad gripper_output mode: {mode}")


class PreviewThread(threading.Thread):
    """Background window that renders the latest RGB frame at ~30 Hz.

    We can't call cv2.imshow/waitKey only once per policy step because Qt's
    event loop needs to be pumped continuously; a blocking input() or a
    500 ms chunk-warmup would otherwise leave the window frozen/black.
    """

    def __init__(self, scale: float = 1.0, window_name: str = "policy obs (exo)") -> None:
        super().__init__(daemon=True)
        self.scale = float(scale)
        self.window_name = window_name
        self._latest_bgr: np.ndarray | None = None
        self._latest_hud: str = ""
        self._lock = threading.Lock()
        # NOTE: attribute name intentionally NOT `_stop` — threading.Thread
        # already uses `_stop()` internally during join() and shadowing it
        # breaks teardown ("TypeError: 'Event' object is not callable").
        self._stop_evt = threading.Event()
        self._q_pressed = threading.Event()
        # 'o' = user-triggered gripper-state override. Only meaningful when
        # --manual_gripper_override is passed; otherwise the main loop ignores
        # this event. Latched (Event, not toggle) so pressing once is enough.
        self._override_pressed = threading.Event()

    def update(self, rgb: np.ndarray, hud: str = "") -> None:
        """Post a new frame; called from the main loop each step."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._latest_bgr = bgr
            self._latest_hud = hud

    def q_pressed(self) -> bool:
        return self._q_pressed.is_set()

    def override_pressed(self) -> bool:
        """True once the user has pressed 'o' in the preview window. Latched:
        stays True for the rest of the run. Only used when the main loop is
        configured with --manual_gripper_override."""
        return self._override_pressed.is_set()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        while not self._stop_evt.is_set():
            with self._lock:
                frame = None if self._latest_bgr is None else self._latest_bgr.copy()
                hud = self._latest_hud
            if frame is not None:
                if self.scale != 1.0:
                    frame = cv2.resize(
                        frame,
                        (int(frame.shape[1] * self.scale),
                         int(frame.shape[0] * self.scale)),
                        interpolation=cv2.INTER_NEAREST,
                    )
                if hud:
                    cv2.putText(frame, hud, (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(self.window_name, frame)
            # This is what actually pumps the Qt event loop.
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                self._q_pressed.set()
                # Don't self-stop — let the main loop notice via q_pressed().
            elif k == ord("o"):
                # 'o' = user-triggered gripper-state override latch. Main
                # loop reads override_pressed() and decides whether to act
                # on it (based on --manual_gripper_override).
                self._override_pressed.set()
        try:
            cv2.destroyWindow(self.window_name)
            for _ in range(4):
                cv2.waitKey(1)
        except Exception:
            pass


def safety_clip(target_q: np.ndarray, current_q: np.ndarray,
                max_delta: float) -> tuple[np.ndarray, bool]:
    """Apply optional per-step joint delta clip + always-on position limits.

    max_delta <= 0 disables the per-step clip (sim default).
    Returns (q, clamped_by_delta?).
    """
    if max_delta > 0.0:
        delta = target_q - current_q
        big = np.abs(delta) > max_delta
        clamped = bool(big.any())
        delta = np.clip(delta, -max_delta, max_delta)
        q = current_q + delta
    else:
        q = target_q.copy()
        clamped = False
    q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    return q.astype(np.float32), clamped


# ---------- Logging ----------

class EpisodeLogger:
    """Dumps per-step JPGs (exo + optional wrist) + a single steps.jsonl.

    Directory layout:
        <log_dir>/frames/0000.jpg         # exo view (kept as-is for back-compat)
        <log_dir>/frames_wrist/0000.jpg   # wrist view (only if wrist rgb given)
        <log_dir>/steps.jsonl

    ~50 KB per view per step at 480x480 / q90.
    """

    def __init__(self, log_dir: Path | None) -> None:
        self.dir = log_dir
        self.fp = None
        self._wrist_dir_ready = False
        if log_dir is not None:
            (log_dir / "frames").mkdir(parents=True, exist_ok=True)
            self.fp = (log_dir / "steps.jsonl").open("w")

    def _ensure_wrist_dir(self) -> None:
        if self.dir is None or self._wrist_dir_ready:
            return
        (self.dir / "frames_wrist").mkdir(parents=True, exist_ok=True)
        self._wrist_dir_ready = True

    def write(self, step: int, rgb: np.ndarray, record: dict,
              rgb_wrist: np.ndarray | None = None) -> None:
        if self.dir is None:
            return
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.dir / "frames" / f"{step:04d}.jpg"), bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if rgb_wrist is not None:
            self._ensure_wrist_dir()
            bgr_w = cv2.cvtColor(rgb_wrist, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self.dir / "frames_wrist" / f"{step:04d}.jpg"),
                        bgr_w, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        self.fp.write(json.dumps(record) + "\n")
        self.fp.flush()

    def close(self) -> None:
        if self.fp is not None:
            self.fp.close()


# ---------- Main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws_url", default=None)
    ap.add_argument("--steps", type=int, default=0,
                    help="max policy steps; 0 = run until Ctrl-C (default). "
                         "At --rate 15 Hz, 450 steps ≈ 30 s.")
    ap.add_argument("--rate", type=float, default=15.0,
                    help="control rate Hz (training used 15)")
    ap.add_argument("--task", required=True,
                    help="Language instruction sent VERBATIM to the model as "
                         "obs['task'] each inference (no prefix/suffix added). "
                         "REQUIRED — this checkpoint (V5) supports diverse "
                         "instructions and multiple objects, so the instruction "
                         "must be specified explicitly per run rather than "
                         "defaulting to a fixed task. "
                         "e.g. --task 'Pick up the red mug'.")
    ap.add_argument("--dry_run", action="store_true",
                    help="don't send to robot; print intended action only")
    ap.add_argument("--max_joint_delta", type=float, default=0.0,
                    help="per-step joint delta clip in rad; 0 disables (sim default).")
    ap.add_argument("--gripper", choices=["panda", "robotiq"], default="panda",
                    help="High-level preset for gripper handling. "
                         "'panda' = current continuous /gripper_move path (matches "
                         "our finetuned pilot's training units). "
                         "'robotiq' = Robotiq-scaled state input, binary threshold on "
                         "the model's action, hysteresis, and force-grasp on close "
                         "(matches DROID-Img training units).")
    ap.add_argument("--model",
                    choices=["molmobot_ft"],
                    default="molmobot_ft",
                    help="Select an A100-side checkpoint. Currently only the "
                         "single finetuned MolmoBot checkpoint (molmobot_ft) "
                         "is served by this script. Point at a different node "
                         "with --a100_host/--a100_port or the env vars "
                         "MOLMOBOTFT_A100_HOST / MOLMOBOTFT_A100_PORT.")
    ap.add_argument("--a100_host",
                    default=os.environ.get("MOLMOBOTFT_A100_HOST",
                                           "10.49.163.26"),
                    help="A100 node hostname/IP hosting the finetuned MolmoBot "
                         "checkpoint. Defaults to $MOLMOBOTFT_A100_HOST or "
                         "10.49.163.26 (current redeployed node).")
    ap.add_argument("--a100_port", type=int,
                    default=int(os.environ.get("MOLMOBOTFT_A100_PORT", "8005")),
                    help="Port on --a100_host serving the MolmoBot-FT policy. "
                         "Default 8005 (env: MOLMOBOTFT_A100_PORT).")
    ap.add_argument("--dual_camera", dest="dual_camera",
                    action="store_true", default=None,
                    help="Send both exo_front and wrist in the obs (required "
                         "by molmobot_ft). Auto-enabled by --model molmobot_ft. "
                         "Requires the wrist ZED (SN %d) to be free "
                         "(stop camera_service)." % WRIST_ZED_SN)
    ap.add_argument("--no_dual_camera", dest="dual_camera",
                    action="store_false",
                    help="Force single-camera obs even if the chosen "
                         "--model default is dual-camera.")
    ap.add_argument("--flip_cams", action="store_true", default=False,
                    help="Swap the two ZED assignments before building the obs "
                         "(and the preview / logged frames). With this flag: "
                         "the wrist-ZED feed is sent to the model as "
                         "'exo_front' and the exo-ZED feed is sent as "
                         "'wrist'. Useful for A/B testing view assignment. "
                         "Only affects --dual_camera runs.")
    ap.add_argument("--crop_bottom_5", action="store_true", default=False,
                    help="Crop 5%% off the BOTTOM of the wrist view BEFORE it "
                         "is sent to the model (keep the top 95%% of rows, "
                         "full width). Applied to the physical wrist ZED "
                         "(i.e. before --flip_cams). The cropped frame is "
                         "then resized back to the policy image size so the "
                         "model still sees the expected resolution. Only "
                         "affects --dual_camera runs.")
    _cam_names = ", ".join(f"{n}={s}" for n, s in ZED_SN_BY_NAME.items())
    ap.add_argument("--front_view_cam_id", choices=list(FRONT_VIEW_PRESETS),
                    default="exo_front",
                    help="Which physical camera to use as the exo view sent "
                         "to the model. exo_front (default) = ZED 2 on the "
                         "tripod, direct front-of-arm view (SN "
                         f"{ZED_SN_BY_NAME['exo_front']}); "
                         "exo_left = ZED 2i mounted on the LEFT of the "
                         f"workspace (SN {ZED_SN_BY_NAME['exo_left']}); "
                         "exo_right = ZED 2i mounted on the RIGHT of the "
                         f"workspace (SN {ZED_SN_BY_NAME['exo_right']}). "
                         "The wrist camera is always the ZED-M "
                         f"(SN {WRIST_ZED_SN}). The obs key sent to the "
                         "server is still 'exo_front' regardless — the model "
                         "doesn't know which physical ZED you chose. "
                         "Overridden by an explicit --exo <SN>.")
    ap.add_argument("--exo", "--exo_sn", dest="exo_sn", default=None,
                    help="Low-level override for the exo ZED serial number. "
                         "Accepts a raw integer SN or one of the presets: "
                         + _cam_names +
                         f". If passed, takes precedence over "
                         f"--front_view_cam_id. Default: resolved from "
                         f"--front_view_cam_id (default exo_front, "
                         f"SN {EXO_ZED_SN}).")
    ap.add_argument("--wrist", "--wrist_sn", dest="wrist_sn", default=None,
                    help="Override the wrist ZED serial number. Same format "
                         "as --exo. Default: zedm_wrist "
                         f"({WRIST_ZED_SN}). Only used under --dual_camera.")
    ap.add_argument("--list_cameras", action="store_true", default=False,
                    help="Enumerate ZED cameras visible to the SDK and exit.")
    ap.add_argument("--no_gripper", action="store_true",
                    help="Skip ALL hand-service calls (54324). Used when the "
                         "Franka Hand is powered off. A fixed dummy value is "
                         "sent to the model as the gripper obs, and the "
                         "gripper action is logged but never dispatched. "
                         "Arm proprioception + arm commanding still work.")
    ap.add_argument("--dummy_gripper_width_m", type=float, default=0.04,
                    help="[--no_gripper] Fixed gripper width in metres to "
                         "report to the model as if the hand were open. "
                         "Default 0.04 m = mid-range of Franka Hand.")
    ap.add_argument("--gripper_input", choices=["normalized", "meters", "raw",
                                                "robotiq_state",
                                                "robotiq_state_kine",
                                                "per_finger_m"],
                    default=None,
                    help="Override the preset's obs encoding. Default: preset-specific "
                         "('per_finger_m' for panda — matches training obs, which is "
                         "the per-finger width in metres ~0.02-0.04; "
                         "'robotiq_state_kine' for robotiq — TRUE 2F-85 parallel-jaw "
                         "kinematics matching what a sim exporter dumping the raw "
                         "knuckle-joint qpos would produce). 'robotiq_state' is the "
                         "older LINEAR proxy, kept only for A/B and back-compat with "
                         "v3-era runs.")
    ap.add_argument("--gripper_output", choices=["normalized", "meters", "raw"],
                    default=None,
                    help="Override the preset's action decoding. Default: "
                         "'normalized' for both panda and robotiq (matches the "
                         "V3 checkpoint's 0..1 action gripper — see "
                         "V3_GRIPPER_AND_IO_CONVENTIONS.md §3.2).")
    # --- robotiq-only tuning (ignored under --gripper panda) ---
    ap.add_argument("--binary_threshold", type=float, default=None,
                    help="[robotiq] threshold on the (decoded) gripper action above "
                         "which we call CLOSE. Default: 127 for raw, 0.5 for "
                         "normalized, 0.04 for meters.")
    ap.add_argument("--gripper_debounce", type=int, default=3,
                    help="[robotiq] number of consecutive same-vote frames required "
                         "before flipping open<->closed. 1 disables debounce.")
    ap.add_argument("--grasp_width_m", type=float, default=0.03,
                    help="[robotiq] target width when closing (m). Object diameter-ish.")
    ap.add_argument("--grasp_force_n", type=float, default=20.0,
                    help="[robotiq] grasp force in Newtons.")
    ap.add_argument("--gripper_speed", type=float, default=0.05,
                    help="[robotiq] speed for grasp/open commands (m/s). "
                         "Legacy Franka-Hand units; ignored when talking to "
                         "the Robotiq service (uses --robotiq_speed_255).")
    ap.add_argument("--robotiq_url", default=ROBOTIQ_URL,
                    help="[robotiq] URL of the Robotiq 2F-85 HTTP service. "
                         "Default http://localhost:54323 (this workstation).")
    ap.add_argument("--robotiq_force_255", type=int, default=255,
                    help="[robotiq] Robotiq force register (0..255, "
                         "0=min, 255=max). Default 255.")
    ap.add_argument("--robotiq_speed_255", type=int, default=255,
                    help="[robotiq] Robotiq speed register (0..255). "
                         "Default 255 (fastest).")
    ap.add_argument("--log_dir", default=None,
                    help="if set, dump per-step frames + steps.jsonl here")
    ap.add_argument("--preview", dest="preview", action="store_true", default=True,
                    help="show live window of the RGB frame sent to the model (default on)")
    ap.add_argument("--no_preview", dest="preview", action="store_false",
                    help="disable the live preview window")
    ap.add_argument("--preview_scale", type=float, default=1.0,
                    help="scale factor for preview window (1.0 = 480x480)")
    ap.add_argument("--step_confirm", action="store_true",
                    help="pause each step after the action is computed and require the "
                         "user to press Enter before it is sent to the robot. Type 's' "
                         "to skip this step (no send), 'g N' to auto-run the next N "
                         "steps without prompting, or 'q'/Ctrl-C to abort.")
    ap.add_argument("--jpeg_obs", action="store_true",
                    help="JPEG-compress image observations before sending to "
                         "the server (~15-24x smaller than raw uint8). Cuts "
                         "the per-chunk upload from ~1.38 MB to ~60-90 KB, "
                         "which is the dominant cost on a bandwidth-limited "
                         "link to the GPU node. REQUIRES a server that decodes "
                         "the JPEG blobs (advertised via metadata "
                         "accepts_jpeg=True); a warning is printed if the "
                         "server does not advertise support.")
    ap.add_argument("--jpeg_quality", type=int, default=90,
                    help="[--jpeg_obs] JPEG quality 1..100 (default 90; "
                         "q90 ~= 29 KB per 480x480 frame, visually lossless "
                         "for this task).")
    ap.add_argument("--chunk_executor", choices=["auto", "on", "off"],
                    default="auto",
                    help="Client-side chunk execution. When the server "
                         "advertises chunk_response=True it returns all 8 "
                         "execute-horizon actions in one response; this "
                         "executor then dispatches them locally on a STRICT "
                         "66.7 ms (=1/rate) clock with NO network in the "
                         "per-action path, and runs the next inference "
                         "sequentially AFTER the chunk finishes (a visible "
                         "~inference-length pause between chunks, arm holds "
                         "the last target during it). 'auto' (default) uses "
                         "this whenever the server supports it and falls back "
                         "to the legacy one-action-per-request loop "
                         "otherwise; 'on' forces it (errors if the server "
                         "can't chunk); 'off' forces the legacy loop.")
    ap.add_argument("--manual_gripper_override", nargs="?", type=float,
                    const=0.535, default=None, metavar="VALUE",
                    help="Diagnostic: once the user presses 'o' in the preview "
                         "window (typically after visually confirming the "
                         "object is firmly grasped), OVERRIDE the reported "
                         "gripper-state scalar sent to the model with VALUE "
                         "and hold it fixed for the rest of the episode. "
                         "Default VALUE is 0.535 (sim LIFT-phase median in "
                         "the raw finger-joint radian convention). Use to "
                         "isolate the 'gripper state doesn't reach sim HOLD "
                         "band' failure mode. Off by default — pass the flag "
                         "with no value for 0.535, or with a value to use "
                         "that value instead.")
    args = ap.parse_args()

    # --- Camera SN resolution (--exo / --wrist / --list_cameras) -----------
    def _resolve_sn(spec: str | None, default_sn: int, label: str) -> int:
        if spec is None:
            return default_sn
        s = str(spec).strip()
        if s in ZED_SN_BY_NAME:
            sn = ZED_SN_BY_NAME[s]
            print(f"[run] --{label} preset {s!r} -> SN {sn}")
            return sn
        try:
            sn = int(s)
        except ValueError:
            raise SystemExit(
                f"--{label}={spec!r} is not a known preset "
                f"({', '.join(ZED_SN_BY_NAME)}) and not an integer SN.")
        print(f"[run] --{label} raw SN {sn}")
        return sn

    if args.list_cameras:
        try:
            devs = sl.Camera.get_device_list()
        except Exception as e:
            print(f"[run] failed to list ZED devices: {e}")
            return 1
        if not devs:
            print("[run] pyzed sees 0 ZED cameras.")
            return 0
        print(f"[run] pyzed sees {len(devs)} ZED camera(s):")
        by_sn = {s: n for n, s in ZED_SN_BY_NAME.items()}
        for d in devs:
            name = by_sn.get(d.serial_number, "(unnamed)")
            print(f"  SN={d.serial_number:<10}  model={d.camera_model}  "
                  f"state={d.camera_state}  path={d.path}  preset={name}")
        return 0

    # Exo SN precedence: explicit --exo > --front_view_cam_id > default (exo_front).
    if args.exo_sn is None:
        # Resolve via the friendly --front_view_cam_id preset.
        preset = args.front_view_cam_id
        if preset not in ZED_SN_BY_NAME:
            raise SystemExit(
                f"--front_view_cam_id={preset!r} not in ZED_SN_BY_NAME "
                f"({list(ZED_SN_BY_NAME)})"
            )
        args.exo_sn = ZED_SN_BY_NAME[preset]
        print(f"[run] --front_view_cam_id={preset} -> exo SN {args.exo_sn}")
    else:
        args.exo_sn = _resolve_sn(args.exo_sn, EXO_ZED_SN, "exo")
        # Warn if the explicit --exo contradicts --front_view_cam_id.
        preset_sn = ZED_SN_BY_NAME.get(args.front_view_cam_id)
        if preset_sn is not None and preset_sn != args.exo_sn:
            print(f"[run] NOTE: --exo {args.exo_sn} overrides "
                  f"--front_view_cam_id={args.front_view_cam_id} "
                  f"(would have been SN {preset_sn})")
    args.wrist_sn = _resolve_sn(args.wrist_sn, WRIST_ZED_SN, "wrist")

    # --- Resolve gripper preset -> concrete encode/decode/thresholds ---
    if args.gripper == "panda":
        if args.gripper_input is None:
            # Training obs was per-finger width in metres, not normalized [0,1].
            args.gripper_input = "per_finger_m"
        if args.gripper_output is None:
            args.gripper_output = "normalized"
    else:  # robotiq
        if args.gripper_input is None:
            # State side: Robotiq finger-joint angle in radians (0≈open,
            # ~0.8≈closed). Matches training state gripper (`qpos.gripper[0]`,
            # rad) per V3_GRIPPER_AND_IO_CONVENTIONS.md §3.1.
            # Uses TRUE 2F-85 parallel-jaw kinematics (closed-form fit to
            # URDF-derived FK anchors) rather than the old linear proxy —
            # see the deploy analysis: on the free-air close ramp AND at
            # held-object widths (57-66 mm) the linear map underreports the
            # angle by ≈0.11-0.12 rad (worst around w=60 mm). Real objects
            # will still not reach the sim's HOLD band without simulator
            # penetration, but this at least removes the convention-
            # mismatch component. Pass --gripper_input robotiq_state to
            # revert to the old linear map for A/B comparison.
            args.gripper_input = "robotiq_state_kine"
        if args.gripper_output is None:
            # Action side: the checkpoint's action gripper is a NORMALIZED
            # 0..1 scalar (0 = open, 1 = closed), with q01/q99 = 0/1 → the
            # server returns un-normalized 0..1 directly. Use 'normalized'
            # decode with threshold 0.5 (V3 conventions §3.2). Previously
            # defaulted to 'raw' (0..255, threshold 127) which was the
            # step1205 bug that never fired the close vote.
            args.gripper_output = "normalized"
    # Binary threshold default depends on decode mode
    if args.binary_threshold is None:
        args.binary_threshold = {
            "raw": 127.0,
            "normalized": 0.5,
            "meters": 0.04,          # halfway between fully open/closed in meters
            "robotiq_state": 0.5,    # not a valid output but keep table complete
            "per_finger_m": 0.02,    # halfway between fully open/closed per finger
        }[args.gripper_output]

    # --- Resolve ws_url: --ws_url > --model > server_info.json ------------
    info = json.loads((ROOT / "sim_serve" / "server_info.json").read_text())
    # Single-model deployment: honour --a100_host/--a100_port.
    MODEL_ENDPOINTS = {
        "molmobot_ft": (args.a100_host, args.a100_port),
    }
    # This finetuned checkpoint takes dual-camera obs (exo_front + wrist).
    DUAL_CAMERA_MODELS = {"molmobot_ft"}
    if args.dual_camera is None:
        args.dual_camera = (args.model in DUAL_CAMERA_MODELS)
    if args.ws_url:
        ws_url = args.ws_url
        # Best-effort healthz derived from ws_url
        healthz = ws_url.replace("ws://", "http://").rstrip("/") + "/healthz"
    elif args.model:
        host, port = MODEL_ENDPOINTS[args.model]
        ws_url = f"ws://{host}:{port}"
        healthz = f"http://{host}:{port}/healthz"
        print(f"[run] --model {args.model} -> {ws_url}")
    else:
        ws_url = info["ws_url_ip"]
        healthz = info["healthz"].replace(info["node_hostname"], info["node_ip"])

    if args.max_joint_delta <= 0.0:
        print("[run] WARNING: per-step joint clamp DISABLED (sim-matched). "
              "Hardware joint position limits still enforced.")
    else:
        print(f"[run] per-step joint clamp ENABLED: {args.max_joint_delta} rad")
    print(f"[run] gripper preset: {args.gripper}  "
          f"(input={args.gripper_input}, output={args.gripper_output})")
    if args.gripper == "robotiq":
        print(f"[run] robotiq mode: url={args.robotiq_url}  "
              f"binary_threshold={args.binary_threshold}  "
              f"debounce={args.gripper_debounce}  "
              f"grasp_width={args.grasp_width_m}m  "
              f"force_255={args.robotiq_force_255}  "
              f"speed_255={args.robotiq_speed_255}")

    # Sanity: server reachable?
    print(f"[run] checking server: {healthz}")
    try:
        r = requests.get(healthz, timeout=3)
        if r.text.strip() != "OK":
            print(f"[run] healthz unexpected: {r.text!r}")
    except Exception as e:
        print(f"[run] healthz failed: {e}")
        return 1

    # Sanity: franky reachable (arm).
    arm_q, stopped = get_joint_state()
    print(f"[run] FR3 joints: {arm_q.round(3).tolist()}  stop_latched={stopped}")
    # Hand reachable (or explicitly skipped)?
    if args.no_gripper:
        width0 = float(args.dummy_gripper_width_m)
        print(f"[run] --no_gripper: gripper service SKIPPED. "
              f"reporting fixed width={width0:.4f} m to model "
              f"(policy state [{args.gripper_input}]: "
              f"{gripper_encode_input(width0, args.gripper_input):.3f}). "
              f"Gripper actions from the model will be LOGGED ONLY (not sent).")
    elif args.gripper == "robotiq":
        width0 = get_robotiq_width(args.robotiq_url)
        print(f"[run] robotiq width: {width0:.4f} m  "
              f"(policy state [{args.gripper_input}]: "
              f"{gripper_encode_input(width0, args.gripper_input):.3f})")
    else:
        width0 = get_gripper_width()
        print(f"[run] gripper width: {width0:.4f} m  "
              f"(policy state [{args.gripper_input}]: "
              f"{gripper_encode_input(width0, args.gripper_input):.3f})")

    # Open ZED(s). Under dual_camera we open both cameras up-front so a
    # ZED-M USB claim failure fails loudly BEFORE we connect the policy.
    zed = open_zed(serial_number=args.exo_sn if args.dual_camera else None,
                   tag="zed_exo")
    buf = sl.Mat()
    zed_wrist: sl.Camera | None = None
    buf_wrist: sl.Mat | None = None
    if args.dual_camera:
        zed_wrist = open_zed(serial_number=args.wrist_sn, tag="zed_wrist")
        buf_wrist = sl.Mat()

    # Connect (= reset policy)
    print(f"[run] connecting to {ws_url}")
    client = MolmoBotClient(ws_url, jpeg_obs=args.jpeg_obs,
                            jpeg_quality=args.jpeg_quality)
    meta = client.connect()
    print(f"[run] server metadata: {meta}")

    # Determine camera key name(s). MolmoBot-FT expects the pair
    # (exo_front, wrist); meta doesn't list them so we hard-wire the schema.
    if args.dual_camera:
        camera_key = "exo_front"
        wrist_camera_key = "wrist"
        print(f"[run] dual-camera obs: exo={camera_key!r} "
              f"wrist={wrist_camera_key!r}")
        if args.flip_cams:
            print(f"[run] --flip_cams: wrist-ZED (SN {args.wrist_sn}) -> "
                  f"{camera_key!r}, exo-ZED (SN {args.exo_sn}) -> "
                  f"{wrist_camera_key!r}")
        if args.crop_bottom_5:
            print("[run] --crop_bottom_5: cropping 5% off the bottom of the "
                  "physical wrist frame (keep top 95%, full width), resizing "
                  f"back to {POLICY_IMG_SIZE}x{POLICY_IMG_SIZE}")
    elif args.flip_cams:
        print("[run] --flip_cams: ignored (single-camera mode)")
    elif args.crop_bottom_5:
        print("[run] --crop_bottom_5: ignored (single-camera mode; wrist not used)")
    else:
        cam_names = meta.get("camera_names") if isinstance(meta, dict) else None
        if cam_names and len(cam_names) >= 1:
            camera_key = cam_names[0]
        else:
            camera_key = "exo_front"
        wrist_camera_key = None
        print(f"[run] using camera obs key: {camera_key!r}")

    log_dir = Path(args.log_dir).resolve() if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "run_meta.json").write_text(json.dumps({
            "ws_url": ws_url, "args": vars(args), "server_meta": meta,
            "joint_limits": JOINT_LIMITS.tolist(),
            "start_state": {
                "arm": arm_q.tolist(),
                "gripper_width_m": width0,
                "stop_latched": stopped,
                "wall_time_unix": time.time(),
            },
        }, indent=2))
        print(f"[run] logging to {log_dir}")
    logger = EpisodeLogger(log_dir)

    print()
    print(f"[run] Task: {args.task!r}")
    step_limit_str = "until Ctrl-C" if args.steps <= 0 else str(args.steps)
    print(f"[run] Steps: {step_limit_str}   Rate: {args.rate} Hz   Dry run: {args.dry_run}")
    # No 'yes' prompt: this script is used for benchmark sweeps where the
    # human confirmation is scripted upstream. Use --dry_run for a safe dry
    # rollout and --step_confirm to gate each individual action.
    print("[run] starting immediately (no 'yes' prompt in benchmark script). "
          "Ctrl-C to abort.")

    signal.signal(signal.SIGINT, sigint_handler)

    # Background preview window (Qt needs continuous waitKey pumping, which we
    # can't do from the main loop while it's blocked in input() / a chunk warmup).
    preview_thread: PreviewThread | None = None
    if args.preview:
        preview_thread = PreviewThread(scale=args.preview_scale)
        preview_thread.start()

    # State latch for --manual_gripper_override. Once set to True (via the
    # user pressing 'o' in the preview window), the gripper-state scalar
    # sent to the model is FROZEN at args.manual_gripper_override for the
    # rest of the episode, regardless of physical hand position.
    manual_grip_override_active = False
    if args.manual_gripper_override is not None:
        print(f"[grip] manual gripper override armed: value={args.manual_gripper_override:.3f} "
              f"(press 'o' in the preview window once the object is firmly "
              f"grasped; the reported gripper state will then be pinned at "
              f"this value until episode end).")
        if not args.preview:
            print("[grip]   WARN --manual_gripper_override requires --preview to "
                  "capture the 'o' keypress. Preview is currently disabled, so "
                  "the override can never trigger.")

    if not args.dry_run:
        # Give the deadman more slack when we're gating each step on user input;
        # otherwise franky_service latches a stop between prompts.
        set_command_timeout(600.0 if args.step_confirm else 1.0)

    period = 1.0 / args.rate
    t_start = time.time()
    # --- panda path state ---
    last_grip_cmd_width = None
    GRIP_CMD_DEADBAND_M = 0.005
    # --- robotiq path state (binary state machine) ---
    # Seed the current state from measured width: closed if fingers are well below max.
    gripper_state = "closed" if width0 < 0.5 * GRIPPER_MAX_WIDTH_M else "open"
    pending_state = gripper_state
    pending_votes = 0
    # --- step-confirm state ---
    # If >0, run this many more steps without prompting (from a 'g N' answer).
    auto_run_remaining = 0
    # If True, keep auto-running until the next inference (chunk boundary),
    # then re-prompt on the inference step itself (from a 'c' answer).
    run_to_next_infer = False
    # --- chunk-tracking state (for [INFER]/[REPLAY] log banners) ---
    # A step is an "inference step" when server_timing.total_ms > 0, meaning
    # the server actually ran the model and produced a fresh chunk. Between
    # inference steps the server just pops the next queued action from that
    # chunk. We use meta.execute_horizon (typically 8) purely for the
    # "REPLAY k/H" position display; the ground truth for inference is the
    # server_timing signal, not our own counter.
    exec_horizon = int(meta.get("execute_horizon", 8)) if isinstance(meta, dict) else 8
    action_horizon = int(meta.get("action_horizon", 16)) if isinstance(meta, dict) else 16
    print(f"[run] chunking: action_horizon={action_horizon}  "
          f"execute_horizon={exec_horizon}  "
          f"(a full chunk covers ~{exec_horizon} steps between inferences)")
    infer_count = 0        # 1-based index of chunks predicted this episode
    chunk_step_idx = 0     # position within the current chunk (0..exec_horizon-1)
    last_infer_ms = None   # ms of the most recent inference call

    # --- JPEG obs negotiation --------------------------------------------------
    if args.jpeg_obs:
        server_accepts_jpeg = (bool(meta.get("accepts_jpeg"))
                               if isinstance(meta, dict) else False)
        if server_accepts_jpeg:
            print(f"[run] --jpeg_obs ON (q{args.jpeg_quality}): image obs sent "
                  f"as JPEG; server advertises accepts_jpeg=True.")
        else:
            print(f"[run] --jpeg_obs ON (q{args.jpeg_quality}) but server does "
                  f"NOT advertise accepts_jpeg. Sending JPEG anyway — if the "
                  f"server can't decode it the request will error. Confirm the "
                  f"server was launched with JPEG-obs support.")

    # --- Decide execution mode -------------------------------------------------
    server_can_chunk = bool(meta.get("chunk_response")) if isinstance(meta, dict) else False
    if args.chunk_executor == "on":
        if not server_can_chunk:
            raise SystemExit(
                "[run] --chunk_executor on but server metadata does not "
                "advertise chunk_response=True. This server returns one action "
                "per request; use --chunk_executor off/auto for the legacy loop."
            )
        chunk_mode = True
    elif args.chunk_executor == "off":
        chunk_mode = False
    else:  # auto
        chunk_mode = server_can_chunk
    if chunk_mode:
        print(f"[run] CHUNK EXECUTOR ON: server returns {exec_horizon}-action "
              f"chunks; dispatching locally at a STRICT {period*1000:.1f} ms "
              f"clock, inference runs sequentially between chunks (arm holds "
              f"the last target during the ~inference-length gap).")
    else:
        print("[run] legacy one-action-per-request loop (no client-side chunk "
              "executor).")

    # --- Nested helpers used by the chunk executor -----------------------------
    def _capture_obs():
        """Read arm+gripper state + camera frame(s) for one inference request.
        Applies --crop_bottom_5 and --flip_cams exactly like the legacy loop."""
        arm_q_l, stopped_l = get_joint_state()
        if args.no_gripper:
            grip_w_l = float(args.dummy_gripper_width_m)
        elif args.gripper == "robotiq":
            try:
                w = get_robotiq_width(args.robotiq_url)
                grip_w_l = (float(args.dummy_gripper_width_m)
                            if not np.isfinite(w) else w)
            except requests.RequestException as e:
                print(f"[run] robotiq state read failed: {e}; dummy width")
                grip_w_l = float(args.dummy_gripper_width_m)
        else:
            grip_w_l = get_gripper_width()
        rgb_l = grab_rgb_square(zed, buf)
        rgb_wrist_l = grab_rgb_square(zed_wrist, buf_wrist) if zed_wrist is not None else None
        if args.crop_bottom_5 and rgb_wrist_l is not None:
            h, w, _ = rgb_wrist_l.shape
            dy = int(round(h * 0.05))
            rgb_wrist_l = np.ascontiguousarray(
                cv2.resize(rgb_wrist_l[:h - dy, :], (w, h),
                           interpolation=cv2.INTER_AREA))
        if args.flip_cams and rgb_wrist_l is not None:
            rgb_l, rgb_wrist_l = rgb_wrist_l, rgb_l
        return arm_q_l, stopped_l, grip_w_l, rgb_l, rgb_wrist_l

    def _gripper_edge(grip_policy):
        """Robotiq binary vote + debounce. Mutates gripper_state/pending_*.
        Returns (vote, edge)."""
        nonlocal gripper_state, pending_state, pending_votes
        vote = "closed" if grip_policy > args.binary_threshold else "open"
        if vote == pending_state:
            pending_votes += 1
        else:
            pending_state = vote
            pending_votes = 1
        edge = False
        if (pending_votes >= max(1, args.gripper_debounce)
                and pending_state != gripper_state):
            gripper_state = pending_state
            edge = True
        return vote, edge

    try:
        step = 0
        while True:
            if abort_flag.is_set():
                break
            if args.steps > 0 and step >= args.steps:
                break

            # ================= CLIENT-SIDE CHUNK EXECUTOR =====================
            if chunk_mode:
                # (1) Capture the observation reflecting the CURRENT robot state
                arm_q, stopped, grip_w, rgb, rgb_wrist = _capture_obs()
                grip_state_scalar = gripper_encode_input(grip_w, args.gripper_input)
                if (args.manual_gripper_override is not None
                        and preview_thread is not None):
                    if (not manual_grip_override_active
                            and preview_thread.override_pressed()):
                        manual_grip_override_active = True
                        print(f"[grip] step {step}: MANUAL OVERRIDE ACTIVATED "
                              f"-> pinned to {args.manual_gripper_override:.3f}")
                    if manual_grip_override_active:
                        grip_state_scalar = float(args.manual_gripper_override)

                if preview_thread is not None:
                    pv = (np.concatenate([rgb, rgb_wrist], axis=1)
                          if rgb_wrist is not None else rgb)
                    hud = f"chunk req @ step {step}"
                    if manual_grip_override_active:
                        hud += f"  [OVR {args.manual_gripper_override:.2f}]"
                    preview_thread.update(np.ascontiguousarray(pv), hud)
                    if preview_thread.q_pressed():
                        print("[run] preview 'q' -> aborting")
                        abort_flag.set()
                        break

                # (2) Request the FULL chunk (one blocking inference)
                obs = {
                    camera_key: rgb,
                    "qpos": {
                        "arm": arm_q.astype(np.float32),
                        "gripper": np.array([grip_state_scalar], dtype=np.float32),
                    },
                    "task": args.task,
                }
                if wrist_camera_key is not None and rgb_wrist is not None:
                    obs[wrist_camera_key] = rgb_wrist
                t_call = time.time()
                chunk_actions = client.get_action_chunk(obs)
                call_ms = (time.time() - t_call) * 1000

                # get_action_chunk returns a list of per-step dicts
                # [{"arm":(7,), "gripper":(G,), "server_timing":...}, ...].
                K = len(chunk_actions)
                arm_chunk = np.stack(
                    [np.asarray(a["arm"], dtype=np.float32).reshape(-1)[:7]
                     for a in chunk_actions], axis=0)                       # (K,7)
                grip_chunk = np.stack(
                    [np.asarray(a["gripper"], dtype=np.float32).reshape(-1)
                     for a in chunk_actions], axis=0)                       # (K,G)
                timing = chunk_actions[0].get("server_timing", {}) or {}
                try:
                    srv_ms = float(timing.get("total_ms", 0) or 0)
                except (TypeError, ValueError):
                    srv_ms = 0.0
                infer_count += 1
                last_infer_ms = srv_ms
                print(f"\n[INFER #{infer_count}] step {step}  srv={srv_ms:.0f}ms "
                      f"wall={call_ms:.0f}ms  -> dispatch {K} action(s) @ "
                      f"{period*1000:.1f}ms (chunk covers {K*period*1000:.0f}ms)")

                # (3) Optional per-chunk confirm gate
                should_send = not args.dry_run
                if args.step_confirm and not args.dry_run:
                    dq0 = float(np.max(np.abs(
                        safety_clip(arm_chunk[0], arm_q, args.max_joint_delta)[0]
                        - arm_q)))
                    try:
                        r = input(
                            f"[confirm chunk #{infer_count} @step {step}]  "
                            f"K={K}  first_arm_delta_max={dq0:.3f} rad\n"
                            f"[Enter]=dispatch  s=skip chunk  q=abort > "
                        ).strip().lower()
                    except EOFError:
                        r = "q"
                    if r == "q":
                        print("[run] step_confirm: 'q' -> aborting")
                        abort_flag.set()
                        break
                    if r == "s":
                        print("[run] step_confirm: skipping this chunk (no send)")
                        should_send = False

                # (4) Dispatch the K actions on a STRICT monotonic clock.
                # ref_q clips action[0] against measured q, then each action
                # against the previous target (limits inter-action jerk without
                # a per-action state read on the critical path).
                ref_q = arm_q.copy()
                chunk_t0 = time.monotonic()
                for k in range(K):
                    if abort_flag.is_set():
                        break
                    if args.steps > 0 and step >= args.steps:
                        break
                    scheduled = chunk_t0 + k * period
                    actual = time.monotonic()
                    jitter_ms = (actual - scheduled) * 1000.0

                    arm_target_raw = arm_chunk[k]
                    grip_policy = float(grip_chunk[k, 0])
                    arm_target, clamped = safety_clip(
                        arm_target_raw, ref_q, args.max_joint_delta)
                    grip_width_cmd = gripper_decode_output(
                        grip_policy, args.gripper_output)

                    binary_vote = None
                    gripper_action_taken = None
                    if should_send:
                        send_arm_target(arm_target.tolist())
                    if args.gripper == "robotiq":
                        binary_vote, edge = _gripper_edge(grip_policy)
                        if should_send and edge and not args.no_gripper:
                            if gripper_state == "closed":
                                send_robotiq_grasp(
                                    width_m=args.grasp_width_m,
                                    force_255=args.robotiq_force_255,
                                    speed_255=args.robotiq_speed_255,
                                    url=args.robotiq_url)
                                gripper_action_taken = (
                                    f"robotiq_grasp(w={args.grasp_width_m:.3f})")
                            else:
                                send_robotiq_open(
                                    speed_255=args.robotiq_speed_255,
                                    url=args.robotiq_url)
                                gripper_action_taken = "robotiq_open()"
                    else:  # panda
                        if should_send and not args.no_gripper:
                            if (last_grip_cmd_width is None
                                    or abs(grip_width_cmd - last_grip_cmd_width)
                                    > GRIP_CMD_DEADBAND_M):
                                send_gripper_move(grip_width_cmd)
                                last_grip_cmd_width = grip_width_cmd
                                gripper_action_taken = f"move({grip_width_cmd:.3f}m)"

                    t_wall = time.time() - t_start
                    logger.write(step, rgb, {
                        "step": step,
                        "t_wall": t_wall,
                        "obs": {
                            "arm": arm_q.tolist(),
                            "gripper_width_m": grip_w,
                            "gripper_state_reported": grip_state_scalar,
                            "gripper_state_scalar": grip_state_scalar,
                            "gripper_input_mode": args.gripper_input,
                            "gripper_manual_override_active": manual_grip_override_active,
                            "gripper_manual_override_value": (
                                float(args.manual_gripper_override)
                                if args.manual_gripper_override is not None else None),
                            "task": args.task,
                            "has_wrist": rgb_wrist is not None,
                            "note": "open-loop chunk frame (obs captured at chunk start)",
                        },
                        "action_raw": {
                            "arm": arm_chunk[k].tolist(),
                            "gripper": grip_policy,
                        },
                        "action_sent": {
                            "arm": arm_target.tolist(),
                            "gripper_width_m": grip_width_cmd,
                            "clamped_by_delta": clamped,
                            "dry_run": args.dry_run,
                            "gripper_preset": args.gripper,
                            "binary_vote": binary_vote,
                            "gripper_state": (gripper_state
                                              if args.gripper == "robotiq" else None),
                            "gripper_action_taken": gripper_action_taken,
                        },
                        "stop_latched": stopped,
                        "call_ms": call_ms if k == 0 else 0.0,
                        "server_timing": timing if k == 0 else {},
                        "chunk": {
                            "mode": "client_executor",
                            "infer_id": infer_count,
                            "step_in_chunk": k + 1,
                            "chunk_size": K,
                            "is_inference_step": (k == 0),
                            "server_inference_ms": srv_ms if k == 0 else None,
                            "exec_horizon": exec_horizon,
                            "action_horizon": action_horizon,
                            # Strict-timing diagnostics (per sim-agent spec):
                            "scheduled_dispatch_s": scheduled - chunk_t0,
                            "dispatch_jitter_ms": jitter_ms,
                            "buffer_remaining": K - k - 1,
                        },
                    }, rgb_wrist=rgb_wrist)

                    ref_q = arm_target
                    step += 1

                    # Strict pacing: sleep to the NEXT absolute deadline
                    # (absolute, not incremental, so jitter doesn't accumulate).
                    next_deadline = chunk_t0 + (k + 1) * period
                    slack = next_deadline - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
                    elif slack < -0.5 * period:
                        print(f"[run] WARN dispatch overrun "
                              f"{-slack*1000:.0f}ms at chunk step {k} "
                              f"(local I/O slower than {period*1000:.0f}ms)")
                continue
            # ================= END CHUNK EXECUTOR =============================

            step_t0 = time.time()
            t_wall = step_t0 - t_start

            arm_q, stopped = get_joint_state()
            if args.no_gripper:
                grip_w = float(args.dummy_gripper_width_m)
            elif args.gripper == "robotiq":
                try:
                    w = get_robotiq_width(args.robotiq_url)
                    grip_w = (float(args.dummy_gripper_width_m)
                              if not np.isfinite(w) else w)
                except requests.RequestException as e:
                    print(f"[run] robotiq state read failed: {e}; "
                          "falling back to dummy width")
                    grip_w = float(args.dummy_gripper_width_m)
            else:
                grip_w = get_gripper_width()
            rgb = grab_rgb_square(zed, buf)
            rgb_wrist = None
            if zed_wrist is not None:
                rgb_wrist = grab_rgb_square(zed_wrist, buf_wrist)
            # --crop_bottom_5: crop 5% off the bottom rows of the physical
            # wrist frame (keep the top 95%, full width) and resize back to
            # the policy image size. Applied BEFORE --flip_cams so it always
            # operates on the physical wrist ZED regardless of assignment.
            if args.crop_bottom_5 and rgb_wrist is not None:
                h, w, _ = rgb_wrist.shape
                dy = int(round(h * 0.05))
                crop = rgb_wrist[:h - dy, :]
                rgb_wrist = np.ascontiguousarray(
                    cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
                )
            # --flip_cams: swap which physical ZED plays "exo" vs "wrist" for
            # this step. Affects preview, model obs, and logged frames since
            # everything downstream keys off (rgb, rgb_wrist).
            if args.flip_cams and rgb_wrist is not None:
                rgb, rgb_wrist = rgb_wrist, rgb
            grip_state_scalar = gripper_encode_input(grip_w, args.gripper_input)
            # --manual_gripper_override: once user presses 'o' in preview,
            # pin the reported gripper state at the override value for the
            # rest of the episode. Print a one-shot notice when we latch.
            if args.manual_gripper_override is not None and preview_thread is not None:
                if not manual_grip_override_active and preview_thread.override_pressed():
                    manual_grip_override_active = True
                    print(f"[grip] step {step}: MANUAL OVERRIDE ACTIVATED — "
                          f"reported gripper state pinned to "
                          f"{args.manual_gripper_override:.3f} for the rest "
                          f"of the episode (was {grip_state_scalar:.3f} at "
                          f"the moment of override).")
                if manual_grip_override_active:
                    grip_state_scalar = float(args.manual_gripper_override)

            # Live preview of the exact frame sent to the model (rendered by
            # PreviewThread; here we just hand it the newest array + HUD text).
            # Under dual_camera we hconcat exo + wrist so both are visible.
            if preview_thread is not None:
                if rgb_wrist is not None:
                    preview_frame = np.ascontiguousarray(
                        np.concatenate([rgb, rgb_wrist], axis=1)
                    )
                else:
                    preview_frame = rgb
                hud = f"step {step}  grip={grip_w:.3f}m"
                if args.manual_gripper_override is not None:
                    if manual_grip_override_active:
                        hud += (f"  [OVERRIDE ACTIVE: "
                                f"gstate={args.manual_gripper_override:.3f}]")
                    else:
                        hud += "  [press 'o' to lock gripper-state]"
                preview_thread.update(preview_frame, hud)
                if preview_thread.q_pressed():
                    print("[run] preview 'q' pressed — aborting episode")
                    abort_flag.set()
                    break

            obs = {
                camera_key: rgb,
                "qpos": {
                    "arm": arm_q.astype(np.float32),
                    "gripper": np.array([grip_state_scalar], dtype=np.float32),
                },
                "task": args.task,
            }
            if wrist_camera_key is not None and rgb_wrist is not None:
                obs[wrist_camera_key] = rgb_wrist
            t_call = time.time()
            action = client.get_action(obs)
            call_ms = (time.time() - t_call) * 1000

            arm_target_raw = np.asarray(action["arm"], dtype=np.float32).reshape(-1)[:7]
            grip_policy = float(np.asarray(action["gripper"]).reshape(-1)[0])
            arm_target, clamped = safety_clip(arm_target_raw, arm_q, args.max_joint_delta)
            grip_width_cmd = gripper_decode_output(grip_policy, args.gripper_output)

            timing = action.get("server_timing", {})
            tag = " CLAMPED" if clamped else ""
            stop_tag = " STOP_LATCHED" if stopped else ""

            # --- Chunk-boundary announcement -----------------------------------
            # The server sets total_ms > 0 whenever it actually invokes the
            # model. Otherwise it popped from the pre-computed chunk buffer.
            srv_ms_raw = timing.get("total_ms", 0)
            try:
                srv_ms = float(srv_ms_raw)
            except (TypeError, ValueError):
                srv_ms = 0.0
            if srv_ms > 0.0:
                infer_count += 1
                chunk_step_idx = 1   # this step IS the first action of the chunk
                last_infer_ms = srv_ms
                print(
                    f"\n[INFER #{infer_count}] step {step}  "
                    f"server inference took {srv_ms:.0f} ms  "
                    f"— next ~{exec_horizon} action(s) served from this chunk "
                    f"until the next [INFER] banner"
                )
            else:
                chunk_step_idx += 1
            chunk_tag = (
                f"[chunk#{max(infer_count,1)} {chunk_step_idx}/{exec_horizon}"
                f"{' INFER' if srv_ms > 0.0 else ' REPLAY'}]"
            )

            # --- Step-confirm gate -------------------------------------------------
            # Pause after action is computed. Skip prompt while --dry_run (nothing
            # is sent anyway) or while auto_run_remaining is burning down.
            should_send = not args.dry_run
            # If we're auto-running to the next inference and this step *is*
            # an inference step, cancel the auto-run so we re-prompt here.
            if run_to_next_infer and srv_ms > 0.0:
                run_to_next_infer = False
                print("[run] step_confirm: reached next inference boundary "
                      f"([INFER #{infer_count}] step {step}) — pausing")
            if args.step_confirm and not args.dry_run:
                if auto_run_remaining > 0:
                    auto_run_remaining -= 1
                elif run_to_next_infer:
                    pass  # keep flying through replay steps silently
                else:
                    dq_max = float(np.max(np.abs(arm_target - arm_q)))
                    kind_banner = (
                        f"*** INFER #{infer_count} (fresh chunk, "
                        f"srv={srv_ms:.0f} ms) ***"
                        if srv_ms > 0.0 else
                        f"    replay {chunk_step_idx}/{exec_horizon} of chunk #"
                        f"{max(infer_count,1)}"
                    )
                    prompt = (
                        f"\n[confirm step {step}]  {kind_banner}\n"
                        f"  arm_delta_max={dq_max:.3f} rad  "
                        f"grip_cmd_width={grip_width_cmd:.3f} m  "
                        f"g_raw={grip_policy:.3f}\n"
                        f"  q_cur   = {arm_q.round(3).tolist()}\n"
                        f"  q_target= {arm_target.round(3).tolist()}\n"
                        f"[Enter]=send  s=skip  c=run-to-next-INFER  "
                        f"'g N'=auto-run N  q=abort > "
                    )
                    try:
                        resp = input(prompt).strip().lower()
                    except EOFError:
                        resp = "q"
                    if resp == "q":
                        print("[run] step_confirm: 'q' -> aborting")
                        abort_flag.set()
                        break
                    elif resp == "s":
                        print("[run] step_confirm: skipping this step (no send)")
                        should_send = False
                    elif resp == "c":
                        run_to_next_infer = True
                        print("[run] step_confirm: running to next inference "
                              "boundary (will pause at the next [INFER] step)")
                    elif resp.startswith("g"):
                        parts = resp.split()
                        try:
                            n = int(parts[1]) if len(parts) > 1 else 10
                        except ValueError:
                            n = 10
                        auto_run_remaining = max(0, n - 1)  # this step also runs
                        print(f"[run] step_confirm: auto-running next {n} step(s)")
                    # else (empty / anything else): fall through and send.

            # --- Gripper dispatch --------------------------------------------------
            # Common to both presets: the *decoded* width_cmd for logging.
            # panda   : send /gripper_move(width_cmd) with a small deadband dedupe.
            # robotiq : threshold policy output -> binary vote,
            #           debounce, and on rising edge issue /gripper_grasp or /gripper_open.
            binary_vote = None       # only meaningful for robotiq preset
            gripper_action_taken = None  # human-readable, for logging

            if args.gripper == "panda":
                print(
                    f"step {step:3d} {chunk_tag}  "
                    f"call={call_ms:5.0f}ms srv={timing.get('total_ms','?'):>4}ms  "
                    f"q[0..2]={arm_target[:3].round(2).tolist()}  "
                    f"grip_cmd={grip_width_cmd:.3f}m (g={grip_policy:.3f}){tag}{stop_tag}"
                )
                if should_send:
                    send_arm_target(arm_target.tolist())
                    if args.no_gripper:
                        gripper_action_taken = (
                            f"SKIPPED move({grip_width_cmd:.3f}m) [--no_gripper]"
                        )
                    elif (last_grip_cmd_width is None
                            or abs(grip_width_cmd - last_grip_cmd_width) > GRIP_CMD_DEADBAND_M):
                        send_gripper_move(grip_width_cmd)
                        last_grip_cmd_width = grip_width_cmd
                        gripper_action_taken = f"move({grip_width_cmd:.3f}m)"

            else:  # robotiq preset
                vote = "closed" if grip_policy > args.binary_threshold else "open"
                binary_vote = vote
                # Debounce: require N consecutive same votes before flipping state.
                if vote == pending_state:
                    pending_votes += 1
                else:
                    pending_state = vote
                    pending_votes = 1
                edge = False
                if (pending_votes >= max(1, args.gripper_debounce)
                        and pending_state != gripper_state):
                    gripper_state = pending_state
                    edge = True

                print(
                    f"step {step:3d} {chunk_tag}  "
                    f"call={call_ms:5.0f}ms srv={timing.get('total_ms','?'):>4}ms  "
                    f"q[0..2]={arm_target[:3].round(2).tolist()}  "
                    f"g_raw={grip_policy:.2f} vote={vote} state={gripper_state}"
                    f"{' EDGE' if edge else ''}{tag}{stop_tag}"
                )
                if should_send:
                    send_arm_target(arm_target.tolist())
                    if edge:
                        if args.no_gripper:
                            gripper_action_taken = (
                                f"SKIPPED {'grasp' if gripper_state=='closed' else 'open'}"
                                f" [--no_gripper]"
                            )
                        elif gripper_state == "closed":
                            # Route to the Robotiq 2F-85 HTTP service.
                            send_robotiq_grasp(
                                width_m=args.grasp_width_m,
                                force_255=args.robotiq_force_255,
                                speed_255=args.robotiq_speed_255,
                                url=args.robotiq_url,
                            )
                            gripper_action_taken = (
                                f"robotiq_grasp(w={args.grasp_width_m:.3f}m,"
                                f"F={args.robotiq_force_255})"
                            )
                        else:
                            send_robotiq_open(
                                speed_255=args.robotiq_speed_255,
                                url=args.robotiq_url,
                            )
                            gripper_action_taken = "robotiq_open()"
            # ----------------------------------------------------------------------

            logger.write(step, rgb, {
                "step": step,
                "t_wall": t_wall,
                "obs": {
                    "arm": arm_q.tolist(),
                    "gripper_width_m": grip_w,
                    # The scalar we sent to the policy as obs.qpos.gripper[0].
                    # Encoding is `args.gripper_input` (also saved in
                    # run_meta.json). Kept side-by-side with the raw width so
                    # offline analysis can (a) verify the on-the-fly encoding,
                    # (b) rebuild θ under a different convention without a
                    # rerun, (c) compare against the sim training distribution
                    # of qpos.gripper[0]. Alias `gripper_state_scalar` retained
                    # for back-compat with older log parsers.
                    "gripper_state_reported": grip_state_scalar,
                    "gripper_state_scalar":   grip_state_scalar,
                    "gripper_input_mode":     args.gripper_input,
                    # --manual_gripper_override state (True after the user
                    # pressed 'o' in preview; false otherwise). When True,
                    # gripper_state_reported above is pinned at
                    # args.manual_gripper_override rather than the true
                    # encoded value from grip_w. Both remain visible for
                    # offline analysis.
                    "gripper_manual_override_active": manual_grip_override_active,
                    "gripper_manual_override_value": (
                        float(args.manual_gripper_override)
                        if args.manual_gripper_override is not None else None
                    ),
                    "task": args.task,
                    "has_wrist": rgb_wrist is not None,
                },
                "action_raw": {
                    "arm": arm_target_raw.tolist(),
                    "gripper": grip_policy,
                },
                "action_sent": {
                    "arm": arm_target.tolist(),
                    "gripper_width_m": grip_width_cmd,
                    "clamped_by_delta": clamped,
                    "dry_run": args.dry_run,
                    "gripper_preset": args.gripper,
                    "binary_vote": binary_vote,
                    "gripper_state": (gripper_state
                                      if args.gripper == "robotiq" else None),
                    "gripper_action_taken": gripper_action_taken,
                },
                "stop_latched": stopped,
                "call_ms": call_ms,
                "server_timing": timing,
                "chunk": {
                    "infer_id": infer_count,       # 1-based chunk number
                    "step_in_chunk": chunk_step_idx,  # 1..exec_horizon
                    "is_inference_step": srv_ms > 0.0,
                    "server_inference_ms": srv_ms if srv_ms > 0.0 else None,
                    "exec_horizon": exec_horizon,
                    "action_horizon": action_horizon,
                },
            }, rgb_wrist=rgb_wrist)

            # Pace loop
            elapsed = time.time() - step_t0
            if elapsed < period:
                time.sleep(period - elapsed)
            elif elapsed > 2 * period and not args.step_confirm:
                print(f"[run] WARN step took {elapsed*1000:.0f}ms > {2*period*1000:.0f}ms")
            step += 1
    finally:
        wall = time.time() - t_start
        print(f"[run] episode ended after {wall:.1f}s")
        # Chunk stats: how many times did the model actually run, and roughly
        # what average action budget did each chunk cover?
        if infer_count > 0:
            avg_steps_per_chunk = step / infer_count if step > 0 else 0.0
            print(f"[run] chunks: {infer_count} inference call(s) served "
                  f"{step} step(s) -> ~{avg_steps_per_chunk:.1f} step(s)/chunk "
                  f"(exec_horizon={exec_horizon})")
        if not args.dry_run:
            stop_robot()
        try:
            client.close()
        except Exception:
            pass
        try:
            zed.close()
        except Exception:
            pass
        if zed_wrist is not None:
            try:
                zed_wrist.close()
            except Exception:
                pass
        if preview_thread is not None:
            preview_thread.stop()
            preview_thread.join(timeout=2.0)
        logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
