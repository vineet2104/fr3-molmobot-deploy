import logging
import os
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pyrobotiqgripper import RobotiqGripper

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Robotiq Gripper Service", version="0.1.0")


class _State:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.gripper: RobotiqGripper | None = None
        self.last_connect_error: str | None = None
        self.calibrating: bool = False
        self.motion_thread: threading.Thread | None = None
        self.motion: dict[str, Any] = {
            "busy": False,
            "last_action": None,  # "open" | "close" | "go_to" | "go_to_mm" | ...
            "last_request": None,
            "last_result": None,
            "last_error": None,
        }
        self.ui_session: str | None = None
        self.ui_session_last_seen: float = 0.0


STATE = _State()


def _require_gripper() -> RobotiqGripper:
    with STATE.lock:
        if STATE.gripper is None:
            detail = {"error": "Gripper not connected", "last_connect_error": STATE.last_connect_error}
            raise HTTPException(status_code=503, detail=detail)
        return STATE.gripper


def _calibrate_gripper(g: RobotiqGripper) -> None:
    """Calibrate and reset the stuck processing flag left by pyRobotiqGripper."""
    logger.info("Calibrating gripper (2F-85: close=0mm, open=85mm)...")
    with STATE.lock:
        STATE.calibrating = True
    try:
        g.calibrate(0, 85)
        logger.info("Gripper calibration complete.")
    except Exception as e:
        logger.error("Gripper calibration failed: %s", e)
    finally:
        # pyRobotiqGripper.goTo() sets processing=True but never resets it.
        g.processing = False
        with STATE.lock:
            STATE.calibrating = False


def _connect() -> None:
    port = os.getenv("GRIPPER_PORT", "auto")
    slave_addr = int(os.getenv("GRIPPER_SLAVE_ADDRESS", "9"))
    with STATE.lock:
        # Release any existing gripper first so its serial port is free before we
        # probe again. Otherwise, on port="auto", the new RobotiqGripper() will
        # try to open /dev/ttyUSB0 which is still held by the old instance,
        # fail on every port, raise, and wipe out the previously-working handle.
        old = STATE.gripper
        STATE.gripper = None
        if old is not None:
            try:
                old.disconnect()
            except Exception as e:
                logger.warning("Ignoring error while disconnecting old gripper: %s", e)
        try:
            STATE.gripper = RobotiqGripper(portname=port, slaveAddress=slave_addr)
            STATE.last_connect_error = None
        except Exception as e:
            STATE.gripper = None
            STATE.last_connect_error = repr(e)
    if STATE.gripper is not None:
        g = STATE.gripper
        try:
            if not g.isActivated():
                logger.info("Activating gripper...")
                g.activate()
                logger.info("Gripper activation complete.")
        except Exception as e:
            logger.error("Gripper activation failed: %s", e)
        _calibrate_gripper(g)


@app.on_event("startup")
def _startup() -> None:
    _connect()


class GoToBitsRequest(BaseModel):
    position: int = Field(..., ge=0, le=255)
    speed: int = Field(255, ge=0, le=255)
    force: int = Field(255, ge=0, le=255)


class GoToMmRequest(BaseModel):
    position_mm: float
    speed: int = Field(255, ge=0, le=255)
    force: int = Field(255, ge=0, le=255)


class CalibrateRequest(BaseModel):
    close_mm: float
    open_mm: float


def _motion_busy_locked(g: RobotiqGripper) -> bool:
    # `pyRobotiqGripper` maintains `processing`; also track a server-side busy bit.
    # Calibration also holds the serial port, so treat it as busy.
    try:
        processing = bool(getattr(g, "processing", False))
    except Exception:
        processing = False
    return bool(STATE.motion.get("busy")) or processing or STATE.calibrating


def _start_motion(action: str, request: dict[str, Any], fn) -> dict[str, Any]:
    """
    Start motion in a background thread and return immediately.
    Reject if a motion is already running.
    """
    g = _require_gripper()
    with STATE.lock:
        if _motion_busy_locked(g):
            raise HTTPException(status_code=409, detail={"error": "Gripper busy"})

        STATE.motion.update(
            {
                "busy": True,
                "last_action": action,
                "last_request": dict(request),
                "last_result": None,
                "last_error": None,
            }
        )

        def _run() -> None:
            # NOTE: Do NOT hold STATE.lock across the entire motion duration.
            # `pyRobotiqGripper.goTo(...)` can block until motion completes; if we
            # hold the lock while waiting, clients polling `/gripper_state` will
            # block too, which can pause real-time control loops.
            try:
                # Execute the blocking motion without holding STATE.lock.
                # While `STATE.motion["busy"]` is True, endpoints should avoid
                # issuing additional gripper I/O.
                res = fn()
                with STATE.lock:
                    STATE.motion["last_result"] = res
            except Exception as e:
                with STATE.lock:
                    STATE.motion["last_error"] = repr(e)
            finally:
                with STATE.lock:
                    # pyRobotiqGripper sets `processing=True` at motion start but may not reset it
                    # on exceptions. Ensure we don't get stuck in a permanently-busy state.
                    try:
                        setattr(g, "processing", False)
                    except Exception:
                        pass
                    STATE.motion["busy"] = False

        t = threading.Thread(target=_run, daemon=True)
        STATE.motion_thread = t
        t.start()

        return {"ok": True, "accepted": True, "busy": True, "action": action, "request": dict(request)}


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    # Minimal single-file UI: slider drives /go_to.
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Robotiq Gripper</title>
  <style>
    :root { color-scheme: dark; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 0; background: #0b1020; color: #e8eefc; }
    .wrap { max-width: 820px; margin: 0 auto; padding: 28px 18px; }
    .card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; padding: 16px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
    .row > * { flex: 0 0 auto; }
    h1 { font-size: 18px; margin: 0 0 12px; }
    button { background: #2c6cff; color: #fff; border: 0; padding: 9px 12px; border-radius: 10px; cursor: pointer; font-weight: 600; }
    button.secondary { background: rgba(255,255,255,0.12); }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    input[type="range"] { width: min(520px, 92vw); }
    input, code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .pill { padding: 6px 10px; border-radius: 999px; background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.10); }
    .muted { color: rgba(232,238,252,0.75); }
    .log { white-space: pre-wrap; font-size: 12px; line-height: 1.35; max-height: 240px; overflow: auto; margin-top: 12px; padding: 10px; border-radius: 12px; background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.08); }
    .kv { display: grid; grid-template-columns: 160px 1fr; gap: 6px 12px; margin-top: 12px; }
    .kv div { padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Robotiq Gripper Service</h1>
      <div class="row" style="margin-bottom: 10px;">
        <button id="btnConnect" class="secondary">Connect</button>
        <button id="btnActivate">Activate</button>
        <button id="btnCalibrate">Calibrate</button>
        <span class="pill">Connected: <strong id="connected">?</strong></span>
        <span class="pill">Activated: <strong id="activated">?</strong></span>
        <span class="pill">Calibrated: <strong id="calibrated">?</strong></span>
      </div>

      <div class="row" style="margin: 12px 0 6px;">
        <label class="muted" for="pos">Position (bits 0=open, 255=close)</label>
      </div>
      <div class="row">
        <input id="pos" type="range" min="0" max="255" step="1" value="0" />
        <span class="pill">pos=<strong id="posVal">0</strong></span>
      </div>
      <div class="row" style="margin-top: 10px;">
        <label class="muted">speed</label>
        <input id="speed" type="number" min="0" max="255" step="1" value="255" style="width: 90px;" />
        <label class="muted">force</label>
        <input id="force" type="number" min="0" max="255" step="1" value="255" style="width: 90px;" />
        <button id="btnSend" class="secondary">Send</button>
      </div>

      <div class="kv">
        <div class="muted">Current position (bits)</div><div><code id="curPos">?</code></div>
        <div class="muted">Current position (mm)</div><div><code id="curPosMm">?</code></div>
        <div class="muted">Last response</div><div><code id="lastResp">—</code></div>
      </div>

      <div id="log" class="log"></div>
      <div class="muted" style="margin-top: 10px; font-size: 12px;">
        Tip: drag the slider; it will send at most ~10 requests/sec while dragging.
      </div>
    </div>
  </div>

<script>
  const el = (id) => document.getElementById(id);
  const logEl = el("log");
  const pos = el("pos");
  const posVal = el("posVal");
  const btnConnect = el("btnConnect");
  const btnActivate = el("btnActivate");
  const btnSend = el("btnSend");

  function log(line) {
    const ts = new Date().toISOString().slice(11, 19);
    logEl.textContent = `[${ts}] ${line}\n` + logEl.textContent;
  }

  async function jfetch(path, opts) {
    const r = await fetch(path, Object.assign({ headers: {"Content-Type": "application/json"} }, opts || {}));
    const txt = await r.text();
    let json = null;
    try { json = txt ? JSON.parse(txt) : null; } catch (e) {}
    if (!r.ok) {
      const msg = json ? JSON.stringify(json) : txt;
      throw new Error(`${r.status} ${r.statusText}: ${msg}`);
    }
    return json;
  }

  async function refresh() {
    try {
      const h = await jfetch("/health");
      el("connected").textContent = String(!!h.connected);
      el("activated").textContent = String(!!h.is_activated);
      el("calibrated").textContent = String(!!h.is_calibrated);

      try {
        const p = await jfetch("/position");
        el("curPos").textContent = String(p.position);
      } catch (_) {}
      try {
        const pm = await jfetch("/position_mm");
        el("curPosMm").textContent = pm.position_mm != null ? String(pm.position_mm) : (pm.calibrated ? "busy" : "n/a");
      } catch (_) {}
    } catch (e) {
      el("connected").textContent = "false";
      el("activated").textContent = "?";
      el("calibrated").textContent = "?";
      el("curPos").textContent = "?";
      el("curPosMm").textContent = "?";
      log(`health error: ${e.message}`);
    }
  }

  let lastSentAt = 0;
  let pending = null;
  async function sendGoTo(position) {
    const now = performance.now();
    const minIntervalMs = 100; // throttle to ~10Hz
    if (now - lastSentAt < minIntervalMs) {
      pending = position;
      return;
    }
    lastSentAt = now;

    const speed = Number(el("speed").value || 255);
    const force = Number(el("force").value || 255);
    try {
      const resp = await jfetch("/go_to?wait=false", { method: "POST", body: JSON.stringify({ position, speed, force }) });
      el("lastResp").textContent = JSON.stringify(resp);
    } catch (e) {
      el("lastResp").textContent = e.message;
      log(`go_to error: ${e.message}`);
    }
  }

  function scheduleFlush() {
    if (pending === null) return;
    const p = pending;
    pending = null;
    sendGoTo(p);
  }

  pos.addEventListener("input", () => {
    const v = Number(pos.value);
    posVal.textContent = String(v);
    sendGoTo(v);
    setTimeout(scheduleFlush, 120);
  });

  btnSend.addEventListener("click", () => {
    const v = Number(pos.value);
    sendGoTo(v);
  });

  btnConnect.addEventListener("click", async () => {
    try {
      const resp = await jfetch("/connect", { method: "POST" });
      el("lastResp").textContent = JSON.stringify(resp);
      log("connect ok");
      await refresh();
    } catch (e) {
      el("lastResp").textContent = e.message;
      log(`connect error: ${e.message}`);
    }
  });

  btnActivate.addEventListener("click", async () => {
    try {
      const resp = await jfetch("/activate", { method: "POST" });
      el("lastResp").textContent = JSON.stringify(resp);
      log("activate ok");
      await refresh();
    } catch (e) {
      el("lastResp").textContent = e.message;
      log(`activate error: ${e.message}`);
    }
  });

  el("btnCalibrate").addEventListener("click", async () => {
    try {
      const resp = await jfetch("/calibrate", { method: "POST", body: JSON.stringify({ close_mm: 0, open_mm: 85 }) });
      el("lastResp").textContent = JSON.stringify(resp);
      log("calibrate ok (2F-85: close=0mm, open=85mm)");
      await refresh();
    } catch (e) {
      el("lastResp").textContent = e.message;
      log(`calibrate error: ${e.message}`);
    }
  });

  let sessionId = null;
  let pollTimer = null;

  async function claimSession() {
    try {
      const resp = await jfetch("/ui_session", { method: "POST" });
      sessionId = resp.session;
      log("UI session claimed");
      refresh();
      pollTimer = setInterval(async () => {
        try {
          await jfetch(`/ui_heartbeat?session=${sessionId}`, { method: "POST" });
          await refresh();
        } catch (e) {
          clearInterval(pollTimer);
          log(`Session lost: ${e.message}`);
          document.body.innerHTML = '<div style="padding:40px;text-align:center;color:#ff6b6b;font-size:20px;font-family:sans-serif;">Another UI session is already active.<br>Close the other tab and reload this page.</div>';
        }
      }, 1000);
    } catch (e) {
      log(`Failed to claim session: ${e.message}`);
      document.body.innerHTML = '<div style="padding:40px;text-align:center;color:#ff6b6b;font-size:20px;font-family:sans-serif;">Another UI session is already active.<br>Close the other tab and reload this page.</div>';
    }
  }

  claimSession();
</script>
</body>
</html>
"""


_UI_SESSION_TIMEOUT = 5.0  # seconds of inactivity before session expires


@app.post("/ui_session")
def ui_session() -> dict[str, Any]:
    """Claim a UI session. Only one UI client allowed at a time."""
    with STATE.lock:
        now = time.monotonic()
        # Expire stale session
        if STATE.ui_session is not None and (now - STATE.ui_session_last_seen) > _UI_SESSION_TIMEOUT:
            STATE.ui_session = None
        if STATE.ui_session is not None:
            raise HTTPException(status_code=409, detail="Another UI session is already active. Close the other tab and reload.")
        STATE.ui_session = uuid.uuid4().hex
        STATE.ui_session_last_seen = now
        return {"session": STATE.ui_session}


@app.post("/ui_heartbeat")
def ui_heartbeat(session: str = Query(...)) -> dict[str, Any]:
    """Heartbeat from the active UI client. Rejects unknown sessions."""
    with STATE.lock:
        now = time.monotonic()
        if STATE.ui_session is not None and (now - STATE.ui_session_last_seen) > _UI_SESSION_TIMEOUT:
            STATE.ui_session = None
        if STATE.ui_session is None or STATE.ui_session != session:
            raise HTTPException(status_code=409, detail="Another UI session is active, or session expired. Close other tabs and reload.")
        STATE.ui_session_last_seen = now
        return {"ok": True}


@app.get("/info")
def info() -> dict[str, Any]:
    with STATE.lock:
        connected = STATE.gripper is not None
    return {"service": app.title, "version": app.version, "connected": connected}


@app.get("/health")
def health() -> dict[str, Any]:
    with STATE.lock:
        g = STATE.gripper
        err = STATE.last_connect_error
        if g is None:
            return {"ok": False, "connected": False, "last_connect_error": err}
        if _motion_busy_locked(g):
            return {"ok": True, "connected": True, "is_activated": None, "is_calibrated": None, "busy": True}
        return {"ok": True, "connected": True, "is_activated": g.isActivated(), "is_calibrated": g.isCalibrated()}

@app.get("/gripper_state")
def gripper_state(*, closed_threshold: int = Query(128, ge=0, le=255)) -> dict[str, Any]:
    """
    Lightweight state endpoint for clients:
      - position_bits: gPO (0=open, 255=close)
      - is_closed: position_bits >= closed_threshold
      - is_open: position_bits < closed_threshold
      - busy: server/gripper busy indicator
    """
    g = _require_gripper()
    with STATE.lock:
        busy = _motion_busy_locked(g)

        # Avoid hammering the bus while a motion is in progress.
        if busy:
            return {
                "connected": True,
                "position_bits": None,
                "max_position_bits": 255,
                "position_frac": None,
                "is_closed": None,
                "is_open": None,
                "is_activated": None,
                "is_calibrated": None,
                "busy": True,
            }

        try:
            pos = int(g.getPosition())
        except Exception as e:
            # Treat as transient (serial contention / device hiccup).
            raise HTTPException(status_code=503, detail={"error": "Failed to read gripper position", "exc": repr(e)})

        try:
            activated = bool(g.isActivated())
        except Exception:
            activated = None

        try:
            calibrated = bool(g.isCalibrated())
        except Exception:
            calibrated = None

    is_closed = bool(pos >= int(closed_threshold))
    max_position_bits = 255
    position_frac = float(pos) / float(max_position_bits)
    return {
        "connected": True,
        "position_bits": pos,
        "max_position_bits": max_position_bits,
        "position_frac": position_frac,
        "is_closed": is_closed,
        "is_open": not is_closed,
        "is_activated": activated,
        "is_calibrated": calibrated,
        "busy": busy,
    }


@app.post("/connect")
def connect() -> dict[str, Any]:
    _connect()
    with STATE.lock:
        return {
            "connected": STATE.gripper is not None,
            "last_connect_error": STATE.last_connect_error,
        }


@app.post("/reset")
def reset() -> dict[str, Any]:
    g = _require_gripper()
    with STATE.lock:
        g.reset()
    return {"ok": True}


@app.post("/activate")
def activate() -> dict[str, Any]:
    g = _require_gripper()
    with STATE.lock:
        if not g.isActivated():
            g.activate()
        activated = g.isActivated()
    return {"ok": True, "is_activated": activated}


@app.post("/reset_activate")
def reset_activate() -> dict[str, Any]:
    g = _require_gripper()
    with STATE.lock:
        g.resetActivate()
        activated = g.isActivated()
    return {"ok": True, "is_activated": activated}


@app.post("/open")
def open_gripper(req: GoToBitsRequest | None = None, wait: bool = Query(True)) -> dict[str, Any]:
    g = _require_gripper()
    if req is None:
        req = GoToBitsRequest(position=0)
    if not wait:
        return _start_motion(
            "open",
            {"position": 0, "speed": int(req.speed), "force": int(req.force)},
            lambda: g.goTo(0, speed=req.speed, force=req.force),
        )
    with STATE.lock:
        end_pos, object_detected = g.goTo(0, speed=req.speed, force=req.force)
        g.processing = False  # goTo() never resets this
        # Track last commanded action even in blocking mode.
        STATE.motion.update({"last_action": "open", "last_request": {"position": 0, "speed": int(req.speed), "force": int(req.force)}})
    return {"ok": True, "position": end_pos, "object_detected": object_detected, "accepted": False}


@app.post("/close")
def close_gripper(req: GoToBitsRequest | None = None, wait: bool = Query(True)) -> dict[str, Any]:
    g = _require_gripper()
    if req is None:
        req = GoToBitsRequest(position=255)
    if not wait:
        return _start_motion(
            "close",
            {"position": 255, "speed": int(req.speed), "force": int(req.force)},
            lambda: g.goTo(255, speed=req.speed, force=req.force),
        )
    with STATE.lock:
        end_pos, object_detected = g.goTo(255, speed=req.speed, force=req.force)
        g.processing = False
        STATE.motion.update({"last_action": "close", "last_request": {"position": 255, "speed": int(req.speed), "force": int(req.force)}})
    return {"ok": True, "position": end_pos, "object_detected": object_detected, "accepted": False}


@app.post("/go_to")
def go_to(req: GoToBitsRequest, wait: bool = Query(True)) -> dict[str, Any]:
    g = _require_gripper()
    if not wait:
        return _start_motion(
            "go_to",
            {"position": int(req.position), "speed": int(req.speed), "force": int(req.force)},
            lambda: g.goTo(req.position, speed=req.speed, force=req.force),
        )
    with STATE.lock:
        end_pos, object_detected = g.goTo(req.position, speed=req.speed, force=req.force)
        g.processing = False
        STATE.motion.update(
            {"last_action": "go_to", "last_request": {"position": int(req.position), "speed": int(req.speed), "force": int(req.force)}}
        )
    return {"ok": True, "position": end_pos, "object_detected": object_detected, "accepted": False}


@app.post("/go_to_mm")
def go_to_mm(req: GoToMmRequest, wait: bool = Query(True)) -> dict[str, Any]:
    g = _require_gripper()
    if not wait:
        return _start_motion(
            "go_to_mm",
            {"position_mm": float(req.position_mm), "speed": int(req.speed), "force": int(req.force)},
            lambda: g.goTomm(req.position_mm, speed=req.speed, force=req.force),
        )
    with STATE.lock:
        g.goTomm(req.position_mm, speed=req.speed, force=req.force)
        g.processing = False
        pos_mm = g.getPositionmm()
        STATE.motion.update(
            {
                "last_action": "go_to_mm",
                "last_request": {"position_mm": float(req.position_mm), "speed": int(req.speed), "force": int(req.force)},
            }
        )
    return {"ok": True, "position_mm": pos_mm, "accepted": False}


@app.post("/calibrate")
def calibrate(req: CalibrateRequest) -> dict[str, Any]:
    g = _require_gripper()
    with STATE.lock:
        g.calibrate(req.close_mm, req.open_mm)
        g.processing = False
        calibrated = g.isCalibrated()
    return {"ok": True, "is_calibrated": calibrated}


@app.get("/is_activated")
def is_activated() -> dict[str, Any]:
    g = _require_gripper()
    with STATE.lock:
        busy = _motion_busy_locked(g)
        if busy:
            return {"is_activated": None, "busy": True}
        return {"is_activated": g.isActivated(), "busy": False}


@app.get("/is_calibrated")
def is_calibrated() -> dict[str, Any]:
    g = _require_gripper()
    with STATE.lock:
        busy = _motion_busy_locked(g)
        if busy:
            return {"is_calibrated": None, "busy": True}
        return {"is_calibrated": g.isCalibrated(), "busy": False}


@app.get("/position")
def position() -> dict[str, Any]:
    g = _require_gripper()
    with STATE.lock:
        busy = _motion_busy_locked(g)
        if busy:
            return {"position": None, "busy": True}
        return {"position": g.getPosition(), "busy": False}


@app.get("/position_mm")
def position_mm() -> dict[str, Any]:
    g = _require_gripper()
    with STATE.lock:
        if not g.isCalibrated():
            return {"position_mm": None, "calibrated": False, "busy": False}
        busy = _motion_busy_locked(g)
        if busy:
            return {"position_mm": None, "calibrated": True, "busy": True}
        return {"position_mm": g.getPositionmm(), "calibrated": True, "busy": False}


@app.get("/status")
def status() -> dict[str, Any]:
    g = _require_gripper()
    with STATE.lock:
        if _motion_busy_locked(g):
            motion = dict(STATE.motion)
            motion["busy"] = True
            return {"param": None, "decoded": None, "motion": motion}
        g.readAll()
        # Return raw registers + human-readable lookup maps where possible.
        param = dict(g.paramDic)
        decoded = {k: g.registerDic.get(k, {}).get(v) for k, v in param.items()}
        motion = dict(STATE.motion)
        motion["busy"] = bool(motion.get("busy")) or bool(getattr(g, "processing", False))
    return {"param": param, "decoded": decoded, "motion": motion}


def main():
    """Entry point for gripper-service CLI."""
    import uvicorn

    uvicorn.run("droid_plus.services.gripper_service:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "54323")))


if __name__ == "__main__":
    main()
