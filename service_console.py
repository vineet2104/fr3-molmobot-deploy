"""fr3-molmobot-deploy — lab web console (workstation).

A small, self-contained FastAPI page to run MolmoBot / MolmoBot-FT rollouts
against a MODEL SERVER RUNNING ELSEWHERE (the GPU host). Unlike the reference
console it does NOT auto-deploy models locally — the GPU host owns the model.

It:
  - lists policies (official / finetuned / custom endpoint),
  - runs a preflight (arm + gripper + model reachable),
  - launches the matching client/ rollout script as a subprocess (with the task
    and endpoint from the UI), streams its log, and shows the latest frame,
  - Stop kills the rollout.

Endpoints come from the environment (source configs/robot.env first) and can be
overridden per-run in the UI.

Run:  python service_console.py --port 7070   ->   http://<workstation-ip>:7070
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

ROOT = Path(__file__).resolve().parent
CLIENT = ROOT / "client"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
PY = sys.executable

# --- endpoints (env-driven; override in UI) ---
FRANKY_URL = os.environ.get("FRANKY_URL", "http://127.0.0.1:54321")
HAND_URL = os.environ.get("HAND_URL", "http://127.0.0.1:54324")
ROBOTIQ_URL = os.environ.get("ROBOTIQ_URL", "http://127.0.0.1:54323")
GRIPPER_BACKEND = os.environ.get("GRIPPER_BACKEND", "franka_hand")
MODEL_HOST = os.environ.get("MOLMOBOTFT_A100_HOST", "127.0.0.1")
MODEL_PORT = os.environ.get("MOLMOBOTFT_A100_PORT", "8000")

POLICIES = {
    "molmobot_ft": {
        "label": "MolmoBot Finetuned (remote GPU host)",
        "script": "run_molmobot_ft.py",
        "gripper": "panda",
        "default_task": "Pick up the pineapple slices can",
    },
    "molmobot_official": {
        "label": "MolmoBot Official — Img-DROID (remote GPU host)",
        "script": "run_molmobot_official.py",
        "gripper": "panda",
        "default_task": "Pick up the object",
    },
}

# ---------------- runtime state ----------------
PROC: dict = {"popen": None, "log_dir": None, "log_path": None, "policy": None}
app = FastAPI(title="fr3-molmobot console")


def _get(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode()
            try:
                return r.status, json.loads(body)
            except Exception:
                return r.status, body
    except Exception as e:
        return None, str(e)


def _running() -> bool:
    p = PROC["popen"]
    return p is not None and p.poll() is None


@app.get("/api/policies")
def api_policies():
    return {"policies": {k: v["label"] for k, v in POLICIES.items()},
            "defaults": {"model_host": MODEL_HOST, "model_port": MODEL_PORT,
                         "franky_url": FRANKY_URL, "hand_url": HAND_URL,
                         "gripper_backend": GRIPPER_BACKEND}}


@app.get("/api/preflight")
def api_preflight(model_host: str = MODEL_HOST, model_port: str = MODEL_PORT):
    out = {}
    st, js = _get(f"{FRANKY_URL}/health")
    out["arm"] = {"ok": st == 200, "detail": js}
    st, js = _get(f"{FRANKY_URL}/joint_state")
    out["joint_state"] = {"ok": st == 200 and isinstance(js, dict) and len(js.get("positions", [])) == 7,
                          "detail": js if st != 200 else "7 joints ok"}
    gurl = ROBOTIQ_URL if GRIPPER_BACKEND == "robotiq" else HAND_URL
    if GRIPPER_BACKEND == "none":
        out["gripper"] = {"ok": True, "detail": "disabled"}
    else:
        st, js = _get(f"{gurl}/health")
        out["gripper"] = {"ok": st == 200, "detail": js}
    st, body = _get(f"http://{model_host}:{model_port}/healthz")
    out["model"] = {"ok": st == 200, "detail": body}
    out["all_ok"] = all(v.get("ok") for v in out.values() if isinstance(v, dict))
    return out


@app.post("/api/run")
async def api_run(req: Request):
    if _running():
        return JSONResponse({"ok": False, "error": "a rollout is already running"}, status_code=409)
    body = await req.json()
    policy = body.get("policy", "molmobot_ft")
    spec = POLICIES.get(policy)
    if not spec:
        return JSONResponse({"ok": False, "error": f"unknown policy {policy}"}, status_code=400)
    task = (body.get("task") or spec["default_task"]).strip()
    model_host = (body.get("model_host") or MODEL_HOST).strip()
    model_port = str(body.get("model_port") or MODEL_PORT).strip()
    front_cam = (body.get("front_view_cam_id") or "exo_front").strip()
    extra = (body.get("extra_flags") or "").split()

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_dir = LOGS / f"{ts}_{policy}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "console.log"

    env = dict(os.environ)
    env["MOLMOBOTFT_A100_HOST"] = model_host
    env["MOLMOBOTFT_A100_PORT"] = model_port
    cmd = [PY, str(CLIENT / spec["script"]),
           "--task", task,
           "--gripper", spec["gripper"],
           "--front_view_cam_id", front_cam,
           "--log_dir", str(log_dir)] + extra

    lf = open(log_path, "w")
    lf.write(f"# {' '.join(cmd)}\n# model={model_host}:{model_port}\n"); lf.flush()
    popen = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                             stdout=lf, stderr=subprocess.STDOUT,
                             start_new_session=True)
    PROC.update({"popen": popen, "log_dir": log_dir, "log_path": log_path, "policy": policy})
    return {"ok": True, "pid": popen.pid, "log_dir": str(log_dir), "cmd": cmd}


@app.post("/api/stop")
def api_stop():
    p = PROC["popen"]
    if p is None or p.poll() is not None:
        return {"ok": True, "note": "not running"}
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)  # let the client stop the arm cleanly
        try:
            p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/status")
def api_status():
    p = PROC["popen"]
    return {"running": _running(), "policy": PROC["policy"],
            "log_dir": str(PROC["log_dir"]) if PROC["log_dir"] else None,
            "returncode": (p.poll() if p else None)}


@app.get("/api/log")
def api_log(lines: int = 200):
    lp = PROC["log_path"]
    if not lp or not Path(lp).exists():
        return {"log": ""}
    with open(lp, errors="replace") as f:
        return {"log": "".join(f.readlines()[-lines:])}


@app.get("/api/frame")
def api_frame():
    ld = PROC["log_dir"]
    if not ld:
        return Response(status_code=204)
    frames = sorted((Path(ld) / "frames").glob("*.jpg")) if (Path(ld) / "frames").exists() else []
    if not frames:
        return Response(status_code=204)
    return Response(content=frames[-1].read_bytes(), media_type="image/jpeg")


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>fr3-molmobot console</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,sans-serif;margin:0;background:#0e1116;color:#e6edf3}
.wrap{max-width:1000px;margin:0 auto;padding:16px}h1{font-size:18px}
label{display:block;margin:8px 0 2px;font-size:13px;color:#9aa4af}
input,select,textarea{width:100%;padding:8px;border:1px solid #30363d;border-radius:6px;background:#161b22;color:#e6edf3;box-sizing:border-box}
.row{display:flex;gap:10px}.row>div{flex:1}
button{padding:9px 14px;border:0;border-radius:6px;cursor:pointer;font-weight:600;margin-top:10px}
.run{background:#238636;color:#fff}.stop{background:#da3633;color:#fff}.pre{background:#1f6feb;color:#fff}
pre{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px;height:260px;overflow:auto;font-size:12px}
img{max-width:100%;border:1px solid #30363d;border-radius:6px}.muted{color:#9aa4af;font-size:12px}
.ok{color:#3fb950}.bad{color:#f85149}</style></head><body><div class=wrap>
<h1>FR3 · MolmoBot console</h1>
<div class=row><div>
  <label>Policy</label><select id=policy></select>
  <label>Task instruction</label><textarea id=task rows=2></textarea>
  <div class=row><div><label>Model host</label><input id=mh></div>
  <div><label>Model port</label><input id=mp></div></div>
  <label>Front camera</label><select id=cam>
    <option>exo_front</option><option>exo_left</option><option>exo_right</option></select>
  <label>Extra client flags (advanced)</label><input id=extra placeholder="--step_confirm --no_gripper --max_joint_delta 0.05">
  <div class=row>
    <button class=pre onclick=preflight()>Preflight</button>
    <button class=run onclick=run()>Run</button>
    <button class=stop onclick=stop()>Stop</button>
  </div>
  <div id=pf class=muted></div>
</div><div>
  <label>Live frame</label><img id=frame src="" alt="(no frame)">
  <div id=stat class=muted></div>
</div></div>
<label>Log</label><pre id=log></pre>
</div><script>
async function j(u,o){const r=await fetch(u,o);return r.json()}
async function init(){const d=await j('/api/policies');const s=document.getElementById('policy');
 for(const k in d.policies){const o=document.createElement('option');o.value=k;o.text=d.policies[k];s.add(o)}
 document.getElementById('mh').value=d.defaults.model_host;document.getElementById('mp').value=d.defaults.model_port;
 document.getElementById('task').value='Pick up the pineapple slices can';}
async function preflight(){const mh=mhv(),mp=mpv();
 const d=await j(`/api/preflight?model_host=${mh}&model_port=${mp}`);
 let h='';for(const k of ['arm','joint_state','gripper','model']){const v=d[k];
  h+=`<div class=${v.ok?'ok':'bad'}>${v.ok?'[ok]':'[FAIL]'} ${k}: ${JSON.stringify(v.detail).slice(0,120)}</div>`}
 document.getElementById('pf').innerHTML=h;}
function mhv(){return document.getElementById('mh').value}
function mpv(){return document.getElementById('mp').value}
async function run(){const b={policy:policy.value,task:task.value,model_host:mhv(),model_port:mpv(),
  front_view_cam_id:cam.value,extra_flags:extra.value};
 const d=await j('/api/run',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)});
 document.getElementById('pf').innerText=d.ok?('running pid '+d.pid):('ERROR: '+d.error);}
async function stop(){await j('/api/stop',{method:'POST'});}
async function poll(){try{const s=await j('/api/status');
 document.getElementById('stat').innerText=(s.running?'● running ':'○ idle ')+(s.policy||'')+(s.returncode!=null?(' rc='+s.returncode):'');
 const l=await j('/api/log');document.getElementById('log').innerText=l.log;
 const el=document.getElementById('log');el.scrollTop=el.scrollHeight;
 if(s.running){document.getElementById('frame').src='/api/frame?'+Date.now();}}catch(e){}}
init();setInterval(poll,1000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7070)
    args = ap.parse_args()
    print(f"[console] http://{args.host}:{args.port}   model default {MODEL_HOST}:{MODEL_PORT}")
    print(f"[console] arm={FRANKY_URL}  gripper={GRIPPER_BACKEND}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
